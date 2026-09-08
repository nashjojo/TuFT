from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from opentelemetry.trace import StatusCode
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from tinker import types

from .backends import BaseSamplingBackend
from .checkpoints import CheckpointRecord
from .config import AppConfig, ModelConfig
from .exceptions import (
    CheckpointAccessDeniedException,
    CheckpointException,
    CheckpointNotFoundException,
    MissingSequenceIDException,
    SessionNotFoundException,
    UnknownModelException,
    UserMismatchException,
)
from .persistence import (
    get_redis_store,
    is_persistence_enabled,
    load_record,
    save_record,
)
from .sequence_executor import SequenceExecutor
from .telemetry.metrics import get_metrics
from .telemetry.tracing import get_tracer


_get_tracer = lambda: get_tracer("tuft.sampling_controller")  # noqa: E731


logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Throttle for sampling-session persistence. History entries are diagnostics
# (prompt hash/count per request); persisting them on every asample made the
# per-request cost O(session length) — quadratic over an RL run with tens of
# thousands of samples in one session. 5s loses at most a few entries on a crash.
_SESSION_SAVE_INTERVAL_SEC = 5.0


class SamplingHistoryEntry(BaseModel):
    """Entry in the sampling history."""

    seq_id: int
    prompt_token_count: int
    prompt_hash: str
    created_at: datetime = Field(default_factory=_now)


class SamplingSessionRecord(BaseModel):
    """Sampling session record with persistence support.

    Sessions are permanent records (no TTL) as they represent active
    sampling sessions that users may access at any time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sampling_session_id: str
    session_id: str
    model_id: str
    base_model: str
    user_id: str
    model_path: str | None = None
    session_seq_id: int
    last_seq_id: int = -1
    history: list[SamplingHistoryEntry] = Field(default_factory=list)
    executor: SequenceExecutor = Field(default_factory=SequenceExecutor, exclude=True)
    training_mode: str = "lora"
    checkpoint_id: str | None = None

    _history_by_seq_id: dict[int, SamplingHistoryEntry] = PrivateAttr(default_factory=dict)
    _last_session_save: float = PrivateAttr(default=0.0)


class SamplingController:
    """Manages sampling sessions and connects them to the correct training or base-model backend."""

    REDIS_KEY_PREFIX = "sampling_session"

    def __init__(self, config: AppConfig, training_controller: object | None = None) -> None:
        self.config = config
        self.sampling_sessions: Dict[str, SamplingSessionRecord] = {}
        self._base_backends: Dict[str, BaseSamplingBackend] = self._create_backends(
            config.supported_models
        )
        self._active_full_deploy_time: Dict[str, datetime] = {}
        self._training_controller = training_controller
        self._restore_from_redis()

    def _build_key(self, session_id: str) -> str:
        return get_redis_store().build_key(self.REDIS_KEY_PREFIX, session_id)

    async def reset_full_deployment(self, base_model: str) -> None:
        """Revert a base model's replicas to base weights after its run ends.

        While full-param weights are deployed, base-model sampling sessions are
        rejected so clients cannot silently get the wrong revision. That guard
        must be lifted when the owning run goes away, otherwise the next run
        cannot create its initial base-model session.
        """
        backend = self._base_backends.get(base_model)
        if backend is None:
            return
        revert: Optional[Callable[[], Awaitable[None]]] = getattr(
            backend, "revert_to_base_weights", None
        )
        if revert is not None:
            await revert()
        self._active_full_deploy_time.pop(base_model, None)

    def _rebuild_history_index(self, record: SamplingSessionRecord) -> None:
        history_by_seq_id: dict[int, SamplingHistoryEntry] = {}
        for entry in record.history:
            history_by_seq_id[entry.seq_id] = entry
        record._history_by_seq_id = history_by_seq_id
        if history_by_seq_id:
            record.last_seq_id = max(record.last_seq_id, max(history_by_seq_id))

    def _restore_from_redis(self) -> None:
        """Restore sampling sessions from Redis on startup."""
        if not is_persistence_enabled():
            return
        store = get_redis_store()
        pattern = store.build_key(self.REDIS_KEY_PREFIX, "*")
        invalid_sessions = []
        for key in store.keys(pattern):
            # Match only top-level sessions (3 parts)
            if len(key.split("::")) != 3:
                continue
            record = load_record(key, SamplingSessionRecord)
            if record is None:
                continue
            if record.base_model and record.base_model not in self._base_backends:
                invalid_sessions.append(record.sampling_session_id)
                continue
            record.executor = SequenceExecutor()
            self._rebuild_history_index(record)
            self.sampling_sessions[record.sampling_session_id] = record
        for session_id in invalid_sessions:
            store.delete(self._build_key(session_id))

    def _save_session(self, session_id: str) -> None:
        """Save session to Redis (no TTL - permanent record)."""
        if not is_persistence_enabled():
            return
        record = self.sampling_sessions.get(session_id)
        if record is not None:
            save_record(self._build_key(session_id), record)

    def _delete_session(self, session_id: str) -> None:
        if not is_persistence_enabled():
            return
        get_redis_store().delete(self._build_key(session_id))

    async def async_init(self) -> None:
        """Perform any async initialization here, including adapter reloading."""
        init_tasks = [backend.async_init() for backend in self._base_backends.values()]
        await asyncio.gather(*init_tasks)

        # Re-add adapters for restored sessions
        await self._rebuild_sampling_backends()

    async def shutdown(self) -> None:
        """Shut down all sampling backends and release GPU/Ray resources."""
        for backend in self._base_backends.values():
            try:
                await backend.shutdown()
            except Exception:
                logger.exception("Failed to shut down sampling backend for %s", backend.base_model)

    async def _rebuild_sampling_backends(self) -> None:
        """Rebuild sampling backends for restored sessions."""
        invalid_sessions = []
        for session_id, record in list(self.sampling_sessions.items()):
            if record.model_path and record.base_model:
                adapter_path = Path(record.model_path)
                if not adapter_path.exists():
                    invalid_sessions.append(session_id)
                    continue
                if record.base_model not in self._base_backends:
                    invalid_sessions.append(session_id)
                    continue
                try:
                    backend = self._base_backends[record.base_model]
                    await backend.add_adapter(
                        lora_id=record.sampling_session_id, adapter_path=adapter_path
                    )
                except Exception:
                    logger.exception(
                        "Failed to rebuild adapter for sampling session %s", session_id
                    )
                    invalid_sessions.append(session_id)
        for session_id in invalid_sessions:
            del self.sampling_sessions[session_id]
            self._delete_session(session_id)

    def _create_backends(self, model_configs: List[ModelConfig]) -> Dict[str, BaseSamplingBackend]:
        backends: Dict[str, BaseSamplingBackend] = {}
        for config in model_configs:
            backends[config.model_name] = BaseSamplingBackend.create_backend(
                config, worker_venv_path=self.config.worker_venv_path
            )
        return backends

    async def create_sampling_session(
        self,
        *,
        session_id: str,
        user_id: str,
        base_model: str | None,
        model_path: str | None,
        session_seq_id: int,
    ) -> str:
        base_model_ref: str | None = None
        adapter_path: Path | None = None
        sampling_session_id = str(uuid.uuid4())
        deployment_id: Optional[str] = None

        with _get_tracer().start_as_current_span(
            "sampling_controller.create_sampling_session"
        ) as span:
            span.set_attribute("tuft.session_id", session_id)
            span.set_attribute("tuft.sampling_session_id", sampling_session_id)
            if base_model:
                span.set_attribute("tuft.base_model", base_model)
            try:
                if model_path:
                    # model_path should have higher priority than base_model
                    try:
                        assert self.config.checkpoint_dir is not None
                        parsed_checkpoint = CheckpointRecord.from_tinker_path(
                            model_path,
                            self.config.checkpoint_dir,
                        )
                    except FileNotFoundError as exc:
                        raise CheckpointNotFoundException(checkpoint_id=model_path) from exc
                    if not parsed_checkpoint.path.exists():
                        raise CheckpointNotFoundException(
                            checkpoint_id=parsed_checkpoint.checkpoint_id,
                        )
                    metadata = parsed_checkpoint.metadata
                    base_model_ref = metadata.base_model
                    is_public = parsed_checkpoint.public
                    model_owner = parsed_checkpoint.owner_name
                    if not is_public and model_owner != user_id:
                        raise CheckpointAccessDeniedException(
                            checkpoint_id=parsed_checkpoint.checkpoint_id,
                        )
                    if base_model_ref not in self._base_backends:
                        raise UnknownModelException(model_name=base_model_ref)
                    sampling_backend = self._base_backends[base_model_ref]
                    checkpoint_mode = metadata.training_mode
                    deployment_id = parsed_checkpoint.tinker_path
                    if checkpoint_mode == "full_param":
                        # A checkpoint from a different training run must never be
                        # served: rollout would sample one run's weights while
                        # training updates another's, silently breaking the RL
                        # loop. The timestamp guard below cannot catch this on its
                        # own because _active_full_deploy_time is in-memory and
                        # empty after a restart, so validate run ownership against
                        # the training runs this server knows about, which are
                        # restored from Redis and therefore survive restarts.
                        known_runs = getattr(self._training_controller, "training_runs", None)
                        if known_runs is not None and (
                            parsed_checkpoint.training_run_id not in known_runs
                        ):
                            raise CheckpointException(
                                status_code=409,
                                detail=(
                                    f"Checkpoint {parsed_checkpoint.checkpoint_id} belongs to "
                                    f"training run {parsed_checkpoint.training_run_id}, which is "
                                    "not an active training run on this server; refusing to "
                                    f"deploy weights for {base_model_ref} from a foreign run."
                                ),
                                checkpoint_id=parsed_checkpoint.checkpoint_id,
                            )
                        # Latest checkpoint wins. Only a stale checkpoint is
                        # refused, because deploying it would roll the served
                        # weights backward; a newer one must be allowed through.
                        active_time = self._active_full_deploy_time.get(base_model_ref)
                        if active_time is not None and parsed_checkpoint.created_at < active_time:
                            raise CheckpointException(
                                status_code=409,
                                detail=(
                                    f"Checkpoint {parsed_checkpoint.checkpoint_id} is older "
                                    "than the full-param weights currently deployed for "
                                    f"{base_model_ref}; refusing to roll weights backward."
                                ),
                                checkpoint_id=parsed_checkpoint.checkpoint_id,
                            )
                        await sampling_backend.deploy_full_weights(
                            parsed_checkpoint.model_path, deployment_id
                        )
                        self._active_full_deploy_time[base_model_ref] = parsed_checkpoint.created_at
                    else:
                        active_id = getattr(sampling_backend, "get_active_deployment_id", None)
                        if callable(active_id) and active_id() is not None:
                            raise CheckpointAccessDeniedException(
                                checkpoint_id=parsed_checkpoint.checkpoint_id
                            )
                        await sampling_backend.add_adapter(
                            lora_id=sampling_session_id, adapter_path=parsed_checkpoint.adapter_path
                        )
                        adapter_path = parsed_checkpoint.adapter_path
                    # TODO: remove adapter when session is deleted
                elif base_model:
                    base_model_ref = base_model
                    if base_model_ref not in self._base_backends:
                        raise UnknownModelException(model_name=base_model_ref)
                    sampling_backend = self._base_backends[base_model_ref]
                    active_id = getattr(sampling_backend, "get_active_deployment_id", None)
                    if callable(active_id) and active_id() is not None:
                        raise CheckpointAccessDeniedException(checkpoint_id="base-model")
                    checkpoint_mode = "lora"
                else:
                    raise UnknownModelException(model_name="None")
                self.sampling_sessions[sampling_session_id] = SamplingSessionRecord(
                    sampling_session_id=sampling_session_id,
                    session_id=session_id,
                    user_id=user_id,
                    model_id=sampling_session_id,
                    base_model=base_model_ref,
                    model_path=str(adapter_path) if adapter_path else None,
                    session_seq_id=session_seq_id,
                    training_mode=checkpoint_mode,
                    checkpoint_id=deployment_id if model_path else None,
                )
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._save_session, sampling_session_id)

                # Update metrics
                get_metrics().sampling_sessions_active.add(1, {"base_model": base_model_ref or ""})
                logger.info("Sampling session created: %s", sampling_session_id)
                return sampling_session_id
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR)
                raise

    def _hash_prompt(self, prompt: types.ModelInput) -> str:
        # array.to_bytes is one C-level pass; the previous
        # ",".join(str(t) for t in ...) ran a Python-level generator over
        # every token and dominated per-request event-loop cost.
        import array

        buf = array.array("q", prompt.to_ints())
        return hashlib.sha1(buf.tobytes()).hexdigest()[:16]

    async def _record_sequence(
        self, record: SamplingSessionRecord, seq_id: int, prompt: types.ModelInput
    ) -> None:
        record._history_by_seq_id[seq_id] = SamplingHistoryEntry(
            seq_id=seq_id,
            prompt_token_count=prompt.length,
            prompt_hash=self._hash_prompt(prompt),
        )
        record.last_seq_id = max(record.last_seq_id, seq_id)

        now = time.monotonic()
        if now - record._last_session_save < _SESSION_SAVE_INTERVAL_SEC:
            return
        record._last_session_save = now
        # Materialize the sorted history only at (throttled) save time.
        # Rebuilding it on every request was O(n log n) per asample and
        # O(n^2) per session — with RL sessions reaching tens of thousands
        # of samples this alone pegged the single event loop.
        record.history = [record._history_by_seq_id[k] for k in sorted(record._history_by_seq_id)]

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_session, record.sampling_session_id)

    async def _resolve_backend(
        self, request: types.SampleRequest, user_id: str
    ) -> Tuple[BaseSamplingBackend, str | None]:
        """Resolve the appropriate backend for the sampling request.

        Args:
            request: The sampling request.

        Returns:
            A tuple of the resolved backend and the LoRA ID if applicable.
        """
        if request.sampling_session_id:
            record = self.sampling_sessions.get(request.sampling_session_id)
            if record is None:
                raise SessionNotFoundException(session_id=request.sampling_session_id)
            if record.user_id != user_id:
                raise UserMismatchException()
            if request.seq_id is None:
                raise MissingSequenceIDException()

            await record.executor.submit(
                sequence_id=request.seq_id,
                func=self._record_sequence,
                record=record,
                seq_id=request.seq_id,
                prompt=request.prompt,
            )
            if record.base_model not in self._base_backends:
                raise UnknownModelException(model_name=record.base_model)
            if record.model_path is None or record.training_mode == "full_param":
                lora_id = None
            else:
                lora_id = record.sampling_session_id
            return self._base_backends[record.base_model], lora_id
        raise SessionNotFoundException(session_id="None")

    async def run_sample(
        self,
        request: types.SampleRequest,
        user_id: str,
    ) -> types.SampleResponse:
        with _get_tracer().start_as_current_span("sampling_controller.run_sample") as span:
            sampling_session_id = request.sampling_session_id or ""
            span.set_attribute("tuft.sampling_session_id", sampling_session_id)
            # Get session_id from sampling session record if available
            if request.sampling_session_id:
                record = self.sampling_sessions.get(request.sampling_session_id)
                if record:
                    span.set_attribute("tuft.session_id", record.session_id)
            span.set_attribute("tuft.num_samples", request.num_samples)

            logger.info("Sampling begin for %s", sampling_session_id)
            start_time = time.perf_counter()

            backend, lora_id = await self._resolve_backend(request, user_id=user_id)
            await backend.wait_ready()
            prompt = request.prompt
            sampling_params = request.sampling_params
            num_samples = request.num_samples
            include_prompt_logprobs = bool(request.prompt_logprobs)
            topk_prompt_logprobs = request.topk_prompt_logprobs or 0

            response = await backend.sample(
                prompt=prompt,
                num_samples=num_samples,
                sampling_params=sampling_params,
                include_prompt_logprobs=include_prompt_logprobs,
                topk_prompt_logprobs=topk_prompt_logprobs,
                lora_id=lora_id,
            )

            duration = time.perf_counter() - start_time
            logger.info("Sampling completed for %s", sampling_session_id)

            # Get base_model for metrics
            record = self.sampling_sessions.get(request.sampling_session_id or "")
            base_model = record.base_model if record else ""

            # Update metrics
            metrics = get_metrics()
            metrics.sampling_requests.add(1, {"base_model": base_model})
            metrics.sampling_duration.record(duration, {"base_model": base_model})

            # Record output tokens for each sequence
            total_output_tokens = 0
            for seq in response.sequences:
                if seq.tokens:
                    metrics.sampling_output_tokens.record(len(seq.tokens))
                    total_output_tokens += len(seq.tokens)

            # Record tokens per second if we have output tokens and positive duration
            if total_output_tokens > 0 and duration > 0:
                tokens_per_second = total_output_tokens / duration
                metrics.sampling_tokens_per_second.record(
                    tokens_per_second, {"base_model": base_model}
                )

            return response

    async def evict_model(self, model_id: str, user_id: str) -> None:
        for sampling_id, record in list(self.sampling_sessions.items()):
            if record.model_id == model_id and record.user_id == user_id:
                base_model = record.base_model
                del self.sampling_sessions[sampling_id]
                self._delete_session(sampling_id)
                # Update metrics
                get_metrics().sampling_sessions_active.add(-1, {"base_model": base_model or ""})

    def get_sampler_info(
        self, sampler_id: str, user_id: str, default_base_model: str
    ) -> types.GetSamplerResponse:
        record = self.sampling_sessions.get(sampler_id)
        if record is None:
            raise SessionNotFoundException(session_id=sampler_id)
        if record.user_id != user_id:
            raise UserMismatchException()
        base = record.base_model
        return types.GetSamplerResponse(
            sampler_id=sampler_id,
            base_model=base or default_base_model,
            model_path=record.model_path,
        )
