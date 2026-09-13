"""Training controller for managing training runs and routing requests."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Dict, List, Optional, TypeVar

from opentelemetry.trace import StatusCode
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from tinker import types

from .backends import BaseTrainingBackend
from .checkpoints import CheckpointRecord, compute_tree_size
from .config import AppConfig, ModelConfig
from .exceptions import (
    CheckpointAccessDeniedException,
    CheckpointMetadataReadException,
    CheckpointNotFoundException,
    SequenceConflictException,
    UnknownModelException,
    UserMismatchException,
)
from .persistence import (
    delete_record,
    get_redis_store,
    is_persistence_enabled,
    load_record,
    save_record,
    save_records_atomic,
)
from .telemetry.metrics import get_metrics
from .telemetry.tracing import get_tracer


_get_tracer = lambda: get_tracer("tuft.training_controller")  # noqa: E731


logger = logging.getLogger(__name__)

T = TypeVar("T")


# How long an out-of-order request waits for its missing lower seq_ids before
# falling back to the gap/fast-forward semantics (covers a client draining a
# concurrent submission window; bounded so a lost request cannot stall a run).
_SEQ_WAIT_TIMEOUT_S = 180.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TrainingRunRecord(BaseModel):
    """Training run record with persistence support.

    Runtime-only fields (backend, _execution_lock) are excluded from serialization.
    Checkpoints are stored separately with their own keys.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    training_run_id: str
    base_model: str
    lora_rank: int | None = None
    session_id: str
    model_owner: str
    user_metadata: dict[str, str] | None = None
    training_mode: str = "lora"
    created_at: datetime = Field(default_factory=_now)
    last_request_time: datetime = Field(default_factory=_now)
    # Checkpoints are stored separately, excluded from serialization
    checkpoints: Dict[str, CheckpointRecord] = Field(default_factory=dict, exclude=True)
    sampler_checkpoints: Dict[str, CheckpointRecord] = Field(default_factory=dict, exclude=True)
    next_training_checkpoint: int = 1
    next_sampler_checkpoint: int = 1
    corrupted: bool = False
    next_seq_id: int = 1
    # Runtime-only fields, excluded from serialization
    backend: BaseTrainingBackend | None = Field(default=None, exclude=True)
    # Private attribute for execution lock (not a model field)
    _execution_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    # Notifies seq_id waiters when next_seq_id advances (concurrent submissions)
    _seq_cond: asyncio.Condition = PrivateAttr(default_factory=asyncio.Condition)

    def to_training_run(self) -> types.TrainingRun:
        training_checkpoint = self._latest_checkpoint(self.checkpoints)
        sampler_checkpoint = self._latest_checkpoint(self.sampler_checkpoints)
        return types.TrainingRun(
            training_run_id=self.training_run_id,
            base_model=self.base_model,
            model_owner=self.model_owner,
            is_lora=self.training_mode == "lora",
            corrupted=self.corrupted,
            lora_rank=self.lora_rank,
            last_request_time=self.last_request_time,
            last_checkpoint=training_checkpoint,
            last_sampler_checkpoint=sampler_checkpoint,
            user_metadata=self.user_metadata,
        )

    def _latest_checkpoint(self, items: Dict[str, CheckpointRecord]) -> types.Checkpoint | None:
        if not items:
            return None
        latest = max(items.values(), key=lambda record: record.created_at)
        return latest.tinker_checkpoint


class TrainingController:
    """Tracks training runs, enforces request ordering.

    Routes work into ModelBackend instances.
    """

    REDIS_KEY_PREFIX = "training_run"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.training_mode = os.getenv("TUFT_TRAINING_MODE", "lora")
        if self.training_mode not in {"lora", "full_param"}:
            raise ValueError(
                f"TUFT_TRAINING_MODE must be 'lora' or 'full_param', got '{self.training_mode}'"
            )
        self.training_backends = self._create_backends(config.supported_models)
        # TODO: add a mechanism to manage training_runs
        self.training_runs: Dict[str, TrainingRunRecord] = {}
        # Set by ServerState to SamplingController.reset_full_deployment. Kept
        # optional because TrainingController is constructed first and tests
        # build it standalone.
        self.on_full_param_run_unloaded: Optional[Callable[[str], Awaitable[None]]] = None
        self._restore_from_redis()

    def _create_backends(self, model_configs: List[ModelConfig]) -> Dict[str, BaseTrainingBackend]:
        backends: Dict[str, BaseTrainingBackend] = {}
        # FSDP port allocation: 29500, 29501, ... by order of FSDP models in supported_models
        fsdp_model_names = [
            c.model_name for c in model_configs if getattr(c, "training_backend", "hf") == "fsdp"
        ]
        for config in model_configs:
            fsdp_index: Optional[int] = None
            if config.model_name in fsdp_model_names:
                fsdp_index = fsdp_model_names.index(config.model_name)
            backends[config.model_name] = BaseTrainingBackend.create_backend(
                config,
                fsdp_index=fsdp_index,
                worker_venv_path=self.config.worker_venv_path,
                training_mode=self.training_mode,
            )
        return backends

    async def shutdown(self) -> None:
        """Shut down all training backends and release GPU/Ray resources."""
        for backend in self.training_backends.values():
            try:
                await backend.shutdown()
            except Exception:
                logger.exception("Failed to shut down training backend for %s", backend.base_model)

    def _build_key(self, model_id: str) -> str:
        return get_redis_store().build_key(self.REDIS_KEY_PREFIX, model_id)

    def _build_checkpoint_key(self, model_id: str, checkpoint_id: str) -> str:
        return get_redis_store().build_key(self.REDIS_KEY_PREFIX, model_id, "ckpt", checkpoint_id)

    def _build_sampler_checkpoint_key(self, model_id: str, checkpoint_id: str) -> str:
        return get_redis_store().build_key(
            self.REDIS_KEY_PREFIX, model_id, "sampler_ckpt", checkpoint_id
        )

    def _restore_from_redis(self) -> None:
        """Restore training runs from Redis on startup."""
        if not is_persistence_enabled():
            return
        store = get_redis_store()
        # Match only top-level training runs (3 parts: namespace::prefix::model_id)
        for key in store.keys(store.build_key(self.REDIS_KEY_PREFIX, "*")):
            parts = key.split("::")
            if len(parts) != 3:
                continue
            record = load_record(key, TrainingRunRecord)
            if record is None:
                continue
            model_id = record.training_run_id
            # Restore checkpoints (stored separately, not subject to TTL)
            self._restore_checkpoints(model_id, record)
            self._restore_sampler_checkpoints(model_id, record)
            # A process is either LoRA or full-param; leave the other mode's
            # records persisted but unbound so they are not loaded into the
            # wrong backend or marked corrupted.
            if record.training_mode != self.training_mode:
                continue
            # Restore backend reference
            if record.base_model in self.training_backends:
                record.backend = self.training_backends[record.base_model]
            else:
                record.corrupted = True
            self.training_runs[model_id] = record

    def _restore_checkpoints(self, model_id: str, record: TrainingRunRecord) -> None:
        store = get_redis_store()
        pattern = self._build_checkpoint_key(model_id, "*")
        record.checkpoints = {}
        for key in store.keys(pattern):
            ckpt = load_record(key, CheckpointRecord)
            if ckpt is not None:
                record.checkpoints[ckpt.checkpoint_id] = ckpt

    def _restore_sampler_checkpoints(self, model_id: str, record: TrainingRunRecord) -> None:
        store = get_redis_store()
        pattern = self._build_sampler_checkpoint_key(model_id, "*")
        record.sampler_checkpoints = {}
        for key in store.keys(pattern):
            ckpt = load_record(key, CheckpointRecord)
            if ckpt is not None:
                record.sampler_checkpoints[ckpt.checkpoint_id] = ckpt

    def _save_training_run(self, model_id: str) -> None:
        """Save training run to Redis (no TTL - permanent record)."""
        if not is_persistence_enabled():
            return
        record = self.training_runs.get(model_id)
        if record is not None:
            save_record(self._build_key(model_id), record)

    def _save_checkpoint(self, model_id: str, checkpoint_id: str) -> None:
        """Save checkpoint to Redis (no TTL - permanent record)."""
        if not is_persistence_enabled():
            return
        record = self.training_runs.get(model_id)
        if record is not None:
            ckpt = record.checkpoints.get(checkpoint_id)
            if ckpt is not None:
                save_record(self._build_checkpoint_key(model_id, checkpoint_id), ckpt)

    def _save_sampler_checkpoint(self, model_id: str, checkpoint_id: str) -> None:
        """Save sampler checkpoint to Redis (no TTL - permanent record)."""
        if not is_persistence_enabled():
            return
        record = self.training_runs.get(model_id)
        if record is not None:
            ckpt = record.sampler_checkpoints.get(checkpoint_id)
            if ckpt is not None:
                save_record(self._build_sampler_checkpoint_key(model_id, checkpoint_id), ckpt)

    def _save_training_run_with_checkpoint(
        self, model_id: str, checkpoint_id: str, checkpoint_type: types.CheckpointType
    ) -> None:
        """Save training run and checkpoint atomically using Redis transaction.

        This ensures consistency if the server crashes between saves.
        No TTL is used for these records as they are permanent.
        """
        if not is_persistence_enabled():
            return
        record = self.training_runs.get(model_id)
        if record is None:
            return

        if checkpoint_type == "training":
            ckpt = record.checkpoints.get(checkpoint_id)
            ckpt_key = self._build_checkpoint_key(model_id, checkpoint_id)
        else:
            ckpt = record.sampler_checkpoints.get(checkpoint_id)
            ckpt_key = self._build_sampler_checkpoint_key(model_id, checkpoint_id)

        if ckpt is None:
            # Defensive fallback: checkpoint should exist at this point since
            # _save_training_run_with_checkpoint is called after adding the checkpoint
            # to the target_map. This branch handles unexpected edge cases (e.g., code
            # refactoring that changes call order) to ensure the training run is still
            # persisted even if the checkpoint lookup fails.
            logger.warning(
                "Checkpoint %s not found for model %s during persistence, "
                "saving training run without checkpoint",
                checkpoint_id,
                model_id,
            )
            save_record(self._build_key(model_id), record)
            return

        # Save both atomically (no TTL for permanent records)
        save_records_atomic(
            [
                (self._build_key(model_id), record),
                (ckpt_key, ckpt),
            ]
        )

    def _delete_training_run(self, model_id: str) -> None:
        if not is_persistence_enabled():
            return
        store = get_redis_store()
        store.delete(self._build_key(model_id))
        store.delete_pattern(self._build_checkpoint_key(model_id, "*"))
        store.delete_pattern(self._build_sampler_checkpoint_key(model_id, "*"))

    def _delete_checkpoint_record(self, model_id: str, checkpoint_id: str) -> None:
        if not is_persistence_enabled():
            return
        delete_record(self._build_checkpoint_key(model_id, checkpoint_id))

    def _delete_sampler_checkpoint_record(self, model_id: str, checkpoint_id: str) -> None:
        if not is_persistence_enabled():
            return
        delete_record(self._build_sampler_checkpoint_key(model_id, checkpoint_id))

    async def _with_sequence_guard(
        self,
        record: TrainingRunRecord,
        seq_id: int | None,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        if seq_id is not None and seq_id > record.next_seq_id:
            # Out-of-order arrival (e.g. a client submitting chunks concurrently)
            # with missing lower seq_ids: wait for the predecessors so they can
            # execute first. Without this, the fast path below fast-forwards past
            # the gap and later rejects the late lower seq_ids as conflicts.
            deadline = time.monotonic() + _SEQ_WAIT_TIMEOUT_S
            async with record._seq_cond:
                while seq_id > record.next_seq_id:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break  # fall through to the gap/fast-forward path
                    try:
                        await asyncio.wait_for(record._seq_cond.wait(), remaining)
                    except asyncio.TimeoutError:
                        break

        async with record._execution_lock:
            if seq_id is not None:
                expected = record.next_seq_id
                if seq_id < expected:
                    raise SequenceConflictException(expected=expected, got=seq_id)
                if seq_id > expected:
                    logger.warning(
                        "Sequence gap on training run %s: expected %s, got %s; "
                        "fast-forwarding (an earlier request must have failed)",
                        record.training_run_id,
                        expected,
                        seq_id,
                    )
                    record.next_seq_id = seq_id

            result = await operation()

            if seq_id is not None:
                record.next_seq_id += 1
                async with record._seq_cond:
                    record._seq_cond.notify_all()
            # Save the updated next_seq_id to Redis
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_training_run, record.training_run_id)
            return result

    async def create_model(
        self,
        session_id: str,
        base_model: str,
        lora_config: types.LoraConfig,
        model_owner: str,
        user_metadata: dict[str, str] | None,
        model_id: str | None = None,
    ) -> TrainingRunRecord:
        model_id = model_id or str(uuid.uuid4())
        requested_mode = (user_metadata or {}).get("training_mode", "lora")
        if requested_mode not in {"lora", "full_param"}:
            raise ValueError(f"Unknown training_mode '{requested_mode}'")
        if requested_mode != self.training_mode:
            raise ValueError(
                f"TuFT server is running in '{self.training_mode}' mode; "
                f"cannot create a '{requested_mode}' training run."
            )
        if requested_mode == "full_param":
            # One shared full-param model and optimizer means only one run can be
            # bound at a time. A resumed client arrives under a new session and
            # the SDK derives model_id from it, so it can never reuse the
            # existing run's id; release the previous binding and let the new run
            # take the slot. The old record stays registered because
            # load_checkpoint resolves checkpoints from the training_run_id
            # encoded in the tinker:// path, so the new run can still load it.
            for record in self.training_runs.values():
                if record.training_mode != "full_param" or record.model_owner != model_owner:
                    continue
                if record.training_run_id == model_id:
                    continue
                if record.backend is not None:
                    # release_run is FSDP-specific, not part of the base contract.
                    release = getattr(record.backend, "release_run", None)
                    if release is not None:
                        await release(record.training_run_id)
                record.backend = None
                logger.info(
                    "Released full-param run %s so %s can take the single slot",
                    record.training_run_id,
                    model_id,
                )
        with _get_tracer().start_as_current_span("training_controller.create_model") as span:
            span.set_attribute("tuft.training_run_id", model_id)
            span.set_attribute("tuft.session_id", session_id)
            span.set_attribute("tuft.base_model", base_model)
            span.set_attribute("tuft.training_mode", requested_mode)
            span.set_attribute("tuft.lora_rank", lora_config.rank)
            try:
                logger.info("Creating model %s", model_id)

                if base_model not in self.training_backends:
                    raise UnknownModelException(model_name=base_model)
                backend = self.training_backends[base_model]
                record = TrainingRunRecord(
                    training_run_id=model_id,
                    base_model=base_model,
                    # Keep the carrier rank even in full-param mode: restore
                    # rebuilds a LoraConfig from it, and is_lora comes from
                    # training_mode rather than from this field.
                    lora_rank=lora_config.rank,
                    session_id=session_id,
                    model_owner=model_owner,
                    user_metadata=user_metadata,
                    training_mode=requested_mode,
                    backend=backend,
                )
                await backend.create_adapter(model_id, lora_config)
                self.training_runs[model_id] = record
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._save_training_run, model_id)

                # Update metrics
                get_metrics().training_models_active.add(1, {"base_model": base_model})
                return record
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR)
                raise

    def get_run_record(
        self,
        model_id: str,
        user_id: str,
        enforce_user_match: bool = True,
    ) -> TrainingRunRecord:
        record = self.training_runs.get(model_id)
        if record is None:
            raise UnknownModelException(model_name=model_id)
        if enforce_user_match and record.model_owner != user_id:
            raise UserMismatchException()
        return record

    def build_supported_models(self) -> list[types.SupportedModel]:
        return [
            types.SupportedModel(model_name=model.model_name)
            for model in self.config.supported_models
        ]

    def update_activity(self, model_id: str, user_id: str) -> None:
        record = self.get_run_record(model_id, user_id)
        record.last_request_time = datetime.now(timezone.utc)
        self._save_training_run(model_id)

    async def run_forward(
        self,
        model_id: str,
        user_id: str,
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        seq_id: int | None,
        *,
        backward: bool,
    ) -> types.ForwardBackwardOutput:
        record = self.get_run_record(model_id, user_id)
        self.update_activity(model_id, user_id)

        span_name = (
            "training_controller.run_forward_backward"
            if backward
            else "training_controller.run_forward"
        )
        with _get_tracer().start_as_current_span(span_name) as span:
            span.set_attribute("tuft.training_run_id", model_id)
            span.set_attribute("tuft.session_id", record.session_id)
            span.set_attribute("tuft.backward", backward)
            span.set_attribute("tuft.data_count", len(data))
            span.set_attribute("tuft.loss_fn", loss_fn)

            logger.info("Forward/backward begin for %s", model_id)
            start_time = time.perf_counter()
            t_op_start: float | None = None

            # Count total input tokens for metrics
            total_tokens = sum(len(datum.model_input.to_ints()) for datum in data)

            async def _operation() -> types.ForwardBackwardOutput:
                nonlocal t_op_start
                t_op_start = time.perf_counter()
                if record.backend is None:
                    raise UnknownModelException(model_name=model_id)
                result = await record.backend.forward(
                    data,
                    lora_id=model_id,
                    loss_fn=loss_fn,
                    loss_fn_config=loss_fn_config,
                    backward=backward,
                )
                return result

            result = await self._with_sequence_guard(record, seq_id, _operation)

            # Record tokens per second metric
            duration = time.perf_counter() - start_time
            wait_s = (t_op_start - start_time) if t_op_start is not None else duration
            exec_s = duration - wait_s
            if total_tokens > 0 and duration > 0:
                tokens_per_second = total_tokens / duration
                get_metrics().training_tokens_per_second.record(
                    tokens_per_second, {"base_model": record.base_model}
                )
            logger.info(
                "[call-timing] backward=%s data=%d tokens=%d wait=%.3fs exec=%.3fs total=%.3fs",
                backward,
                len(data),
                total_tokens,
                wait_s,
                exec_s,
                duration,
            )

            return result

    async def run_optim_step(
        self, model_id: str, user_id: str, params: types.AdamParams, seq_id: int | None
    ) -> types.OptimStepResponse:
        record = self.get_run_record(model_id, user_id)
        self.update_activity(model_id, user_id)

        with _get_tracer().start_as_current_span("training_controller.run_optim_step") as span:
            span.set_attribute("tuft.training_run_id", model_id)
            span.set_attribute("tuft.session_id", record.session_id)
            span.set_attribute("tuft.learning_rate", params.learning_rate)

            logger.info("Optimizer step begin for %s", model_id)

            async def _operation() -> types.OptimStepResponse:
                if record.backend is None:
                    raise UnknownModelException(model_name=model_id)
                result = await record.backend.optim_step(adam_params=params, lora_id=model_id)
                logger.info("Optimizer step completed for %s", model_id)
                return result

            return await self._with_sequence_guard(record, seq_id, _operation)

    async def unload_model(self, model_id: str, user_id: str) -> None:
        # NOTE: unload removes the run record and its checkpoint registry
        # entries, so any `tinker://<run>/...` path under this run ID stops
        # resolving afterwards. Resuming after an unload (e.g. to force an
        # actor rebuild for a code refresh) therefore requires a checkpoint
        # that stays registered under a different, still-live run.
        # TODO: Ensure that all created training runs can be unloaded to reduce
        # GPU memory usage.
        if model_id not in self.training_runs:
            raise UnknownModelException(model_name=model_id)
        record = self.training_runs[model_id]
        if record.model_owner != user_id:
            raise UserMismatchException()
        base_model = record.base_model
        if record.backend is not None:
            await record.backend.remove_adapter(model_id)
            if record.training_mode == "full_param":
                await record.backend.shutdown()
        del self.training_runs[model_id]
        self._delete_training_run(model_id)

        if record.training_mode == "full_param" and self.on_full_param_run_unloaded is not None:
            await self.on_full_param_run_unloaded(base_model)

        # Update metrics
        get_metrics().training_models_active.add(-1, {"base_model": base_model})

    def list_training_runs(
        self, *, user_id: str, limit: int | None = None, offset: int = 0
    ) -> types.TrainingRunsResponse:
        runs = [
            record.to_training_run()
            for record in self.training_runs.values()
            if record.model_owner == user_id
        ]
        runs.sort(key=lambda run: run.last_request_time, reverse=True)
        total = len(runs)
        start = min(offset, total)
        end = total if limit is None else min(start + limit, total)
        paged = runs[start:end]
        cursor = types.Cursor(offset=offset, limit=limit or total, total_count=total)
        return types.TrainingRunsResponse(training_runs=paged, cursor=cursor)

    def get_training_run_view(self, model_id: str, user_id: str) -> types.TrainingRun:
        record = self.get_run_record(model_id=model_id, user_id=user_id)
        return record.to_training_run()

    def get_model_info(self, model_id: str, user_id: str) -> types.GetInfoResponse:
        record = self.get_run_record(model_id=model_id, user_id=user_id)
        model_data = types.ModelData(
            arch="toy-transformer",
            model_name=record.base_model,
            tokenizer_id=record.base_model,
        )
        return types.GetInfoResponse(
            model_data=model_data,
            model_id=model_id,
            is_lora=record.training_mode == "lora",
            lora_rank=record.lora_rank,
            model_name=record.base_model,
        )

    async def save_checkpoint(
        self,
        model_id: str,
        user_id: str,
        name: str | None,
        checkpoint_type: types.CheckpointType,
        future_id: int = 0,
        seq_id: int | None = None,
    ) -> CheckpointRecord:
        """Save a checkpoint for the given training run."""
        training_run = self.get_run_record(model_id=model_id, user_id=user_id)

        with _get_tracer().start_as_current_span("training_controller.save_checkpoint") as span:
            span.set_attribute("tuft.training_run_id", model_id)
            span.set_attribute("tuft.session_id", training_run.session_id)
            span.set_attribute("tuft.checkpoint_type", checkpoint_type)

            async def _operation() -> CheckpointRecord:
                counter_attr = (
                    "next_training_checkpoint"
                    if checkpoint_type == "training"
                    else "next_sampler_checkpoint"
                )
                counter = getattr(training_run, counter_attr)
                checkpoint_name = name or f"checkpoint-{counter:04d}"
                checkpoint_id = f"{model_id}/{checkpoint_name}"
                logger.info("Checkpoint save begin: %s", checkpoint_id)

                setattr(training_run, counter_attr, counter + 1)
                assert self.config.checkpoint_dir is not None
                checkpoint = CheckpointRecord.from_training_run(
                    training_run_id=training_run.training_run_id,
                    checkpoint_name=checkpoint_name,
                    owner_name=training_run.model_owner,
                    checkpoint_type=checkpoint_type,
                    checkpoint_root_dir=self.config.checkpoint_dir,
                    exist_ok=True,
                )
                checkpoint.future_id = future_id
                checkpoint.seq_id = seq_id
                target_map = (
                    training_run.checkpoints
                    if checkpoint_type == "training"
                    else training_run.sampler_checkpoints
                )
                if training_run.backend is not None:
                    await training_run.backend.save_state(
                        lora_id=training_run.training_run_id,
                        checkpoint_record=checkpoint,
                        optimizer=(checkpoint_type == "training"),
                    )

                # Write metadata once so metadata.json exists
                checkpoint.training_mode = training_run.training_mode
                checkpoint.save_metadata(
                    base_model=training_run.base_model,
                    session_id=training_run.session_id,
                    lora_rank=training_run.lora_rank,
                    training_mode=training_run.training_mode,
                )

                # Compute total size including metadata.json
                checkpoint.size_bytes = compute_tree_size(checkpoint.path)

                # Persist the correct size into metadata.json
                checkpoint.save_metadata(
                    base_model=training_run.base_model,
                    session_id=training_run.session_id,
                    lora_rank=training_run.lora_rank,
                    training_mode=training_run.training_mode,
                )
                # save the checkpoint record in the training run
                target_map[checkpoint_name] = checkpoint

                # Save training run and checkpoint atomically to prevent inconsistency
                # if server crashes between saves
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._save_training_run_with_checkpoint,
                    model_id,
                    checkpoint_name,
                    checkpoint_type,
                )

                # Update metrics
                metrics = get_metrics()
                metrics.training_checkpoints_saved.add(
                    1, {"model_id": model_id, "checkpoint_type": checkpoint_type}
                )
                logger.info("Checkpoint saved: %s", checkpoint_id)
                metrics.training_checkpoint_size.record(
                    checkpoint.size_bytes,
                    {"model_id": model_id, "checkpoint_type": checkpoint_type},
                )

                return checkpoint

            return await self._with_sequence_guard(training_run, seq_id, _operation)

    async def load_checkpoint(
        self,
        model_id: str,
        user_id: str,
        path: str,
        optimizer: bool,
        seq_id: int | None = None,
    ) -> None:
        """Load a checkpoint."""
        try:
            assert self.config.checkpoint_dir is not None
            parsed_checkpoint = CheckpointRecord.from_tinker_path(
                path,
                self.config.checkpoint_dir,
            )
        except FileNotFoundError as exc:
            raise CheckpointNotFoundException(checkpoint_id=model_id) from exc
        source_model_id = parsed_checkpoint.training_run_id or model_id
        source_run = self.get_run_record(source_model_id, user_id, enforce_user_match=False)

        collection = (
            source_run.checkpoints
            if parsed_checkpoint.checkpoint_type == "training"
            else source_run.sampler_checkpoints
        )

        checkpoint = collection.get(parsed_checkpoint.checkpoint_id)
        if checkpoint is None:
            raise CheckpointNotFoundException(checkpoint_id=parsed_checkpoint.checkpoint_id)
        try:
            metadata = checkpoint.metadata
        except FileNotFoundError as exc:
            raise CheckpointMetadataReadException(
                checkpoint_id=parsed_checkpoint.checkpoint_id
            ) from exc
        if metadata.public or (metadata.owner_name == user_id):
            # The checkpoint's source run locates the files and authorizes
            # access; the weights load into the CALLER's run when it is a
            # different, existing training run. That is the resume flow:
            # create_training_client_from_state creates a fresh run and then
            # loads a previous checkpoint into it — without this the weights
            # would land in the (possibly stale or slot-less) source run
            # while the caller trains on random initialization.
            if model_id in self.training_runs and model_id != source_model_id:
                target_run = self.get_run_record(model_id, user_id)
            else:
                target_run = source_run

            if target_run.backend is None:
                raise UnknownModelException(model_name=model_id)

            checkpoint_id = parsed_checkpoint.checkpoint_id
            logger.info(
                "Checkpoint load begin: %s into run %s", checkpoint_id, target_run.training_run_id
            )

            async def _operation() -> None:
                assert target_run.backend is not None
                await target_run.backend.load_state(
                    lora_id=target_run.training_run_id,
                    checkpoint_record=checkpoint,
                    optimizer=optimizer,
                )
                logger.info("Checkpoint loaded: %s", checkpoint_id)

            await self._with_sequence_guard(target_run, seq_id, _operation)
        else:
            raise CheckpointAccessDeniedException(checkpoint_id=parsed_checkpoint.checkpoint_id)

    def delete_checkpoint(self, model_id: str, user_id: str, checkpoint_id: str) -> None:
        training_run = self.get_run_record(model_id, user_id)
        removed = training_run.checkpoints.pop(checkpoint_id, None)
        is_sampler = False
        if removed is None:
            removed = training_run.sampler_checkpoints.pop(checkpoint_id, None)
            is_sampler = True
        if removed is None:
            raise CheckpointNotFoundException(checkpoint_id=checkpoint_id)
        removed.delete()

        self._save_training_run(model_id)
        if is_sampler:
            self._delete_sampler_checkpoint_record(model_id, checkpoint_id)
        else:
            self._delete_checkpoint_record(model_id, checkpoint_id)

    def list_checkpoints(self, model_id: str, user_id: str) -> list[types.Checkpoint]:
        training_run = self.get_run_record(model_id, user_id)
        checkpoints = [item.tinker_checkpoint for item in training_run.checkpoints.values()]
        checkpoints += [
            item.tinker_checkpoint for item in training_run.sampler_checkpoints.values()
        ]
        checkpoints.sort(key=lambda ckpt: ckpt.time)
        return checkpoints

    def list_user_checkpoints(
        self,
        user_id: str,
    ) -> list[types.Checkpoint]:
        checkpoints: list[types.Checkpoint] = []
        training_runs = [run for run in self.training_runs.values() if run.model_owner == user_id]
        for run in training_runs:
            checkpoints.extend([item.tinker_checkpoint for item in run.checkpoints.values()])
        checkpoints.sort(key=lambda item: item.time, reverse=True)
        return checkpoints

    def set_visibility(
        self, model_id: str, checkpoint_id: str, user_id: str, *, public: bool
    ) -> None:
        training_run = self.get_run_record(model_id=model_id, user_id=user_id)
        target = training_run.checkpoints.get(checkpoint_id)
        is_sampler = False
        if target is None:
            target = training_run.sampler_checkpoints.get(checkpoint_id)
            is_sampler = True
        if target is None:
            raise CheckpointNotFoundException(checkpoint_id=checkpoint_id)
        target.set_visibility(public)

        if is_sampler:
            self._save_sampler_checkpoint(model_id, checkpoint_id)
        else:
            self._save_checkpoint(model_id, checkpoint_id)

    def build_archive_url(
        self,
        model_id: str,
        user_id: str,
        checkpoint_id: str,
    ) -> types.CheckpointArchiveUrlResponse:
        training_run = self.get_run_record(model_id, user_id)
        checkpoint = training_run.checkpoints.get(
            checkpoint_id
        ) or training_run.sampler_checkpoints.get(checkpoint_id)
        if checkpoint is None:
            raise CheckpointNotFoundException(checkpoint_id=checkpoint_id)
        expires = datetime.now(timezone.utc) + timedelta(minutes=15)
        return types.CheckpointArchiveUrlResponse(url=checkpoint.path.as_uri(), expires=expires)

    def get_weights_info(self, model_id: str, user_id: str) -> types.WeightsInfoResponse:
        training_run = self.get_run_record(model_id, user_id)
        return types.WeightsInfoResponse(
            base_model=training_run.base_model,
            is_lora=training_run.training_mode == "lora",
            lora_rank=training_run.lora_rank,
        )

    def get_latest_checkpoint(self, model_id: str) -> CheckpointRecord | None:
        record = self.training_runs.get(model_id)
        if record is None:
            return None
        all_checkpoints = list(record.checkpoints.values()) + list(
            record.sampler_checkpoints.values()
        )
        if not all_checkpoints:
            return None
        return max(all_checkpoints, key=lambda c: c.created_at)

    async def restore_from_checkpoint(self, model_id: str) -> CheckpointRecord | None:
        record = self.training_runs.get(model_id)
        if record is None or record.backend is None:
            return None

        if record.training_mode == "full_param":
            # Opposite ordering from LoRA: the worker refuses load_state until
            # the run is bound, and only training checkpoints are loadable, so
            # bind first and resume from the latest training checkpoint.
            try:
                # Full-param workers ignore the rank; it is only a carrier so the
                # shared create_adapter signature stays uniform.
                await record.backend.create_adapter(
                    model_id, types.LoraConfig(rank=record.lora_rank or 1)
                )
            except Exception:  # pylint: disable=broad-except
                logger.exception("Failed to bind full-param run %s during restore", model_id)
            # Only training checkpoints are loadable in full-param mode: sampler
            # checkpoints are HF model dirs, not distributed-checkpoint state.
            if not record.checkpoints:
                return None
            resume_ckpt = max(record.checkpoints.values(), key=lambda c: c.created_at)
            try:
                await record.backend.load_state(
                    lora_id=model_id, checkpoint_record=resume_ckpt, optimizer=True
                )
            except Exception:  # pylint: disable=broad-except
                record.corrupted = True
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._save_training_run, model_id)
                logger.warning(
                    "Checkpoint load failed for %s; returning checkpoint "
                    "with future_id=%d for future cleanup",
                    model_id,
                    resume_ckpt.future_id,
                )
            return resume_ckpt

        latest_ckpt = self.get_latest_checkpoint(model_id)
        if latest_ckpt is None:
            return None
        # load_state calls load_adapter which creates the adapter from the
        # checkpoint on disk.  Calling create_adapter first causes PEFT's
        # load_adapter to fail with "adapter already exists" on HFTrainingBackend.
        # Only call create_adapter as a fallback if load_state fails.
        try:
            await record.backend.load_state(
                lora_id=model_id,
                checkpoint_record=latest_ckpt,
                optimizer=(latest_ckpt.checkpoint_type == "training"),
            )
        except Exception:  # pylint: disable=broad-except
            # load_state failed – try create_adapter + load_state as fallback
            logger.warning("load_state failed for %s, trying create_adapter fallback", model_id)
            try:
                rank = record.lora_rank
                if rank is None:
                    raise ValueError(f"LoRA run {model_id} has no recorded rank")
                await record.backend.create_adapter(model_id, types.LoraConfig(rank=rank))
            except Exception:
                logger.exception("Failed to create adapter for model %s during restore", model_id)
            try:
                await record.backend.load_state(
                    lora_id=model_id,
                    checkpoint_record=latest_ckpt,
                    optimizer=(latest_ckpt.checkpoint_type == "training"),
                )
            except Exception:
                # If loading still fails, mark as corrupted but still return
                # the checkpoint so that futures AFTER the checkpoint are
                # marked as failed (not ALL futures).
                record.corrupted = True
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._save_training_run, model_id)
                logger.warning(
                    "Checkpoint load failed for %s; returning checkpoint "
                    "with future_id=%d for future cleanup",
                    model_id,
                    latest_ckpt.future_id,
                )
                return latest_ckpt

        return latest_ckpt
