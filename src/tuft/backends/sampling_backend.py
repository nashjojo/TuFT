"""Sampling backend implemented using vLLM"""

import asyncio
import json
import os
import socket
import threading
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Optional

from opentelemetry.trace import StatusCode
from tinker import types

from ..config import ModelConfig
from ..telemetry.tracing import get_tracer
from .base_backend import BaseSamplingBackend
from .vllm_engine import VLLMEngine, VLLMEngineConfig


_get_tracer = lambda: get_tracer("tuft.sampling_backend")  # noqa: E731


logger = getLogger(__name__)


def _build_sample_response(
    req_output: Any,
    include_prompt_logprobs: bool = False,
    topk_prompt_logprobs: int = 0,
) -> types.SampleResponse:
    """Build a tinker 0.18.2 SampleResponse from a raw vLLM RequestOutput.

    The engine actor (``vllm_engine.VLLMEngine.generate``) returns vLLM's
    ``RequestOutput`` untouched; this function is the single adapter between
    vLLM's output format and tinker's frozen dataclasses (constructed via
    their list-based ``_tokens_list=`` / ``_logprobs_list=`` fields in
    tinker 0.18.2).
    """
    sequences: list[types.SampledSequence] = []
    topk_prompt_logprobs_list: list[list[tuple[int, float]] | None] = [None]
    prompt_logprobs: list[float | None] = [None]

    # collect prompt logprobs
    if include_prompt_logprobs:
        for logprob_dict in req_output.prompt_logprobs[1:]:
            prompt_logprobs.append(next(iter(logprob_dict.values())).logprob)
            if topk_prompt_logprobs > 0:
                logprob_items = sorted(logprob_dict.items(), key=lambda x: x[1].rank)
                topk = logprob_items[:topk_prompt_logprobs]
                topk_prompt_logprobs_list.append(
                    [(token_id, logprob.logprob) for token_id, logprob in topk]
                )

    # collect response sequences
    for seq_output in req_output.outputs:
        seq = types.SampledSequence(
            stop_reason="length" if seq_output.finish_reason == "length" else "stop",
            _tokens_list=seq_output.token_ids,
            _logprobs_list=[
                next(iter(logprob_dict.values())).logprob for logprob_dict in seq_output.logprobs
            ],
        )
        sequences.append(seq)

    return types.SampleResponse(
        sequences=sequences,
        _prompt_logprobs_list=prompt_logprobs if include_prompt_logprobs else None,
        _topk_prompt_logprobs_list=(
            topk_prompt_logprobs_list
            if include_prompt_logprobs and topk_prompt_logprobs > 0
            else None
        ),
    )


class VLLMSamplingBackend(BaseSamplingBackend):
    """A sampling backend using vLLM.

    User side `sample`, `sample_async`, `compute_logprobs` and
    `compute_logprobs_async` are all supported by the sample method.
    """

    def __init__(
        self,
        config: ModelConfig,
        worker_venv_path: Optional[str] = None,
        *,
        instance_index: int = 0,
    ) -> None:
        from vllm.lora.request import LoRARequest

        super().__init__(config)
        self._worker_venv_path = worker_venv_path
        self._instance_index = instance_index
        self.engine = self._create_engine(config)
        self.lora_adapters: dict[str, LoRARequest] = {}
        self._counter = 1
        self._lock = asyncio.Lock()
        self._openai_api_url: Optional[str] = None
        self._gate_open = True
        self._active_requests = 0
        self._gate_condition = asyncio.Condition()
        self._deploy_lock = asyncio.Lock()
        self._base_weights_path = str(config.model_path)
        self._active_weights_path: Optional[str] = None
        self._active_deployment_id: Optional[str] = None
        # Set when a request surfaces a dead EngineCore. The gate alone cannot
        # express this: it reopens after a deploy even if the restart failed,
        # which would let healthz report 200 while nothing can serve requests.
        self._engine_dead = False

    def _create_engine(self, config: ModelConfig):
        if config.colocate:
            return self._create_colocated_engine(config)
        else:
            return self._create_standalone_engine(config)

    def _build_engine_config(
        self,
        config: ModelConfig,
        *,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        bundle_indices: str = "",
    ) -> VLLMEngineConfig:
        return VLLMEngineConfig(
            model_path=str(config.model_path),
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=(
                config.sampling_max_model_len
                if config.sampling_max_model_len is not None
                else config.max_model_len
            ),
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=config.sampling_enforce_eager,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            logprobs=config.logprobs,
            min_response_tokens=config.min_response_tokens,
            repetition_penalty=1.0,
            enable_lora=True,
            max_lora_rank=config.max_lora_rank,
            max_loras=config.max_loras,
            quantization=config.quantization,
            enable_runtime_lora_updating=True,
            enable_openai_api=True,
            enable_auto_tool_choice=config.enable_auto_tool_choice,
            tool_call_parser=config.tool_call_parser,
            reasoning_parser=config.reasoning_parser,
            bundle_indices=bundle_indices,
        )

    def _create_colocated_engine(self, config: ModelConfig):
        import ray

        if not self._worker_venv_path or not self._worker_venv_path.strip():
            _runtime_env = {}
        else:
            _path = os.environ.get("PATH", "")
            _venv_python = str(Path(self._worker_venv_path) / "bin" / "python")
            _runtime_env = {
                "py_executable": _venv_python,
                "env_vars": {
                    "VIRTUAL_ENV": self._worker_venv_path,
                    "PATH": f"{self._worker_venv_path}/bin:{_path}",
                },
            }
        return (
            ray.remote(VLLMEngine)
            .options(
                name="sampling_model_" + self.base_model,
                num_gpus=config.sampling_memory_fraction,
                runtime_env=_runtime_env,
            )
            .remote(
                config=self._build_engine_config(
                    config,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=config.sampling_memory_fraction,
                )
            )
        )

    def _create_standalone_engine(self, config: ModelConfig):
        import ray

        # Assign tensor_parallel_size GPUs to the actor itself
        # so that Ray populates CUDA_VISIBLE_DEVICES correctly.  vLLM then
        # creates its own placement group inside the EngineCore process where
        # the GPUs are visible.
        num_gpus = config.tensor_parallel_size
        bundle_indices = ",".join(str(i) for i in range(config.tensor_parallel_size))

        if not self._worker_venv_path or not self._worker_venv_path.strip():
            _runtime_env = {}
        else:
            _path = os.environ.get("PATH", "")
            _venv_python = str(Path(self._worker_venv_path) / "bin" / "python")
            _env_vars = {
                "VIRTUAL_ENV": self._worker_venv_path,
                "PATH": f"{self._worker_venv_path}/bin:{_path}",
            }
            # Optional GPU allowlist for sampling engines, e.g. "1,2,3" to
            # isolate a suspected-faulty GPU. Ray schedules the actor only on
            # a listed device; inside the actor it still sees one GPU (cuda:0).
            _sampling_gpus = os.environ.get("TUFT_SAMPLING_GPUS", "").strip()
            if _sampling_gpus:
                _env_vars["CUDA_VISIBLE_DEVICES"] = _sampling_gpus
            _runtime_env = {
                "py_executable": _venv_python,
                "env_vars": _env_vars,
            }

        # Use instance_index to differentiate Ray actor names for DP replicas
        actor_name = f"sampling_model_{self.base_model}"
        if self._instance_index > 0:
            actor_name = f"{actor_name}_dp{self._instance_index}"

        # In standalone/DP mode, each vLLM instance has a dedicated GPU.
        # Use 0.85 (slightly below vLLM default 0.9) to leave ~8GB headroom
        # for prefix-caching KV eviction churn and peak prefill allocations.
        # At 0.9 the engine hits 76.9/79.2GB and a 4GB prefill allocation
        # triggers CUDA OOM (2026-09-06 18:30 crash, prefix_caching ON).
        standalone_gpu_memory_utilization = 0.80

        return (
            ray.remote(VLLMEngine)
            .options(
                name=actor_name,
                num_gpus=num_gpus,
                runtime_env=_runtime_env,
            )
            .remote(
                config=self._build_engine_config(
                    config,
                    tensor_parallel_size=config.tensor_parallel_size,
                    gpu_memory_utilization=standalone_gpu_memory_utilization,
                    bundle_indices=bundle_indices,
                )
            )
        )

    async def async_init(self) -> None:
        """Initialize the backend for sampling."""
        # Ray @ray.remote decorator adds .remote() method dynamically
        await self.engine.prepare.remote()  # type: ignore[attr-defined]
        self._openai_api_url = await self.engine.get_api_server_url.remote()  # type: ignore[attr-defined]
        logger.info(
            f"SamplingBackend for model {self.base_model} initialized. "
            f"OpenAI API URL: {self._openai_api_url}"
        )
        # Wait for the OpenAI API server to be ready to accept connections
        if self._openai_api_url:
            await self._wait_for_openai_server(self._openai_api_url)

    async def _wait_for_openai_server(self, url: str, timeout: float = 120.0) -> None:
        """Poll the vLLM OpenAI server until it accepts connections."""
        import asyncio

        import httpx as _httpx

        deadline = asyncio.get_event_loop().time() + timeout
        check_url = f"{url}/v1/models"
        attempt = 0
        async with _httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                attempt += 1
                try:
                    resp = await client.get(check_url, timeout=3.0)
                    if resp.status_code == 200:
                        logger.info(f"vLLM OpenAI server at {url} is ready (attempt {attempt})")
                        return
                except (_httpx.ConnectError, _httpx.TimeoutException, OSError):
                    pass
                if attempt % 10 == 0:
                    logger.info(f"Waiting for vLLM OpenAI server at {url}... attempt {attempt}")
                await asyncio.sleep(2.0)
        logger.warning(f"vLLM OpenAI server at {url} not ready after {timeout}s")

    def get_openai_api_url(self) -> Optional[str]:
        """Return the vLLM OpenAI API base URL."""
        return self._openai_api_url

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        """Sampling using vLLM engine."""
        with _get_tracer().start_as_current_span("sampling_backend.sample") as span:
            span.set_attribute("tuft.num_samples", num_samples)
            span.set_attribute("tuft.has_lora", lora_id is not None)
            try:
                async with self._gate_condition:
                    await self._gate_condition.wait_for(lambda: self._gate_open)
                    self._active_requests += 1
                try:
                    return await self._sample_locked(
                        prompt=prompt,
                        num_samples=num_samples,
                        sampling_params=sampling_params,
                        include_prompt_logprobs=include_prompt_logprobs,
                        topk_prompt_logprobs=topk_prompt_logprobs,
                        lora_id=lora_id,
                    )
                finally:
                    async with self._gate_condition:
                        self._active_requests -= 1
                        self._gate_condition.notify_all()
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR)
                # Ray wraps the actor-side exception, so match on the rendered
                # text rather than the exception type.
                if "EngineDeadError" in str(e) or "EngineDeadError" in repr(e):
                    self._engine_dead = True
                    logger.error("vLLM EngineCore is dead; marking backend unavailable")
                raise

    async def _sample_locked(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        async with self._lock:
            if lora_id is not None and lora_id not in self.lora_adapters:
                raise ValueError(f"LoRA adapter {lora_id} not found in backend.")
            lora_request = self.lora_adapters[lora_id] if lora_id is not None else None

        prompt_token_ids = prompt.to_ints()
        # A prompt-logprob request needs no generated tokens, but vLLM still
        # requires at least one. Defaulting to 16 made it demand
        # prompt_len + 16 <= max_model_len, rejecting any prompt longer than
        # max_model_len - 16; 1 keeps the usable window at max_model_len - 1 and
        # avoids generating 16 tokens that are thrown away.
        if sampling_params.max_tokens is not None:
            max_tokens = sampling_params.max_tokens
        elif include_prompt_logprobs:
            max_tokens = 1
        else:
            max_tokens = 16
        params = {
            "max_tokens": max_tokens,
            "seed": sampling_params.seed,
            "top_k": sampling_params.top_k,
            "top_p": sampling_params.top_p,
            "temperature": sampling_params.temperature,
            "n": num_samples,
            "prompt_logprobs": (topk_prompt_logprobs if include_prompt_logprobs else None),
            "logprobs": 0,
        }
        # Avoid prefix cache reads when computing prompt logprobs
        # (cached prompt chunks would otherwise skip logit computation
        # and corrupt the returned logprobs). Native vLLM SamplingParams
        # field since 0.12; newer vLLM auto-sets it when prompt_logprobs
        # is requested -- kept explicit here for clarity.
        if include_prompt_logprobs:
            params["skip_reading_prefix_cache"] = True
        if sampling_params.stop is not None:
            # tinker.SamplingParams.stop accepts a heterogeneous list of
            # str (substring stops) and int (token-id stops). vLLM
            # however expects them in two separate fields: `stop`
            # (list[str]) and `stop_token_ids` (list[int]). Forwarding
            # the raw list to vLLM's `stop` triggers `TypeError: object
            # of type 'int' has no len()` inside
            # SamplingParams.__post_init__ (vllm/sampling_params.py:358).
            # Split by isinstance to route each kind correctly.
            str_stops: list[str] = [s for s in sampling_params.stop if isinstance(s, str)]
            int_stops: list[int] = [
                s for s in sampling_params.stop if isinstance(s, int) and not isinstance(s, bool)
            ]
            if str_stops:
                params["stop"] = str_stops
            if int_stops:
                params["stop_token_ids"] = int_stops

        # Ray @ray.remote decorator adds .remote() method dynamically
        req_output = await self.engine.generate.remote(  # type: ignore[attr-defined]
            prompt={"prompt_token_ids": prompt_token_ids},
            lora_request=lora_request,
            **params,
        )
        return _build_sample_response(
            req_output=req_output,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
        )

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        from vllm.lora.request import LoRARequest

        with _get_tracer().start_as_current_span("sampling_backend.add_adapter") as span:
            span.set_attribute("tuft.lora_id", lora_id)
            try:
                async with self._lock:
                    if not adapter_path.exists():
                        raise ValueError(f"LoRA adapter path {adapter_path} does not exist.")
                    self._counter += 1
                    request = LoRARequest(
                        lora_int_id=self._counter + 1,
                        lora_name=lora_id,
                        lora_path=str(adapter_path),
                    )
                    # Register with vLLM first; only record the adapter in the local
                    # registry after success. Otherwise a failure (missing path, or the
                    # engine raising) would leave a stale entry that sample() would later
                    # pass to vLLM even though it was never registered.
                    added = await self.engine.add_lora.remote(request)  # type: ignore[attr-defined]
                    if added is False:
                        raise RuntimeError(
                            f"vLLM did not register LoRA adapter {lora_id} from {adapter_path}."
                        )
                    self.lora_adapters[lora_id] = request
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR)
                raise

    async def remove_adapter(self, lora_id: str) -> None:
        with _get_tracer().start_as_current_span("sampling_backend.remove_adapter") as span:
            span.set_attribute("tuft.lora_id", lora_id)
            async with self._lock:
                if lora_id in self.lora_adapters:
                    # vLLM removes adapters by their integer id, not name.
                    lora_request = self.lora_adapters.pop(lora_id)
                    await self.engine.remove_lora.remote(lora_request.lora_int_id)  # type: ignore[attr-defined]

    async def reload_weights(self, weights_path: str) -> None:
        await self.engine.reload_full_weights.remote(weights_path)  # type: ignore[attr-defined]
        # The reload restarts the engine, which rebinds an ephemeral port for
        # the embedded OpenAI server, so the cached URL goes stale.
        self._openai_api_url = await self.engine.get_api_server_url.remote()  # type: ignore[attr-defined]
        if self._openai_api_url:
            await self._wait_for_openai_server(self._openai_api_url)
        # A restart that got this far produced a live engine again.
        self._engine_dead = False

    async def get_health(self) -> bool:
        return await self.engine.get_health.remote()  # type: ignore[attr-defined]

    async def deploy_full_weights(self, weights_path: Path, deployment_id: str) -> None:
        async with self._deploy_lock:
            if self._active_deployment_id == deployment_id:
                return
            previous_weights = self._active_weights_path or str(self.config.model_path)
            previous_deployment_id = self._active_deployment_id
            await self.close_gate_and_drain()
            try:
                await self.reload_weights(str(weights_path))
                if not await self.get_health():
                    raise RuntimeError(f"vLLM engine unhealthy after reloading {weights_path}")
            except Exception:
                try:
                    await self.reload_weights(previous_weights)
                    self._active_weights_path = previous_weights
                    self._active_deployment_id = previous_deployment_id
                except Exception:
                    logger.error("Full-weight rollback to %s failed", previous_weights)
                    self._active_weights_path = None
                    self._active_deployment_id = None
                raise
            finally:
                async with self._gate_condition:
                    self._gate_open = True
                    self._gate_condition.notify_all()
            self._active_weights_path = str(weights_path)
            self._active_deployment_id = deployment_id
            logger.info("Deployed full weights %s (%s)", weights_path, deployment_id)

    def get_active_deployment_id(self) -> Optional[str]:
        return self._active_deployment_id

    def get_active_weights_path(self) -> Optional[str]:
        return self._active_weights_path

    async def revert_to_base_weights(self) -> None:
        """Reload the base model and clear the full-weight deployment state.

        Without this the backend keeps rejecting base-model sampling sessions
        after a full-param run deploys weights, even once that run is gone,
        which blocks the next run from creating its initial base session.
        """
        if self._active_deployment_id is None:
            return
        async with self._deploy_lock:
            await self.close_gate_and_drain()
            try:
                await self.reload_weights(str(self.config.model_path))
            finally:
                async with self._gate_condition:
                    self._gate_open = True
                    self._gate_condition.notify_all()
            self._active_weights_path = None
            self._active_deployment_id = None
            logger.info("Reverted to base weights %s", self.config.model_path)

    def is_ready(self) -> bool:
        return self._gate_open and not self._engine_dead

    async def wait_ready(self) -> None:
        async with self._gate_condition:
            await self._gate_condition.wait_for(lambda: self._gate_open)

    async def close_gate_and_drain(self) -> None:
        """Stop admitting requests and wait for the in-flight ones to finish."""
        async with self._gate_condition:
            self._gate_open = False
            await self._gate_condition.wait_for(lambda: self._active_requests == 0)

    async def shutdown(self) -> None:
        """Shut down the vLLM engine Ray actor and release GPU resources."""
        import ray

        try:
            await self.engine.shutdown.remote()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            ray.kill(self.engine, no_restart=True)  # type: ignore[arg-type]
        except Exception:
            pass
        self.lora_adapters.clear()


class DPSamplingBackend(BaseSamplingBackend):
    """Data-Parallel sampling backend: N independent vLLM instances with round-robin LB.

    Each instance runs on its own GPU(s) (tensor_parallel_size per instance).
    Requests are distributed across instances using round-robin for uniform load.
    All instances share the same LoRA adapters.
    """

    def __init__(
        self,
        config: ModelConfig,
        worker_venv_path: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self._dp_size = config.data_parallel_size
        self._instances: list[VLLMSamplingBackend] = []
        for i in range(self._dp_size):
            instance = VLLMSamplingBackend(
                config, worker_venv_path=worker_venv_path, instance_index=i
            )
            self._instances.append(instance)
        # Atomic round-robin counter
        self._rr_counter = 0
        self._rr_lock = asyncio.Lock()
        self._deploy_lock = asyncio.Lock()
        self._active_deployment_id: Optional[str] = None
        self._active_weights_path: Optional[str] = None
        self._deploy_task: Optional[asyncio.Task] = None
        self._deploy_task_id: Optional[str] = None
        logger.info(
            "DPSamplingBackend: created %d instances for model %s",
            self._dp_size,
            config.model_name,
        )

    def _next_instance(self) -> VLLMSamplingBackend:
        """Round-robin selection (no lock needed for simple modular arithmetic)."""
        idx = self._rr_counter % self._dp_size
        self._rr_counter += 1
        return self._instances[idx]

    async def async_init(self) -> None:
        """Initialize all DP instances in parallel."""
        await asyncio.gather(*[inst.async_init() for inst in self._instances])
        logger.info(
            "DPSamplingBackend: all %d instances initialized for model %s",
            self._dp_size,
            self.base_model,
        )

    def get_openai_api_url(self) -> Optional[str]:
        """Return a single URL (first instance) for backward compat.

        For DP-aware routing, use get_openai_api_urls() instead.
        """
        return self._instances[0].get_openai_api_url() if self._instances else None

    def get_openai_api_urls(self) -> list[str]:
        """Return all vLLM instance URLs for DP-aware load balancing."""
        urls = []
        for inst in self._instances:
            url = inst.get_openai_api_url()
            if url:
                urls.append(url)
        return urls

    def get_next_openai_api_url(self) -> Optional[str]:
        """Return the next URL via round-robin for load-balanced proxying."""
        inst = self._next_instance()
        return inst.get_openai_api_url()

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        """Route sampling to a DP instance via round-robin."""
        instance = self._next_instance()
        return await instance.sample(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
            lora_id=lora_id,
        )

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        """Add LoRA adapter to ALL DP instances; roll back on partial failure.

        Without rollback, a failure on one replica would leave the DP instances with
        inconsistent adapter sets (some registered, some not), causing round-robin
        sampling to intermittently fail with an unknown lora_id.
        """
        results = await asyncio.gather(
            *[inst.add_adapter(lora_id, adapter_path) for inst in self._instances],
            return_exceptions=True,
        )
        failures = [i for i, res in enumerate(results) if isinstance(res, BaseException)]
        if failures:
            # Roll back the instances that succeeded so the adapter set stays consistent.
            for i, res in enumerate(results):
                if not isinstance(res, BaseException):
                    try:
                        await self._instances[i].remove_adapter(lora_id)
                    except Exception:  # pylint: disable=broad-except
                        logger.warning("Failed to roll back adapter %s on instance %d", lora_id, i)
            first_error = results[failures[0]]
            if isinstance(first_error, BaseException):
                raise first_error

    async def remove_adapter(self, lora_id: str) -> None:
        """Remove LoRA adapter from ALL DP instances."""
        await asyncio.gather(*[inst.remove_adapter(lora_id) for inst in self._instances])

    async def deploy_full_weights(self, weights_path: Path, deployment_id: str) -> None:
        if self._active_deployment_id == deployment_id:
            return
        # A concurrent caller for the same revision joins the running deploy
        # instead of queueing a second engine restart behind the lock.
        if self._deploy_task is not None and self._deploy_task_id == deployment_id:
            await asyncio.shield(self._deploy_task)
            return
        async with self._deploy_lock:
            if self._active_deployment_id == deployment_id:
                return
            self._deploy_task = asyncio.current_task()
            self._deploy_task_id = deployment_id
            try:
                await self._deploy_full_weights_locked(weights_path, deployment_id)
            finally:
                self._deploy_task = None
                self._deploy_task_id = None

    async def _deploy_full_weights_locked(self, weights_path: Path, deployment_id: str) -> None:
        previous_weights = self._active_weights_path or str(self.config.model_path)
        await asyncio.gather(*[inst.close_gate_and_drain() for inst in self._instances])
        try:
            logger.info("Reloading %d vLLM replicas from %s", self._dp_size, weights_path)
            await asyncio.gather(
                *[inst.reload_weights(str(weights_path)) for inst in self._instances]
            )
            health = await asyncio.gather(*[inst.get_health() for inst in self._instances])
            if not all(health):
                raise RuntimeError("One or more vLLM replicas became unhealthy during reload")
        except Exception:
            rollback_errors = await asyncio.gather(
                *[inst.reload_weights(previous_weights) for inst in self._instances],
                return_exceptions=True,
            )
            failures = [e for e in rollback_errors if isinstance(e, BaseException)]
            if failures:
                logger.error("Full-weight rollback failed: %s", failures)
            raise
        finally:
            for inst in self._instances:
                async with inst._gate_condition:
                    inst._gate_open = True
                    inst._gate_condition.notify_all()
        self._active_weights_path = str(weights_path)
        self._active_deployment_id = deployment_id
        logger.info(
            "Deployed full weights %s (%s) to %d vLLM replicas",
            weights_path,
            deployment_id,
            self._dp_size,
        )

    def get_active_deployment_id(self) -> Optional[str]:
        return self._active_deployment_id

    def get_active_weights_path(self) -> Optional[str]:
        return self._active_weights_path

    async def revert_to_base_weights(self) -> None:
        """Revert every DP replica to the base model and clear deployment state."""
        if self._active_deployment_id is None:
            return
        await asyncio.gather(*[inst.revert_to_base_weights() for inst in self._instances])
        self._active_weights_path = None
        self._active_deployment_id = None
        logger.info("Reverted %d DP replicas to base weights", self._dp_size)

    def is_ready(self) -> bool:
        return all(inst.is_ready() for inst in self._instances)

    async def wait_ready(self) -> None:
        await asyncio.gather(*[inst.wait_ready() for inst in self._instances])

    async def shutdown(self) -> None:
        """Shut down all DP vLLM instances."""
        for inst in self._instances:
            try:
                await inst.shutdown()
            except Exception:
                pass


def _build_mock_openai_app(model_name: str):
    """Build a minimal FastAPI app that mimics vLLM's OpenAI-compatible endpoints."""
    import itertools

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    mock_app = FastAPI()
    _id_counter = itertools.count(1)

    def _make_completion_response(body: dict) -> dict:
        model = body.get("model", model_name)
        raw_prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 16)
        n = body.get("n", 1)
        logprobs_n = body.get("logprobs")
        stop = body.get("stop")

        # Handle batch prompts (list of strings)
        prompts = raw_prompt if isinstance(raw_prompt, list) else [raw_prompt]
        choices = []
        idx = 0
        for prompt in prompts:
            prompt_str = str(prompt)[:20]
            for _j in range(n):
                text = f" dummy completion output for '{prompt_str}'"
                finish = "length"
                # Respect stop sequences
                if stop:
                    for s in stop if isinstance(stop, list) else [stop]:
                        pos = text.find(s)
                        if pos != -1:
                            text = text[:pos]
                            finish = "stop"
                            break
                logprobs_data = None
                if logprobs_n is not None:
                    logprobs_data = {
                        "tokens": [" dummy"],
                        "token_logprobs": [-0.5],
                        "top_logprobs": [{"dummy": -0.5}] if logprobs_n > 0 else None,
                        "text_offset": [0],
                    }
                choices.append(
                    {
                        "index": idx,
                        "text": text,
                        "logprobs": logprobs_data,
                        "finish_reason": finish,
                    }
                )
                idx += 1
        prompt_tokens = sum(max(1, len(str(p).split())) for p in prompts)
        return {
            "id": f"cmpl-dummy-{next(_id_counter)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max_tokens * len(choices),
                "total_tokens": prompt_tokens + max_tokens * len(choices),
            },
        }

    def _make_chat_response(body: dict) -> dict:
        model = body.get("model", model_name)
        n = body.get("n", 1)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 16
        choices = []
        for i in range(n):
            choices.append(
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": "This is a dummy response from the mock OpenAI server.",
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            )
        return {
            "id": f"chatcmpl-dummy-{next(_id_counter)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": max_tokens,
                "total_tokens": 10 + max_tokens,
            },
        }

    async def _stream_chat_chunks(body: dict):
        model = body.get("model", model_name)
        chunk_id = f"chatcmpl-dummy-{next(_id_counter)}"
        # First chunk with role
        first = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(first)}\n\n"
        # Content chunks
        words = ["This", " is", " a", " dummy", " streamed", " response."]
        for word in words:
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        # Final chunk
        final = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": len(words),
                "total_tokens": 10 + len(words),
            },
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_completion_chunks(body: dict):
        model = body.get("model", model_name)
        chunk_id = f"cmpl-dummy-{next(_id_counter)}"
        words = [" dummy", " completion", " output"]
        for word in words:
            chunk = {
                "id": chunk_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "text": word, "logprobs": None, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        final = {
            "id": chunk_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": "", "logprobs": None, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": len(words),
                "total_tokens": 5 + len(words),
            },
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    @mock_app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        if body.get("stream"):
            return StreamingResponse(
                _stream_completion_chunks(body), media_type="text/event-stream"
            )
        return JSONResponse(_make_completion_response(body))

    @mock_app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        if body.get("stream"):
            return StreamingResponse(_stream_chat_chunks(body), media_type="text/event-stream")
        return JSONResponse(_make_chat_response(body))

    @mock_app.get("/v1/models")
    async def list_models():
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "dummy",
                    }
                ],
            }
        )

    return mock_app


class DummySamplingBackend(BaseSamplingBackend):
    """A dummy sampling backend that returns fixed responses for unittest."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.lora_adapters: dict[str, Path] = {}
        self._counter = 1
        self._lock = asyncio.Lock()
        self._openai_api_url: Optional[str] = None
        self._mock_server: object | None = None
        self._mock_thread: threading.Thread | None = None

    async def async_init(self) -> None:
        """Start an embedded mock OpenAI server for testing."""
        self._start_mock_openai_server()

    def _start_mock_openai_server(self) -> None:
        """Start a lightweight mock OpenAI-compatible HTTP server in a background thread."""
        import uvicorn

        mock_app = _build_mock_openai_app(str(self.config.model_path))

        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        server = uvicorn.Server(
            uvicorn.Config(mock_app, host="127.0.0.1", port=port, log_level="error")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for the server to become ready
        import httpx

        for _ in range(50):
            try:
                resp = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=0.5)
                if resp.status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)

        self._openai_api_url = f"http://127.0.0.1:{port}"
        self._mock_server = server
        self._mock_thread = thread
        logger.info(f"DummySamplingBackend mock OpenAI server started at {self._openai_api_url}")

    def get_openai_api_url(self) -> Optional[str]:
        """Return the mock OpenAI API base URL."""
        return self._openai_api_url

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        """Return a fixed dummy response."""
        prompt_tokens = prompt.to_ints()
        max_tokens = sampling_params.max_tokens or 16
        sequences: list[types.SampledSequence] = []
        for _ in range(num_samples):
            generated = self._generate_tokens(prompt_tokens, max_tokens)
            seq = types.SampledSequence(
                stop_reason="length",
                _tokens_list=generated,
                _logprobs_list=[-0.3 for _ in generated],
            )
            sequences.append(seq)
        prompt_logprobs = None
        topk_prompt = None
        if include_prompt_logprobs:
            prompt_logprobs = [-0.1 if tok is not None else None for tok in prompt_tokens]
        if topk_prompt_logprobs > 0:
            topk_prompt = [
                [
                    (token, round(-0.05 - idx * 0.01, 4))
                    for idx, token in enumerate(prompt_tokens[:topk_prompt_logprobs])
                ]
                if token is not None
                else None
                for token in prompt_tokens
            ]
        return types.SampleResponse(
            sequences=sequences,
            _prompt_logprobs_list=prompt_logprobs,
            _topk_prompt_logprobs_list=topk_prompt,
        )

    def _generate_tokens(self, prompt_tokens: list[int], max_tokens: int) -> list[int]:
        start = prompt_tokens[-1] if prompt_tokens else (abs(self.config.seed) % 32000) + 1
        return [(start + i) % 32000 for i in range(1, max_tokens + 1)]

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        self.lora_adapters[lora_id] = adapter_path

    async def remove_adapter(self, lora_id: str) -> None:
        if lora_id in self.lora_adapters:
            del self.lora_adapters[lora_id]

    async def shutdown(self) -> None:
        """Stop the mock OpenAI server if running."""
        if self._mock_server is not None:
            try:
                self._mock_server.should_exit = True  # type: ignore[attr-defined]
            except Exception:
                pass
        self._mock_server = None
        self._mock_thread = None
