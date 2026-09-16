from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from tinker import types
from transformers import AutoTokenizer

from .helpers import clear_ray_state


if TYPE_CHECKING:
    from tuft.backends.sampling_backend import DPSamplingBackend


@pytest.fixture(scope="function", autouse=True)
def ray_cluster():
    import ray

    ray.init(ignore_reinit_error=True)
    yield
    clear_ray_state()


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_sampling_backend():
    from tuft.backends.sampling_backend import VLLMSamplingBackend
    from tuft.config import ModelConfig

    assert "TUFT_TEST_MODEL" in os.environ, (
        "Environment variable TUFT_TEST_MODEL must be set for this test."
    )

    model_path = Path(os.environ.get("TUFT_TEST_MODEL", "Qwen/Qwen3-0.6B"))
    model_config = ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        model_path=model_path,
        max_model_len=2048,
        tensor_parallel_size=1,
        sampling_memory_fraction=0.5,
    )
    backend = VLLMSamplingBackend(model_config)
    await backend.async_init()
    assert backend.base_model == "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_special_tokens=False
    )["input_ids"][0].tolist()
    # generate without prompt logprobs
    # max_tokens is generous enough for thinking-style models (e.g. Qwen3.5)
    # to emit their EOS, so "stop" remains deterministically assertable.
    response = await backend.sample(
        prompt=types.ModelInput.from_ints(input_ids),
        num_samples=3,
        sampling_params=types.SamplingParams(max_tokens=1024, temperature=0.7),
    )
    assert response.sequences is not None
    assert len(response.sequences) == 3
    assert response.prompt_logprobs is None
    assert response.topk_prompt_logprobs is None
    for seq in response.sequences:
        assert seq.tokens is not None
        assert seq.logprobs is not None
        assert len(seq.tokens) > 0
        assert len(seq.logprobs) == len(seq.tokens)
        assert seq.stop_reason == "stop"

    # generate with prompt logprobs
    response_with_logprobs = await backend.sample(
        prompt=types.ModelInput.from_ints(input_ids),
        num_samples=2,
        sampling_params=types.SamplingParams(max_tokens=3, temperature=0.7),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=3,
    )
    assert response_with_logprobs.sequences is not None
    assert len(response_with_logprobs.sequences) == 2
    assert response_with_logprobs.prompt_logprobs is not None
    assert response_with_logprobs.topk_prompt_logprobs is not None
    assert len(response_with_logprobs.prompt_logprobs) == len(input_ids)
    assert len(response_with_logprobs.topk_prompt_logprobs) == len(input_ids)
    # first token should have no top-k logprobs
    assert response_with_logprobs.topk_prompt_logprobs[0] is None
    # each subsequent token should have top-k logprobs
    for topk in response_with_logprobs.topk_prompt_logprobs[1:]:
        assert topk is not None
        assert len(topk) == 3
    # check sequences
    for seq in response_with_logprobs.sequences:
        assert seq.tokens is not None
        assert seq.logprobs is not None
        assert len(seq.tokens) > 0
        assert len(seq.logprobs) == len(seq.tokens)
        assert seq.stop_reason == "length"  # stop because of max_tokens=3

    # compute logprobs only
    logprobs_response = await backend.sample(
        prompt=types.ModelInput.from_ints(input_ids),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
    )
    assert len(logprobs_response.sequences) == 1
    assert logprobs_response.prompt_logprobs is not None
    assert len(logprobs_response.prompt_logprobs) == len(input_ids)
    assert logprobs_response.topk_prompt_logprobs is None


class _FakeInstance:
    """Minimal stand-in for VLLMSamplingBackend's engine-facing surface."""

    def __init__(self, healthy: bool = True):
        self._gate_open = True
        self._gate_condition = asyncio.Condition()
        self.reloaded_paths: list[str] = []
        self.healthy = healthy

    async def close_gate_and_drain(self) -> None:
        self._gate_open = False

    async def reload_weights(self, path: str) -> None:
        self.reloaded_paths.append(path)

    async def get_health(self) -> bool:
        return self.healthy


def _make_dp_backend(
    active_deployment_id: str | None, healthy: bool = True
) -> tuple[DPSamplingBackend, list[_FakeInstance]]:
    from tuft.backends.sampling_backend import DPSamplingBackend

    backend = DPSamplingBackend.__new__(DPSamplingBackend)
    instances = [_FakeInstance(healthy=healthy), _FakeInstance(healthy=healthy)]
    backend._dp_size = 2
    backend._instances = cast(Any, instances)
    backend._deploy_lock = asyncio.Lock()
    backend.config = cast(Any, SimpleNamespace(model_path=Path("/base/model")))
    backend._active_weights_path = "/ckpts/state-248/model" if active_deployment_id else None
    backend._active_deployment_id = active_deployment_id
    return backend, instances


@pytest.mark.asyncio
async def test_dp_revert_reloads_base_weights():
    """Regression (2026-09-17): a DP revert must reload base weights itself.

    The per-instance revert guard checks a flag that only the instance's own
    deploy_full_weights sets; the DP deploy path calls inst.reload_weights
    directly, so delegating the revert to the instances silently no-ops and the
    replicas keep serving the last deployed checkpoint (v4 warm-started on
    state-248 this way).
    """
    backend, instances = _make_dp_backend(active_deployment_id="state-248")

    await backend.revert_to_base_weights()

    for inst in instances:
        assert inst.reloaded_paths == ["/base/model"]
        assert inst._gate_open, "gates must reopen after the revert reload"
    assert backend._active_deployment_id is None
    assert backend._active_weights_path is None

    # No deployment on record: nothing to revert, engines untouched.
    for inst in instances:
        inst.reloaded_paths.clear()
    await backend.revert_to_base_weights()
    assert all(inst.reloaded_paths == [] for inst in instances)


@pytest.mark.asyncio
async def test_dp_revert_failure_keeps_deployment_state():
    """An unhealthy replica must fail the revert loudly instead of clearing the
    deployment state and pretending the engines are back on base weights."""
    backend, instances = _make_dp_backend(active_deployment_id="state-248", healthy=False)

    with pytest.raises(RuntimeError, match="unhealthy during base revert"):
        await backend.revert_to_base_weights()

    for inst in instances:
        assert inst.reloaded_paths == ["/base/model"]
        assert inst._gate_open, "gates must reopen even when the revert fails"
    assert backend._active_deployment_id == "state-248"
    assert backend._active_weights_path == "/ckpts/state-248/model"
