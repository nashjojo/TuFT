"""Data-parallel sampling backend that distributes requests across multiple replicas."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from tinker import types

from ..config import ModelConfig
from .base_backend import BaseSamplingBackend


logger = logging.getLogger(__name__)


class DataParallelSamplingBackend(BaseSamplingBackend):
    """Wraps N independent sampling backend replicas with round-robin routing.

    Each replica is a full sampling backend instance (VLLMSamplingBackend or
    DummySamplingBackend) running on its own GPU(s). Requests are distributed
    via round-robin; LoRA adapter operations are broadcast to all replicas.
    """

    def __init__(self, config: ModelConfig, worker_venv_path: Optional[str] = None) -> None:
        super().__init__(config)
        self._replicas: list[BaseSamplingBackend] = []
        self._replica_count = config.data_parallel_size
        self._rr_counter = 0
        self._create_replicas(config, worker_venv_path)

    def _create_replicas(self, config: ModelConfig, worker_venv_path: Optional[str]) -> None:
        if os.getenv("TUFT_CPU_TEST", "0") == "1":
            from .sampling_backend import DummySamplingBackend

            for _ in range(self._replica_count):
                self._replicas.append(DummySamplingBackend(config))
        else:
            from .sampling_backend import VLLMSamplingBackend

            for i in range(self._replica_count):
                self._replicas.append(
                    VLLMSamplingBackend(config, worker_venv_path=worker_venv_path, dp_rank=i)
                )

    async def async_init(self) -> None:
        await asyncio.gather(*[r.async_init() for r in self._replicas])
        logger.info(
            "DataParallelSamplingBackend for %s initialized with %d replicas",
            self.base_model,
            self._replica_count,
        )

    def _select_replica(self) -> BaseSamplingBackend:
        idx = self._rr_counter % self._replica_count
        self._rr_counter += 1
        return self._replicas[idx]

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        replica = self._select_replica()
        return await replica.sample(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
            lora_id=lora_id,
        )

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        results = await asyncio.gather(
            *[r.add_adapter(lora_id, adapter_path) for r in self._replicas],
            return_exceptions=True,
        )
        failures = [(i, e) for i, e in enumerate(results) if isinstance(e, Exception)]
        if failures:
            for i, r in enumerate(self._replicas):
                if not isinstance(results[i], Exception):
                    try:
                        await r.remove_adapter(lora_id)
                    except Exception:
                        pass
            raise failures[0][1]

    async def remove_adapter(self, lora_id: str) -> None:
        results = await asyncio.gather(
            *[r.remove_adapter(lora_id) for r in self._replicas],
            return_exceptions=True,
        )
        for i, e in enumerate(results):
            if isinstance(e, Exception):
                logger.warning("Failed to remove adapter %s from replica %d: %s", lora_id, i, e)

    def get_openai_api_url(self) -> Optional[str]:
        if self._replicas:
            return self._replicas[0].get_openai_api_url()
        return None
