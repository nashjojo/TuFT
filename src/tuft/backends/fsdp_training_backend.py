"""Multi-node/multi-GPU training via FSDP2 and multi-adapter LoRA."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import torch
from packaging import version
from peft import LoraConfig, TaskType, get_peft_model
from tinker import types
from torch.distributed.tensor import DTensor


# FSDP v2 imports (requires PyTorch >= 2.4)
# PyTorch 2.6+ exports from public module; 2.4/2.5 use private _composable.fsdp
if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
elif version.parse(torch.__version__) >= version.parse("2.4"):
    # pyright: ignore[reportPrivateImportUsage]
    from torch.distributed._composable.fsdp import (
        MixedPrecisionPolicy,  # type: ignore[attr-defined]
        fully_shard,  # type: ignore[attr-defined]
    )
else:
    raise ImportError(
        f"FSDP v2 requires PyTorch >= 2.4, but got {torch.__version__}. "
        "Please upgrade PyTorch or use training_backend='hf' instead."
    )
from tuft.backends.base_backend import BaseTrainingBackend
from tuft.backends.fsdp_engine import (
    FSDPModelConfig,
    build_base_model,
    forward_backward as fsdp_forward_backward,
)
from tuft.backends.vllm_lora_compat import (
    add_language_model_aliases,
    vllm_nests_language_model,
)
from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig


# Matches the slot/adapter name embedded in a PEFT LoRA parameter key, e.g.
# '...lora_A.adapter_r8_3.weight'. Used to canonicalize keys so checkpoints are
# independent of which slot saved/loads them.
_LORA_AB_NAME_RE = re.compile(r"\.lora_(A|B)\.[^.]+\.weight$")


def _canonical_lora_key(key: str) -> str:
    """Strip the embedded slot name from a LoRA parameter key.

    '...lora_A.adapter_r8_3.weight' -> '...lora_A.weight'. A key that is already
    canonical ('...lora_A.weight') is returned unchanged.
    """
    return _LORA_AB_NAME_RE.sub(r".lora_\1.weight", key)


def _copy_full_tensor_into_dtensor(param: DTensor, full_tensor: torch.Tensor) -> None:
    """Copy a full (unsharded) tensor into an FSDP2 DTensor parameter.

    Uses ``distribute_tensor`` with the parameter's own device_mesh/placements so the
    sharding matches FSDP2 semantics exactly (handles non-divisible dim0 and
    dim0 < world_size correctly, unlike manual equal-chunk slicing).
    """
    from torch.distributed.tensor import distribute_tensor

    local_device = param.to_local().device
    full = full_tensor.to(device=local_device, dtype=param.dtype)
    sharded = distribute_tensor(full, device_mesh=param.device_mesh, placements=param.placements)
    param.to_local().copy_(sharded.to_local())


def _shard_list(xs: list[Any], n_shards: int) -> list[list[Any]]:
    """Split xs into n_shards contiguous shards (order-preserving)."""
    if n_shards <= 0:
        raise ValueError(f"n_shards must be > 0, got {n_shards}")
    total = len(xs)
    base = total // n_shards
    rem = total % n_shards
    shards = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        shards.append(xs[start : start + size])
        start += size
    return shards


def _uniform_micro_batch_size(shard_lens: list[int], max_mb: int) -> Optional[int]:
    """Largest micro-batch size <= ``max_mb`` giving every rank the same count.

    FSDP2 issues one collective set per micro-batch, so ranks that iterate a
    different number of times deadlock. Requiring ``max_mb`` to divide every
    shard exactly is too strict: it fails on the common uneven-shard case and
    forces a single micro-batch holding the whole shard, which is what OOMs the
    GPU. The engine slices with ``range(0, len(data), mb)``, so an uneven final
    chunk is fine as long as the resulting count matches across ranks.

    Returns ``None`` when no split yields both equal counts and more than one
    micro-batch; callers then fall back to one micro-batch per rank, which is
    safe because it only happens for shards too small to matter.
    """
    lens = [n for n in shard_lens if n > 0]
    if not lens or max_mb <= 0:
        return None
    for mb in range(min(max_mb, max(lens)), 0, -1):
        counts = {(n + mb - 1) // mb for n in lens}
        if len(counts) == 1 and next(iter(counts)) > 1:
            return mb
    return None


def _merge_metrics(
    results: list[dict[str, Any]], weights: list[int] | None = None
) -> dict[str, Any]:
    """Merge per-shard metrics.

    ``:sum`` metrics are summed; ``:mean`` metrics are averaged weighted by the
    corresponding shard size (``weights``) so unequal shards do not skew the mean
    (matches the HF backend, which weights by micro-batch size).
    """
    merged: dict[str, Any] = {}
    mean_acc: dict[str, list[float]] = {}

    for idx, out in enumerate(results):
        w = float(weights[idx]) if weights is not None and idx < len(weights) else 1.0
        metrics = out.get("metrics", {}) or {}
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue

            if k.endswith(":sum"):
                merged[k] = merged.get(k, 0.0) + float(v)
            elif k.endswith(":mean"):
                acc = mean_acc.setdefault(k, [0.0, 0.0])
                acc[0] += float(v) * w
                acc[1] += w
            else:
                merged[k] = merged.get(k, 0.0) + float(v)

    for k, (weighted_sum, total_weight) in mean_acc.items():
        merged[k] = weighted_sum / total_weight if total_weight else 0.0

    return merged


# Default port for torch.distributed init (multi-GPU). ModelConfig.fsdp_master_port should match.
DEFAULT_MASTER_PORT = 29500


def _fsdp_logprobs_to_loss_fn_outputs(
    engine_output: Dict[str, Any],
    data: list[types.Datum],
) -> list[Dict[str, Any]]:
    """Convert engine log-prob tensors to per-datum Tinker outputs."""

    model_output = (engine_output or {}).get("model_output") or {}
    per_sample = model_output.get("log_probs") or []
    if len(per_sample) != len(data):
        raise RuntimeError(
            f"FSDP engine returned {len(per_sample)} log-prob rows for {len(data)} datums"
        )
    return [
        {"logprobs": types.TensorData.from_torch(t.detach().cpu().float().clone())}
        for t in per_sample
    ]


def _write_adapter_weights_file(
    logger: logging.Logger, adapter_name: str, path: Path, peft_state: dict
) -> None:
    """Write adapter weights as safetensors, falling back to ``.bin`` on failure.

    Always removes the other format first so vLLM (which prefers safetensors
    over .bin) can never pick a stale file left behind by a previous save at
    the same checkpoint path.
    """
    try:
        from safetensors.torch import save_file

        (path / "adapter_model.bin").unlink(missing_ok=True)
        save_file(peft_state, path / "adapter_model.safetensors")
    except Exception as e:
        logger.warning(
            "safetensors save failed for adapter '%s', falling back to .bin: %s",
            adapter_name,
            e,
        )
        (path / "adapter_model.safetensors").unlink(missing_ok=True)
        torch.save(peft_state, path / "adapter_model.bin")


# =============================================================================
# Slot configuration and worker-internal data structures
# =============================================================================


@dataclass
class SlotPoolConfig:
    """Multi-adapter slot pool configuration (rank -> number of slots)."""

    rank_slots: Dict[int, int] = field(default_factory=lambda: {8: 5, 16: 2})
    lora_alpha_ratio: int = 2
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def get_lora_alpha(self, rank: int) -> int:
        return rank * self.lora_alpha_ratio


@dataclass
class AdapterInfo:
    """Per-adapter metadata and optimizer."""

    name: str
    rank: int
    lora_alpha: int
    target_modules: List[str]
    optimizer: Any = None
    step_count: int = 0


# =============================================================================
# Serializable config for Ray actors
# =============================================================================


def _get_rank_slots_from_config(config: ModelConfig) -> Dict[int, int]:
    """Get rank_slots from ModelConfig (config preferred; otherwise default).

    rank_slots: LoRA rank -> number of adapter slots (concurrent adapters of that rank).
    Defaults (override via ModelConfig.fsdp_rank_slots):
    - rank 8: 16 slots (common case; more slots for lower memory per adapter).
    - other max_lora_rank: 8 slots (fewer slots for higher rank due to memory).
    """
    max_rank = getattr(config, "max_lora_rank", 8)
    raw = getattr(config, "fsdp_rank_slots", None)
    if raw and len(raw) > 0:
        return {int(k): v for k, v in raw.items()}
    # default slots
    if max_rank == 8:
        return {8: 16}
    return {8: 16, max_rank: 8}


def _config_to_worker_dict(config: ModelConfig) -> dict:
    """Convert ModelConfig to the serializable subset needed by Ray workers."""

    rank_slots = _get_rank_slots_from_config(config)
    return {
        "model_path": str(config.model_path),
        "max_model_len": config.max_model_len,
        "fsdp_override_config": dict(getattr(config, "fsdp_override_config", None) or {}),
        "attn_implementation": getattr(config, "attn_implementation", None),
        "training_mode": getattr(config, "training_mode", "lora"),
        "slot_config": {
            "rank_slots": rank_slots,
            "lora_alpha_ratio": 2,
            "target_modules": list(
                getattr(config, "lora_target_modules", None) or ["q_proj", "v_proj"]
            ),
        },
    }


def _worker_dict_to_configs(config_dict: dict) -> tuple[FSDPModelConfig, SlotPoolConfig, str]:
    """Build the torch-native model and slot configurations inside an actor."""

    override = dict(config_dict.get("fsdp_override_config") or {})
    attn_implementation = override.pop(
        "attn_implementation",
        config_dict.get("attn_implementation") or "sdpa",
    )
    logging.getLogger(__name__).info(
        "[FSDPTrainingBackend] Loading %s with attn_implementation=%s",
        config_dict.get("model_path"),
        attn_implementation,
    )
    model_config = FSDPModelConfig(
        path=config_dict["model_path"],
        max_model_len=int(config_dict["max_model_len"]),
        attn_implementation=attn_implementation,
        override_config=override,
    )
    sc = config_dict.get("slot_config") or {}
    slot_config = SlotPoolConfig(
        rank_slots=dict(sc.get("rank_slots", {8: 4})),
        lora_alpha_ratio=int(sc.get("lora_alpha_ratio", 2)),
        target_modules=list(sc.get("target_modules", ["q_proj", "v_proj"])),
    )
    training_mode = str(config_dict.get("training_mode", "lora"))
    return model_config, slot_config, training_mode


# =============================================================================
# FSDP workers
# =============================================================================


def _fully_shard_model(model: Any) -> Any:
    # fp32 shards are the master weights AdamW updates; MixedPrecisionPolicy
    # below casts them to bf16 for compute, so precision is kept without
    # slowing the forward/backward.
    model = model.to(torch.float32).cuda()
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device_mesh = init_device_mesh("cuda", (world_size,)) if world_size > 1 else None
    transformer_layer_cls_names = getattr(model, "_no_split_modules", None) or [
        "DecoderLayer",
        "TransformerBlock",
        "LlamaDecoderLayer",
        "Qwen2DecoderLayer",
        "Qwen3DecoderLayer",
    ]
    wrapped_modules = [
        module
        for module in model.modules()
        if module.__class__.__name__ in transformer_layer_cls_names
    ]
    for module in wrapped_modules:
        fully_shard(module, mesh=device_mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh, mp_policy=mp_policy)
    return model


class MultiAdapterFSDPWorker:
    """Own a multi-LoRA FSDP2 module and its per-adapter training state."""

    def __init__(
        self,
        model_config: FSDPModelConfig,
        slot_config: SlotPoolConfig,
    ):
        self.model_config = model_config
        self.slot_config = slot_config
        self.module: Any = None
        self._adapters: Dict[str, AdapterInfo] = {}
        self._adapters_by_rank: Dict[int, List[str]] = {}
        self._name_counter: Dict[int, int] = {}
        self._allocated: Dict[str, bool] = {}
        self._initialized = False
        self.logger = logging.getLogger(f"{__name__}.MultiAdapterFSDPWorker")

    def _generate_adapter_name(self, rank: int) -> str:
        if rank not in self._name_counter:
            self._name_counter[rank] = 0
        idx = self._name_counter[rank]
        self._name_counter[rank] += 1
        return f"adapter_r{rank}_{idx}"

    def initialize(self) -> None:
        """Build the base model, PEFT adapter pool, FSDP2 module, and first optimizer."""

        if self._initialized:
            return
        base_model = build_base_model(self.model_config)
        peft_model = None
        for rank, count in self.slot_config.rank_slots.items():
            lora_alpha = self.slot_config.get_lora_alpha(rank)
            for _ in range(count):
                name = self._generate_adapter_name(rank)
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=rank,
                    lora_alpha=lora_alpha,
                    target_modules=list(self.slot_config.target_modules),
                )
                if peft_model is None:
                    peft_model = get_peft_model(
                        base_model,
                        lora_config,
                        adapter_name=name,
                        autocast_adapter_dtype=False,
                    )
                else:
                    peft_model.add_adapter(name, lora_config)
                self._adapters[name] = AdapterInfo(
                    name=name,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    target_modules=list(self.slot_config.target_modules),
                )
                self._adapters_by_rank.setdefault(rank, []).append(name)
                self._allocated[name] = False
        if peft_model is None or not self._adapters:
            raise RuntimeError("slot_config.rank_slots must define at least one slot")

        first = next(iter(self._adapters))
        peft_model.set_adapter(first)
        self.module = _fully_shard_model(peft_model)
        self._create_optimizer_for_adapter(first)
        self._initialized = True

    def _create_optimizer_for_adapter(self, adapter_name: str) -> None:
        info = self._adapters[adapter_name]
        if info.optimizer is not None:
            return
        self._activate_adapter(adapter_name)
        params = [p for p in self.module.parameters() if p.requires_grad]
        info.optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)

    def _activate_adapter(self, adapter_name: str) -> None:
        """Set PEFT active adapter before computation; same as HFTrainingModel._activate_adapter."""
        module = self.module
        # Handle both FSDP v1 (wrapped) and FSDP v2 (not wrapped) cases
        if hasattr(module, "set_adapter"):
            module.set_adapter(adapter_name)
        elif hasattr(module, "module") and hasattr(module.module, "set_adapter"):
            module.module.set_adapter(adapter_name)
        else:
            raise RuntimeError(f"Cannot find set_adapter method on module: {type(module)}")

    def _reinit_adapter_weights(self, adapter_name: str) -> None:
        """Re-initialize an adapter's LoRA weights to PEFT's fresh-init state.

        lora_A -> kaiming_uniform (a=sqrt(5)); lora_B -> zeros. With lora_B zero the
        adapter contributes nothing on top of the base model (identity delta), i.e. a
        clean start. Works for both plain tensors and FSDP2 DTensors (operates on the
        local shard; every rank resets its own shard).
        """
        _match = f".{adapter_name}."
        with torch.no_grad():
            for name, param in self.module.named_parameters():
                if _match not in name:
                    continue
                data = param.to_local() if isinstance(param, DTensor) else param.data
                if ".lora_B." in name:
                    data.zero_()
                elif ".lora_A." in name:
                    torch.nn.init.kaiming_uniform_(data, a=math.sqrt(5))

    def _reset_adapter_slot(self, adapter_name: str) -> None:
        """Reset a slot to a fresh-adapter state.

        A reused slot would otherwise inherit the previous run's trained LoRA weights,
        Adam momentum / step count, and any unconsumed accumulated gradients. The HF
        backend creates a fresh adapter + fresh AdamW per create_adapter; this keeps
        the two backends consistent.
        """
        info = self._adapters.get(adapter_name)
        if info is None:
            return
        # Drop the optimizer so a fresh AdamW (no stale momentum/step) is recreated.
        info.optimizer = None
        info.step_count = 0
        self._reinit_adapter_weights(adapter_name)
        # Clear any unconsumed accumulated gradients on this adapter's parameters.
        _match = f".{adapter_name}."
        for name, param in self.module.named_parameters():
            if _match in name and param.grad is not None:
                param.grad = None

    def forward_backward(
        self,
        adapter_name: str,
        data: list[types.Datum],
        loss_fn_name: str,
        loss_fn_config: dict[str, float] | None,
        micro_batch_size: int,
        forward_only: bool = False,
        replicated: bool = False,
    ) -> Dict[str, Any]:
        """Run forward/backward without stepping or clearing accumulated gradients.

        NOTE: We deliberately do NOT call optimizer.zero_grad() here. Multiple
        consecutive forward_backward calls between two optim_step calls will
        accumulate gradients (standard PyTorch grad-accumulation semantics).
        zero_grad happens inside optim_step() at the end of each training step
        (see optim_step below).

        This is required for two cases:
          1. Per-call internal micro-batch accumulation (callers split a large
             batch into chunks of ModelConfig.micro_batch_size and rely on grads
             persisting across the loop).
          2. Cross-call grad accumulation (e.g. trinity v22: 32 forward_backward
             calls + 1 optim_step), which removes mini-batch SGD intra-step
             off-policy drift in PPO-style RL training.
        """
        self._activate_adapter(adapter_name)
        info = self._adapters[adapter_name]
        if info.optimizer is None:
            self._create_optimizer_for_adapter(adapter_name)
        self.module.train()
        return fsdp_forward_backward(
            self.module,
            data,
            loss_fn_name,
            loss_fn_config,
            micro_batch_size,
            forward_only=forward_only,
            replicated=replicated,
        )

    def optim_step(
        self,
        adapter_name: str,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        betas: Optional[tuple[float, float]] = None,
        eps: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._activate_adapter(adapter_name)
        info = self._adapters[adapter_name]
        if info.optimizer is None:
            self._create_optimizer_for_adapter(adapter_name)
        opt = info.optimizer
        if learning_rate is not None:
            for pg in opt.param_groups:
                pg["lr"] = learning_rate
        if weight_decay is not None:
            for pg in opt.param_groups:
                pg["weight_decay"] = weight_decay
        if betas is not None:
            for pg in opt.param_groups:
                pg["betas"] = tuple(betas)
        if eps is not None:
            for pg in opt.param_groups:
                pg["eps"] = eps
        if grad_clip_norm is not None and grad_clip_norm > 0:
            # Clip only this adapter's parameters (those in its optimizer). Clipping
            # self.module.parameters() would also rescale pending accumulated gradients
            # of other adapters/runs sharing the base model (cross-contamination), and
            # would skew this run's own global norm.
            clip_params = [p for pg in opt.param_groups for p in pg["params"]]
            torch.nn.utils.clip_grad_norm_(clip_params, grad_clip_norm)
        opt.step()
        opt.zero_grad()
        info.step_count += 1
        return {"step_count": info.step_count, "adapter": adapter_name}

    def save_checkpoint(self, adapter_name: str, path: str | Path, optimizer: bool = True) -> None:
        """Save adapter.pt (training load_state) + PEFT format (sampling); optional optimizer."""
        self._activate_adapter(adapter_name)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # FSDP v2: parameters may be DTensor, need to call full_tensor() to get full param
        # Collect full state dict for the adapter keyed by slot-independent canonical
        # names, so load works even when the slot name differs (restart + fallback).
        state = {}
        _match = f".{adapter_name}."
        for name, param in self.module.named_parameters():
            if _match in name:
                key = _canonical_lora_key(name)
                if isinstance(param, DTensor):
                    # FSDP v2: gather full tensor from all ranks
                    state[key] = param.full_tensor().cpu().clone()
                else:
                    state[key] = param.data.cpu().clone()

        # Only rank 0 saves to avoid duplicate writes
        import torch.distributed as dist

        is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0

        # Move any existing path/adapter_name/ contents to path/ first, so our writes below
        # are not overwritten by unstripped PEFT files (vLLM requires keys like .lora_A.weight).
        lora_subdir = path / adapter_name
        if is_rank_0 and lora_subdir.exists() and lora_subdir.is_dir():
            for item in lora_subdir.iterdir():
                dest = path / item.name
                if dest.exists():
                    if dest.is_file():
                        dest.unlink()
                    elif dest.is_dir():
                        shutil.rmtree(dest)
                shutil.move(str(item), str(dest))
            lora_subdir.rmdir()

        if is_rank_0:
            # 1) Internal format for FSDP load_state
            torch.save(state, path / "adapter.pt")

            if optimizer:
                info = self._adapters[adapter_name]
                if info.optimizer is not None:
                    torch.save(info.optimizer.state_dict(), path / "optimizer.pt")

            # 2) PEFT format (adapter_config.json, adapter_model.safetensors)
            # so sampling/VLLM can load
            # FSDP v2: PEFT's save_pretrained cannot handle DTensor directly.
            # We manually save the adapter config and weights in PEFT format.
            import json

            # Save adapter_config.json
            # Prefer the runtime peft_config attribute when accessible, but always
            # fall back to constructing a minimal config from AdapterInfo so the
            # file is ALWAYS written regardless of FSDP v2 attribute visibility.
            module = self.module
            has_nested_peft = hasattr(module, "module") and hasattr(module.module, "peft_config")
            peft_model = module.module if has_nested_peft else module

            config_dict: dict | None = None
            if hasattr(peft_model, "peft_config") and adapter_name in peft_model.peft_config:
                runtime_cfg = peft_model.peft_config[adapter_name]
                config_dict = runtime_cfg.to_dict()
                assert config_dict is not None
                for key, value in config_dict.items():
                    if isinstance(value, set):
                        config_dict[key] = list(value)

            if config_dict is None:
                # Fallback: construct a minimal but vLLM-compatible adapter_config.json
                # from the AdapterInfo that was stored during initialize().
                info = self._adapters[adapter_name]
                config_dict = {
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "base_model_name_or_path": self.model_config.path,
                    "r": info.rank,
                    "lora_alpha": info.lora_alpha,
                    "target_modules": list(info.target_modules),
                    "lora_dropout": 0.0,
                    "fan_in_fan_out": False,
                    "bias": "none",
                    "modules_to_save": None,
                    "init_lora_weights": True,
                    "layers_to_transform": None,
                    "layers_pattern": None,
                    "inference_mode": True,
                }
                self.logger.warning(
                    "peft_config not accessible on module after FSDP v2 wrap; "
                    "writing adapter_config.json from AdapterInfo for adapter '%s'",
                    adapter_name,
                )

            with open(path / "adapter_config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

            # Save adapter weights in safetensors format for sampling/vLLM.
            # vLLM lora/utils.py expects keys ending in ".lora_A.weight" or ".lora_B.weight"
            # (parts[-2] in ["lora_A","lora_B"]). `state` is already keyed canonically
            # (slot name stripped), which is exactly the layout vLLM expects.
            peft_state = dict(state)
            # Only emit `language_model.` alias keys for models whose vLLM
            # implementation nests the text backbone under `language_model`
            # (e.g. Qwen3.5); emitting them unconditionally would double the
            # checkpoint size and vLLM GPU load-time memory for every model.
            if vllm_nests_language_model(self.model_config.path):
                peft_state = add_language_model_aliases(peft_state)

            _write_adapter_weights_file(self.logger, adapter_name, path, peft_state)

    def load_checkpoint(self, adapter_name: str, path: str | Path, optimizer: bool = True) -> None:
        path = Path(path)

        state = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
        # Build a slot-name-independent lookup: canonicalize saved keys so a checkpoint
        # saved under ANY slot name loads into the current slot. Without this, a slot
        # name mismatch (common after restart + create_adapter fallback) would silently
        # match zero parameters and continue training from stale weights.
        canonical_state = {_canonical_lora_key(k): v for k, v in state.items()}
        self._activate_adapter(adapter_name)
        matched = 0
        with torch.no_grad():
            _match = f".{adapter_name}."
            for name, param in self.module.named_parameters():
                if _match not in name:
                    continue
                key = _canonical_lora_key(name)
                if key not in canonical_state:
                    continue
                matched += 1
                loaded_tensor = canonical_state[key]
                if isinstance(param, DTensor):
                    _copy_full_tensor_into_dtensor(param, loaded_tensor)
                else:
                    param.data.copy_(loaded_tensor.to(param.device))
        if matched == 0:
            raise RuntimeError(
                f"load_checkpoint matched 0 parameters for adapter '{adapter_name}' from "
                f"{path / 'adapter.pt'}; the checkpoint key layout is incompatible."
            )
        if optimizer:
            opt_path = path / "optimizer.pt"
            if opt_path.exists():
                if self._adapters[adapter_name].optimizer is None:
                    self._create_optimizer_for_adapter(adapter_name)
                opt_state = torch.load(opt_path, map_location="cpu", weights_only=True)
                self._adapters[adapter_name].optimizer.load_state_dict(opt_state)

    def allocate_slot(self, rank: int) -> Optional[str]:
        """Allocate an unused slot for rank, reset it to a fresh state, return name."""
        for name in self._adapters_by_rank.get(rank, []):
            if not self._allocated.get(name, False):
                self._allocated[name] = True
                self._reset_adapter_slot(name)
                return name
        return None

    def reserve_slot(self, adapter_name: str) -> None:
        """Mark a specific slot allocated and reset it.

        Used to synchronize non-lead ranks to the slot chosen by the lead rank, so
        every rank resets weights/optimizer identically for the same adapter name.
        """
        if adapter_name in self._adapters:
            self._allocated[adapter_name] = True
            self._reset_adapter_slot(adapter_name)

    def release_slot(self, adapter_name: str) -> None:
        # Reset on release so a freed slot does not retain the run's trained weights,
        # optimizer state, or gradients (allocate_slot also resets defensively).
        self._reset_adapter_slot(adapter_name)
        self._allocated.pop(adapter_name, None)

    def list_adapters(self) -> List[str]:
        return list(self._adapters.keys())


class FullParamFSDPWorker:
    """One exclusive full-parameter training model, optimizer, and checkpoint."""

    FULL_RUN_ID = "__full_param__"

    def __init__(self, model_config: FSDPModelConfig) -> None:
        self.model_config = model_config
        self.module: Any = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.step_count = 0
        self.bound_run_id: str | None = None
        self._initialized = False
        self.logger = logging.getLogger(f"{__name__}.FullParamFSDPWorker")

    def initialize(self) -> None:
        if self._initialized:
            return

        model = build_base_model(self.model_config)
        for param in model.parameters():
            param.requires_grad = True
        self.module = _fully_shard_model(model)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.module.parameters() if p.requires_grad],
            lr=1e-4,
            weight_decay=0.01,
        )
        self._initialized = True
        self.logger.info("[FSDP] full-param worker initialized")

    def bind_run(self, run_id: str) -> None:
        if self.bound_run_id is None:
            self.bound_run_id = run_id
            return
        if self.bound_run_id != run_id:
            raise RuntimeError(
                f"Full-param worker is already bound to {self.bound_run_id}; "
                f"refusing to share it with {run_id}."
            )

    def release_run(self, run_id: str) -> None:
        if self.bound_run_id != run_id:
            raise RuntimeError(f"Full-param worker is bound to {self.bound_run_id}, not {run_id}.")
        self.optimizer = None
        self.step_count = 0
        self.bound_run_id = None

    def _require_run(self, run_id: str) -> None:
        if self.bound_run_id != run_id:
            raise RuntimeError(
                f"Full-param worker is not bound to run {run_id}; "
                f"call create_adapter before forward/optim/checkpoint operations."
            )

    def forward_backward(
        self,
        run_id: str,
        data: list[types.Datum],
        loss_fn_name: str,
        loss_fn_config: dict[str, float] | None,
        micro_batch_size: int,
        forward_only: bool = False,
        replicated: bool = False,
    ) -> Dict[str, Any]:
        self._require_run(run_id)
        if self.optimizer is None:
            self._create_optimizer()
        self.module.train()
        return fsdp_forward_backward(
            self.module,
            data,
            loss_fn_name,
            loss_fn_config,
            micro_batch_size,
            forward_only=forward_only,
            replicated=replicated,
        )

    def optim_step(
        self,
        run_id: str,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        betas: Optional[tuple[float, float]] = None,
        eps: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._require_run(run_id)
        if self.optimizer is None:
            self._create_optimizer()
        opt = self.optimizer
        if learning_rate is not None:
            for pg in opt.param_groups:
                pg["lr"] = learning_rate
        if weight_decay is not None:
            for pg in opt.param_groups:
                pg["weight_decay"] = weight_decay
        if betas is not None:
            for pg in opt.param_groups:
                pg["betas"] = tuple(betas)
        if eps is not None:
            for pg in opt.param_groups:
                pg["eps"] = eps
        # Must run before clipping and zero_grad, both of which destroy the
        # pre-clip gradients that the cross-validation needs.
        diag = self.grad_diagnostics()
        self.logger.info(
            "[grad] preclip_norm=%.6f num_params_with_grad=%d probes=%s",
            diag.get("grad_norm_preclip", 0.0),
            int(diag.get("num_params_with_grad", 0)),
            {
                k.split("/", 1)[1]: round(v, 6)
                for k, v in diag.items()
                if k.startswith("grad_norm_probe/")
            },
        )
        if grad_clip_norm is not None and grad_clip_norm > 0:
            params = [p for pg in opt.param_groups for p in pg["params"]]
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
        opt.step()
        opt.zero_grad()
        self.step_count += 1
        return {"step_count": self.step_count, "run_id": run_id, **diag}

    def grad_diagnostics(self) -> dict:
        """Pre-clip gradient norms, for cross-validating the backward pass.

        Squared norms are additive across FSDP2 shards, so each rank sums its
        local contribution and one scalar all-reduce recovers the global value.
        Gathering every gradient with ``full_tensor()`` instead would
        materialize each parameter unsharded (``embed_tokens`` alone is 1.2 GiB
        in fp32) and issue one collective per parameter on every optimizer step.
        """
        probe_names = (
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.27.self_attn.o_proj.weight",
        )
        named = dict(self.module.named_parameters())

        local_total = 0.0
        count = 0
        device = None
        for p in self.module.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            g = p.grad.detach()
            device = g.device
            local_total += float(g.float().pow(2).sum())
            count += 1

        local_probes = []
        for name in probe_names:
            p = named.get(name)
            if p is not None and p.grad is not None:
                local_probes.append(float(p.grad.detach().float().pow(2).sum()))
            else:
                local_probes.append(0.0)

        values = [local_total, *local_probes]
        if device is not None:
            import torch.distributed as dist

            if dist.is_initialized() and dist.get_world_size() > 1:
                buf = torch.tensor(values, device=device, dtype=torch.float64)
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                values = buf.tolist()

        out = {"grad_norm_preclip": values[0] ** 0.5, "num_params_with_grad": float(count)}
        for name, sq in zip(probe_names, values[1:], strict=True):
            out[f"grad_norm_probe/{name}"] = sq**0.5
        return out

    def _create_optimizer(self) -> None:
        self.optimizer = torch.optim.AdamW(
            [p for p in self.module.parameters() if p.requires_grad],
            lr=1e-4,
            weight_decay=0.01,
        )

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            self._create_optimizer()
        if self.optimizer is None:
            raise RuntimeError("optimizer could not be created for full-param worker")
        return self.optimizer

    def save_checkpoint(self, run_id: str, path: Path, optimizer: bool = True) -> None:
        self._require_run(run_id)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_state_dict,
        )

        opt = self._ensure_optimizer()

        model_state, optim_state = get_state_dict(
            self.module,
            opt,
            options=StateDictOptions(strict=True, flatten_optimizer_state_dict=False),
        )
        state: Dict[str, Any] = {
            "model": model_state,
            "step_count": torch.tensor(self.step_count, dtype=torch.long),
        }
        if optimizer:
            state["optimizer"] = optim_state
        dcp.save(state, checkpoint_id=str(path))

    def load_checkpoint(self, run_id: str, path: Path, optimizer: bool = True) -> None:
        self._require_run(run_id)
        path = Path(path)
        if not (path / ".metadata").exists():
            raise FileNotFoundError(f"{path} is not a complete full-parameter training checkpoint.")

        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_state_dict,
            set_state_dict,
        )

        opt = self._ensure_optimizer()

        model_state, optim_state = get_state_dict(
            self.module,
            opt,
            options=StateDictOptions(strict=True, flatten_optimizer_state_dict=False),
        )
        state: Dict[str, Any] = {
            "model": model_state,
            "step_count": torch.tensor(self.step_count, dtype=torch.long),
        }
        if optimizer:
            state["optimizer"] = optim_state
        dcp.load(state, checkpoint_id=str(path))

        if not optimizer:
            optim_state = {}
        set_state_dict(
            self.module,
            opt,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=StateDictOptions(strict=True, flatten_optimizer_state_dict=False),
        )
        self.step_count = int(state["step_count"].item())

    def save_sampler_checkpoint(self, run_id: str, path: Path) -> None:
        self._require_run(run_id)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        import torch.distributed as dist
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        full_state = get_model_state_dict(
            self.module,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True, strict=True),
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            config = AutoConfig.from_pretrained(self.model_config.path)
            export_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.path,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            # Master weights are fp32; vLLM loads bf16, so cast down before
            # writing or the exported directory doubles in size.
            export_state = {
                k: v.to(torch.bfloat16)
                for k, v in full_state.items()
                if isinstance(v, torch.Tensor)
            }
            missing, unexpected = export_model.load_state_dict(export_state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Full sampler export mismatch: missing={missing}, unexpected={unexpected}"
                )
            export_model.save_pretrained(path, state_dict=export_state, safe_serialization=True)
            # vLLM serves this directory standalone, so it needs the tokenizer
            # (including the think tokens its reasoning parser looks up).
            AutoTokenizer.from_pretrained(self.model_config.path).save_pretrained(path)


# =============================================================================
# FSDPWorkerActor: Ray actor, one GPU per process, forms torch.distributed with peers
# =============================================================================


class FSDPWorkerActor:
    """Single-GPU Ray actor; N form process group via init_dist, each holds one worker."""

    def __init__(self, rank: int, world_size: int, config_dict: dict) -> None:
        self.rank = rank
        self.world_size = world_size
        self.config_dict = config_dict
        self._worker: Optional[MultiAdapterFSDPWorker | FullParamFSDPWorker] = None
        self._dist_initialized = False
        self.logger = logging.getLogger(f"{__name__}.FSDPWorkerActor")

    def get_node_ip(self) -> str:
        import ray

        return ray.util.get_node_ip_address()

    def init_dist(self, master_addr: str, master_port: int = DEFAULT_MASTER_PORT) -> None:
        import torch.distributed as dist

        if self._dist_initialized:
            return
        import os

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        # Must set CUDA device before init_process_group to avoid DeviceMesh
        # picking the wrong GPU (PyTorch 2.6+ creates DeviceMesh internally).
        # Each actor is a Ray num_gpus=1 process: Ray sets CUDA_VISIBLE_DEVICES
        # to a single physical GPU that torch sees as cuda:0. Do NOT use the
        # global rank (self.rank) here — rank>=1 would select a nonexistent
        # local device and fail before NCCL init. Exactly one device is visible,
        # so always pin index 0.
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"tcp://{master_addr}:{master_port}",
        )
        self._dist_initialized = True

    def build_worker(self) -> None:
        if self._worker is not None:
            return
        model_config, slot_config, training_mode = _worker_dict_to_configs(self.config_dict)
        if training_mode == "full_param":
            self._worker = FullParamFSDPWorker(model_config=model_config)
        else:
            self._worker = MultiAdapterFSDPWorker(
                model_config=model_config,
                slot_config=slot_config,
            )
        logging.info("[SERVER][Actor] build_worker 调用 initialize 前 rank=%s", self.rank)
        self._worker.initialize()
        logging.info("[SERVER][Actor] build_worker initialize 返回 rank=%s", self.rank)

    def bind_run(self, run_id: str) -> None:
        if isinstance(self._worker, FullParamFSDPWorker):
            self._worker.bind_run(run_id)

    def release_run(self, run_id: str) -> None:
        if isinstance(self._worker, FullParamFSDPWorker):
            self._worker.release_run(run_id)

    def save_sampler_checkpoint(self, run_id: str, path: str) -> None:
        if isinstance(self._worker, FullParamFSDPWorker):
            self._worker.save_sampler_checkpoint(run_id, Path(path))

    def shutdown(self) -> None:
        import torch.distributed as dist

        self._worker = None
        if dist.is_initialized():
            dist.destroy_process_group()
        self._dist_initialized = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def allocate_slot(self, rank: int) -> Optional[str]:
        if isinstance(self._worker, MultiAdapterFSDPWorker):
            return self._worker.allocate_slot(rank)
        return None

    def reserve_slot(self, adapter_name: str) -> None:
        if isinstance(self._worker, MultiAdapterFSDPWorker):
            self._worker.reserve_slot(adapter_name)

    def release_slot(self, adapter_name: str) -> None:
        if isinstance(self._worker, MultiAdapterFSDPWorker):
            self._worker.release_slot(adapter_name)

    def forward_backward(
        self,
        data: list,
        adapter_name: str,
        loss_fn_name: str,
        loss_fn_config: Optional[dict] = None,
        forward_only: bool = False,
        micro_batch_size: Optional[int] = None,
        replicated: bool = False,
    ) -> Dict[str, Any]:
        """Run forward (+backward) on this actor's data shard.

        data: list[types.Datum] (or dicts after Ray serialization).

        The torch-native engine splits this shard into contiguous micro-batches
        and accumulates gradients without stepping or clearing them. The caller
        guarantees every rank runs the same number of micro-batches so FSDP2
        collectives remain symmetric.

        replicated: every rank received the identical full batch (small-batch
        fallback); skips the world_size loss compensation in the engine.

        Caller is responsible for invoking `optim_step` afterwards
        (which will step + zero_grad).
        """
        if data and isinstance(data[0], dict):
            data = [types.Datum(**d) for d in data]

        if not data or self._worker is None:
            return {
                "metrics": {},
                "loss_fn_outputs": [],
            }

        # Cross-rank micro-batch count symmetry is decided by the backend, which
        # sees every shard length and picks a size that yields the same count on
        # all ranks. Honour it as given: falling back to the whole shard here is
        # what puts a large batch into one micro-batch and OOMs the GPU.
        mb = micro_batch_size if micro_batch_size and micro_batch_size > 0 else len(data)
        n_micro = (len(data) + mb - 1) // mb
        out = self._worker.forward_backward(
            adapter_name,
            data,
            loss_fn_name,
            loss_fn_config,
            mb,
            forward_only=forward_only,
            replicated=replicated,
        )
        metrics = dict(out.get("metrics") or {})
        metrics["actor/num_micro_batches"] = float(n_micro)
        all_outputs = _fsdp_logprobs_to_loss_fn_outputs(out, data)

        return {
            "metrics": metrics,
            "loss_fn_outputs": all_outputs,
        }

    def optim_step(
        self,
        adapter_name: str,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        betas: Optional[tuple[float, float]] = None,
        eps: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self._worker is None:
            return {}
        return self._worker.optim_step(
            adapter_name, learning_rate, weight_decay, grad_clip_norm, betas, eps
        )

    def save_checkpoint(
        self,
        adapter_name: str,
        path: str,
        optimizer: bool = True,
        sampler: bool = False,
    ) -> None:
        # FSDP v2: all ranks must participate in collective checkpoint operations.
        if self._worker is None:
            return
        if isinstance(self._worker, FullParamFSDPWorker):
            if sampler:
                self._worker.save_sampler_checkpoint(adapter_name, Path(path))
            else:
                self._worker.save_checkpoint(adapter_name, Path(path), optimizer)
            return
        self._worker.save_checkpoint(adapter_name, Path(path), optimizer)

    def load_checkpoint(self, adapter_name: str, path: str, optimizer: bool = True) -> None:
        if self._worker is None:
            return
        self._worker.load_checkpoint(adapter_name, Path(path), optimizer)


# =============================================================================
# FSDPTrainingBackend (implements BaseTrainingBackend)
# Multi-GPU: N GPUs = N FSDPWorkerActor processes, forming torch.distributed.
# =============================================================================


class FSDPTrainingBackend(BaseTrainingBackend):
    """
    Multi-node/multi-GPU training backend; peer to HFTrainingBackend.
    Selected via ModelConfig.training_backend = "fsdp".
    Uses one Ray actor per GPU, forming a process group for FSDP2.
    """

    def __init__(
        self,
        config: ModelConfig,
        fsdp_index: Optional[int] = None,
        worker_venv_path: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self._fsdp_index = fsdp_index  # Index among FSDP models; port = base + fsdp_index
        self._worker_venv_path = worker_venv_path
        self._training_mode = getattr(config, "training_mode", "lora")
        self._worker: Optional[MultiAdapterFSDPWorker | FullParamFSDPWorker] = None
        self._actors: List[Any] = []
        self._world_size: int = 0
        self._full_run_id: Optional[str] = None
        self._lora_id_to_adapter_name: Dict[str, str] = {}
        self._adapter_name_to_lora_id: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        rank_slots = _get_rank_slots_from_config(config)
        self._slot_config = SlotPoolConfig(
            rank_slots=rank_slots,
        )
        self._config_dict = _config_to_worker_dict(config)
        self.logger = logging.getLogger(f"{__name__}.FSDPTrainingBackend")

    async def shutdown(self) -> None:
        """Kill all FSDP worker Ray actors and release GPU resources."""
        import ray

        for actor in self._actors:
            try:
                await asyncio.to_thread(ray.get, actor.shutdown.remote(), timeout=10)
            except Exception:
                pass
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        self._actors = []
        self._worker = None
        self._world_size = 0
        self._full_run_id = None
        self._lora_id_to_adapter_name.clear()
        self._adapter_name_to_lora_id.clear()

    async def async_init(self) -> None:
        if self._world_size > 0 or self._worker is not None:
            return
        n_gpus = getattr(self.config, "fsdp_num_gpus", 1)
        n_gpus = max(1, int(n_gpus))
        use_ray = os.environ.get("TUFT_FSDP_NO_RAY") != "1"
        if not use_ray and n_gpus != 1:
            raise ValueError(
                "TUFT_FSDP_NO_RAY=1 (no Ray) requires fsdp_num_gpus=1. "
                f"Got fsdp_num_gpus={n_gpus}. Multi-GPU requires Ray."
            )
        if not use_ray:
            # Local single-process: no Ray actors; for standalone tests (train/save logic)
            import torch.distributed as dist

            if not dist.is_available() or not dist.is_initialized():
                import socket

                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                if "MASTER_PORT" not in os.environ:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("", 0))
                        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
                os.environ.setdefault("RANK", "0")
                os.environ.setdefault("WORLD_SIZE", "1")
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    rank=0,
                    world_size=1,
                )
            model_config, slot_config, training_mode = _worker_dict_to_configs(self._config_dict)
            self._worker = (
                FullParamFSDPWorker(model_config=model_config)
                if training_mode == "full_param"
                else MultiAdapterFSDPWorker(model_config=model_config, slot_config=slot_config)
            )
            await asyncio.to_thread(self._worker.initialize)
            self._world_size = 1
            return
        import ray

        config_dict = self._config_dict
        _venv = self._worker_venv_path
        base_env_vars: Dict[str, str] = {}
        if not _venv or not _venv.strip():
            self.logger.warning(
                "worker_venv_path is not set. Recommend using a virtual environment for Ray FSDP; "
                "set worker_venv_path in config if all nodes use the same venv. "
                "Proceeding with empty runtime_env (relying on node-installed packages)."
            )
            venv_python = None
        else:
            _path = os.environ.get("PATH", "")
            venv_python = str(Path(_venv) / "bin" / "python")
            base_env_vars = {
                "VIRTUAL_ENV": _venv,
                "PATH": f"{_venv}/bin:{_path}",
            }

        _fsdp_gpu_list = [
            g.strip() for g in os.environ.get("TUFT_FSDP_GPUS", "").split(",") if g.strip()
        ]
        if _fsdp_gpu_list:
            if len(set(_fsdp_gpu_list)) != len(_fsdp_gpu_list):
                raise ValueError(f"TUFT_FSDP_GPUS contains duplicates: {_fsdp_gpu_list}")
            if len(_fsdp_gpu_list) != n_gpus:
                raise ValueError(
                    f"TUFT_FSDP_GPUS has {len(_fsdp_gpu_list)} GPUs but fsdp_num_gpus={n_gpus}"
                )

        actors = []
        try:
            for r in range(n_gpus):
                actor_env = dict(base_env_vars)
                if _fsdp_gpu_list:
                    actor_env["CUDA_VISIBLE_DEVICES"] = _fsdp_gpu_list[r]
                actor_runtime_env: Dict[str, Any] = {"env_vars": actor_env}
                if venv_python:
                    actor_runtime_env["py_executable"] = venv_python
                actor = (
                    ray.remote(FSDPWorkerActor)
                    .options(
                        num_gpus=0 if _fsdp_gpu_list else 1,
                        runtime_env=actor_runtime_env,
                    )
                    .remote(r, n_gpus, config_dict)
                )
                actors.append(actor)
            # Publish actor state only once all actors exist, so a failure leaves
            # nothing half-initialized that the next create_adapter would reuse.
            # get_node_ip hangs forever when an actor never schedules (e.g. no GPU).
            _GET_NODE_IP_TIMEOUT = 120
            self.logger.info("[FSDP] async_init: created %d actors, calling get_node_ip...", n_gpus)
            master_addr = await asyncio.to_thread(
                ray.get, actors[0].get_node_ip.remote(), timeout=_GET_NODE_IP_TIMEOUT
            )
            self.logger.info("[FSDP] get_node_ip OK: %s, calling init_dist...", master_addr)
            base_port = getattr(self.config, "fsdp_master_port", DEFAULT_MASTER_PORT)
            master_port = (
                base_port + self._fsdp_index if self._fsdp_index is not None else base_port
            )
            await asyncio.gather(
                *[
                    asyncio.to_thread(ray.get, a.init_dist.remote(master_addr, master_port))
                    for a in actors
                ]
            )
            self.logger.info("[FSDP] init_dist OK, calling build_worker...")
            await asyncio.gather(
                *[asyncio.to_thread(ray.get, a.build_worker.remote()) for a in actors]
            )
        except Exception:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    pass
            raise
        self.logger.info("[FSDP] build_worker OK, FSDP backend ready")
        self._actors = actors
        self._world_size = n_gpus

    def _get_adapter_name(self, lora_id: str) -> str:
        if lora_id not in self._lora_id_to_adapter_name:
            raise ValueError(f"Unknown lora_id: {lora_id}; call create_adapter first.")
        return self._lora_id_to_adapter_name[lora_id]

    async def release_run(self, run_id: str) -> None:
        """Give up the single full-param slot so another run can take it.

        Mirrors remove_adapter's full-param branch. A resumed client arrives
        under a new session with a new run id, and create_adapter refuses a
        second full-param run while this backend still hosts one.
        """
        async with self._lock:
            if self._training_mode != "full_param" or self._full_run_id != run_id:
                return
            self._full_run_id = None
            self._lora_id_to_adapter_name.pop(run_id, None)
            self._adapter_name_to_lora_id.pop(run_id, None)
            if isinstance(self._worker, FullParamFSDPWorker):
                await asyncio.to_thread(self._worker.release_run, run_id)
            elif self._actors:
                import ray

                await asyncio.gather(
                    *[
                        asyncio.to_thread(ray.get, a.release_run.remote(run_id))
                        for a in self._actors
                    ]
                )

    async def create_adapter(self, lora_id: str, lora_config: types.LoraConfig) -> None:
        async with self._lock:
            if self._world_size == 0 and self._worker is None and not self._actors:
                await self.async_init()

            if self._training_mode == "full_param":
                if self._full_run_id is None:
                    self._full_run_id = lora_id
                elif self._full_run_id != lora_id:
                    raise ValueError(
                        f"Full-param backend already hosts {self._full_run_id}; "
                        f"refusing to create a second full-param run {lora_id}."
                    )
                if isinstance(self._worker, FullParamFSDPWorker):
                    await asyncio.to_thread(self._worker.bind_run, lora_id)
                elif self._actors:
                    import ray

                    await asyncio.gather(
                        *[
                            asyncio.to_thread(ray.get, a.bind_run.remote(lora_id))
                            for a in self._actors
                        ]
                    )
                self._lora_id_to_adapter_name[lora_id] = lora_id
                self._adapter_name_to_lora_id[lora_id] = lora_id
                return

            rank = getattr(lora_config, "rank", 8)
            if isinstance(self._worker, MultiAdapterFSDPWorker):
                adapter_name = await asyncio.to_thread(self._worker.allocate_slot, rank)
            elif self._actors:
                import ray

                adapter_name: str | None = await asyncio.to_thread(
                    ray.get, self._actors[0].allocate_slot.remote(rank)
                )
                if adapter_name is not None and len(self._actors) > 1:
                    # Every rank shares the base model; sync the remaining ranks to the
                    # same slot and reset weights/optimizer identically on each of them.
                    await asyncio.gather(
                        *[
                            asyncio.to_thread(ray.get, a.reserve_slot.remote(adapter_name))
                            for a in self._actors[1:]
                        ]
                    )
            else:
                raise RuntimeError("FSDPTrainingBackend not initialized.")
            if adapter_name is None:
                raise ValueError(f"No free slot for rank={rank}; all slots allocated.")
            self._lora_id_to_adapter_name[lora_id] = adapter_name
            self._adapter_name_to_lora_id[adapter_name] = lora_id

    async def remove_adapter(self, lora_id: str) -> None:
        async with self._lock:
            adapter_name = self._lora_id_to_adapter_name.pop(lora_id, None)
            if adapter_name:
                self._adapter_name_to_lora_id.pop(adapter_name, None)
                if self._training_mode == "full_param":
                    self._full_run_id = None
                    if isinstance(self._worker, FullParamFSDPWorker):
                        await asyncio.to_thread(self._worker.release_run, lora_id)
                    elif self._actors:
                        import ray

                        await asyncio.gather(
                            *[
                                asyncio.to_thread(ray.get, a.release_run.remote(lora_id))
                                for a in self._actors
                            ]
                        )
                    return
                if isinstance(self._worker, MultiAdapterFSDPWorker):
                    await asyncio.to_thread(self._worker.release_slot, adapter_name)
                elif self._actors:
                    import ray

                    # Release (and reset) the slot on every rank, not just the lead,
                    # so no rank retains the run's weights/optimizer state.
                    await asyncio.gather(
                        *[
                            asyncio.to_thread(ray.get, a.release_slot.remote(adapter_name))
                            for a in self._actors
                        ]
                    )

    def _apply_token_budget(self, mb: int, data: list[types.Datum]) -> int:
        """Cap the per-call micro-batch size by a token budget.

        The cap uses the batch's longest datum so every rank derives the same
        effective mb from the same data (NCCL micro-batch symmetry holds). A
        single datum longer than the budget keeps mb=1 — the caller is inside
        the max_model_len envelope, which the engine has verified fits.
        """
        budget = getattr(self.config, "micro_batch_tokens", None)
        if not budget or budget <= 0 or mb <= 1 or not data:
            return mb
        max_len = max(d.model_input.length for d in data)
        capped = budget // max(max_len, 1)
        if capped < 1:
            return 1
        # Round down to a power of two so the per-micro-batch token count stays
        # predictable across calls rather than tracking each batch's longest
        # datum exactly.
        return min(mb, 1 << min(mb, capped).bit_length() - 1)

    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
    ) -> types.ForwardBackwardOutput:
        adapter_name = self._get_adapter_name(lora_id)
        loss_fn_name = (
            loss_fn if isinstance(loss_fn, str) else getattr(loss_fn, "__name__", "cross_entropy")
        )

        # Per-call internal micro-batch grad accumulation.
        #
        # When ModelConfig.micro_batch_size < len(data) (or, in multi-GPU mode,
        # < shard length), each call splits its (sharded) batch into
        # ceil(len/mb) micro-batches. Each micro-batch runs a full
        # forward+backward, and gradients accumulate across micro-batches
        # because MultiAdapterFSDPWorker.forward_backward never zero_grad's
        # at entry. The final optim_step (called by the user) consumes the
        # accumulated gradients and zero_grad's.
        #
        # This makes the FSDP backend behave like the HF backend's built-in
        # micro-batching (see hf_training_model.HFTrainingModel.forward), and
        # lets callers (e.g. Trinity-RFT v22) pass a large train batch +
        # 1 optim_step instead of N (forward_backward + optim_step) loops --
        # eliminating mini-batch SGD intra-step off-policy drift in PPO/GRPO.
        mb = int(getattr(self.config, "micro_batch_size", 0) or 0)
        mb = self._apply_token_budget(mb, data)

        if self._worker is not None:
            # NO_RAY single-process mode: serialize GPU work across runs. Ray mode is
            # safe because each actor serializes its own method calls, but here two
            # runs' asyncio.to_thread calls could otherwise race on set_adapter.
            async with self._lock:
                if mb > 0 and len(data) % mb == 0:
                    eff_mb = mb
                else:
                    eff_mb = max(len(data), 1)
                n_micro = max(len(data) // eff_mb, 1) if data else 0
                out = await asyncio.to_thread(
                    cast(Any, self._worker).forward_backward,
                    adapter_name,
                    data,
                    loss_fn_name,
                    loss_fn_config,
                    eff_mb,
                    forward_only=not backward,
                )
            metrics = dict(out.get("metrics") or {})
            metrics["actor/num_micro_batches"] = float(n_micro)
            loss_fn_outputs = _fsdp_logprobs_to_loss_fn_outputs(out, data)
        else:
            import ray

            n_actors = len(self._actors)
            if not data:
                return types.ForwardBackwardOutput(
                    loss_fn_output_type=loss_fn_name,
                    loss_fn_outputs=[],
                    metrics={},
                )

            # NCCL symmetry: each actor must run the same number of
            # micro-batches, else FSDP-2 collectives deadlock. The tinker SDK's
            # byte-budget request chunking (5MB / 1024-item greedy packing)
            # can emit 1-3 datum remainder sub-requests, which cannot be
            # sharded across ranks. Dispatch those REPLICATED instead: every
            # actor runs the identical full batch, per-rank gradients are
            # identical, and the FSDP2 reduce-scatter average recovers the
            # exact small-batch gradient (engine skips the world_size loss
            # compensation in replicated mode).
            replicated = len(data) < n_actors

            shards = _shard_list(data, n_actors) if not replicated else [data] * n_actors

            # In multi-actor mode every actor must issue the same number of
            # micro-batches, otherwise FSDP-2 NCCL collectives deadlock
            # (one rank finishes early while others are still iterating).
            # _uniform_micro_batch_size picks the largest size that satisfies
            # that while still bounding per-micro-batch memory.
            if not replicated:
                if n_actors == 1:
                    # Single rank: there is no collective to deadlock, so honour
                    # the configured micro-batch size even when it does not
                    # divide the shard. Passing None here makes the actor fall
                    # back to the whole batch as one micro-batch, which is what
                    # OOMs the GPU on batches whose size is not a multiple of mb.
                    eff_mb = mb if mb > 0 else None
                else:
                    eff_mb = _uniform_micro_batch_size([len(s) for s in shards], mb)
            else:
                # All ranks hold the same data; any mb divides it identically,
                # but tiny batches are cheap — one micro-batch keeps it simple.
                eff_mb = None

            self.logger.info(
                "FSDP multi-actor forward: batch=%d actors=%d mb=%s eff_mb=%s replicated=%s",
                len(data),
                n_actors,
                mb,
                eff_mb,
                replicated,
            )

            refs = []
            ref_weights = []
            for actor, shard in zip(self._actors, shards, strict=False):
                if not shard:
                    continue
                refs.append(
                    actor.forward_backward.remote(
                        list(shard),
                        adapter_name,
                        loss_fn_name,
                        loss_fn_config,
                        not backward,
                        eff_mb,
                        replicated,
                    )
                )
                ref_weights.append(len(shard))

            results = await asyncio.to_thread(ray.get, refs) if refs else []

            if replicated:
                # Every actor returned identical outputs; keep one copy so
                # metrics and loss_fn_outputs are not counted n_actors times.
                first = results[0] if results else {"metrics": {}, "loss_fn_outputs": []}
                metrics = dict(first.get("metrics") or {})
                loss_fn_outputs = list(first.get("loss_fn_outputs") or [])
            else:
                metrics = _merge_metrics(results, ref_weights)
                loss_fn_outputs = []
                for out in results:
                    loss_fn_outputs.extend(out.get("loss_fn_outputs", []))

        # Tinker expects every metric key to be "name:reduction" (e.g. loss:sum)
        metrics = {k: v for k, v in metrics.items() if ":" in k}

        return types.ForwardBackwardOutput(
            loss_fn_output_type=loss_fn_name,
            loss_fn_outputs=loss_fn_outputs,
            metrics=metrics,
        )

    async def optim_step(
        self,
        adam_params: types.AdamParams,
        lora_id: str,
    ) -> types.OptimStepResponse:
        adapter_name = self._get_adapter_name(lora_id)
        betas = (adam_params.beta1, adam_params.beta2)
        if self._worker is not None:
            async with self._lock:
                result = await asyncio.to_thread(
                    cast(Any, self._worker).optim_step,
                    adapter_name,
                    adam_params.learning_rate,
                    adam_params.weight_decay,
                    adam_params.grad_clip_norm,
                    betas,
                    adam_params.eps,
                )
        else:
            import ray

            refs = [
                a.optim_step.remote(
                    adapter_name,
                    adam_params.learning_rate,
                    adam_params.weight_decay,
                    adam_params.grad_clip_norm,
                    betas,
                    adam_params.eps,
                )
                for a in self._actors
            ]
            results = await asyncio.to_thread(ray.get, refs)
            result = results[0] if results else {}
        metrics = {k: float(v) for k, v in (result or {}).items() if isinstance(v, (int, float))}
        return types.OptimStepResponse(metrics=metrics or None)

    async def save_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
    ) -> None:
        adapter_name = self._get_adapter_name(lora_id)
        full_mode = self._training_mode == "full_param"
        sampler = checkpoint_record.checkpoint_type == "sampler"
        if full_mode:
            path = (
                checkpoint_record.model_path if sampler else checkpoint_record.training_state_path
            )
        else:
            path = checkpoint_record.adapter_path

        if self._worker is not None:
            async with self._lock:
                if full_mode and isinstance(self._worker, FullParamFSDPWorker):
                    if sampler:
                        await asyncio.to_thread(
                            self._worker.save_sampler_checkpoint, adapter_name, path
                        )
                    else:
                        await asyncio.to_thread(
                            self._worker.save_checkpoint, adapter_name, path, optimizer
                        )
                else:
                    await asyncio.to_thread(
                        cast(Any, self._worker).save_checkpoint, adapter_name, path, optimizer
                    )
        else:
            import ray

            refs = [
                a.save_checkpoint.remote(adapter_name, str(path), optimizer, sampler)
                for a in self._actors
            ]
            await asyncio.to_thread(ray.get, refs)

    async def load_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
    ) -> None:
        full_mode = self._training_mode == "full_param"
        if checkpoint_record.checkpoint_type != "training":
            raise ValueError("Full-param load_state only supports training checkpoints.")
        if (
            full_mode
            and getattr(checkpoint_record.metadata, "training_mode", "lora") != "full_param"
        ):
            raise ValueError("Cannot load a LoRA checkpoint into a full-param training backend.")

        if lora_id not in self._lora_id_to_adapter_name:
            rank = getattr(checkpoint_record.metadata, "lora_rank", None) or 8
            await self.create_adapter(lora_id, types.LoraConfig(rank=rank))
        adapter_name = self._get_adapter_name(lora_id)
        path = (
            checkpoint_record.training_state_path if full_mode else checkpoint_record.adapter_path
        )
        if self._worker is not None:
            async with self._lock:
                await asyncio.to_thread(
                    cast(Any, self._worker).load_checkpoint, adapter_name, path, optimizer
                )
        else:
            import ray

            refs = [
                a.load_checkpoint.remote(adapter_name, str(path), optimizer) for a in self._actors
            ]
            await asyncio.to_thread(ray.get, refs)
