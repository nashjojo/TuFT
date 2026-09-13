"""Torch-native forward/backward utilities for the FSDP training backend.

This module intentionally owns only the small model-execution surface TuFT needs:
model construction, contiguous micro-batching, next-token log-prob extraction, and
loss/backward execution. FSDP wrapping, adapter management, optimizers, checkpoints,
and distributed orchestration remain in :mod:`fsdp_training_backend`.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from tinker import types
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForCausalLM

from tuft.loss_fn import get_loss_fn


_RLHF_LOSS_FNS = {"ppo", "grpo", "cispo", "importance_sampling", "dro", "trinity_ppo"}

logger = logging.getLogger(__name__)


def _fsdp_world_size() -> int:
    """Number of ranks in the FSDP process group (1 when not distributed)."""

    import torch.distributed as dist

    return dist.get_world_size() if dist.is_initialized() else 1


@dataclass
class FSDPModelConfig:
    """Minimal serializable model configuration used by FSDP workers."""

    path: str
    max_model_len: int
    attn_implementation: str | None = None
    override_config: dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True
    checkpoint_skip_layers: int = 0


@dataclass
class MicroBatch:
    """Padded model inputs/labels plus the true per-sample sequence lengths."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    lengths: list[int]


def _explicit_target_tokens(datum: types.Datum, device: torch.device | str) -> torch.Tensor | None:
    value = (datum.loss_fn_inputs or {}).get("target_tokens")
    if value is None:
        return None
    return value.to_torch().to(device=device, dtype=torch.long).reshape(-1)


def build_base_model(config: FSDPModelConfig) -> Any:
    """Load a single CPU copy of the base model and enable checkpointed training."""

    override = dict(config.override_config)
    attn_implementation = override.pop("attn_implementation", config.attn_implementation)
    hf_config = AutoConfig.from_pretrained(
        config.path,
        trust_remote_code=config.trust_remote_code,
    )
    for key, value in override.items():
        setattr(hf_config, key, value)

    model_kwargs: dict[str, Any] = {
        "config": hf_config,
        # fp32 master weights: with bf16 parameters AdamW's moments are bf16 too,
        # and a step at lr=1e-5 is ~10x smaller than the bf16 ulp for weights of
        # magnitude ~0.027, so updates round away and training stalls. FSDP2's
        # MixedPrecisionPolicy still casts to bf16 for compute.
        "dtype": torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": config.trust_remote_code,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(config.path, **model_kwargs)
    model.enable_input_require_grads()
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
        # Optional partial checkpointing: uncheckpoint the LAST N decoder layers
        # (their activations persist; earlier layers still recompute). Trades
        # some activation memory for a fraction of the recompute FLOPs — the
        # full-OFF variant OOMs at real ALFWorld micro-batch sizes, this is the
        # tunable middle ground. A diagnostic flag file overrides the config so
        # fractions can be swept without a server restart.
        skip = config.checkpoint_skip_layers
        skip_file = "/tmp/tuft_fsdp_skip_ckpt_layers"
        if os.path.exists(skip_file):
            try:
                with open(skip_file) as fh:
                    skip = int(fh.read().strip() or "0")
            except (OSError, ValueError):
                pass
        layers = getattr(getattr(model, "model", None), "layers", None)
        if skip > 0 and layers is not None:
            for layer in list(layers)[-skip:]:
                layer.gradient_checkpointing = False
    return model


def _prepare_micro_batch(data: list[types.Datum], device: torch.device | str) -> MicroBatch:
    """Pad model inputs and labels, honoring explicit Tinker target tokens."""

    if not data:
        raise ValueError("A micro-batch must contain at least one datum")

    sequences = [
        torch.tensor(datum.model_input.to_ints(), dtype=torch.long, device=device) for datum in data
    ]
    lengths = [int(sequence.numel()) for sequence in sequences]
    if any(length == 0 for length in lengths):
        raise ValueError("FSDP forward does not support empty token sequences")

    input_ids = pad_sequence(sequences, batch_first=True, padding_value=0)
    max_len = input_ids.size(1)
    token_positions = torch.arange(max_len, dtype=torch.long, device=device)
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    attention_mask = (token_positions.unsqueeze(0) < length_tensor.unsqueeze(1)).long()
    position_ids = token_positions.unsqueeze(0).expand(len(data), -1)

    # Standard TuFT/Tinker training data carries already-shifted target_tokens
    # in loss_fn_inputs. Use them when present so the final supervised token of
    # each sample is preserved; fall back to the prior flat-roll convention only
    # for legacy/RLHF-style rows that omit explicit targets.
    flat_labels = torch.roll(torch.cat(sequences), shifts=-1, dims=0)
    labels = []
    offset = 0
    for datum, length in zip(data, lengths, strict=True):
        explicit_targets = _explicit_target_tokens(datum, device)
        if explicit_targets is not None:
            if int(explicit_targets.numel()) != length:
                raise ValueError(
                    "target_tokens length must match model_input length: "
                    f"got {int(explicit_targets.numel())} vs {length}"
                )
            labels.append(explicit_targets)
        else:
            labels.append(flat_labels[offset : offset + length])
        offset += length
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    return MicroBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels_padded,
        lengths=lengths,
    )


def _compute_target_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gather label log-probabilities without materializing a full log-softmax.

    A ~32k-token datum against a ~248k vocab makes the [seq, vocab] logits
    tensor ~16 GiB in bf16; a full log_softmax needs a second logits-sized
    tensor that stays alive until backward, which OOMs an 80 GiB GPU. Chunking
    over sequence positions with gather + logsumexp keeps only the logits plus
    one small chunk workspace: both ops save only small tensors for backward,
    unlike log_softmax, which saves its entire output.
    """

    _CHUNK = 1024

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    out = torch.empty(flat_labels.shape, dtype=torch.float32, device=logits.device)
    for start in range(0, flat_logits.size(0), _CHUNK):
        end = start + _CHUNK
        # Both ops must run in fp32: bf16 logsumexp over a ~152k-token vocab
        # loses precision in the log-softmax. The small chunk keeps the single
        # fp32 copy at ~622 MiB instead of a logits-sized tensor.
        chunk = flat_logits[start:end].float()
        with torch.autocast(device_type="cuda", enabled=False):
            label_logits = torch.gather(
                chunk, dim=-1, index=flat_labels[start:end].unsqueeze(-1)
            ).squeeze(-1)
            logsumexp = torch.logsumexp(chunk, dim=-1)
        out[start:end] = label_logits - logsumexp
    return out.view(labels.shape)


def _datum_field(
    datum: types.Datum,
    key: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    value = (datum.loss_fn_inputs or {}).get(key)
    if value is None:
        return None
    return value.to_torch().to(device=device, dtype=dtype).reshape(-1)


def _copy_row(destination: torch.Tensor, row: int, value: torch.Tensor) -> None:
    width = min(destination.size(1), value.numel())
    if width:
        destination[row, :width] = value[:width]


def _prepare_loss_fn_inputs(
    data: list[types.Datum],
    target_logprobs: torch.Tensor,
    loss_fn_name: str,
) -> dict[str, torch.Tensor]:
    """Build padded TuFT loss inputs directly from Datum objects."""

    batch_size, max_len = target_logprobs.shape
    device = target_logprobs.device
    if loss_fn_name.lower() in _RLHF_LOSS_FNS:
        # detach() is critical: sampling_logprobs must be a constant (old policy logprobs),
        # NOT connected to the computation graph. Without detach(), clone() preserves the
        # autograd connection and the gradients of target_logprobs cancel out in
        # prob_ratio = exp(target_logprobs - sampling_logprobs), giving zero net gradient
        # and no weight updates (reward never grows in RL training).
        sampling_logprobs = target_logprobs.detach().clone()
        advantages = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
        has_ref = any("ref_logprobs" in (datum.loss_fn_inputs or {}) for datum in data)
        ref_logprobs = (
            torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
            if has_ref
            else None
        )
        lengths: list[int] = []
        for row, datum in enumerate(data):
            old_logprobs = _datum_field(
                datum,
                "logprobs",
                device=device,
                dtype=torch.float32,
            )
            if old_logprobs is not None:
                _copy_row(sampling_logprobs, row, old_logprobs)
            advantage = _datum_field(
                datum,
                "advantages",
                device=device,
                dtype=torch.float32,
            )
            if advantage is not None:
                _copy_row(advantages, row, advantage)
            if ref_logprobs is not None:
                ref = _datum_field(
                    datum,
                    "ref_logprobs",
                    device=device,
                    dtype=torch.float32,
                )
                if ref is not None:
                    _copy_row(ref_logprobs, row, ref)
            lengths.append(int(datum.model_input.length))

        # Token-validity mask so masked-mean losses (trinity_ppo) can exclude
        # padding; other RLHF losses simply ignore the extra key. A client-sent
        # "mask" (same tail alignment as the other per-token arrays; 1.0 at
        # trained response positions) takes precedence over the length-based
        # fallback: its row sum is the per-datum token-mean divisor of the
        # client's reference loss, which is smaller than the datum length
        # whenever a prompt region is excluded.
        positions = torch.arange(max_len, device=device).unsqueeze(0)
        mask = (positions < torch.tensor(lengths, device=device).unsqueeze(1)).float()
        for row, datum in enumerate(data):
            client_mask = _datum_field(datum, "mask", device=device, dtype=torch.float32)
            if client_mask is not None:
                mask[row].zero_()
                _copy_row(mask, row, client_mask)

        inputs: dict[str, torch.Tensor] = {
            "target_logprobs": target_logprobs,
            "logprobs": sampling_logprobs,
            "advantages": advantages,
            "mask": mask,
        }
        if ref_logprobs is not None:
            inputs["ref_logprobs"] = ref_logprobs
        return inputs

    weights = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    for row, datum in enumerate(data):
        value = _datum_field(datum, "weights", device=device, dtype=torch.float32)
        if value is None:
            length = len(datum.model_input.to_ints())
            value = torch.ones(length, dtype=torch.float32, device=device)
            # Flat-roll fallback: without explicit target_tokens, a sample's final
            # token receives the *next* sample's first token as its label (wrapping
            # to the first sample for the last row). That label is a garbage
            # supervision signal, so zero its weight so it does not enter the loss
            # (the HF backend fails fast instead of silently supervising it).
            if (datum.loss_fn_inputs or {}).get("target_tokens") is None and length > 0:
                value[length - 1] = 0.0
        _copy_row(weights, row, value)
    return {"target_logprobs": target_logprobs, "weights": weights}


def _equal_count_pieces(data: list[types.Datum], num_pieces: int) -> list[list[types.Datum]]:
    """Split data into ``num_pieces`` contiguous equal-datum-count pieces.

    Equal COUNT (not equal tokens) is deliberate: clients send length-sorted
    batches, so contiguous equal-count pieces stay length-homogeneous and pad
    only to each piece's own maximum. Equal token loads across ranks are already
    handled one level up by token-balanced sharding; this only controls
    per-rank padding and memory.
    """

    if num_pieces <= 1 or len(data) <= 1:
        return [list(data)]
    n = len(data)
    base = n // num_pieces
    rem = n % num_pieces
    pieces = []
    start = 0
    for i in range(num_pieces):
        size = base + (1 if i < rem else 0)
        pieces.append(data[start : start + size])
        start += size
    return pieces


def _merge_micro_metrics(metric_list: list[dict[str, Any]]) -> dict[str, float]:
    """Combine numeric micro-batch metrics using their declared reduction."""

    grouped: dict[str, list[float]] = {}
    for metrics in metric_list:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                grouped.setdefault(key, []).append(float(value))

    merged: dict[str, float] = {}
    for key, values in grouped.items():
        if key.endswith(":mean"):
            merged[key] = sum(values) / len(values)
        else:
            merged[key] = sum(values)
    return merged


def forward_backward(
    module: Any,
    data: list[types.Datum],
    loss_fn_name: str,
    loss_fn_config: dict[str, float] | None,
    micro_batch_size: int,
    *,
    forward_only: bool = False,
    replicated: bool = False,
    num_pieces: int | None = None,
) -> dict[str, Any]:
    """Run contiguous micro-batches while preserving summed gradient accumulation.

    ``replicated``: every rank received the identical full batch (used when
    len(data) < world_size so no rank idles in FSDP-2 collectives). Per-rank
    gradients are identical, so the reduce-scatter average already equals the
    full-batch gradient — the world_size loss compensation must be skipped.

    ``num_pieces``: when set (>1), this rank splits its shard into that many
    contiguous token-balanced pieces instead of datum-count micro-batches. The
    backend passes one shared piece count to every rank, keeping FSDP-2
    collective rounds symmetric while equalizing per-round token volume.
    """

    if not data:
        return {"model_output": {"log_probs": []}, "metrics": {}}
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")

    device = next(module.parameters()).device
    loss_callable = get_loss_fn(loss_fn_name)
    config = loss_fn_config or {}
    per_sample_logprobs: list[torch.Tensor] = []
    metric_list: list[dict[str, Any]] = []

    # Diagnostic gate for the bwd-split study: a flag file (or env var) turns on
    # a chrome trace for this call, exported to $TUFT_FSDP_PROFILE_OUT_<pid>_<ts>.
    profiler = None
    if os.environ.get("TUFT_FSDP_PROFILE") == "1" or os.path.exists("/tmp/tuft_fsdp_profile_on"):
        import torch.profiler as torch_profiler

        profiler = torch_profiler.profile(
            activities=[torch_profiler.ProfilerActivity.CPU, torch_profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        )
        profiler.start()

    t_call = time.perf_counter()
    prep_s = fwd_s = loss_s = bwd_s = 0.0
    real_tokens = 0
    padded_tokens = 0

    if num_pieces and num_pieces > 1:
        if num_pieces > len(data):
            # Every rank must run the same piece count or FSDP-2 collectives
            # deadlock; never silently drop pieces. Cannot happen while each
            # datum is <= micro_batch_tokens (num_pieces is derived from the
            # token budget), so fail loudly on a misconfigured combination.
            raise ValueError(
                f"num_pieces={num_pieces} exceeds datum count {len(data)}; "
                "check micro_batch_tokens against the longest datum"
            )
        micro_batches: list[list[types.Datum]] = _equal_count_pieces(data, num_pieces)
    else:
        micro_batches = [
            data[start : start + micro_batch_size]
            for start in range(0, len(data), micro_batch_size)
        ]

    grad_context = torch.no_grad() if forward_only else nullcontext()
    with grad_context:
        for micro_data in micro_batches:
            t_phase = time.perf_counter()
            batch = _prepare_micro_batch(micro_data, device)
            prep_s += time.perf_counter() - t_phase
            autocast = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            )
            t_phase = time.perf_counter()
            with autocast:
                outputs = module(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                # Match HFTrainingModel: honor loss_fn_config["temperature"]. Without
                # this, FSDP target_logprobs are computed at a different temperature
                # than the RL sampling distribution, biasing importance ratios.
                if "temperature" in config and config["temperature"]:
                    logits = logits / config["temperature"]
                target_logprobs = _compute_target_logprobs(logits, batch.labels)
            fwd_s += time.perf_counter() - t_phase

            t_phase = time.perf_counter()
            loss_inputs = _prepare_loss_fn_inputs(micro_data, target_logprobs, loss_fn_name)
            loss, metrics = loss_callable(loss_inputs, config)
            loss_s += time.perf_counter() - t_phase
            if not forward_only:
                t_phase = time.perf_counter()
                # FSDP2 `fully_shard` averages gradients across ranks during the
                # reduce-scatter, but our loss functions are sum-reductions. Without
                # compensation the effective gradient is 1/world_size of the
                # single-GPU / HF value, silently rescaling the effective learning
                # rate whenever the GPU count changes. Multiply back by world_size so
                # the reduced gradient equals the full-batch sum gradient.
                # Exception: replicated mode — every rank holds the SAME full batch,
                # so the average of identical gradients already IS the full-batch
                # gradient; compensating would scale it by world_size.
                world_size = _fsdp_world_size()
                if world_size > 1 and not replicated:
                    (loss * world_size).backward()
                else:
                    loss.backward()
                bwd_s += time.perf_counter() - t_phase
            metric_list.append(metrics)
            per_sample_logprobs.extend(
                target_logprobs[row, :length].detach() for row, length in enumerate(batch.lengths)
            )
            real_tokens += sum(batch.lengths)
            padded_tokens += int(batch.input_ids.numel())

    total_s = time.perf_counter() - t_call
    n_micro = len(micro_batches)
    pad_ratio = (padded_tokens - real_tokens) / padded_tokens if padded_tokens else 0.0
    if profiler is not None:
        profiler.stop()
        out = os.environ.get("TUFT_FSDP_PROFILE_OUT", "/tmp/tuft_fsdp_prof")
        path = f"{out}_{os.getpid()}_{int(time.time())}.json"
        try:
            profiler.export_chrome_trace(path)
            logger.info("[fsdp-profile] trace exported: %s", path)
        except Exception:  # noqa: BLE001 - diagnostics must never break training
            logger.exception("[fsdp-profile] trace export failed")
    logger.info(
        "[fsdp-timing] backward=%s batch=%d micros=%d real_tok=%d padded_tok=%d "
        "pad_ratio=%.3f prep=%.3fs fwd=%.3fs loss=%.3fs bwd=%.3fs total=%.3fs",
        not forward_only,
        len(data),
        n_micro,
        real_tokens,
        padded_tokens,
        pad_ratio,
        prep_s,
        fwd_s,
        loss_s,
        bwd_s,
        total_s,
    )
    # Ship timings through the return value as well: worker stdout forwarding is
    # unreliable (some actors' lines never reach the driver log), while the
    # ray.get result path always works. The backend logs these and strips them
    # before the metrics reach the client, so the client surface is unchanged.
    metrics_out = _merge_micro_metrics(metric_list)
    metrics_out["timing/prep:mean"] = prep_s
    metrics_out["timing/fwd:mean"] = fwd_s
    metrics_out["timing/loss:mean"] = loss_s
    metrics_out["timing/bwd:mean"] = bwd_s
    metrics_out["timing/total:mean"] = total_s
    metrics_out["timing/pad_ratio:mean"] = pad_ratio
    metrics_out["timing/real_tokens:sum"] = float(real_tokens)
    metrics_out["timing/padded_tokens:sum"] = float(padded_tokens)
    metrics_out["timing/micro_batches:sum"] = float(n_micro)

    return {
        "model_output": {"log_probs": per_sample_logprobs},
        "metrics": metrics_out,
    }
