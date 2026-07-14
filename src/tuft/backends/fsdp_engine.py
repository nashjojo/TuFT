"""Native training engine for the FSDP backend (no verl dependency).

Provides the pieces the FSDP backend previously borrowed from verl's
``FSDPEngineWithLMHead``: building the base causal-LM module and running the
micro-batched forward/backward loop.

Semantics deliberately mirror the HF training backend
(``hf_training_model.HFTrainingModel``), which is TuFT's canonical
implementation of the Tinker training contract:

- inputs are right-padded dense batches with a length-based attention mask;
- ``loss_fn_config["temperature"]`` divides the logits before the loss;
- per-token ``target_logprobs`` are gathered at the client-provided
  ``target_tokens`` (memory-efficient gather/log_softmax);
- loss functions come from ``tuft.loss_fn`` and metrics use the
  ``name:reduction`` convention aggregated via ``metrics_reduction``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import torch
from tinker import types
from torch.nn.utils.rnn import pad_sequence

from tuft.loss_fn import get_loss_fn, metrics_reduction


logger = logging.getLogger(__name__)


def shard_list(xs: list[Any], n_shards: int) -> list[list[Any]]:
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


def compute_num_micro_batches(shard_sizes: List[int], micro_batch_size: int | None) -> int:
    """Pick one micro-batch count that every data-parallel rank must use.

    Each ``loss.backward()`` under FSDP-2 issues gradient collectives, so all
    ranks have to run the same number of micro-batches per forward_backward
    call or NCCL deadlocks. We take the MIN over ranks of
    ``ceil(shard_size / micro_batch_size)``: that count is feasible on every
    rank (it never exceeds the smallest shard size), at the cost of
    micro-batches on larger shards holding at most ``micro_batch_size + 1``
    samples when contiguous sharding leaves shards unequal by one.

    Returns 0 when every shard is empty.
    """
    sizes = [s for s in shard_sizes if s > 0]
    if not sizes:
        return 0
    if not micro_batch_size or micro_batch_size <= 0:
        return 1
    return min(max(1, -(-s // micro_batch_size)) for s in sizes)


@dataclass
class FSDPEngineConfig:
    """Configuration for building the base causal-LM module.

    override_config entries are applied to the model's AutoConfig, except
    ``attn_implementation`` which is passed to ``from_pretrained`` directly.
    """

    model_path: str
    attn_implementation: str = "eager"
    override_config: Dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = True
    enable_gradient_checkpointing: bool = True


def build_causal_lm(config: FSDPEngineConfig) -> Any:
    """Build the base model on CPU in bf16 with pretrained weights loaded.

    The FSDP worker applies PEFT adapters and FSDP-2 sharding afterwards.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    overrides = dict(config.override_config or {})
    overrides.pop("attn_implementation", None)

    hf_config = AutoConfig.from_pretrained(
        config.model_path, trust_remote_code=config.trust_remote_code
    )
    for key, value in overrides.items():
        setattr(hf_config, key, value)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        config=hf_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
        low_cpu_mem_usage=True,
        trust_remote_code=config.trust_remote_code,
    )
    if config.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def compute_logprobs_from_target_tokens(
    logits: torch.Tensor, target_tokens: torch.Tensor
) -> torch.Tensor:
    """Compute log probabilities of target tokens from logits with low memory usage.

    Same implementation as HFTrainingModel._compute_logprobs_from_target_tokens
    (https://github.com/OpenRLHF/OpenRLHF/pull/718); duplicated so the FSDP
    path does not import the HF backend module.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack(
            [torch.logsumexp(logit, dim=-1) for logit in logits]  # loop to reduce peak mem
        )
        log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(
            logits, target_tokens, strict=True
        ):  # loop to reduce peak mem consumption
            row_log_probs = torch.nn.functional.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels


def prepare_loss_fn_inputs(
    data: list[types.Datum], device: str | torch.device
) -> Dict[str, torch.Tensor]:
    """Pad per-datum loss_fn_inputs into batch tensors (mirrors the HF backend).

    Datums missing ``target_tokens``/``weights`` get defaults matching the old
    FSDP behavior: next-token labels derived from the input ids (final position
    labeled with pad id 0) and all-ones weights.
    """
    keys = list(data[0].loss_fn_inputs.keys()) if data[0].loss_fn_inputs else []
    loss_fn_input_dict: Dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [datum.loss_fn_inputs[key].to_torch() for datum in data]
        # If tensor is 1D, pad to max length; if already same shape, stack directly
        if all(t.dim() == 1 for t in tensors):
            padded = pad_sequence(tensors, batch_first=True, padding_value=0)
            loss_fn_input_dict[key] = padded.to(device)
        else:
            try:
                stacked = torch.stack(tensors)
                loss_fn_input_dict[key] = stacked.to(device)
            except Exception:
                # Pad every dim to the per-dim max length
                max_shape = list(tensors[0].shape)
                for t in tensors:
                    for i, s in enumerate(t.shape):
                        if s > max_shape[i]:
                            max_shape[i] = s
                padded_tensors = []
                for t in tensors:
                    pad_width = [(0, m - s) for s, m in zip(t.shape, max_shape, strict=False)]
                    pad_args: list[int] = []
                    for p in reversed(pad_width):
                        pad_args.extend(p)
                    padded_tensors.append(torch.nn.functional.pad(t, pad_args, value=0))
                loss_fn_input_dict[key] = torch.stack(padded_tensors).to(device)

    if "target_tokens" not in loss_fn_input_dict:
        target_list = []
        for datum in data:
            ids = torch.tensor(datum.model_input.to_ints(), dtype=torch.long)
            target_list.append(torch.cat([ids[1:], ids.new_zeros(1)]))
        loss_fn_input_dict["target_tokens"] = pad_sequence(
            target_list, batch_first=True, padding_value=0
        ).to(device)
    if "weights" not in loss_fn_input_dict:
        weight_list = [
            torch.ones(len(datum.model_input.to_ints()), dtype=torch.float32) for datum in data
        ]
        loss_fn_input_dict["weights"] = pad_sequence(
            weight_list, batch_first=True, padding_value=0.0
        ).to(device)
    return loss_fn_input_dict


def _forward_micro_batch(
    module: Any,
    data: list[types.Datum],
    loss_fn_callable: Callable,
    loss_fn_config: Dict[str, float],
    backward: bool,
    device: str | torch.device,
) -> tuple[float, Dict[str, float], list[Dict[str, Any]]]:
    """Forward (+backward) one micro-batch; returns (loss, metrics, loss_fn_outputs)."""
    input_ids_list = [torch.tensor(datum.model_input.to_ints(), dtype=torch.long) for datum in data]
    lengths = [t.size(0) for t in input_ids_list]
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0).to(device)
    batch_size, max_len = input_ids.shape

    # Length-based mask rather than `input_ids != 0`: token id 0 can be real content.
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, n in enumerate(lengths):
        attention_mask[i, :n] = 1
    attention_mask = attention_mask.to(device)
    position_ids = (
        torch.arange(max_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).to(device)
    )

    outputs = module(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits
    del outputs

    if "temperature" in loss_fn_config:
        logits = logits / loss_fn_config["temperature"]

    loss_fn_inputs = prepare_loss_fn_inputs(data, device)
    target_logprobs = compute_logprobs_from_target_tokens(logits, loss_fn_inputs["target_tokens"])
    del logits
    loss_fn_inputs["target_logprobs"] = target_logprobs

    loss, metrics = loss_fn_callable(loss_fn_inputs, loss_fn_config)
    if backward:
        loss.backward()

    detached = target_logprobs.detach()
    loss_fn_outputs = [
        {"logprobs": types.TensorData.from_torch(detached[i, :n].cpu().float().clone())}
        for i, n in enumerate(lengths)
    ]
    return float(loss.detach().item()), metrics, loss_fn_outputs


def forward_backward_batch(
    module: Any,
    data: list[types.Datum],
    loss_fn_name: str,
    loss_fn_config: Dict[str, float] | None,
    num_micro_batches: int,
    forward_only: bool = False,
    device: str | torch.device = "cuda",
) -> Dict[str, Any]:
    """Run a micro-batched forward (+backward) pass over data.

    Gradients accumulate across micro-batches: each micro-batch calls
    ``loss.backward()`` and nothing here zero-grads; the caller's optim_step
    consumes and clears the accumulated gradients (standard PyTorch
    grad-accumulation semantics, same as the HF backend).

    IMPORTANT (multi-rank): ``num_micro_batches`` must be identical on every
    data-parallel rank for a given call — each backward triggers FSDP-2
    gradient collectives, and mismatched counts across ranks hang NCCL.
    Compute it once from all shard sizes via ``compute_num_micro_batches``.
    """
    if not data or num_micro_batches <= 0:
        return {"metrics": {}, "loss_fn_outputs": []}
    if num_micro_batches > len(data):
        raise ValueError(
            f"num_micro_batches={num_micro_batches} exceeds batch size {len(data)}; "
            "every micro-batch must be non-empty."
        )
    loss_fn_callable = get_loss_fn(loss_fn_name)
    loss_fn_config = dict(loss_fn_config or {})

    micro_batches = shard_list(data, num_micro_batches)
    metric_list: list[Dict[str, float]] = []
    micro_batch_weights: list[float] = []
    all_loss_fn_outputs: list[Dict[str, Any]] = []

    for micro_data in micro_batches:
        with torch.no_grad() if forward_only else nullcontext():
            _loss, metrics, outputs = _forward_micro_batch(
                module,
                micro_data,
                loss_fn_callable,
                loss_fn_config,
                backward=not forward_only,
                device=device,
            )
        metric_list.append(metrics)
        micro_batch_weights.append(len(micro_data))
        all_loss_fn_outputs.extend(outputs)

    return {
        "metrics": metrics_reduction(metric_list, micro_batch_weights),
        "loss_fn_outputs": all_loss_fn_outputs,
    }
