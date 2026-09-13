from typing import Dict, Tuple

import torch

from . import _check_loss_fn_inputs


def _mean_or_zero(values: torch.Tensor) -> float:
    """Mean as a plain float, 0.0 when empty.

    An all-zero mask (a chunk with no response tokens) leaves an empty slice;
    torch's mean of an empty tensor is NaN, which serializes to JSON null and
    gets rejected by strict clients. 0.0 keeps every metric a finite float.
    """
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def trinity_ppo_loss(
    loss_fn_inputs: Dict[str, torch.Tensor], loss_fn_config: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Verl-style PPO with dual clipping plus an optional K2 KL term.

    Server-side equivalent of the Trinity trainer's custom loss
    (tinker_trainer._loss_func_impl): moving it here removes the client's
    separate logprob-forward pass (the ratio's dependence on the current
    weights is evaluated inside this call).

    Args:
        loss_fn_inputs: "target_logprobs" (current weights, server-computed),
            "logprobs" (sampling-time logprobs), "advantages", plus optional
            "ref_logprobs" (base-model logprobs for the KL term) and "mask"
            (1.0 at trained response-token positions, 0.0 at prompt/padding;
            the engine forwards the client's per-token mask when present).
        loss_fn_config: clip_range (default 0.2, symmetric),
            clip_ratio_c (default 3.0), kl_coef (default 0.001), and
            num_total_datums (full-batch datum count, e.g. 7680).

    Aggregation matches the client's reference exactly: per-datum masked token
    mean, summed across datums, then divided by num_total_datums. Every chunk of
    a batch carries the same global divisor, so summing the accumulated chunk
    gradients reproduces the single-batch gradient. Without num_total_datums it
    falls back to the per-datum mean within the call.
    """
    _check_loss_fn_inputs(
        loss_fn_inputs, ("target_logprobs", "logprobs", "advantages"), check_shapes=True
    )
    target_logprobs = loss_fn_inputs["target_logprobs"]
    sampling_logprobs = loss_fn_inputs["logprobs"]
    advantages = loss_fn_inputs["advantages"]
    ref_logprobs = loss_fn_inputs.get("ref_logprobs")
    mask = loss_fn_inputs.get("mask")
    config = loss_fn_config or {}
    clip_range = float(config.get("clip_range", 0.2))
    clip_ratio_c = float(config.get("clip_ratio_c", 3.0))
    kl_coef = float(config.get("kl_coef", 0.001))
    num_total_datums = float(config.get("num_total_datums", 0) or 0)

    ratio = torch.exp(torch.clamp(target_logprobs - sampling_logprobs, -20.0, 20.0))
    p1 = -advantages * ratio
    p2 = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    c1 = torch.maximum(p1, p2)
    c2 = torch.minimum(-advantages * clip_ratio_c, c1)
    per_token = torch.where(advantages < 0, c2, c1)
    if ref_logprobs is not None and kl_coef:
        per_token = per_token + 0.5 * kl_coef * (target_logprobs - ref_logprobs) ** 2

    if mask is not None:
        token_counts = mask.sum(dim=1).clamp(min=1.0)
        per_datum = (per_token * mask).sum(dim=1) / token_counts
    else:
        per_datum = per_token.sum(dim=1)
    if num_total_datums > 0:
        loss = per_datum.sum() / num_total_datums
    else:
        loss = per_datum.mean()

    with torch.no_grad():
        flat_ratio = ratio.reshape(-1)
        if mask is not None:
            flat_ratio = ratio[mask > 0]
        metrics: Dict[str, float] = {
            "loss:sum": float(loss.item()),
            "trinity/ratio_mean:mean": _mean_or_zero(flat_ratio),
            "trinity/ratio_std:mean": float(flat_ratio.std().item())
            if flat_ratio.numel() > 1
            else 0.0,
            "trinity/clip_frac:mean": _mean_or_zero(
                (flat_ratio < 1.0 - clip_range) | (flat_ratio > 1.0 + clip_range)
            ),
        }
        if ref_logprobs is not None:
            kl = 0.5 * (target_logprobs - ref_logprobs) ** 2
            if mask is not None:
                kl = kl[mask > 0]
            metrics["trinity/kl_mean:mean"] = _mean_or_zero(kl)

    return loss, metrics
