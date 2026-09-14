"""Benchmark forward_backward throughput with Trinity's alfworld datum shape.

Replicates the RL training workload: variable-length datums submitted as ONE
forward_backward call + ONE optim_step (the tinker trainer protocol).

Datum lengths mirror the peer's measured experience distribution:
  min 485, median 1371, p75 1777, p90 2191, p95 2487, p99 3070, max 16842,
  mean ~1464 (right-skewed).

Production baseline (mb=1, unsorted, real GRPO run): 167 ms/datum.

mb is a SERVER config value (tuft_config.yaml micro_batch_size) — each mb
setting needs a server restart; this script only controls input shape.

Usage:
    python scripts/_bench_fwd_bwd.py [--n 64] [--unsorted] [--tail] [--seed 7]

Modes:
    default   sorted, alfworld length distribution
    --unsorted same lengths, shuffled order (quantifies padding penalty)
    --tail     N datums all at ~16.8k tokens (the sorted long-tail bucket:
              worst-case micro-batch for the client token-cap contract)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import threading
import time

import tinker
from tinker import types


BASE_URL = os.getenv("TUFT_BASE_URL", "http://127.0.0.1:10610")
API_KEY = os.getenv("TUFT_API_KEY", "tml-tuft-dev-key")
MODEL = os.getenv("TUFT_MODEL", "Qwen/Qwen3-1.7B")
TRAIN_GPUS = os.getenv("TRAIN_GPUS", "4,5,6,7")

# Piecewise-linear interpolation of the peer's empirical length CDF.
_QUANTILES = [
    (0.00, 485),
    (0.25, 1120),
    (0.50, 1371),
    (0.75, 1777),
    (0.90, 2191),
    (0.95, 2487),
    (0.99, 3070),
    (1.00, 16842),
]


def sample_length(u: float) -> int:
    for (lo_q, lo_v), (hi_q, hi_v) in zip(_QUANTILES, _QUANTILES[1:], strict=False):
        if u <= hi_q:
            frac = (u - lo_q) / (hi_q - lo_q)
            return int(lo_v + frac * (hi_v - lo_v))
    return _QUANTILES[-1][1]


def make_datum(rng: random.Random, seq_len: int) -> types.Datum:
    input_ids = [rng.randint(10, 150000) for _ in range(seq_len)]
    target_tokens = input_ids[1:] + [input_ids[-1]]
    weights = [1.0] * seq_len
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=target_tokens, dtype="int64", shape=[seq_len]),
            "weights": types.TensorData(data=weights, dtype="float32", shape=[seq_len]),
        },
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--unsorted", action="store_true")
    p.add_argument(
        "--tail", action="store_true", help="all datums ~16.8k tokens (sorted long-tail bucket)"
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--keep", action="store_true", help="keep the training run loaded (for repeat calls)"
    )
    args = p.parse_args()

    peaks: dict[int, int] = {}
    utils: dict[int, list[int]] = {}
    stop = threading.Event()

    def sample_gpus() -> None:
        idxs = [int(g) for g in TRAIN_GPUS.split(",")]
        while not stop.is_set():
            for i in idxs:
                out = (
                    subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                            "-i",
                            str(i),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    .stdout.strip()
                    .split(",")
                )
                if len(out) == 2 and out[0].strip().isdigit():
                    peaks[i] = max(peaks.get(i, 0), int(out[0]))
                    utils.setdefault(i, []).append(int(out[1]))
            time.sleep(0.5)

    t = threading.Thread(target=sample_gpus, daemon=True)
    t.start()

    rng = random.Random(args.seed)
    if args.tail:
        # Peer truncates at max_model_len=8192: the sorted long-tail bucket.
        lengths = [8192] * args.n
    else:
        lengths = sorted(sample_length(rng.random()) for _ in range(args.n))
        if args.unsorted:
            rng.shuffle(lengths)
    data = [make_datum(rng, n) for n in lengths]
    total_tokens = sum(lengths)
    padded = args.n * max(lengths)
    mode = "tail" if args.tail else ("unsorted" if args.unsorted else "sorted")
    print(f"benchmark: n={args.n} mode={mode}")
    print(
        f"  tokens: total={total_tokens} mean={total_tokens // args.n} "
        f"max={max(lengths)} | padding factor={padded / total_tokens:.2f}x"
    )

    sc = tinker.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
    tc = sc.create_lora_training_client(
        base_model=MODEL,
        rank=8,
        train_attn=True,
        train_mlp=True,
        user_metadata={"training_mode": "full_param"},
    )

    t_fb = time.time()
    res_fb = tc.forward_backward(data, loss_fn="cross_entropy").result()
    fb_elapsed = time.time() - t_fb

    t_opt = time.time()
    res_opt = tc.optim_step(types.AdamParams(learning_rate=1e-6)).result()
    opt_elapsed = time.time() - t_opt

    stop.set()
    t.join(timeout=2)

    print("optim metrics:")
    opt_metrics = res_opt.metrics or {}
    for k in sorted(opt_metrics):
        if "grad_norm" in k or "num_params" in k:
            print(f"  {k}: {opt_metrics[k]}")
    print("fwd_bwd metrics:", (res_fb.metrics or {}))

    if not args.keep:
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{BASE_URL}/api/v1/unload_model",
                data=json.dumps({"model_id": tc.model_id}).encode(),
                headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=60)
        except Exception as e:  # noqa: BLE001 - cleanup must never fail the bench
            print(f"(unload failed: {e})")

    per_datum = fb_elapsed / args.n
    print(
        f"forward_backward: {fb_elapsed:.2f}s  ({per_datum * 1000:.0f} ms/datum; "
        f"production baseline 167 ms/datum)"
    )
    print(f"optim_step:       {opt_elapsed:.2f}s")
    print(f"total:            {fb_elapsed + opt_elapsed:.2f}s")
    for i in sorted(peaks):
        u = utils.get(i, [])
        avg_u = sum(u) / len(u) if u else 0
        print(f"GPU {i}: peak_mem={peaks[i]} MiB, avg_util={avg_u:.0f}%")


if __name__ == "__main__":
    main()
