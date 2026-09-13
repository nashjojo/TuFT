"""Timing smoke for the per-chunk instrumentation.

SMOKE_MODE=fwd  : one forward-only request (no gradients, no optimizer state).
SMOKE_MODE=bwd  : one IMPORTANCE_SAMPLING forward_backward (accumulates grads on a
                  throwaway run; use with SMOKE_KEEP=0 to unload afterwards).
Prints the client-side wall time; server-side decomposition lands in logs/server_v3.log
as [call-timing] / [fsdp-call] / [fsdp-engine] lines (aggregate with
scripts/_agg_fsdp_timing.py).
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.request

import tinker
import torch
from tinker import types


BASE_URL = os.getenv("TUFT_BASE_URL", "http://127.0.0.1:10610")
API_KEY = os.getenv("TUFT_API_KEY", "tml-tuft-dev-key")
BASE_MODEL = os.getenv("TUFT_MODEL", "Qwen/Qwen3-1.7B")
SEQ_LEN = int(os.getenv("SMOKE_SEQ_LEN", "4096"))
BATCH = int(os.getenv("SMOKE_BATCH", "40"))
MODE = os.getenv("SMOKE_MODE", "fwd")


def _http_post(path: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _make_datum(seq_len: int, rlhf: bool) -> types.Datum:
    input_ids = [random.randint(10, 150000) for _ in range(seq_len)]
    target_tokens = input_ids[1:] + [input_ids[-1]]
    inputs: dict = {
        "target_tokens": types.TensorData(data=target_tokens, dtype="int64", shape=[seq_len]),
        "weights": types.TensorData(data=[1.0] * seq_len, dtype="float32", shape=[seq_len]),
    }
    if rlhf:
        inputs["logprobs"] = types.TensorData.from_torch(torch.zeros(seq_len, dtype=torch.float32))
        inputs["advantages"] = types.TensorData.from_torch(torch.ones(seq_len, dtype=torch.float32))
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_ids),
        loss_fn_inputs=inputs,
    )


def _padtest(client) -> None:
    """Warm up, then compare a length-sorted vs shuffled variable-length chunk."""
    lens = sorted(random.randint(445, 8191) for _ in range(107))
    print(
        f"[timing-smoke] padtest lens: n={len(lens)} min={lens[0]} median={lens[len(lens) // 2]} "
        f"max={lens[-1]} total={sum(lens)}"
    )

    warm = [_make_datum(2048, rlhf=True) for _ in range(8)]
    t0 = time.time()
    client.forward_backward(warm, loss_fn="importance_sampling").result()
    print(f"[timing-smoke] warmup done: {time.time() - t0:.2f}s")

    t0 = time.time()
    sorted_datums = [_make_datum(n, rlhf=True) for n in lens]
    client.forward_backward(sorted_datums, loss_fn="importance_sampling").result()
    print(f"[timing-smoke] SORTED (asc) done: {time.time() - t0:.2f}s")

    t0 = time.time()
    shuffled = lens[:]
    random.shuffle(shuffled)
    shuffled_datums = [_make_datum(n, rlhf=True) for n in shuffled]
    client.forward_backward(shuffled_datums, loss_fn="importance_sampling").result()
    print(f"[timing-smoke] SHUFFLED done: {time.time() - t0:.2f}s")


def main() -> None:
    svc = tinker.ServiceClient(base_url=BASE_URL, api_key=API_KEY, timeout=900)
    print(f"[timing-smoke] mode={MODE} creating full-param run (batch={BATCH} x {SEQ_LEN} tokens)")
    client = svc.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=8,
        user_metadata={"training_mode": "full_param"},
    )
    print(f"[timing-smoke] model_id={client.model_id}")
    try:
        if MODE == "padtest":
            _padtest(client)
        else:
            datums = [_make_datum(SEQ_LEN, rlhf=(MODE == "bwd")) for _ in range(BATCH)]
            t0 = time.time()
            if MODE == "bwd":
                result = client.forward_backward(datums, loss_fn="importance_sampling").result()
                kind = "fwd+bwd"
            else:
                result = client.forward(datums, loss_fn="cross_entropy").result()
                kind = "forward"
            dt = time.time() - t0
            n_out = len(result.loss_fn_outputs)
            print(f"[timing-smoke] {kind} OK: {dt:.2f}s client-side, outputs={n_out}")
    finally:
        if os.getenv("SMOKE_KEEP") == "1":
            print("[timing-smoke] SMOKE_KEEP=1: leaving run bound (warm actors for takeover)")
        else:
            try:
                _http_post("/api/v1/unload_model", {"model_id": client.model_id})
                print("[timing-smoke] unloaded own run")
            except Exception as e:  # noqa: BLE001
                print(f"[timing-smoke] unload failed: {e}")


if __name__ == "__main__":
    main()
