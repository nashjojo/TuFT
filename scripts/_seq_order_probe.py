"""Validate ordered seq_id pickup: send seq_ids out of order, expect no conflicts.

Creates a fresh full-param run, posts three small forward_backward requests with
seq_ids 3, 1, 2 (in that arrival order), and checks that all three complete:
the seq=3 request must wait for 1 and 2 instead of fast-forwarding and then
rejecting them. Then unloads its run.
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
SEQ_LEN = int(os.getenv("SEQ_PROBE_LEN", "512"))
SEQ_BATCH = int(os.getenv("SEQ_PROBE_BATCH", "4"))


def _http_post(path: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _datum(seq_len: int) -> types.Datum:
    input_ids = [random.randint(10, 150000) for _ in range(seq_len)]
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=input_ids[1:] + [input_ids[-1]], dtype="int64", shape=[seq_len]
            ),
            "weights": types.TensorData(data=[1.0] * seq_len, dtype="float32", shape=[seq_len]),
            "logprobs": types.TensorData.from_torch(torch.zeros(seq_len, dtype=torch.float32)),
            "advantages": types.TensorData.from_torch(torch.ones(seq_len, dtype=torch.float32)),
        },
    )


def main() -> None:
    from tinker._compat import model_dump
    from tinker.lib._pydantic_conv import to_pydantic_request

    svc = tinker.ServiceClient(base_url=BASE_URL, api_key=API_KEY, timeout=900)
    client = svc.create_lora_training_client(
        base_model=BASE_MODEL, rank=8, user_metadata={"training_mode": "full_param"}
    )
    print(f"[seq-probe] model_id={client.model_id}")
    try:
        data = [_datum(SEQ_LEN) for _ in range(SEQ_BATCH)]
        request_ids: dict[int, str] = {}
        # Post seq 3 first so it arrives with a gap; then the missing 1 and 2.
        for seq in (3, 1, 2):
            req = types.ForwardBackwardRequest(
                forward_backward_input=types.ForwardBackwardInput(
                    data=[d if seq != 3 else data[0] for d in data],
                    loss_fn="importance_sampling",
                    loss_fn_config=None,
                ),
                model_id=client.model_id,
                seq_id=seq,
            )
            body = model_dump(
                to_pydantic_request(req), exclude_unset=False, exclude_none=True, mode="json"
            )
            fut = _http_post("/api/v1/forward_backward", body)
            request_ids[seq] = fut["request_id"]
            print(f"[seq-probe] posted seq={seq} -> {fut['request_id'][:8]}")
            time.sleep(0.5)

        results: dict[int, str] = {}
        deadline = time.time() + 240
        while (len(results) < 3) and time.time() < deadline:
            for seq, rid in request_ids.items():
                if seq in results:
                    continue
                r = _http_post("/api/v1/retrieve_future", {"request_id": rid})
                blob = json.dumps(r)
                if "loss_fn_outputs" in blob or '"result"' in blob:
                    results[seq] = "OK"
                elif "error" in blob or "detail" in blob:
                    results[seq] = f"FAILED: {blob[:200]}"
            time.sleep(3)

        for seq in sorted(request_ids):
            print(f"[seq-probe] seq={seq}: {results.get(seq, 'PENDING/timeout')}")
        if all(v == "OK" for v in results.values()) and len(results) == 3:
            print("[seq-probe] PASS: out-of-order submission drained in order")
        else:
            print("[seq-probe] FAIL")
    finally:
        try:
            _http_post("/api/v1/unload_model", {"model_id": client.model_id})
            print("[seq-probe] unloaded own run")
        except Exception as e:  # noqa: BLE001
            print(f"[seq-probe] unload failed: {e}")


if __name__ == "__main__":
    main()
