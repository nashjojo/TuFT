"""Send ONE raw forward_backward request with length-sorted data (no SDK chunking).

Isolates server-side sharding/piece padding from the SDK's byte-budget chunking:
if this shows ~0 pad_ratio while the SDK path shows ~34%, the SDK chunker is
reordering datums before sending.
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
N = int(os.getenv("PROBE_N", "24"))
ORDER = os.getenv("PROBE_ORDER", "sorted")


def _http_post(path: str, payload: dict, timeout: int = 600) -> dict:
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
    svc = tinker.ServiceClient(base_url=BASE_URL, api_key=API_KEY, timeout=900)
    client = svc.create_lora_training_client(
        base_model=BASE_MODEL, rank=8, user_metadata={"training_mode": "full_param"}
    )
    try:
        lens = sorted(random.randint(526, 8089) for _ in range(N))
        if ORDER == "shuffled":
            random.shuffle(lens)
        print(f"[probe] order={ORDER} n={N} lens={lens[:4]}..{lens[-4:]} total={sum(lens)}")
        data = [_datum(n) for n in lens]
        req = types.ForwardBackwardRequest(
            forward_backward_input=types.ForwardBackwardInput(
                data=data, loss_fn="importance_sampling", loss_fn_config=None
            ),
            model_id=client.model_id,
            seq_id=1,
        )
        from tinker._compat import model_dump
        from tinker.lib._pydantic_conv import to_pydantic_request

        body = model_dump(
            to_pydantic_request(req), exclude_unset=False, exclude_none=True, mode="json"
        )
        t0 = time.time()
        fut = _http_post("/api/v1/forward_backward", body)
        rid = fut["request_id"]
        for _ in range(120):
            r = _http_post("/api/v1/retrieve_future", {"request_id": rid})
            if "result" in r or "path" in r or "loss_fn_outputs" in r:
                print(f"[probe] OK in {time.time() - t0:.1f}s")
                break
            if r.get("status") == "failed" or "error" in r:
                print(f"[probe] FAILED: {str(r)[:300]}")
                break
            time.sleep(5)
        else:
            print("[probe] timeout waiting for future")
    finally:
        if os.getenv("PROBE_KEEP") == "1":
            print("[probe] PROBE_KEEP=1: leaving run bound")
        else:
            try:
                _http_post("/api/v1/unload_model", {"model_id": client.model_id})
                print("[probe] unloaded own run")
            except Exception as e:  # noqa: BLE001
                print(f"[probe] unload failed: {e}")


if __name__ == "__main__":
    main()
