"""Comprehensive TuFT server smoke test.

Tests 4 critical capabilities:
  1. Inference (basic chat completion)
  2. Tool call (function calling with finish_reason=tool_calls)
  3. Training (LoRA forward_backward + optim_step, verifies flash-attn etc.)
  4. Long context training (8K+ token sequence, stress-tests flash-attention)

Usage:
    MACHINE_ID=v1 python scripts/_test_tuft_full.py
    MACHINE_ID=v1 python scripts/_test_tuft_full.py --long-seq-len 16384
    MACHINE_ID=v1 python scripts/_test_tuft_full.py --skip-long-ctx

Environment variables:
    MACHINE_ID     - Machine identifier (default: v1)
    TUFT_BASE_URL  - TuFT server URL (default: http://localhost:10610)
    TUFT_API_KEY   - API key (default: tml-tuft-dev-key)
    TUFT_MODEL     - Model name (default: Qwen/Qwen3-4B-Thinking-2507)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MACHINE_ID = os.getenv("MACHINE_ID", "v1")
BASE_URL = os.getenv("TUFT_BASE_URL", "http://localhost:10610")
API_KEY = os.getenv("TUFT_API_KEY", "tml-tuft-dev-key")
MODEL = os.getenv("TUFT_MODEL", "Qwen/Qwen3-4B-Thinking-2507")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SKIP = "\033[93m⊘ SKIP\033[0m"


def http_post(path: str, body: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    """POST JSON to TuFT server and return parsed response."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode("utf-8"))


def http_get(path: str, timeout: int = 30) -> Dict[str, Any]:
    """GET from TuFT server and return parsed response."""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(
        url,
        headers={"X-API-Key": API_KEY},
        method="GET",
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode("utf-8"))


def _unload(model_id: str) -> None:
    """Release the training slot held by a test client (slots never self-free)."""
    try:
        http_post("/api/v1/unload_model", {"model_id": model_id}, timeout=60)
    except Exception as e:  # noqa: BLE001 - cleanup must never fail a test
        print(f"         (unload {model_id[:8]} failed: {e})")


# FSDP shards data across ranks: forward_backward requires len(data) >=
# fsdp_num_gpus (world_size). Default 4 matches the 4-GPU FSDP layout; a
# single-GPU server also accepts batches of this size.
TRAIN_BATCH = int(os.getenv("TUFT_TEST_BATCH", "4"))


def _make_datum(seq_len: int):
    """Build one synthetic next-token-prediction Datum of seq_len tokens."""
    from tinker import types

    input_ids = [random.randint(10, 150000) for _ in range(seq_len)]
    target_tokens = input_ids[1:] + [input_ids[-1]]
    weights = [1.0] * seq_len
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=target_tokens, dtype="int64", shape=[seq_len]),
            "weights": types.TensorData(data=weights, dtype="float32", shape=[seq_len]),
        },
    )


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.skipped = False
        self.error: str | None = None
        self.elapsed: float = 0.0
        self.details: str = ""

    def __str__(self) -> str:
        status = PASS if self.passed else (SKIP if self.skipped else FAIL)
        line = f"  {status} [{self.name}] ({self.elapsed:.2f}s)"
        if self.details:
            line += f" — {self.details}"
        if self.error:
            line += f"\n         Error: {self.error}"
        return line


# ---------------------------------------------------------------------------
# Test 1: Health Check
# ---------------------------------------------------------------------------


def test_health() -> TestResult:
    r = TestResult("Health Check")
    t0 = time.time()
    try:
        resp = http_get("/api/v1/healthz")
        r.elapsed = time.time() - t0
        if resp.get("status") == "ok":
            r.passed = True
            r.details = "status=ok"
        else:
            r.error = f"unexpected response: {resp}"
    except Exception as e:
        r.elapsed = time.time() - t0
        r.error = str(e)
    return r


# ---------------------------------------------------------------------------
# Test 2: Basic Inference
# ---------------------------------------------------------------------------


def test_inference() -> TestResult:
    r = TestResult("Inference (chat completion)")
    t0 = time.time()
    try:
        resp = http_post(
            "/oai/api/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                "max_tokens": 512,
                "temperature": 0.7,
            },
        )
        r.elapsed = time.time() - t0
        choices = resp.get("choices", [])
        if not choices:
            r.error = "no choices in response"
            return r
        msg = choices[0].get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        finish = choices[0].get("finish_reason", "")
        has_output = len(content) > 0 or len(reasoning) > 0
        if finish in ("stop", "length") and has_output:
            r.passed = True
            r.details = (
                f"finish_reason={finish}, content_len={len(content)}, "
                f"reasoning_len={len(reasoning)}"
            )
        else:
            r.error = (
                f"finish_reason={finish}, content='{content[:80]}', reasoning='{reasoning[:80]}'"
            )
    except Exception as e:
        r.elapsed = time.time() - t0
        r.error = str(e)
    return r


# ---------------------------------------------------------------------------
# Test 3: Tool Call
# ---------------------------------------------------------------------------


def test_tool_call() -> TestResult:
    r = TestResult("Tool Call (function calling)")
    t0 = time.time()
    try:
        resp = http_post(
            "/oai/api/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "What is the weather in Beijing today?"}],
                "max_tokens": 512,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the current weather for a given city",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string", "description": "The city name"},
                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                                },
                                "required": ["city"],
                            },
                        },
                    }
                ],
            },
        )
        r.elapsed = time.time() - t0
        choices = resp.get("choices", [])
        if not choices:
            r.error = "no choices in response"
            return r
        finish = choices[0].get("finish_reason", "")
        tool_calls = choices[0].get("message", {}).get("tool_calls") or []
        if finish == "tool_calls" and len(tool_calls) > 0:
            fn = tool_calls[0].get("function", {})
            r.passed = True
            r.details = f"fn={fn.get('name')}, args={fn.get('arguments')}"
        else:
            r.error = f"finish_reason={finish}, tool_calls={tool_calls}"
    except Exception as e:
        r.elapsed = time.time() - t0
        r.error = str(e)
    return r


# ---------------------------------------------------------------------------
# Test 4: Training (short sequence)
# ---------------------------------------------------------------------------


def test_training() -> TestResult:
    r = TestResult("Training (LoRA fwd+bwd+optim)")
    t0 = time.time()
    tc = None
    try:
        import tinker
        from tinker import types

        sc = tinker.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
        tc = sc.create_lora_training_client(
            base_model=MODEL,
            rank=8,
            train_attn=True,
            train_mlp=True,
            user_metadata={"training_mode": "full_param"},
        )

        # Short synthetic batch (128 tokens x TRAIN_BATCH)
        seq_len = 128
        data = [_make_datum(seq_len) for _ in range(TRAIN_BATCH)]

        # forward_backward
        tc.forward_backward(data, loss_fn="cross_entropy").result()

        # optim_step
        tc.optim_step(types.AdamParams(learning_rate=1e-4)).result()

        r.elapsed = time.time() - t0
        r.passed = True
        r.details = f"seq_len={seq_len}, batch={TRAIN_BATCH}, 1 step OK"
    except Exception as e:
        r.elapsed = time.time() - t0
        r.error = f"{type(e).__name__}: {e}"
    if tc is not None:
        _unload(tc.model_id)
    return r


# ---------------------------------------------------------------------------
# Test 5: Long Context Training
# ---------------------------------------------------------------------------


def test_long_context(seq_len: int = 8192) -> TestResult:
    r = TestResult(f"Long Context Training (seq_len={seq_len})")
    t0 = time.time()
    tc = None
    try:
        import tinker
        from tinker import types

        sc = tinker.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
        tc = sc.create_lora_training_client(
            base_model=MODEL,
            rank=8,
            train_attn=True,
            train_mlp=True,
            user_metadata={"training_mode": "full_param"},
        )

        # Synthesize a long-sequence batch
        data = [_make_datum(seq_len) for _ in range(TRAIN_BATCH)]

        # forward_backward (this is where flash-attn matters)
        t_fb = time.time()
        tc.forward_backward(data, loss_fn="cross_entropy").result()
        fb_elapsed = time.time() - t_fb

        # optim_step
        t_opt = time.time()
        tc.optim_step(types.AdamParams(learning_rate=1e-4)).result()
        opt_elapsed = time.time() - t_opt

        r.elapsed = time.time() - t0
        r.passed = True
        r.details = f"batch={TRAIN_BATCH}, fwd+bwd={fb_elapsed:.1f}s, optim={opt_elapsed:.1f}s"
    except Exception as e:
        r.elapsed = time.time() - t0
        r.error = f"{type(e).__name__}: {e}"
        # Detect flash-attn missing
        err_str = str(e).lower()
        if "flash" in err_str or "attn" in err_str:
            r.error += " [HINT: flash-attn may not be installed]"
    if tc is not None:
        _unload(tc.model_id)
    return r


# ---------------------------------------------------------------------------
# Test 6: Gradient Accumulation (batch_size > 1, long context)
# ---------------------------------------------------------------------------


def test_grad_accumulation(seq_len: int = 8192, batch_size: int = 4) -> TestResult:
    r = TestResult(f"Grad Accum (batch={batch_size}, seq_len={seq_len})")
    t0 = time.time()
    tc = None
    try:
        import tinker
        from tinker import types

        sc = tinker.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
        tc = sc.create_lora_training_client(
            base_model=MODEL,
            rank=8,
            train_attn=True,
            train_mlp=True,
            user_metadata={"training_mode": "full_param"},
        )

        # Build multiple long datums
        datums = [_make_datum(seq_len) for _ in range(batch_size)]

        total_tokens = seq_len * batch_size

        # forward_backward with multiple datums triggers grad accumulation
        t_fb = time.time()
        tc.forward_backward(datums, loss_fn="cross_entropy").result()
        fb_elapsed = time.time() - t_fb

        # optim_step
        t_opt = time.time()
        tc.optim_step(types.AdamParams(learning_rate=1e-4)).result()
        opt_elapsed = time.time() - t_opt

        r.elapsed = time.time() - t0
        r.passed = True
        r.details = (
            f"total_tokens={total_tokens}, fwd+bwd={fb_elapsed:.1f}s, optim={opt_elapsed:.1f}s"
        )
    except Exception as e:
        r.elapsed = time.time() - t0
        r.error = f"{type(e).__name__}: {e}"
        err_str = str(e).lower()
        if "oom" in err_str or "out of memory" in err_str:
            r.error += " [HINT: OOM during grad accumulation]"
    if tc is not None:
        _unload(tc.model_id)
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comprehensive TuFT smoke test")
    p.add_argument(
        "--long-seq-len",
        type=int,
        default=8192,
        help="Token count for long-context test (default: 8192)",
    )
    p.add_argument(
        "--grad-accum-seq-len",
        type=int,
        default=8192,
        help="Token count per datum for grad-accum test (default: 8192)",
    )
    p.add_argument(
        "--grad-accum-batch",
        type=int,
        default=4,
        help="Batch size (num datums) for grad-accum test (default: 4)",
    )
    p.add_argument(
        "--skip-long-ctx", action="store_true", help="Skip the long-context training test"
    )
    p.add_argument("--skip-train", action="store_true", help="Skip all training tests")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print(f"  TuFT Full Smoke Test (machine={MACHINE_ID})")
    print(f"  base_url={BASE_URL}  model={MODEL}")
    print("=" * 60)
    print()

    results: List[TestResult] = []

    total_tests = 6

    # 1. Health
    print(f"[1/{total_tests}] Health check...")
    results.append(test_health())
    print(results[-1])
    if not results[-1].passed:
        print("\n  ❌ Server unreachable. Aborting remaining tests.")
        sys.exit(1)

    # 2. Inference
    print(f"[2/{total_tests}] Inference...")
    results.append(test_inference())
    print(results[-1])

    # 3. Tool call
    print(f"[3/{total_tests}] Tool call...")
    results.append(test_tool_call())
    print(results[-1])

    # 4. Training
    if args.skip_train:
        r = TestResult("Training (LoRA fwd+bwd+optim)")
        r.skipped = True
        r.details = "skipped by --skip-train"
        results.append(r)
        print(f"[4/{total_tests}] {results[-1]}")
    else:
        print(f"[4/{total_tests}] Training (short seq)...")
        results.append(test_training())
        print(results[-1])

    # 5. Long context
    if args.skip_train or args.skip_long_ctx:
        r = TestResult(f"Long Context Training (seq_len={args.long_seq_len})")
        r.skipped = True
        r.details = "skipped by flag"
        results.append(r)
        print(f"[5/{total_tests}] {results[-1]}")
    else:
        print(f"[5/{total_tests}] Long context (seq_len={args.long_seq_len})...")
        results.append(test_long_context(seq_len=args.long_seq_len))
        print(results[-1])

    # 6. Gradient accumulation
    if args.skip_train:
        r = TestResult(
            f"Grad Accum (batch={args.grad_accum_batch}, seq_len={args.grad_accum_seq_len})"
        )
        r.skipped = True
        r.details = "skipped by --skip-train"
        results.append(r)
        print(f"[6/{total_tests}] {results[-1]}")
    else:
        print(
            f"[6/{total_tests}] Grad accumulation "
            f"(batch={args.grad_accum_batch}, seq_len={args.grad_accum_seq_len})..."
        )
        results.append(
            test_grad_accumulation(
                seq_len=args.grad_accum_seq_len, batch_size=args.grad_accum_batch
            )
        )
        print(results[-1])

    # Summary
    print()
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.passed and not r.skipped)
    total_time = sum(r.elapsed for r in results)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped  ({total_time:.1f}s)")
    if failed > 0:
        print("  ❌ SOME TESTS FAILED")
        for r in results:
            if not r.passed and not r.skipped:
                print(f"     - {r.name}: {r.error}")
        sys.exit(1)
    else:
        print("  ✅ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
