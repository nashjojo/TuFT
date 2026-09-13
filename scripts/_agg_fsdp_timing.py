#!/usr/bin/env python
"""Aggregate TuFT FSDP timing instrumentation lines into a decomposition report.

Inputs (after the instrumented server restart):
  - server_v3.log:  [call-timing] (controller) and [fsdp-call] (backend) lines
  - Ray worker logs: [fsdp-timing] (engine) lines, one line per rank per call
    (glob /tmp/ray/session_*/logs/python-core-worker-*.log)

Usage:
    .venv/bin/python scripts/_agg_fsdp_timing.py \
        --server-log logs/server_v3.log \
        --engine-glob "/tmp/ray/session_2026-*/logs/python-core-worker-*.log" \
        [--since "2026-09-13T06"]   # optional ISO prefix filter (none: whole file)

A/B runbook (restart per config change):
    # v3 REQUIRES exporting TUFT_CONFIG (script default is tuft_config.yaml, not lowmem):
    #   export TUFT_CONFIG=<worktree>/config/tuft_config_lowmem.yaml
    # baseline (grad-ckpt on, tokens 32768):
    TUFT_MACHINE_ID=v3 bash scripts/_start_tuft.sh
    # variant (grad-ckpt off): set fsdp_gradient_checkpointing: false in the config, restart
    # variant (bigger micro-batch budget): micro_batch_tokens: 49152, restart
    #
    # NOTE: adding/removing ModelConfig fields changes the SUPPORTED_MODELS signature in
    # Redis and startup is refused (by design). Re-sync with:
    #   python -c "from tuft.config import load_yaml_config;
    #              from tuft.persistence import get_redis_store;
    #              from tuft.persistence.redis_store import save_config_signature;
    #              c=load_yaml_config('config/tuft_config_lowmem.yaml');
    #              get_redis_store().configure(c.persistence); save_config_signature(c)"
"""

from __future__ import annotations

import argparse
import glob
import re
import statistics


CALL_RE = re.compile(
    r"\[fsdp-call\] backward=(?P<bw>\w+) batch=(?P<batch>\d+) eff_mb=(?P<mb>\S+) "
    r"ray=(?P<ray>[\d.]+)s total=(?P<total>[\d.]+)s"
)
CTRL_RE = re.compile(
    r"\[call-timing\] backward=(?P<bw>\w+) data=(?P<n>\d+) tokens=(?P<tok>\d+) "
    r"wait=(?P<wait>[\d.]+)s exec=(?P<exec>[\d.]+)s total=(?P<total>[\d.]+)s"
)
ENG_RE = re.compile(
    r"\[fsdp-timing\] backward=(?P<bw>\w+) batch=(?P<batch>\d+) micros=(?P<micros>\d+) "
    r"real_tok=(?P<real>\d+) padded_tok=(?P<padded>\d+) pad_ratio=(?P<pad>[\d.]+) "
    r"prep=(?P<prep>[\d.]+)s fwd=(?P<fwd>[\d.]+)s loss=(?P<loss>[\d.]+)s "
    r"bwd=(?P<bwd>[\d.]+)s total=(?P<total>[\d.]+)s"
)
ENGINE_RE = re.compile(
    r"\[fsdp-engine\] fwd_mean=(?P<fwd>[\d.]+)s bwd_mean=(?P<bwd>[\d.]+)s "
    r"prep_mean=(?P<prep>[\d.]+)s loss_mean=(?P<loss>[\d.]+)s total_mean=(?P<total>[\d.]+)s "
    r"pad_ratio=(?P<pad>[\d.]+) real_tok=(?P<real>\d+) padded_tok=(?P<padded>\d+) "
    r"micros=(?P<micros>\d+)"
)


def _stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    p90 = sorted(values)[int(0.9 * (len(values) - 1))]
    return (
        f"n={len(values)} mean={statistics.mean(values):.3f} "
        f"median={statistics.median(values):.3f} p90={p90:.3f} "
        f"sum={sum(values):.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-log", default="logs/server_v3.log")
    ap.add_argument("--engine-glob", default="/tmp/ray/session_*/logs/python-core-worker-*.log")
    args = ap.parse_args()

    ctrl = {"wait": [], "exec": [], "total": [], "tok": 0, "n_calls": 0}
    call = {"ray": [], "total": [], "batch": []}
    eng = {
        "total": [],
        "prep": [],
        "fwd": [],
        "loss": [],
        "bwd": [],
        "pad": [],
        "real": 0,
        "padded": 0,
        "n": 0,
        "micros": [],
    }

    eng_lines = 0
    with open(args.server_log, errors="replace") as fh:
        for line in fh:
            if "[call-timing]" in line:
                m = CTRL_RE.search(line)
                if m:
                    ctrl["wait"].append(float(m["wait"]))
                    ctrl["exec"].append(float(m["exec"]))
                    ctrl["total"].append(float(m["total"]))
                    ctrl["tok"] += int(m["tok"])
                    ctrl["n_calls"] += 1
            elif "[fsdp-call]" in line:
                m = CALL_RE.search(line)
                if m:
                    call["ray"].append(float(m["ray"]))
                    call["total"].append(float(m["total"]))
                    call["batch"].append(int(m["batch"]))
            elif "[fsdp-engine]" in line:
                m = ENGINE_RE.search(line)
                if m:
                    eng_lines += 1
                    eng["total"].append(float(m["total"]))
                    eng["prep"].append(float(m["prep"]))
                    eng["fwd"].append(float(m["fwd"]))
                    eng["loss"].append(float(m["loss"]))
                    eng["bwd"].append(float(m["bwd"]))
                    eng["pad"].append(float(m["pad"]))
                    eng["real"] += int(m["real"])
                    eng["padded"] += int(m["padded"])
                    eng["micros"].append(int(m["micros"]))
                    eng["n"] += 1

    # Secondary source: [fsdp-timing] lines, if worker stdout forwarding delivered any.
    eng_files = sorted(glob.glob(args.engine_glob))
    for path in eng_files:
        with open(path, errors="replace") as fh:
            for line in fh:
                if "[fsdp-timing]" not in line:
                    continue
                m = ENG_RE.search(line)
                if not m:
                    continue
                eng_lines += 1
                eng["total"].append(float(m["total"]))
                eng["prep"].append(float(m["prep"]))
                eng["fwd"].append(float(m["fwd"]))
                eng["loss"].append(float(m["loss"]))
                eng["bwd"].append(float(m["bwd"]))
                eng["pad"].append(float(m["pad"]))
                eng["real"] += int(m["real"])
                eng["padded"] += int(m["padded"])
                eng["micros"].append(int(m["micros"]))
                eng["n"] += 1

    print("== controller [call-timing] (per chunk RPC) ==")
    print(f"  calls={ctrl['n_calls']} tokens={ctrl['tok']}")
    print(f"  wait(parse+queue) s: {_stats(ctrl['wait'])}")
    print(f"  exec(backend)     s: {_stats(ctrl['exec'])}")
    print(f"  total             s: {_stats(ctrl['total'])}")
    if ctrl["exec"] and ctrl["tok"]:
        print(f"  exec aggregated µs/token: {sum(ctrl['exec']) / ctrl['tok'] * 1e6:.1f}")

    print("== backend [fsdp-call] (server-side per chunk) ==")
    print(
        f"  calls={len(call['total'])} batch mean={statistics.mean(call['batch']):.1f}"
        if call["batch"]
        else "  calls=0"
    )
    print(f"  ray.get s : {_stats(call['ray'])}")
    print(f"  total   s : {_stats(call['total'])}")

    print(f"== engine [fsdp-timing] (rank-local, files={len(eng_files)}, lines={eng_lines}) ==")
    print(f"  per-rank calls={eng['n']}  (×4 ranks ≈ {eng['n'] * 4} rank-calls)")
    print(f"  total s: {_stats(eng['total'])}")
    print(f"  prep s : {_stats(eng['prep'])}")
    print(f"  fwd s  : {_stats(eng['fwd'])}")
    print(f"  loss s : {_stats(eng['loss'])}")
    print(f"  bwd s  : {_stats(eng['bwd'])}")
    print(f"  pad_ratio: {_stats(eng['pad'])}")
    if eng["padded"]:
        print(
            f"  tokens real={eng['real']} padded={eng['padded']} "
            f"waste={(eng['padded'] - eng['real']) / eng['padded'] * 100:.1f}%"
        )
    print(f"  micros/call: {_stats([float(x) for x in eng['micros']])}")
    if eng["total"]:
        gpu_like = sum(eng["fwd"]) + sum(eng["loss"]) + sum(eng["bwd"])
        print(f"  GPU-ish share of engine wall: {gpu_like / sum(eng['total']) * 100:.1f}%")


if __name__ == "__main__":
    main()
