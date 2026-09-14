#!/bin/bash
# Start (or restart) TuFT and its machine-local redis for ONE machine.
#
# Machine identity comes from TUFT_MACHINE_ID (v1 / v2 / v3 / ...), never
# hardcoded, so v1/v2/v3 can share this script from the shared filesystem.
#
# Usage:
#   TUFT_MACHINE_ID=v3 bash scripts/_start_tuft.sh
#
# Optional overrides:
#   TUFT_PORT (10610)  TUFT_HOST (0.0.0.0)  TUFT_CONFIG  TUFT_BIN
#   TUFT_REDIS_DIR (/mnt/workspace/kaixiang/tuft_redis_$TUFT_MACHINE_ID)
#   TUFT_REDIS_PORT (6379)  TUFT_START_TIMEOUT (1800 seconds)
set -uo pipefail

: "${TUFT_MACHINE_ID:?TUFT_MACHINE_ID must be set, e.g. TUFT_MACHINE_ID=v3 bash $0}"

TUFT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TUFT_BIN="${TUFT_BIN:-$TUFT_ROOT/.venv/bin/tuft}"
CONFIG="${TUFT_CONFIG:-$TUFT_ROOT/config/tuft_config.yaml}"
HOST="${TUFT_HOST:-0.0.0.0}"
PORT="${TUFT_PORT:-10610}"
LOG_DIR="$TUFT_ROOT/logs"
LOG="$LOG_DIR/server_${TUFT_MACHINE_ID}.log"
REDIS_DIR="${TUFT_REDIS_DIR:-/mnt/workspace/kaixiang/tuft_redis_${TUFT_MACHINE_ID}}"
REDIS_PORT="${TUFT_REDIS_PORT:-6379}"
START_TIMEOUT="${TUFT_START_TIMEOUT:-1800}"
API_KEY="${TUFT_API_KEY:-tml-tuft-dev-key}"

# nohup-launched bash does not source rc files, so export everything here.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-$HF_ENDPOINT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# GPU partition: vLLM sampling on 0-3, FSDP training on 4-7 (the verl
# zero-crash layout). Both sets are pinned explicitly because vLLM engines
# live outside Ray, so Ray would otherwise see all GPUs as free and colocate
# FSDP actors onto sampling GPUs - that colocate was the true root cause of
# the seven engine crashes on 2026-09-06 (vLLM+FSDP shared GPU 3).
export TUFT_SAMPLING_GPUS="${TUFT_SAMPLING_GPUS:-0,1,2,3}"
export TUFT_FSDP_GPUS="${TUFT_FSDP_GPUS:-4,5,6,7}"
export TUFT_TRAINING_MODE="${TUFT_TRAINING_MODE:-full_param}"

echo "[start:$TUFT_MACHINE_ID] root=$TUFT_ROOT port=$PORT log=$LOG"

if [ ! -x "$TUFT_BIN" ]; then
    echo "[start:$TUFT_MACHINE_ID] ERROR: $TUFT_BIN missing; run 'uv sync' first." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# --- Stop only THIS worktree's server (match on tuft launch + our config) ----
# Use a precise pattern "tuft launch.*<config>" so that other commands
# referencing the config path (e.g. `tuft clear persistence -c <config>`)
# are not accidentally killed.
OLD_PIDS=$(pgrep -f "tuft launch.*$CONFIG" 2>/dev/null)
if [ -n "$OLD_PIDS" ]; then
    echo "[start:$TUFT_MACHINE_ID] stopping existing server: $(echo "$OLD_PIDS" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -TERM $OLD_PIDS 2>/dev/null
    for _ in $(seq 20); do
        pgrep -f "tuft launch.*$CONFIG" >/dev/null 2>&1 || break
        sleep 1
    done
    STILL=$(pgrep -f "tuft launch.*$CONFIG" 2>/dev/null)
    if [ -n "$STILL" ]; then
        # shellcheck disable=SC2086
        kill -KILL $STILL 2>/dev/null
        sleep 2
    fi
fi

# Leftover GPU processes from a previous run of THIS instance hold memory and
# must be cleared, but several TuFT instances share this worktree, so the
# cleanup has to be scoped to the GPUs this instance is about to claim.
# Walking sibling process trees is not enough: Ray daemonizes its GCS and
# raylet, so they are not descendants of the sibling's "tuft launch" process
# and a blanket venv-path match kills a live sibling's workers.
GPU_ORPHANS=$(python3 - "$TUFT_ROOT" "$TUFT_SAMPLING_GPUS" "$TUFT_FSDP_GPUS" <<'PY'
import os
import subprocess
import sys

root, sampling, fsdp = sys.argv[1], sys.argv[2], sys.argv[3]
ours = {g.strip() for g in (sampling + "," + fsdp).split(",") if g.strip()}


def query(*args):
    return subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True
    ).stdout.strip()


uuid_to_index = {}
for line in query("--query-gpu=index,uuid", "--format=csv,noheader").splitlines():
    if not line.strip():
        continue
    index, uuid = [part.strip() for part in line.split(",")]
    uuid_to_index[uuid] = index

orphans = []
for line in query("--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader").splitlines():
    if not line.strip():
        continue
    pid, uuid = [part.strip() for part in line.split(",")]
    if uuid_to_index.get(uuid) not in ours:
        continue
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        continue
    if cwd == root or cwd.startswith(root):
        orphans.append(pid)

print(" ".join(orphans))
PY
)
if [ -n "$GPU_ORPHANS" ]; then
    echo "[start:$TUFT_MACHINE_ID] killing leftover workers on GPUs $TUFT_SAMPLING_GPUS,$TUFT_FSDP_GPUS: $GPU_ORPHANS"
    # shellcheck disable=SC2086
    kill -KILL $GPU_ORPHANS 2>/dev/null
    sleep 2
fi

# --- Redis (machine-local persistence backend) ------------------------------
if redis-cli -p "$REDIS_PORT" PING >/dev/null 2>&1; then
    echo "[start:$TUFT_MACHINE_ID] redis already up on :$REDIS_PORT (dir=$REDIS_DIR)"
else
    if [ ! -f "$REDIS_DIR/redis.conf" ]; then
        echo "[start:$TUFT_MACHINE_ID] ERROR: $REDIS_DIR/redis.conf not found." >&2
        exit 1
    fi
    echo "[start:$TUFT_MACHINE_ID] starting redis with $REDIS_DIR/redis.conf"
    redis-server "$REDIS_DIR/redis.conf" || exit 1
    sleep 2
    redis-cli -p "$REDIS_PORT" PING >/dev/null 2>&1 || {
        echo "[start:$TUFT_MACHINE_ID] ERROR: redis did not come up; see $REDIS_DIR/redis.log" >&2
        exit 1
    }
fi

# --- Launch -----------------------------------------------------------------
echo "=== [$TUFT_MACHINE_ID] launch $(date -Is) ===" >> "$LOG"
cd "$TUFT_ROOT"
nohup "$TUFT_BIN" launch --host "$HOST" --port "$PORT" --config "$CONFIG" >> "$LOG" 2>&1 &
PID=$!
echo "[start:$TUFT_MACHINE_ID] pid=$PID; waiting up to ${START_TIMEOUT}s for healthz"

DEADLINE=$((SECONDS + START_TIMEOUT))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
    if ! ps -p "$PID" >/dev/null 2>&1; then
        echo "[start:$TUFT_MACHINE_ID] ERROR: process exited. Last log lines:" >&2
        tail -30 "$LOG" >&2
        exit 1
    fi
    if curl -sf -m 5 -H "X-API-Key: $API_KEY" \
        "http://127.0.0.1:$PORT/api/v1/healthz" >/dev/null 2>&1; then
        echo "[start:$TUFT_MACHINE_ID] healthy after ${SECONDS}s: http://127.0.0.1:$PORT"
        exit 0
    fi
    sleep 5
done

echo "[start:$TUFT_MACHINE_ID] ERROR: not healthy within ${START_TIMEOUT}s; tail $LOG" >&2
tail -30 "$LOG" >&2
exit 1
