#!/bin/bash
# Run the full TuFT smoke test against this machine's server.
#
# Machine identity comes from TUFT_MACHINE_ID; the model defaults to the first
# entry in the server config, so v1/v2/v3 can share this script unchanged.
#
# Usage:
#   TUFT_MACHINE_ID=v3 bash scripts/_run_full_test.sh [extra args for _test_tuft_full.py]
set -uo pipefail

: "${TUFT_MACHINE_ID:?TUFT_MACHINE_ID must be set, e.g. TUFT_MACHINE_ID=v3 bash $0}"

TUFT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TUFT_PYTHON:-$TUFT_ROOT/.venv/bin/python}"
CONFIG="${TUFT_CONFIG:-$TUFT_ROOT/config/tuft_config.yaml}"
PORT="${TUFT_PORT:-10610}"
LOG_DIR="$TUFT_ROOT/logs"
LOG="$LOG_DIR/test_${TUFT_MACHINE_ID}.log"

export MACHINE_ID="$TUFT_MACHINE_ID"
export TUFT_BASE_URL="${TUFT_BASE_URL:-http://127.0.0.1:$PORT}"
export TUFT_API_KEY="${TUFT_API_KEY:-tml-tuft-dev-key}"

if [ -z "${TUFT_MODEL:-}" ]; then
    TUFT_MODEL=$("$PYTHON" - "$CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
print(cfg.supported_models[0].model_name)
PY
    ) || exit 1
fi
export TUFT_MODEL

mkdir -p "$LOG_DIR"
echo "=== [$TUFT_MACHINE_ID] full test $(date -Is) model=$TUFT_MODEL url=$TUFT_BASE_URL ===" | tee -a "$LOG"
"$PYTHON" -u "$TUFT_ROOT/scripts/_test_tuft_full.py" "$@" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
