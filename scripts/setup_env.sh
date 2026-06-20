#!/usr/bin/env bash
# =============================================================================
# VLATune — local WSL2 environment setup (Phase 1)
#
# Idempotent + re-runnable. Creates a fresh Python 3.12 conda env, installs
# ffmpeg via conda-forge, then pip-installs LeRobot with [smolvla,libero] extras
# + wandb. Captures exact versions into docs/versions.md and docs/pip_freeze.txt,
# and dumps the authoritative `lerobot-eval/--train --help` into docs/.
#
# RUN THIS INSIDE WSL2 UBUNTU (or any Linux). LIBERO requires sys_platform=linux.
#
# Usage:
#   bash scripts/setup_env.sh                 # PyPI install (default)
#   LEROBOT_FROM_SOURCE=1 bash scripts/setup_env.sh   # editable from a git clone
#   ENV_NAME=vlatune bash scripts/setup_env.sh
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-vlatune}"
PY_VERSION="${PY_VERSION:-3.12}"
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-smolvla,libero}"
LEROBOT_FROM_SOURCE="${LEROBOT_FROM_SOURCE:-0}"
LEROBOT_SRC_DIR="${LEROBOT_SRC_DIR:-$HOME/lerobot}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
mkdir -p "$DOCS_DIR"

log() { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup:warn]\033[0m %s\n' "$*"; }

# --- 0. Platform sanity --------------------------------------------------------
if [[ "$(uname -s)" != "Linux" ]]; then
  warn "Not on Linux (uname=$(uname -s)). LIBERO needs Linux. On Windows, run this inside WSL2 Ubuntu."
fi

# --- 1. conda available? -------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH. Install Miniconda/Miniforge first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- 2. Create env if missing (idempotent) ------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "conda env '$ENV_NAME' already exists — reusing."
else
  log "Creating conda env '$ENV_NAME' (python=$PY_VERSION)."
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi
conda activate "$ENV_NAME"

ACTUAL_PY="$(python -c 'import platform;print(platform.python_version())')"
log "Active python: $ACTUAL_PY  (env: $CONDA_DEFAULT_ENV)"
case "$ACTUAL_PY" in
  3.12*|3.13*) : ;;
  *) warn "Python $ACTUAL_PY is < 3.12; LeRobot v0.5 requires 3.12+." ;;
esac

# --- 3. ffmpeg via conda-forge BEFORE pip lerobot (required ordering) ---------
if conda list ffmpeg 2>/dev/null | grep -qi '^ffmpeg'; then
  log "ffmpeg already installed in env."
else
  log "Installing ffmpeg from conda-forge."
  conda install -y -c conda-forge ffmpeg
fi

# --- 4. Install LeRobot with extras (idempotent-ish; pip skips satisfied) ------
python -m pip install --upgrade pip
if [[ "$LEROBOT_FROM_SOURCE" == "1" ]]; then
  if [[ ! -d "$LEROBOT_SRC_DIR/.git" ]]; then
    log "Cloning lerobot into $LEROBOT_SRC_DIR"
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_SRC_DIR"
  fi
  log "Editable install: lerobot[$LEROBOT_EXTRAS] from $LEROBOT_SRC_DIR"
  python -m pip install -e "$LEROBOT_SRC_DIR[$LEROBOT_EXTRAS]"
else
  log "PyPI install: lerobot[$LEROBOT_EXTRAS]"
  python -m pip install "lerobot[$LEROBOT_EXTRAS]"
fi

# --- 5. wandb + hub cli -------------------------------------------------------
log "Installing wandb + huggingface_hub CLI."
python -m pip install wandb "huggingface_hub[cli]"

# --- 6. System libs note (render) ---------------------------------------------
warn "For local LIBERO rendering you may also need (needs sudo):"
warn "  sudo apt-get install -y libegl1-mesa-dev libgl1-mesa-glx libosmesa6-dev libglfw3"
warn "Rendering is timeboxed — if it fights you, run sim on Colab (see notebooks/)."

# --- 7. Capture exact versions ------------------------------------------------
log "Capturing versions -> $DOCS_DIR/versions.md and pip_freeze.txt"
VERSIONS_TXT="$(python - <<'PY'
import platform
def v(mod):
    try:
        m = __import__(mod); return getattr(m, "__version__", "?")
    except Exception as e:
        return f"NOT-INSTALLED ({e.__class__.__name__})"
import importlib
lines = []
lines.append(f"- python: {platform.python_version()}")
for mod in ("lerobot","torch","transformers"):
    lines.append(f"- {mod}: {v(mod)}")
try:
    import torch
    lines.append(f"- torch.cuda.is_available: {torch.cuda.is_available()}")
    lines.append(f"- torch.version.cuda: {torch.version.cuda}")
    if torch.cuda.is_available():
        lines.append(f"- gpu: {torch.cuda.get_device_name(0)}")
except Exception as e:
    lines.append(f"- torch: error {e}")
print("\n".join(lines))
PY
)"
echo "$VERSIONS_TXT"

python -m pip freeze > "$DOCS_DIR/pip_freeze.txt" || warn "pip freeze failed"

# Splice the captured versions into docs/versions.md between the markers.
STAMP="captured $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(uname -srm)"
python - "$DOCS_DIR/versions.md" <<PY || warn "versions.md autofill skipped"
import sys, pathlib
p = pathlib.Path(sys.argv[1])
txt = p.read_text()
block = """<!-- VERSIONS_AUTOFILL_START -->
\`\`\`
$VERSIONS_TXT
\`\`\`
_$STAMP_
<!-- VERSIONS_AUTOFILL_END -->"""
import re
new = re.sub(r"<!-- VERSIONS_AUTOFILL_START -->.*?<!-- VERSIONS_AUTOFILL_END -->",
             block.replace("\\\\","\\\\"), txt, flags=re.S)
p.write_text(new)
print("versions.md updated")
PY

# --- 8. Reconcile CLI flags (authoritative --help dump) -----------------------
log "Dumping lerobot --help into docs/ (authoritative flag reference)."
bash "$REPO_ROOT/scripts/reconcile_flags.sh" || warn "reconcile_flags.sh failed (is lerobot-eval on PATH?)"

log "DONE. Next: bash scripts/run_smoke_tests.sh"
log "Remember: export MUJOCO_GL=egl before any eval/render."
