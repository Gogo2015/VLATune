#!/usr/bin/env bash
# Run all local smoke tests in order. GPU/model/dataset are gating; render is
# timeboxed/non-gating (failure => sim is Colab-only, which is an accepted outcome).
#
# Usage:  conda activate vlatune && bash scripts/run_smoke_tests.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

declare -A RESULT
run() {  # name, cmd...
  local name="$1"; shift
  echo; echo "============================================================"
  echo "  RUN: $name"
  echo "============================================================"
  if "$@"; then RESULT[$name]="PASS"; else RESULT[$name]="FAIL($?)"; fi
}

run "smoke_gpu"     python "$HERE/smoke_gpu.py"
run "smoke_model"   python "$HERE/smoke_model.py"
run "smoke_dataset" python "$HERE/smoke_dataset.py"
echo; echo ">>> render is timeboxed & non-gating (Colab is the fallback) <<<"
run "smoke_render"  python "$HERE/smoke_render.py"

echo; echo "============================================================"
echo "  SMOKE TEST SUMMARY"
echo "============================================================"
for k in smoke_gpu smoke_model smoke_dataset smoke_render; do
  printf "  %-16s %s\n" "$k" "${RESULT[$k]:-SKIPPED}"
done
echo
echo "  gating = gpu/model/dataset. render FAIL is OK -> run sim on Colab."
echo "  see results/*.json and docs/ for details."

# Gate only on the three core tests.
for k in smoke_gpu smoke_model smoke_dataset; do
  [[ "${RESULT[$k]:-}" == "PASS" ]] || exit 1
done
exit 0
