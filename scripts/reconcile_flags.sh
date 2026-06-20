#!/usr/bin/env bash
# Dump the INSTALLED version's --help so the flags we use are reconciled against
# reality (not docs/training-data). Writes raw help to docs/ and prints a quick
# grep of the flags this project depends on.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
mkdir -p "$DOCS_DIR"

dump() {  # tool, outfile
  local tool="$1" out="$2"
  if command -v "$tool" >/dev/null 2>&1; then
    echo "Dumping '$tool --help' -> $out"
    { echo "# $tool --help  (captured $(date -u +%Y-%m-%dT%H:%M:%SZ))"; echo;
      "$tool" --help 2>&1; } > "$out"
  else
    echo "WARN: '$tool' not on PATH — skipping (run inside the activated env)."
  fi
}

dump lerobot-eval  "$DOCS_DIR/flags_eval_help_raw.txt"
dump lerobot-train "$DOCS_DIR/flags_train_help_raw.txt"

echo
echo "=== Flags this project depends on — confirm each appears above ==="
for f in policy.path env.type env.task env.control_mode env.max_parallel_tasks \
         eval.n_episodes eval.batch_size rename_map output_dir policy.n_action_steps; do
  if grep -qi -- "$f" "$DOCS_DIR"/flags_*_help_raw.txt 2>/dev/null; then
    echo "  [FOUND]   --$f"
  else
    echo "  [MISSING] --$f   <-- reconcile docs/flags.md, name may differ on this version"
  fi
done
echo
echo "If any are MISSING, the installed version's name wins — update docs/flags.md,"
echo "configs/eval_libero_goal.json, and notebooks/baseline_eval_colab.ipynb."
