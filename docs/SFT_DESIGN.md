# Phase 3: SFT design (drafted 2026-06-12)

Fine-tune `lerobot/smolvla_base` on `HuggingFaceVLA/libero` filtered to
`libero_goal`, on a Colab Pro+ A100. Comparison target:
85.0% (`results/baseline_libero_goal.json`, the official
`HuggingFaceVLA/smolvla_libero` checkpoint under our validated protocol).

## Why these choices

- **A100 for training**. SFT is GPU-bound (unlike eval, which is
  CPU-sim-bound). The official SmolVLA fine-tune recipe is batch 64 on A100.
- **Goal-suite-only data**, specialization over scale: LIBERO ships 50 demos
  per task → ~500 episodes for the 10 Goal tasks. The official checkpoint
  trained on more (likely all suites). We don't need to beat 85.0%; we need a
  clean, well-understood SFT checkpoint as the RL launching point. Landing
  within a few points of 85% validates the pipeline.
- **No in-training env eval** (`eval_freq=0`). LIBERO env eval inside the
  training job drags CPU/RAM into the A100 session and burns expensive units
  on sim. All evals run offline on L4 with the Phase-2 protocol.

## Protocol (gated steps; don't skip the smoke run)

### Step 0: Train notebook + flag reconcile (free runtime, ~0 units)
- New `notebooks/01_sft_train.ipynb`, same defensive structure as the eval
  notebook (GPU gate cell, Drive mount, version record, throttled output + log
  tee).
- Reconcile `lerobot-train --help` against the command below (same pattern as
  `docs/flags.md`; installed help wins). Flag names below are from the plan
  and the SmolVLA docs, unverified against lerobot 0.5.1 until this step.

### Step 1: Dataset recon (metadata only, free runtime)
- Load `HuggingFaceVLA/libero` metadata only (no 34GB pull): confirm how
  suites are distinguished (per-episode task strings vs. a suite column) and
  extract the episode indices for the 10 `libero_goal` tasks.
- Verify keys/shapes match the policy (two cameras → `observation.images.image`
  / `image2`, `observation.state` 8-dim). This is expected to match (Phase 2
  confirmed env-side), but the training data path is new.
- Output: `results/dataset_goal_episodes.json` (the episode index list) so the
  train cell is deterministic.

### Step 2: A100 smoke run (~500–1000 steps, ~1 h, gate)
- Measure: VRAM headroom at batch 64 (drop to 32 if needed), steps/sec, and
  compute-unit burn rate (read the Colab meter before/after to calibrate
  the full-run cost estimate).
- Verify: loss decreases; checkpoint saves to Drive; resume from checkpoint
  works; checkpoint loads in `lerobot-eval` (run 1–2 episodes on the A100
  just to prove the load path, not for numbers).
- Gate: don't launch the full run until save→resume→eval-load all pass.

### Step 3: Full SFT run (A100, background execution)
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.episodes="[<goal indices from step 1>]" \
  --output_dir=<Drive>/VLATune/sft/smolvla_goal_run1 \
  --steps=20000 --batch_size=64 \
  --policy.device=cuda \
  --save_freq=2000 \
  --eval_freq=0 \
  --wandb.enable=true --job_name=smolvla_sft_goal_run1
```
- 20k steps / batch 64 is the official SmolVLA fine-tune recipe; keep SmolVLA
  policy defaults for lr/scheduler/freezing (vision encoder frozen) unless the
  smoke run says otherwise.
- Expected wall time ~4–8 h on A100 (smoke run gives the real number).
- Checkpoints: save to local runtime disk if Drive-direct writes prove slow,
  with a periodic copy of the latest to Drive; keep last 2 + every 5k.
- Resume after preemption: re-run the cell pointing `--policy.path` (or
  `--resume`) at the latest Drive checkpoint.

### Step 4: Offline eval (L4, Phase-2 protocol exactly)
- Reuse `notebooks/02_eval.ipynb` with `POLICY_PATH` switched to
  the SFT checkpoint. Same knobs: `control_mode=relative`,
  `n_action_steps=10`, `batch_size=4`, n=100, `MUJOCO_GL=egl`. ~2.5 h each.
- Eval the final (20k) checkpoint first; eval 10k only if 20k looks off
  (over/underfit triage).
- Optional (cheap insight, +2.5 h): eval `lerobot/smolvla_base` zero-shot for
  the full zero-shot → SFT delta. Useful for the writeup.

### Step 5: Bank it
- `results/sft_libero_goal.json` (same schema as the baseline file) + W&B link.
- Writeup section: zero-shot → SFT → official-checkpoint comparison, per-task.
- Commit notebook + configs + results directly on `main` (solo repo, no PRs).

## Budget estimate (Pro+ units; calibrate in step 2)

| Item | Hardware | Wall time |
|---|---|---|
| Smoke run | A100 | ~1 h |
| Full SFT (20k steps) | A100 | ~4–8 h |
| Eval 20k ckpt (+ optional zero-shot) | L4 | 2.5–5 h |

A100 burns units ~4–5× faster than L4. Even so this fits comfortably in one
month of Pro+; the $200–500 budget mainly buys retries and Phase 4.

## Risks / open questions

- **Dataset delivery**: 34GB repo, filtered to ~"goal" episodes. Colab runtime
  disk holds it, but it re-downloads each session (~15–30 min), which is
  acceptable. Do NOT put the HF cache on Drive (many-small-files IO is terrible).
  Step 1 determines whether `--dataset.episodes` filtering downloads only the
  needed shards or the whole set.
- **Flag drift**: `--dataset.episodes` syntax (draccus list with brackets, per
  the Phase-2 lesson) and `--resume` semantics need the step-0 reconcile.
- **Batch 64 VRAM on A100 40GB**: expected to fit (~450M params + frozen
  vision encoder), but unverified until the smoke run.
- **Underperformance vs 85%** is not failure: goal-only data + 20k steps vs
  the official multi-suite checkpoint. The project's headline is the SFT→RL
  delta (Phase 4), with SFT-vs-zero-shot as the supporting result.
