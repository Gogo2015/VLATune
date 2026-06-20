# Phase 3 SFT — VALIDATED runbook (smoke passed 2026-06-15)

The 100-step smoke run passed every gate on a Colab Pro+ **A100-80GB**:
loss ↓ (0.88→0.43), checkpoint saved to Drive, batch 64 fits (no OOM),
**~2 steps/s → 20k-step run ≈ 2.8 h**, and the checkpoint loads + runs in
`lerobot-eval`. Below is the exact, working recipe and every non-obvious fix the
smoke surfaced. Stack: lerobot 0.5.1, torch 2.10.0+cu128, transformers 5.3.0,
python 3.12.13, datasets 4.0.0.
(Torch pin reconciled 2026-06-16 to match the as-run record in
`results/sft_libero_goal.json` / `results/baseline_libero_goal.json`; the
`lerobot[smolvla,libero]` install is not torch-pinned, so the resolver picked
`2.10.0+cu128` on the run day.)

## Five gotchas the smoke caught (all fixed)

1. **`--policy.push_to_hub=false` is REQUIRED.** Without it `lerobot-train`
   aborts at config validation: `'policy.repo_id' argument missing`. We keep
   checkpoints on Drive, not the Hub.

2. **`HuggingFaceVLA/libero` has a BROKEN episode→file mapping.** The metadata
   says episode 379 lives in `data/chunk-000/file-025.parquet`, but that file
   actually holds episodes 61–63 (verified with a pristine force-download; broken
   on **every** revision: `main`, `v3`, `v3.0`). So `--dataset.episodes`
   episode-filtered *download* fetches the wrong files and the `episode_index`
   filter then matches zero rows → `ValueError: Instruction "train" corresponds
   to no data!`. **Fix: download the full `data/` + `meta/` once (~35 GB, images
   are embedded in parquet — `video_keys: []`, no separate videos), then the
   episode_index filter selects the 428 goal episodes correctly** (lerobot globs
   all present parquet and filters on the real `episode_index` column;
   `_build_index_mapping` also uses real loaded data, so this is correct).

3. **Camera mismatch — `smolvla_base` wants 3 cameras, LIBERO has 2.** Base
   `input_features` = `camera1/camera2/camera3` (+ state[6]); LIBERO provides
   `observation.images.image`/`image2` (+ state[8]). `make_policy` only rebuilds
   input_features from the dataset `if not cfg.input_features` — and the base
   ships them populated, so it keeps camera1/2/3. **Fix: `--rename_map` mapping
   image→camera1, image2→camera2.** Then `{camera1,camera2} ⊆ {camera1,2,3}` so
   the visual-feature check passes (`validate_visual_features_consistency` accepts
   either-direction subset), and at runtime the missing camera3 is simply skipped
   (`empty_cameras=0` → padding loop breaks). State[8]/action[7] auto-pad to
   `max_state_dim`/`max_action_dim`=32, so only cameras need renaming.
   (`--policy.input_features={}` does NOT work — `from_pretrained` reloads the
   checkpoint's own config and ignores the CLI clear.)

4. **EVAL of our SFT checkpoint needs the SAME `--rename_map`.** Our checkpoint
   keeps camera1/2/3 input_features, so when evaluating it the LIBERO env's
   image/image2 must be renamed too. (This differs from the official
   `HuggingFaceVLA/smolvla_libero`, whose config natively uses image/image2 and
   needs no rename — that's why the 85% baseline eval didn't use one.)

5. **Pre-create `~/.libero/config.yaml` before any eval subprocess.** LIBERO's
   first import calls `input()` (custom-dataset-path prompt); in a subprocess
   with no stdin that's `EOFError`. Write the default config first (see the
   notebook cell) so the prompt never fires.

Flag note: rename_map is the **top-level** `--rename_map` for both
`lerobot-train` and `lerobot-eval` (not `--dataset.rename_map`).

## Validated FULL train command (20k steps)

```bash
# Prereq: full data on disk —
#   snapshot_download("HuggingFaceVLA/libero", repo_type="dataset",
#                     revision="v3.0", allow_patterns=["data/*","meta/*"])
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.episodes='[<428 goal episode indices from results/dataset_goal_episodes.json>]' \
  --output_dir=<Drive>/VLATune/sft/smolvla_sft_goal_run1 \
  --job_name=smolvla_sft_goal_run1 \
  --steps=20000 --batch_size=64 \
  --save_freq=2000 --log_freq=200 --num_workers=4 \
  --wandb.enable=false
```

Goal episode indices: `results/dataset_goal_episodes.json` (428 episodes, 10
tasks; the task-string match is correct even though file_index is broken).

## Validated EVAL command (our SFT checkpoint, run on L4)

```bash
# Pre-create ~/.libero/config.yaml first (see notebook). Then:
lerobot-eval \
  --policy.path=<Drive>/VLATune/sft/smolvla_sft_goal_run1/checkpoints/last/pretrained_model \
  --env.type=libero --env.task=libero_goal \
  --env.control_mode=relative --env.max_parallel_tasks=1 \
  --policy.n_action_steps=10 \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --eval.batch_size=4 --eval.n_episodes=10 \
  --output_dir=<Drive>/VLATune/sft/eval_smolvla_sft_goal_run1
```

Compare the resulting `pc_success` against the **85.0%** baseline
(`results/baseline_libero_goal.json`). Landing within a few points validates the
pipeline; the headline is the Phase-4 SFT→RL delta.

## Colab execution notes

- Colab runs cells serially in ONE kernel; a blocking `subprocess.run` cell (e.g. eval)
  holds the kernel and queues later cells. Launch long jobs as a background
  `subprocess.Popen` (+ a separate poll loop) so the kernel stays responsive, and write
  progress/checkpoints to Drive every few thousand steps so a disconnect loses minutes, not hours.
