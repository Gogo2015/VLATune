# LeRobot CLI flags — reconciled reference

> **Source of truth precedence:** the installed version's `--help` **wins** over
> anything written here. This file is reconciled against the official LeRobot v0.5
> docs (huggingface.co/docs/lerobot/en/libero and .../smolvla) as of **2026-06-09**.
> Run `scripts/reconcile_flags.sh` after install to dump the *actual* `--help`
> into `docs/flags_help_raw.txt` and update this file if anything differs.

## Entry points

- `lerobot-eval` — run policy rollouts in an env, report success rates.
- `lerobot-train` — fine-tune a policy on a dataset (used for SFT; see `docs/SFT_RUNBOOK.md`).

## `lerobot-eval` — verified flags (from official LIBERO docs)

| Flag | Value / example | Notes |
|---|---|---|
| `--policy.path` | `HuggingFaceVLA/smolvla_libero` | Hub id or local checkpoint dir. |
| `--env.type` | `libero` | |
| `--env.task` | `libero_goal` | Suite name. Comma-separate for multi-suite. |
| `--env.task_ids` | `[0]`, `[1,2,3]` | Optional: restrict to specific task indices. Omit = all tasks in suite. |
| `--env.control_mode` | `relative` (default) or `absolute` | **Failure mode #2.** Must match how the checkpoint was trained. |
| `--env.max_parallel_tasks` | `1` | Docs use 1 for reproducible benchmarking. |
| `--eval.n_episodes` | `50` | Episodes per task. Docs use 10 for the published 400-episode protocol; we use 50 on Goal. |
| `--eval.batch_size` | `1` | Parallel envs. Keep 1 on a T4 to be safe. |
| `--policy.n_action_steps` | `10` | **CONFIRMED needed for smolvla (2026-06-09).** Without it the policy re-inferred every sim step (~400s/episode); with `=10` → ~100s/episode at batch_size=2. Accepted by the installed version. Keep this for all smolvla eval/SFT. |
| `--rename_map` | `'{"observation.images.OLD":"observation.images.NEW"}'` | **Failure mode #1.** JSON dict, dataset/env key -> policy-expected key. See `docs/rename_map_notes.md`. |
| `--output_dir` | `./eval_logs/` | Where success JSON + rollout videos are written. |

### Valid `--env.task` suite names (official)

| Suite | CLI name | Tasks |
|---|---|---|
| LIBERO-Spatial | `libero_spatial` | 10 |
| LIBERO-Object | `libero_object` | 10 |
| LIBERO-Goal | `libero_goal` | 10 |
| LIBERO-90 | `libero_90` | 90 |
| LIBERO-Long | `libero_10` | 10 |

### Canonical single-suite eval command (our baseline)

```bash
export MUJOCO_GL=egl
lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_goal \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --policy.n_action_steps=10 \
  --eval.batch_size=4 \
  --eval.n_episodes=10 \
  --output_dir=./eval_logs/smolvla_libero_goal
```
> Validated 2026-06-09 on Colab T4: this config produced 83% on a 6-episode probe
> (control_mode=relative, no rename_map, n_action_steps=10). `n_action_steps=10`
> is the key speed flag — without it ~400s/episode, with it ~100s/episode.

## Policy inputs / outputs (LIBERO env, official)

- `observation.state` — **8-dim** proprioception (eef pos, axis-angle orientation, gripper qpos).
- `observation.images.image` — main camera (`agentview_image`), HWC uint8.
- `observation.images.image2` — wrist camera (`robot0_eye_in_hand_image`), HWC uint8.
- **action** — `Box(-1, 1, shape=(7,))` = 6D eef delta + 1D gripper.
- LeRobot **enforces the `observation.images.*` prefix**. The policy's
  `input_features` keys must match the (possibly renamed) observation keys,
  because key names are baked into the normalization-stats layer.

## `lerobot-train` (training reference)

Official LIBERO training example:
```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id=${HF_USER}/libero-test \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero --env.task=libero_10 \
  --output_dir=./outputs/ --steps=100000 \
  --batch_size=4 --eval.batch_size=1 --eval.n_episodes=1 --eval_freq=1000
```
SmolVLA fine-tune example (own data): `--policy.path=lerobot/smolvla_base --batch_size=64 --steps=20000 --policy.device=cuda --wandb.enable=true`.

## What to verify after install (fill in from `--help`)

- [ ] `--rename_map` exact accepted format (JSON string vs `k=v` pairs).
- [ ] Whether `--env.control_mode` is the exact flag name (vs `--env.action_mode`).
- [ ] Whether smolvla eval needs `--policy.n_action_steps`.
- [ ] Default `--output_dir` and the name of the results JSON it writes.
