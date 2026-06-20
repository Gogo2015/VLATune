# rename_map & control_mode — the two 0%-success failure modes

Two documented bugs cause **0% eval success with no crash**. Both are checked by
the smoke tests. This file records the analysis and the decision rule.

## Failure mode #1 — image-key mismatch (`--rename_map`)

The policy's `input_features` keys must exactly match the observation keys coming
from the dataset/env, because **key names are encoded inside the normalization
statistics layer**. A mismatch silently feeds zeros/wrong normalization → 0%.

**Policy-expected keys for the LIBERO env (official):**
- `observation.images.image`   (main / agentview)
- `observation.images.image2`  (wrist / eye-in-hand)
- `observation.state`          (8-dim)

**rename_map syntax (lerobot 0.5):**
```
--rename_map='{"observation.images.OLD":"observation.images.NEW"}'
```
JSON dict mapping the **source** key (what the dataset/env produces) to the
**policy-expected** key. Applied by `RenameObservationsProcessor` before the obs
reaches the model.

### Decision rule (implemented in scripts/smoke_dataset.py)
1. Read the dataset's actual observation keys (from `HuggingFaceVLA/smol-libero`).
2. Read the policy's expected keys (written by `scripts/smoke_model.py` to
   `results/model_keys.json`).
3. If they match → **no rename_map needed** (print `--rename_map: NONE`).
4. If they differ → print a suggested `--rename_map='{...}'` and mark **FAIL**.

> Expectation: the `HuggingFaceVLA/*` LIBERO datasets are already in LeRobot
> format using `observation.images.image` / `image2`, and the official
> `HuggingFaceVLA/smolvla_libero` checkpoint was trained on exactly those keys.
> So for the **standard baseline path, rename_map is most likely NOT needed.**
> It becomes necessary if you swap in a checkpoint trained with different camera
> names, or a raw/community LIBERO dataset. (cf. lerobot issue #2318 where a
> rename_map raised a spurious visual-mismatch error — confirm against `--help`.)

## Failure mode #2 — control_mode mismatch (relative vs absolute)

LIBERO supports `relative` (default) and `absolute` action parameterization. A
checkpoint trained with one and evaluated with the other produces valid-looking
but wrong actions → the arm drifts → 0% success, no crash.

```
--env.control_mode=relative   # default
--env.control_mode=absolute
```

**Decision rule:** start with `relative` (the default, and what most LeRobot
LIBERO checkpoints use). If Goal success is ~0% but the policy loads and produces
finite, in-range actions, flip to `absolute` and re-run a 2-episode probe before
committing to a 50-episode run. Record which one gave non-zero success here:

- Observed working control_mode for `HuggingFaceVLA/smolvla_libero`: **`relative`**
  (CONFIRMED 2026-06-09 on Colab T4 — 83% success on a 6-episode probe, so
  non-zero → `relative` is correct, no need to try `absolute`).
- Observed rename_map need: **NONE** (CONFIRMED — eval ran clean with no
  `--rename_map`; dataset/env keys already matched the policy).

## Quick triage table

| Symptom | Likely cause | Action |
|---|---|---|
| Crash: "key not found" / shape mismatch on load | image-key mismatch | set `--rename_map` |
| Loads fine, 0% success, arm drifts/overshoots | control_mode mismatch | flip `--env.control_mode` |
| Loads fine, 0% success, arm barely moves | normalization / wrong state dim | check `observation.state` is 8-dim, check rename_map |
| ~78–84% Goal | **working as intended** | this is the reproduced baseline |
