#!/usr/bin/env python
"""smoke_dataset.py — inspect LIBERO dataset keys/shapes, diff vs the policy.

Uses LeRobotDatasetMetadata (downloads only info/stats JSON, NOT the ~GB videos)
so this is cheap and never pulls the 34GB full dataset. Default repo is the
compact HuggingFaceVLA/smol-libero (~1.8GB if fully materialized; we only read
metadata here).

Outputs:
  - dataset observation keys, image shapes, state dim, action dim
  - DIFF vs the LIBERO env-standard keys (what eval actually feeds the policy)
  - DIFF vs results/model_keys.json (the loaded policy's expected keys)
  - explicit PASS/FAIL and a suggested --rename_map if keys mismatch

Env vars:
  DATASET   default 'HuggingFaceVLA/smol-libero'
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
MODEL_KEYS = RESULTS / "model_keys.json"
DATASET = os.environ.get("DATASET", "HuggingFaceVLA/smol-libero")

# LIBERO env-standard policy inputs (official LeRobot docs). This is what the
# eval harness feeds the policy, so it's the operationally relevant target.
ENV_STD_IMAGE_KEYS = ["observation.images.image", "observation.images.image2"]
ENV_STD_STATE_KEY = "observation.state"
ENV_STD_STATE_DIM = 8
ENV_STD_ACTION_DIM = 7


def load_metadata(repo_id):
    last = None
    for path in (
        "lerobot.datasets.lerobot_dataset",        # lerobot 0.5.x
        "lerobot.common.datasets.lerobot_dataset",  # older layout
    ):
        try:
            mod = __import__(path, fromlist=["LeRobotDatasetMetadata"])
            cls = getattr(mod, "LeRobotDatasetMetadata")
            print(f"[smoke_dataset] using {path}.LeRobotDatasetMetadata")
            return cls(repo_id)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"Could not load LeRobotDatasetMetadata: {last}")


def main() -> int:
    print(f"[smoke_dataset] inspecting metadata for '{DATASET}'")
    try:
        meta = load_metadata(DATASET)
    except Exception as e:  # noqa: BLE001
        print(f"[smoke_dataset] FAIL: {e}")
        return 1

    features = dict(getattr(meta, "features", {}) or {})
    if not features:
        print("[smoke_dataset] FAIL: no features in metadata.")
        return 1

    ds_image_keys, ds_state_key, ds_state_dim = [], None, None
    ds_action_dim = None
    print("[smoke_dataset] dataset features:")
    for k, spec in sorted(features.items()):
        shape = spec.get("shape") if isinstance(spec, dict) else getattr(spec, "shape", None)
        dtype = spec.get("dtype") if isinstance(spec, dict) else getattr(spec, "dtype", None)
        print(f"[smoke_dataset]   {k:34s} dtype={dtype} shape={tuple(shape) if shape else None}")
        if "image" in k or dtype in ("image", "video"):
            ds_image_keys.append(k)
        elif k == "observation.state" or k.endswith(".state"):
            ds_state_key = k
            if shape:
                ds_state_dim = int(shape[-1])
        elif k == "action":
            if shape:
                ds_action_dim = int(shape[-1])

    print(f"[smoke_dataset] -> image_keys={ds_image_keys} state_key={ds_state_key} "
          f"state_dim={ds_state_dim} action_dim={ds_action_dim}")

    # ---- DIFF 1: dataset vs LIBERO env-standard (the eval-relevant check) -----
    print("\n[smoke_dataset] === DIFF vs LIBERO env-standard keys ===")
    rename_map: dict[str, str] = {}
    img_ok = sorted(ds_image_keys) == sorted(ENV_STD_IMAGE_KEYS)
    if not img_ok:
        print(f"[smoke_dataset]   image keys differ: dataset={ds_image_keys} "
              f"expected={ENV_STD_IMAGE_KEYS}")
        # Best-effort positional suggestion (review before using!).
        for src, dst in zip(sorted(ds_image_keys), ENV_STD_IMAGE_KEYS):
            if src != dst:
                rename_map[src] = dst
    else:
        print(f"[smoke_dataset]   image keys MATCH {ENV_STD_IMAGE_KEYS}")

    state_ok = (ds_state_key == ENV_STD_STATE_KEY) and (ds_state_dim == ENV_STD_STATE_DIM)
    print(f"[smoke_dataset]   state: dataset({ds_state_key},{ds_state_dim}) "
          f"vs env({ENV_STD_STATE_KEY},{ENV_STD_STATE_DIM}) -> "
          f"{'MATCH' if state_ok else 'MISMATCH'}")
    action_ok = ds_action_dim == ENV_STD_ACTION_DIM
    print(f"[smoke_dataset]   action dim: dataset={ds_action_dim} "
          f"vs env={ENV_STD_ACTION_DIM} -> {'MATCH' if action_ok else 'MISMATCH'}")

    # ---- DIFF 2: dataset vs the loaded policy (informational) -----------------
    if MODEL_KEYS.exists():
        mk = json.loads(MODEL_KEYS.read_text())
        print(f"\n[smoke_dataset] === DIFF vs loaded policy '{mk.get('model')}' "
              "(informational) ===")
        print(f"[smoke_dataset]   policy image_keys={mk.get('image_keys')} "
              f"state_key={mk.get('state_key')} state_dim={mk.get('state_dim')}")
        if mk.get("model", "").endswith("smolvla_base"):
            print("[smoke_dataset]   NOTE: smolvla_base is NOT libero-finetuned, so "
                  "its keys may differ from the env — that's expected. The eval "
                  "target HuggingFaceVLA/smolvla_libero is trained on env-standard keys.")
    else:
        print(f"\n[smoke_dataset] (no {MODEL_KEYS.name}; run smoke_model.py first "
              "for the policy-side diff)")

    # ---- Verdict + rename_map suggestion --------------------------------------
    print("\n[smoke_dataset] ================ VERDICT ================")
    all_ok = img_ok and state_ok and action_ok
    if all_ok:
        print("[smoke_dataset] PASS — dataset keys match the LIBERO env-standard.")
        print("[smoke_dataset] suggested --rename_map: NONE NEEDED")
    else:
        print("[smoke_dataset] FAIL — key/shape mismatch vs env-standard.")
        if rename_map:
            print("[smoke_dataset] suggested (REVIEW!) --rename_map="
                  f"'{json.dumps(rename_map)}'")
        if not state_ok or not action_ok:
            print("[smoke_dataset] dims differ — a rename_map won't fix a wrong "
                  "state/action dim; you likely have the wrong dataset/suite.")

    (RESULTS / "dataset_keys.json").write_text(json.dumps({
        "dataset": DATASET,
        "image_keys": ds_image_keys,
        "state_key": ds_state_key, "state_dim": ds_state_dim,
        "action_dim": ds_action_dim,
        "matches_env_standard": all_ok,
        "suggested_rename_map": rename_map,
    }, indent=2))
    print(f"[smoke_dataset] wrote {(RESULTS / 'dataset_keys.json').relative_to(REPO_ROOT)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
