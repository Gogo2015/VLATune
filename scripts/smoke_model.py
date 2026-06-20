#!/usr/bin/env python
"""smoke_model.py — load SmolVLA, report expected input keys, run one inference.

Deliverables:
  1. Print the policy's expected observation keys + shapes (cameras, state).
     These drive the rename_map check in smoke_dataset.py.
  2. Build a dummy observation matching those keys and call select_action once;
     confirm an action of sane shape comes back.
  3. Write results/model_keys.json for smoke_dataset.py to diff against.

Env vars:
  MODEL   default 'lerobot/smolvla_base'  (use 'HuggingFaceVLA/smolvla_libero'
          to inspect the libero-finetuned checkpoint's expected keys instead)

Exit 0 only if both keys were extracted AND an action tensor came back.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(exist_ok=True)
OUT_JSON = RESULTS / "model_keys.json"

MODEL = os.environ.get("MODEL", "lerobot/smolvla_base")


def load_policy(model_id: str):
    """Import SmolVLAPolicy across known lerobot module layouts and load it."""
    last_err = None
    for path in (
        "lerobot.policies.smolvla.modeling_smolvla",       # lerobot 0.5.x
        "lerobot.common.policies.smolvla.modeling_smolvla",  # older layout
    ):
        try:
            mod = __import__(path, fromlist=["SmolVLAPolicy"])
            cls = getattr(mod, "SmolVLAPolicy")
            print(f"[smoke_model] using {path}.SmolVLAPolicy")
            return cls.from_pretrained(model_id)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Could not import/load SmolVLAPolicy: {last_err}")


def extract_features(policy):
    """Return {key: shape_tuple} for input features, robust to config layout."""
    cfg = getattr(policy, "config", None)
    feats: dict[str, tuple] = {}
    src = None
    for attr in ("input_features", "input_shapes"):
        d = getattr(cfg, attr, None)
        if d:
            src = attr
            for k, v in d.items():
                shape = getattr(v, "shape", v)  # PolicyFeature.shape or raw tuple
                feats[k] = tuple(shape) if shape is not None else None
            break
    # Fallback: read keys baked into the normalization buffers.
    if not feats:
        norm = getattr(policy, "normalize_inputs", None)
        if norm is not None:
            for name, _ in norm.named_buffers():
                # buffers look like 'buffer_observation_state.mean' etc.
                key = name.split(".")[0].replace("buffer_", "").replace("_", ".")
                feats.setdefault(key, None)
            src = "normalize_inputs.buffers"
    return feats, src


def main() -> int:
    record: dict = {"model": MODEL}
    try:
        import torch
        import lerobot
        record["lerobot_version"] = getattr(lerobot, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        print(f"[smoke_model] FAIL: import error: {e}")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    record["device"] = device
    print(f"[smoke_model] loading '{MODEL}' on {device} "
          f"(lerobot {record['lerobot_version']})")

    try:
        policy = load_policy(MODEL)
        policy.to(device)
        policy.eval()
    except Exception as e:  # noqa: BLE001
        print(f"[smoke_model] FAIL: {e}")
        traceback.print_exc()
        return 1

    feats, src = extract_features(policy)
    print(f"[smoke_model] expected input_features (from {src}):")
    image_keys, state_key, state_dim = [], None, None
    for k, shape in sorted(feats.items()):
        print(f"[smoke_model]   {k:32s} shape={shape}")
        if "image" in k or "pixels" in k:
            image_keys.append(k)
        elif k.endswith("state") or "state" in k:
            state_key = k
            if shape:
                state_dim = int(shape[-1])

    # Output / action feature, if exposed.
    action_shape = None
    out = getattr(policy.config, "output_features", None) or \
        getattr(policy.config, "output_shapes", None)
    if out:
        for k, v in out.items():
            if "action" in k:
                action_shape = tuple(getattr(v, "shape", v) or ())
    print(f"[smoke_model] image_keys={image_keys} state_key={state_key} "
          f"state_dim={state_dim} action_feature_shape={action_shape}")

    record.update({
        "input_keys": sorted(feats.keys()),
        "image_keys": image_keys,
        "state_key": state_key,
        "state_dim": state_dim,
        "input_feature_shapes": {k: list(v) if v else None for k, v in feats.items()},
        "action_feature_shape": list(action_shape) if action_shape else None,
    })

    # ---- one inference on a dummy obs built from the reported shapes ----------
    inference_ok = False
    try:
        import torch
        B = 1
        batch = {}
        for k, shape in feats.items():
            if shape is None:
                continue
            if k in image_keys:
                # (C,H,W) -> (B,C,H,W) float in [0,1]
                c, h, w = (shape if len(shape) == 3 else (3, 256, 256))
                batch[k] = torch.rand(B, c, h, w, device=device)
            elif k == state_key:
                batch[k] = torch.rand(B, shape[-1], device=device)
        batch["task"] = ["pick up the object and place it on the target"]
        if hasattr(policy, "reset"):
            policy.reset()
        with torch.no_grad():
            action = policy.select_action(batch)
        print(f"[smoke_model] select_action OK -> action shape={tuple(action.shape)} "
              f"dtype={action.dtype}")
        finite = bool(torch.isfinite(action).all().item())
        print(f"[smoke_model] action all-finite: {finite}  "
              f"range=[{action.min().item():.3f}, {action.max().item():.3f}]")
        record["inference_action_shape"] = list(action.shape)
        record["inference_action_finite"] = finite
        inference_ok = finite and action.numel() > 0
    except Exception as e:  # noqa: BLE001
        print(f"[smoke_model] WARN: inference call failed: {e}")
        traceback.print_exc()
        record["inference_error"] = str(e)

    OUT_JSON.write_text(json.dumps(record, indent=2))
    print(f"[smoke_model] wrote {OUT_JSON.relative_to(REPO_ROOT)}")

    keys_ok = len(feats) > 0
    if keys_ok and inference_ok:
        print("[smoke_model] PASS")
        return 0
    print(f"[smoke_model] {'PASS(keys)/FAIL(inference)' if keys_ok else 'FAIL'} "
          "— keys were saved for the dataset diff regardless.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
