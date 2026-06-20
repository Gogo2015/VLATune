#!/usr/bin/env python
"""smoke_render.py — TIMEBOXED: render one LIBERO frame headlessly.

MuJoCo's GL backend (MUJOCO_GL) must be set BEFORE importing mujoco, and Python
caches imports, so to try egl then osmesa we re-exec this script in a fresh
process with the fallback backend. Saves a PNG on success; logs the traceback
into docs/wsl2_render_notes.md on failure and exits non-zero.

This is the WSL2 pain-point test. If it fails after both backends, that's fine —
sim runs on Colab. Do NOT sink hours into WSL2 graphics drivers.
"""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(exist_ok=True)
NOTES = REPO_ROOT / "docs" / "wsl2_render_notes.md"

BACKENDS = ["egl", "osmesa"]
SUITE = os.environ.get("RENDER_SUITE", "libero_goal")


def log_failure(backend: str, err: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = (f"\n### {stamp} — MUJOCO_GL={backend} FAILED\n"
             f"```\n{err.strip()}\n```\n")
    try:
        txt = NOTES.read_text()
        marker = "<!-- RENDER_LOG_START -->"
        if marker in txt:
            txt = txt.replace("<!-- RENDER_LOG_START -->",
                              "<!-- RENDER_LOG_START -->" + entry, 1)
            txt = txt.replace("_(none yet)_", "", 1)
            NOTES.write_text(txt)
        else:
            NOTES.write_text(txt + entry)
    except FileNotFoundError:
        NOTES.write_text(f"# WSL2 render notes\n{entry}")


def try_render(backend: str) -> bool:
    os.environ["MUJOCO_GL"] = backend
    print(f"[smoke_render] attempting render with MUJOCO_GL={backend}")
    try:
        import numpy as np
        from PIL import Image
        # Canonical LIBERO offscreen-render path (installed via lerobot[libero]).
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[SUITE]()
        task = suite.get_task(0)
        bddl_dir = get_libero_path("bddl_files")
        bddl = os.path.join(bddl_dir, task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_heights=256, camera_widths=256
        )
        env.seed(0)
        obs = env.reset()
        # agentview_image is HWC uint8; flip vertically (MuJoCo convention).
        frame = np.asarray(obs["agentview_image"])[::-1]
        out = RESULTS / f"render_smoke_{backend}.png"
        Image.fromarray(frame).save(out)
        env.close()
        nonblack = bool(frame.any())
        print(f"[smoke_render] saved {out.name} shape={frame.shape} "
              f"non-black={nonblack}")
        if not nonblack:
            raise RuntimeError("rendered frame is all-black (EGL likely can't see GPU)")
        print(f"[smoke_render] PASS (backend={backend})")
        return True
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        print(f"[smoke_render] backend {backend} failed:\n{err}")
        log_failure(backend, err)
        return False


def main() -> int:
    attempt = int(os.environ.get("VLATUNE_RENDER_ATTEMPT", "0"))
    backend = BACKENDS[attempt] if attempt < len(BACKENDS) else None
    if backend is None:
        print("[smoke_render] FAIL: all backends exhausted (egl, osmesa).")
        print("[smoke_render] -> Mark sim as Colab-only. See docs/wsl2_render_notes.md")
        return 1

    if try_render(backend):
        return 0

    # Re-exec with the next backend in a fresh process (clean MuJoCo import).
    next_attempt = attempt + 1
    if next_attempt < len(BACKENDS):
        print(f"[smoke_render] re-exec with {BACKENDS[next_attempt]} ...")
        env = dict(os.environ, VLATUNE_RENDER_ATTEMPT=str(next_attempt))
        env.pop("MUJOCO_GL", None)  # let the child set it
        os.execve(sys.executable, [sys.executable, __file__], env)
    print("[smoke_render] FAIL: egl and osmesa both failed. Sim is Colab-only.")
    print("[smoke_render] (non-fatal — this is the documented WSL2 pain point.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
