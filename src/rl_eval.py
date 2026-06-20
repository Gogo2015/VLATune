#!/usr/bin/env python
"""
VLATune Phase 4 (RL) -- per-task evaluation driver (the proxy/claim eval workhorse).

Runs `lerobot-eval` on an RL (or SFT) checkpoint and aggregates per-task pc_success
vs the 89.0% SFT baseline. Each task is evaluated in its OWN single-task subprocess.

Why per-task subprocesses (NOT one multi-task eval): on a long-lived Colab L4 the
multi-task eval crashes at the task->task transition with an EGL context-teardown
error (OpenGL.raw.EGL._errors.EGLError in robosuite egl_context / MjRenderContext)
on long-up runtimes. Isolating each task in its own process avoids the mid-eval
teardown entirely (validated: single-task evals complete cleanly). It also makes the
eval resumable (skip tasks whose eval_info.json already exists) and bounds the EGL
context accumulation that otherwise slows rendering over a long run.

Contract (lerobot 0.5.1): eval_info.json layout is
  {"per_task": [{"task_id": int, "metrics": {"successes": [bool,...], ...}}],
   "per_group": {...}, "overall": {"pc_success": float, "n_episodes": int, ...}}.
Same eval protocol as Phase 3 (relative control, n_action_steps=10, rename_map,
~/.libero/config.yaml preinit, MUJOCO_GL=egl).

Usage:
  python rl_eval.py --ckpt-pm <ckpt>/pretrained_model --task-ids 0,1,2,3,4,5,6,7,8,9 \
    --n-episodes 10 --out-root <Drive>/VLATune/rl/eval_<run> [--baseline-json results/sft_libero_goal.json]
"""
import argparse
import json
import os
import subprocess
import time

RENAME_MAP_JSON = (
    '{"observation.images.image":"observation.images.camera1",'
    '"observation.images.image2":"observation.images.camera2"}'
)
# SFT per-task baseline (results/sft_libero_goal.json), for the delta column.
SFT_BASELINE = {0: 100, 1: 90, 2: 90, 3: 80, 4: 100, 5: 100, 6: 70, 7: 100, 8: 100, 9: 60}


def _poll(path, msg):
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    if path:
        try:
            with open(path, "a") as f:
                print(line, file=f)
        except Exception:
            pass


def _find_info(out_dir):
    for root, _, files in os.walk(out_dir):
        if "eval_info.json" in files:
            return os.path.join(root, "eval_info.json")
    return None


def eval_one_task(ckpt_pm, tid, n_episodes, out_dir, batch_size=4, suite="libero_goal"):
    """Run a single-task lerobot-eval subprocess. Returns (successes, n, ok)."""
    if _find_info(out_dir):  # resumable: already evaluated
        d = json.load(open(_find_info(out_dir)))
        succ = sum(int(bool(s)) for s in d["per_task"][0]["metrics"]["successes"])
        return succ, len(d["per_task"][0]["metrics"]["successes"]), True
    cmd = [
        "lerobot-eval",
        "--policy.path=" + ckpt_pm,
        "--env.type=libero",
        "--env.task=" + suite,
        "--env.control_mode=relative",
        "--env.task_ids=[%d]" % tid,
        "--env.max_parallel_tasks=1",
        "--policy.n_action_steps=10",
        "--rename_map=" + RENAME_MAP_JSON,
        "--eval.batch_size=%d" % batch_size,
        "--eval.n_episodes=%d" % n_episodes,
        "--output_dir=" + out_dir,
    ]
    log = open(os.path.join(out_dir + ".log"), "w")
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    info = _find_info(out_dir)
    if r.returncode != 0 or info is None:
        return 0, 0, False
    d = json.load(open(info))
    succ = sum(int(bool(s)) for s in d["per_task"][0]["metrics"]["successes"])
    n = len(d["per_task"][0]["metrics"]["successes"])
    return succ, n, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-pm", required=True, help="path to the checkpoint's pretrained_model dir")
    ap.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--n-episodes", type=int, default=10, help="episodes PER TASK")
    ap.add_argument("--out-root", required=True, help="dir to hold per-task eval outputs + summary")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--poll", default=None, help="optional progress-log file (default: stdout only)")
    ap.add_argument("--label", default="rl", help="tag for the summary (e.g. iter1_b10)")
    ap.add_argument("--suite", default="libero_goal", help="LIBERO suite = --env.task (libero_goal, libero_10, ...)")
    args = ap.parse_args()
    is_goal = args.suite == "libero_goal"  # SFT per-task baseline + 89.0 delta only meaningful on Goal
    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip() != ""]
    os.makedirs(args.out_root, exist_ok=True)
    if args.poll:
        open(args.poll, "w").close()
    _poll(args.poll, f"EVAL START ckpt={args.ckpt_pm} tasks={task_ids} n={args.n_episodes}")

    per_task, tot_s, tot_n, failed = {}, 0, 0, []
    for tid in task_ids:
        out_dir = os.path.join(args.out_root, f"t{tid}")
        os.makedirs(out_dir, exist_ok=True)
        succ, n, ok = eval_one_task(args.ckpt_pm, tid, args.n_episodes, out_dir, args.batch_size, args.suite)
        if not ok:
            failed.append(tid)
            _poll(args.poll, f"task{tid} EVAL-FAILED (EGL? rerun this tid)")
            continue
        tot_s += succ
        tot_n += n
        pc = round(100.0 * succ / max(1, n), 1)
        base = SFT_BASELINE.get(tid) if is_goal else None
        per_task[str(tid)] = {"successes": succ, "n": n, "pc_success": pc,
                              "sft_pc": base, "delta": (None if base is None else round(pc - base, 1))}
        _poll(args.poll, f"task{tid} {succ}/{n} = {pc}%  (SFT {base}%, delta {per_task[str(tid)]['delta']})")

    overall = round(100.0 * tot_s / max(1, tot_n), 1)
    summary = {"label": args.label, "suite": args.suite, "ckpt": args.ckpt_pm, "n_per_task": args.n_episodes,
               "overall_pc_success": overall, "n_episodes": tot_n,
               "per_task": per_task, "failed_tasks": failed,
               "sft_overall": (89.0 if is_goal else None),
               "overall_delta": (round(overall - 89.0, 1) if is_goal else None)}
    json.dump(summary, open(os.path.join(args.out_root, "eval_summary.json"), "w"), indent=2)
    _poll(args.poll, "EVAL-DONE " + json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
