#!/usr/bin/env python
"""
VLATune Phase 4 (RL) -- rollout collection for filtered / reward-weighted BC.

Runs rollouts of the pinned SFT policy in LIBERO and records, per env step, the
observation the policy *saw* together with the action it *took*, the terminal
binary success, and the step-count to termination. Successful trajectories are
filtered and saved to Drive; the fine-tune step then builds (obs, action-chunk)
samples from them and continue-trains the policy on its own successes.

Design contract (verified against lerobot 0.5.1):
  * Mirrors lerobot's eval rollout() loop EXACTLY for the policy/env contract
    (relative control, n_action_steps=10, rename_map, MUJOCO_GL=egl), so the
    success flag here is the same predicate lerobot-eval scores -- the smoke
    gate "rollout success == pc_success" is then near-tautological.
  * The ONE deliberate difference: we snapshot the observation AFTER the LIBERO
    env-processor (LiberoProcessorStep) and BEFORE the policy normalizer, i.e.
    in the SFT training representation: observation.images.image / image2
    (float32 BCHW in [0,1], flipped 180 deg) + observation.state (B,8) =
    [eef_pos(3), quat->axis-angle(3), gripper.qpos(2)]. Stock rollout()
    snapshots one stage earlier, which is why we need a custom loop.
  * Images are stored as uint8 (re-scaled from [0,1]) to keep Drive size sane;
    the fine-tune loader converts back to float32 [0,1] -- the policy only ever
    consumes [0,1], so this round-trips losslessly at image precision.

Usage (smoke): python rl_rollout.py --task-ids 0,9 --n-per-task 10
               --base-pm <Drive>/VLATune/rl/rl_base_16k/pretrained_model --out <out dir>

"""
import argparse
import json
import os
import time

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

# Our SFT checkpoint keeps the base's camera1/2/3 input_features (runbook gotcha #3),
# so the LIBERO image/image2 keys must be renamed at inference -- same map as eval.
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
# Optional progress-log file (default None = stdout only). Pass --poll to also
# append timestamped progress lines to a file when monitoring long unattended runs.
POLL = None


def _poll(msg):
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    if not POLL:
        return
    try:
        with open(POLL, "a") as f:
            print(line, file=f)
    except Exception:
        pass


def build(task_ids, n_envs, base_pm, device="cuda", suite="libero_goal"):
    """Construct env + policy + the four processor pipelines exactly as eval_main."""
    env_cfg = LiberoEnv(task=suite, task_ids=list(task_ids), control_mode="relative")
    envs = make_env(env_cfg, n_envs=n_envs, use_async_envs=False)

    policy_cfg = PreTrainedConfig.from_pretrained(base_pm)
    policy_cfg.pretrained_path = base_pm
    policy_cfg.device = device
    policy_cfg.n_action_steps = 10  # match SFT/eval; chunk_size stays 50
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=RENAME_MAP)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=base_pm,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    return env_cfg, envs, policy, preprocessor, postprocessor, env_pre, env_post


def _img_keys(obs_env):
    """Image keys present after the env-processor (expected: image, image2)."""
    return sorted(k for k in obs_env if k.startswith("observation.images."))


@torch.no_grad()
def rollout_vec(env, policy, pre, post, env_pre, env_post, seeds):
    """One batched rollout of a LIBERO vec env (n parallel episodes of one task).

    Returns a list of per-episode dicts: each holds the obs snapshot sequence
    (uint8 images + state) the policy saw, the executed action sequence, the
    terminal success (0/1), the step-count, and the task string.
    """
    policy.reset()
    obs, _ = env.reset(seed=seeds)
    n = env.num_envs
    max_steps = int(env.call("_max_episode_steps")[0])

    imgs = None  # lazily keyed by the actual image keys
    state_acc = [[] for _ in range(n)]
    act_acc = [[] for _ in range(n)]
    img_acc = None
    tasks = [None] * n
    done = np.zeros(n, dtype=bool)
    success = np.zeros(n, dtype=bool)
    steps = np.zeros(n, dtype=int)

    step = 0
    while not np.all(done) and step < max_steps:
        o = preprocess_observation(obs)
        o = add_envs_task(env, o)
        o_env = env_pre(o)  # <-- SFT-representation snapshot point

        ik = _img_keys(o_env)
        assert ik, f"no observation.images.* after env-processor; keys={list(o_env.keys())}"
        if img_acc is None:
            imgs = ik
            img_acc = {k: [[] for _ in range(n)] for k in imgs}
        # store images as uint8 (B,C,H,W) and state as float32 (B,8)
        u8 = {k: (o_env[k].clamp(0, 1) * 255).round().to(torch.uint8).cpu() for k in imgs}
        st = o_env["observation.state"].float().cpu()

        for i in range(n):
            if not done[i]:
                for k in imgs:
                    img_acc[k][i].append(u8[k][i])
                state_acc[i].append(st[i])
                if tasks[i] is None:
                    tasks[i] = o["task"][i]

        o_norm = pre(o_env)
        action = policy.select_action(o_norm)
        action = post(action)
        action = env_post({ACTION: action})[ACTION]
        a_np = action.to("cpu").numpy()
        for i in range(n):
            if not done[i]:
                act_acc[i].append(torch.from_numpy(a_np[i]).float())

        obs, _reward, term, trunc, info = env.step(a_np)
        if "final_info" in info:
            succ_now = np.asarray(info["final_info"]["is_success"]).astype(bool)
        else:
            succ_now = np.zeros(n, dtype=bool)

        newly = (term | trunc) & (~done)
        for i in range(n):
            if newly[i]:
                steps[i] = step + 1
                success[i] = bool(succ_now[i])
        done = term | trunc | done
        if step + 1 == max_steps:
            for i in range(n):
                if not done[i]:
                    steps[i] = max_steps
            done = np.ones(n, dtype=bool)
        step += 1

    episodes = []
    for i in range(n):
        T = len(act_acc[i])
        if T == 0:
            continue
        ep = {
            "images": {k: torch.stack(img_acc[k][i][:T]) for k in imgs},  # each (T,C,H,W) uint8
            "state": torch.stack(state_acc[i][:T]),  # (T,8) float32
            "action": torch.stack(act_acc[i]),  # (T,7) float32
            "success": int(success[i]),
            "steps": int(steps[i]),
            "task": tasks[i],
            "image_keys": imgs,
        }
        episodes.append(ep)
    return episodes


def collect(task_ids, n_per_task, n_envs, base_pm, out, device="cuda", suite="libero_goal"):
    os.makedirs(out, exist_ok=True)
    if POLL:
        open(POLL, "w").close()
    _poll(f"START suite={suite} tasks={task_ids} n_per_task={n_per_task} n_envs={n_envs}")
    _, envs, policy, pre, post, env_pre, env_post = build(task_ids, n_envs, base_pm, device, suite)
    _poll("built env+policy+processors")

    summary = {}
    for tid in task_ids:
        outp = os.path.join(out, f"rollouts_task{tid}.pt")
        if os.path.exists(outp):  # resumable: a prior (possibly preempted) run already did this task
            _poll(f"task{tid} SKIP (exists)")
            continue
        vec = envs[suite][tid]
        kept, n_done, n_succ, b = [], 0, 0, 0
        while n_done < n_per_task:
            seeds = list(range(1000 + b * n_envs, 1000 + (b + 1) * n_envs))
            eps = rollout_vec(vec, policy, pre, post, env_pre, env_post, seeds)
            for ep in eps[: n_per_task - n_done]:
                n_done += 1
                if ep["success"] == 1:
                    n_succ += 1
                    kept.append(ep)
            b += 1
            _poll(f"task{tid} {n_done}/{n_per_task} succ={n_succ} kept={len(kept)}")
        torch.save(kept, outp)
        summary[str(tid)] = {
            "n": n_done,
            "successes": n_succ,
            "pc_success": round(100.0 * n_succ / max(1, n_done), 1),
            "kept_episodes": len(kept),
            "kept_frames": int(sum(ep["action"].shape[0] for ep in kept)),
        }
        _poll(f"task{tid} DONE {summary[str(tid)]}")

    json.dump(summary, open(os.path.join(out, "rollout_summary.json"), "w"), indent=2)
    _poll("ALL-DONE " + json.dumps(summary))
    return summary


def main():
    global POLL
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-ids", type=str, required=True, help="comma-separated, e.g. 0,9")
    ap.add_argument("--n-per-task", type=int, default=10)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--base-pm", type=str, required=True, help="path to rl_base_16k/pretrained_model")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--suite", type=str, default="libero_goal", help="LIBERO suite = LiberoEnv `task` field (libero_goal, libero_10, ...)")
    ap.add_argument("--poll", type=str, default=POLL, help="poll-file path (set per-process when running collectors in parallel)")
    args = ap.parse_args()
    POLL = args.poll
    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip() != ""]
    try:
        collect(task_ids, args.n_per_task, args.n_envs, args.base_pm, args.out, args.device, args.suite)
    except Exception as e:
        import traceback

        _poll("ERROR " + repr(e))
        _poll(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
