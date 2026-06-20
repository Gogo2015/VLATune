#!/usr/bin/env python
"""
VLATune Phase 4 (RL) -- custom fine-tune step: reward-weighted / filtered BC
on the policy's own successful rollouts, with a trust-region anchor to pi_SFT.

This is the core RL step (design doc docs/METHOD.md, sec 2-4). One outer
iteration's fine-tune: load the pinned SFT policy, build (obs, action-chunk, pad)
samples from the *kept* (successful) rollout episodes, and continue-train with the
SAME flow-matching regression loss as SFT -- each sample weighted by a function of
its return -- plus an anchor that keeps the policy near pi_SFT.

Loss (design sec 2-3), all hosted in the SmolVLA flow-matching loss:
  * reward-weighted FM:  L_fm = mean_b( w_b * per_sample_loss_b )
      per_sample_loss = policy.forward(batch, reduction="none")  # the lib's RA-BC hook
      w_b = exp(-steps_b / tau_task)   (time-to-success weight, sec 2; w=1 -> pure filtering)
      Under sparse binary reward the *filter* (keep successes) carries the signal;
      the time-to-success weight is the one cheap, non-degenerate reward shape.
      Weights are globally normalized to mean 1 so w=1 exactly recovers filtering
      and the reward shape is mean-preserving (does not silently rescale the LR).
  * anchor to pi_SFT -- two tractable surrogates for the (intractable) marginal KL:
      (a) demo-mix  (lambda, PRIMARY for the full loop, sec 3a): mix a lambda-fraction
          of original SFT demo batches, add their forward(reduction="mean"). Needs the
          428 goal demos on disk -> injected via an optional demo-batch sampler.
      (b) vector-field distillation (beta, GATED add, sec 3b; dataset-free): on the
          SAME (noise, time), penalize ||v_theta - v_SFT||^2 between the live and a
          frozen pi_SFT velocity field. v_t is captured with a forward hook on
          action_out_proj (its output IS v_t in VLAFlowMatching.forward) -- no
          reimplementation of the forward, no monkey-patching.

Contract verified against lerobot 0.5.1 source (modeling_smolvla.py,
policies_factory.py, env_processor.py):
  * SmolVLAPolicy.forward(batch, noise=None, time=None, reduction="mean"|"none").
    reduction="none" -> (per_sample_loss[B], loss_dict); it natively zeroes losses on
    padded chunk steps via batch["action_is_pad"].
  * VLAFlowMatching.forward: x_t=t*noise+(1-t)*a; u_t=noise-a; v_t=action_out_proj(suffix_out);
    losses=mse(u_t,v_t,none). Passing the SAME (noise,time) to live+frozen makes the vf
    penalty well defined.
  * LiberoProcessorStep (env_pre) is OBSERVATION-only; the policy preprocessor `pre`
    (loaded from the checkpoint) is what renames image->camera1/2, normalizes state AND
    action, tokenizes `task`, and moves to device. Stored rollout actions are env-space =
    dataset-space for LIBERO relative control, so `pre` re-normalizes them into the FM target.

Usage:
  probe (synchronous, no training -- gates the action representation):
    python rl_finetune.py --probe --base-pm <rl_base_16k/pretrained_model> \
      --kept-dir <Drive>/VLATune/rl/rollouts_smoke --task-ids 0,9
  smoke fine-tune (background Popen; vf anchor; tiny):
    python rl_finetune.py --base-pm <..> --kept-dir <..> --task-ids 0,9 \
      --out <Drive>/VLATune/rl/rl_smoke_run --max-steps 200 --batch-size 16 \
      --lr 1e-5 --beta 10.0
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

# Our SFT checkpoint kept the base's camera1/2/3 input_features (runbook gotcha #3/#4),
# so the LIBERO image/image2 keys must be renamed -- same map as SFT train + eval.
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
# Optional progress-log file (default None = stdout only); pass --poll to also append to a file.
POLL_DEFAULT = None


def _poll(path, msg):
    line = time.strftime("%H:%M:%S") + " " + msg
    print(line, flush=True)
    if not path:
        return
    try:
        with open(path, "a") as f:
            print(line, file=f)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Build: policy (+ optional frozen pi_SFT) and the policy preprocessor.
# make_policy derives features from env_cfg WITHOUT instantiating MuJoCo (env_to_policy_
# features reads the config only), so no env is spun up here -- the fine-tune is GPU-only.
# --------------------------------------------------------------------------------------
def build_policy(base_pm, device="cuda", train=True, suite="libero_goal"):
    cfg = PreTrainedConfig.from_pretrained(base_pm)
    cfg.pretrained_path = base_pm
    cfg.device = device
    cfg.n_action_steps = 10  # match SFT/eval; chunk_size stays 50
    # LIBERO suites share the obs/action space, so the suite only labels the feature spec
    # (make_policy reads env_cfg WITHOUT spinning up MuJoCo) -- parametrized for loop-on-libero_10.
    env_cfg = LiberoEnv(task=suite, task_ids=[0], control_mode="relative")
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=RENAME_MAP)
    if train:
        policy.train()
    else:
        policy.eval()
        for p in policy.parameters():
            p.requires_grad_(False)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=base_pm,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    return cfg, policy, pre, post


# --------------------------------------------------------------------------------------
# Dataset: flatten kept episodes into (episode, t) samples + per-sample reward weights.
# --------------------------------------------------------------------------------------
class RolloutSamples:
    def __init__(self, kept_dir, task_ids, tau_floor=1.0):
        self.eps = []           # list of episode dicts (kept successes)
        self.index = []         # list of (ep_idx, t)
        self.ep_steps = []      # per-episode terminal step count
        self.ep_task = []       # per-episode task string
        for tid in task_ids:
            p = os.path.join(kept_dir, f"rollouts_task{tid}.pt")
            eps = torch.load(p, map_location="cpu", weights_only=False)
            for e in eps:
                assert int(e["success"]) == 1, "kept set must be all successes"
                ei = len(self.eps)
                self.eps.append(e)
                self.ep_steps.append(int(e["steps"]))
                self.ep_task.append(e["task"])
                T = e["action"].shape[0]
                for t in range(T):
                    self.index.append((ei, t))
        # per-task tau = median success length; per-sample w = exp(-steps/tau_task),
        # then globally normalized to mean 1 (mean-preserving reward shape).
        steps = np.array(self.ep_steps, dtype=np.float64)
        tasks = np.array(self.ep_task)
        self.tau = {}
        ep_w = np.ones(len(self.eps), dtype=np.float64)
        for tk in np.unique(tasks):
            m = tasks == tk
            tau = max(float(np.median(steps[m])), tau_floor)
            self.tau[str(tk)] = tau
            ep_w[m] = np.exp(-steps[m] / tau)
        ep_w = ep_w / ep_w.mean()  # global mean-1 normalization
        self.ep_w = ep_w
        self.sample_w = np.array([ep_w[ei] for ei, _ in self.index], dtype=np.float32)

    def __len__(self):
        return len(self.index)

    def collate(self, ids, chunk_size, n_action_dim):
        """Build the PRE-preprocessor batch dict for a list of flat sample ids."""
        imgs = {k: [] for k in self.eps[0]["image_keys"]}
        states, tasks, acts, pads, ws = [], [], [], [], []
        for sid in ids:
            ei, t = self.index[sid]
            e = self.eps[ei]
            for k in imgs:
                imgs[k].append(e["images"][k][t].float() / 255.0)  # (C,H,W) in [0,1]
            states.append(e["state"][t])                            # (8,)
            tasks.append(e["task"])
            T = e["action"].shape[0]
            n = min(chunk_size, T - t)
            chunk = torch.zeros(chunk_size, n_action_dim, dtype=torch.float32)
            chunk[:n] = e["action"][t:t + n]
            pad = torch.ones(chunk_size, dtype=torch.bool)
            pad[:n] = False
            acts.append(chunk)
            pads.append(pad)
            ws.append(self.sample_w[sid])
        batch = {k: torch.stack(v) for k, v in imgs.items()}        # (B,C,H,W)
        batch["observation.state"] = torch.stack(states)            # (B,8)
        batch["task"] = tasks                                        # list[str]
        batch[ACTION] = torch.stack(acts)                           # (B,chunk,7)
        action_is_pad = torch.stack(pads)                           # (B,chunk)
        w = torch.tensor(ws, dtype=torch.float32)                   # (B,)
        return batch, action_is_pad, w


# --------------------------------------------------------------------------------------
# vf anchor: capture v_t via a forward hook on action_out_proj (its output IS v_t).
# --------------------------------------------------------------------------------------
class VtHook:
    def __init__(self, policy):
        self.v = None
        self.h = policy.model.action_out_proj.register_forward_hook(self._cb)

    def _cb(self, module, inp, out):
        self.v = out

    def remove(self):
        self.h.remove()


def run_loss(policy, pre, batch, action_is_pad, w, device, n_action_dim,
             frozen=None, beta=0.0, hook=None, frozen_hook=None):
    """One reward-weighted FM loss (+ optional vf anchor). Returns (total, parts dict)."""
    nb = pre(batch)
    nb["action_is_pad"] = action_is_pad.to(device)
    # Sample (noise, time) ONCE so the optional frozen pass shares them exactly.
    actions = policy.prepare_action(nb)                          # (B,chunk,max_action_dim)
    noise = policy.model.sample_noise(actions.shape, actions.device)
    t = policy.model.sample_time(actions.shape[0], actions.device)
    per_sample, ld = policy.forward(nb, noise=noise, time=t, reduction="none")
    w = w.to(per_sample.device)
    l_fm = (w * per_sample).mean()
    parts = {"l_fm": float(l_fm.item()), "fm_raw": float(per_sample.mean().item())}
    total = l_fm
    if beta > 0.0 and frozen is not None:
        v_live = hook.v[:, :, :n_action_dim]
        # Reuse the SAME normalized batch nb + the SAME (noise, time): live and frozen
        # share identical normalization stats (both rl_base_16k), so at init v_live==v_frozen
        # and r_vf==0 exactly; it grows only as the live policy drifts. forward() does not
        # mutate nb (adapt_to_pi_aloha is False for LIBERO), so reuse is safe.
        with torch.no_grad():
            frozen.forward(nb, noise=noise, time=t, reduction="none")
        v_frozen = frozen_hook.v[:, :, :n_action_dim]
        r_vf = torch.nn.functional.mse_loss(v_live, v_frozen.detach())
        total = total + beta * r_vf
        parts["r_vf"] = float(r_vf.item())
    return total, parts


def save_ckpt(out, step, policy, base_pm):
    import shutil
    d = os.path.join(out, "checkpoints", f"{step:06d}", "pretrained_model")
    os.makedirs(d, exist_ok=True)
    # Copy aux files (processor configs + their normalization-stat .safetensors that eval needs),
    # skipping ONLY the policy weight file; then write the fresh policy weights + config over them.
    for fn in os.listdir(base_pm):
        if fn in ("model.safetensors", "model.bin", "pytorch_model.bin"):
            continue
        src = os.path.join(base_pm, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(d, fn))
    policy.save_pretrained(d)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-pm", required=True, help="rl_base_16k/pretrained_model (pi_SFT = start = anchor)")
    ap.add_argument("--sft-pm", default=None, help="frozen pi_SFT for vf anchor (default = base-pm)")
    ap.add_argument("--kept-dir", required=True, help="dir with rollouts_task{tid}.pt")
    ap.add_argument("--task-ids", default="0,9")
    ap.add_argument("--suite", default="libero_goal", help="LIBERO suite for policy feature spec (libero_goal, libero_10, ...)")
    ap.add_argument("--out", default=None, help="ckpt output dir (Drive)")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0, help="vf anchor strength")
    ap.add_argument("--save-every", type=int, default=0, help="0 = only at end")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--poll", default=POLL_DEFAULT)
    ap.add_argument("--probe", action="store_true", help="1 batch, print diagnostics, exit")
    args = ap.parse_args()
    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip() != ""]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.poll:
        open(args.poll, "w").close()

    try:
        _poll(args.poll, f"START tasks={task_ids} steps={args.max_steps} bs={args.batch_size} "
                         f"lr={args.lr} beta={args.beta} probe={args.probe}")
        cfg, policy, pre, post = build_policy(args.base_pm, args.device, train=not args.probe, suite=args.suite)
        chunk = int(cfg.chunk_size)
        n_act = int(cfg.action_feature.shape[0])
        data = RolloutSamples(args.kept_dir, task_ids)
        _poll(args.poll, f"built policy + {len(data)} samples from {len(data.eps)} eps; "
                         f"chunk={chunk} n_act={n_act} tau={data.tau}")

        frozen = frozen_pre = hook = frozen_hook = None
        if args.beta > 0.0 or args.probe:
            sft_pm = args.sft_pm or args.base_pm
            _, frozen, _, _ = build_policy(sft_pm, args.device, train=False, suite=args.suite)
            hook = VtHook(policy)
            frozen_hook = VtHook(frozen)
            _poll(args.poll, f"loaded frozen pi_SFT from {os.path.basename(os.path.dirname(sft_pm))} + vf hooks")

        # ---- PROBE: gate the action representation + loss machinery, then exit ----
        if args.probe:
            ids = list(np.random.choice(len(data), size=min(8, len(data)), replace=False))
            batch, pad, w = data.collate(ids, chunk, n_act)
            a_raw = batch[ACTION]
            nb = pre({k: (v.clone() if torch.is_tensor(v) else list(v)) for k, v in batch.items()})
            a_norm = nb[ACTION]
            _poll(args.poll, f"PROBE files@base_pm={sorted(os.listdir(args.base_pm))}")
            _poll(args.poll, f"PROBE pre.steps={[type(s).__name__ for s in pre.steps]}")
            _poll(args.poll, f"PROBE action raw  min/max/mean {a_raw.min():.3f}/{a_raw.max():.3f}/{a_raw.float().mean():.3f}")
            _poll(args.poll, f"PROBE action norm min/max/mean {a_norm.min():.3f}/{a_norm.max():.3f}/{a_norm.float().mean():.3f}")
            _poll(args.poll, f"PROBE state norm  min/max {nb['observation.state'].min():.3f}/{nb['observation.state'].max():.3f}")
            _poll(args.poll, f"PROBE nb keys={sorted(nb.keys())}")
            with torch.no_grad():
                total, parts = run_loss(policy, pre, batch, pad, w, args.device, n_act,
                                        frozen=frozen, beta=max(args.beta, 1.0),
                                        hook=hook, frozen_hook=frozen_hook)
            _poll(args.poll, f"PROBE loss parts={parts} (vf should be ~0 since live==frozen at init)")
            _poll(args.poll, f"PROBE w stats min/max/mean {float(w.min()):.3f}/{float(w.max()):.3f}/{float(w.mean()):.3f}")
            _poll(args.poll, "PROBE-DONE")
            return

        # ---- TRAIN ----
        params = [p for p in policy.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr)
        n_trainable = sum(p.numel() for p in params)
        _poll(args.poll, f"trainable params={n_trainable/1e6:.1f}M; optimizer=AdamW lr={args.lr}")
        # weight snapshot for the anchor-movement gate (L2 drift of action expert from pi_SFT)
        w0 = torch.nn.utils.parameters_to_vector(params).detach().clone()

        order = np.random.permutation(len(data))
        ptr = 0
        losshist = []
        for step in range(1, args.max_steps + 1):
            if ptr + args.batch_size > len(order):
                order = np.random.permutation(len(data))
                ptr = 0
            ids = order[ptr:ptr + args.batch_size].tolist()
            ptr += args.batch_size
            batch, pad, w = data.collate(ids, chunk, n_act)
            total, parts = run_loss(policy, pre, batch, pad, w, args.device, n_act,
                                    frozen=frozen, beta=args.beta,
                                    hook=hook, frozen_hook=frozen_hook)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losshist.append(float(total.item()))
            if step % 20 == 0 or step == 1:
                drift = float((torch.nn.utils.parameters_to_vector(params).detach() - w0).norm().item())
                _poll(args.poll, f"step {step}/{args.max_steps} loss={total.item():.4f} {parts} drift={drift:.4f}")

        drift = float((torch.nn.utils.parameters_to_vector(params).detach() - w0).norm().item())
        ckpt = None
        if args.out:
            ckpt = save_ckpt(args.out, args.max_steps, policy, args.base_pm)
        summary = {
            "task_ids": task_ids, "steps": args.max_steps, "batch_size": args.batch_size,
            "lr": args.lr, "beta": args.beta, "tau": data.tau,
            "loss_first": round(float(np.mean(losshist[:5])), 4),
            "loss_last": round(float(np.mean(losshist[-5:])), 4),
            "param_drift_l2": round(drift, 4), "ckpt": ckpt,
        }
        if args.out:
            json.dump(summary, open(os.path.join(args.out, "finetune_summary.json"), "w"), indent=2)
        _poll(args.poll, "DONE " + json.dumps(summary))
    except Exception as e:
        import traceback
        _poll(args.poll, "ERROR " + repr(e))
        _poll(args.poll, traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
