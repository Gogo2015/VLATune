# RL Method: reward-weighted / filtered behavior cloning with a vector-field anchor

Reward-weighted / KL-to-SFT **filtered behavior cloning** on top of the SFT checkpoint
`smolvla_sft_goal_run1/checkpoints/last` (**89.0%**, step-16k, on Drive under `VLATune/sft/`).
This is the de-risked filtered/reward-weighted BC route, chosen over the riskier on-policy
flow-matching RL alternatives. Comparison target:
**89.0%** (`results/sft_libero_goal.json`). The RL headline is the **SFT→RL delta on top of
this checkpoint**, not a re-comparison to the 85.0% baseline.

Why this method and not PPO, in one line: SmolVLA's action expert is a **flow-matching
denoiser** with **no cheaply/stably available per-action log-probability**, so PPO's ratio
term can't be formed. Filtered/reward-weighted BC needs no such likelihood (§1).

---

## Decisions locked

| Decision | Choice | Where |
|---|---|---|
| Algorithm | Reward-weighted / filtered BC = AWR/RWR adapted to flow matching (ORW-CFM family) | §1 |
| Reward | Sparse binary success (sim ground-truth predicate); **+ time-to-success weight** to make weighting non-degenerate | §2 |
| KL-to-SFT surrogate | **Vector-field distillation** (hosted in the custom loss) + SFT-demo mixing; NOT exact marginal KL | §3 |
| Implementation | **Custom training loop**: per-sample weighted FM loss + anchor (NOT `lerobot-train` reuse); exact wiring pending VM source inspection | §3 |
| The one sweep | **Anchor strength** (demo-mix fraction λ, or vector-field β if used) | §4 |
| Compute | **Single-card on L4** for the whole loop (rollouts are the sink, fine-tune is short) | §5 |
| Start point | **Start RL from the 16k checkpoint**; do NOT resume SFT to 20k first | §7 |
| Where to measure | **Per-task on Goal + a LIBERO-Long (`libero_10`) headroom probe**, not Goal-aggregate alone | §8 |
| Claim-grade eval | **n=500+ episodes, ≥3 seeds**, confounds stated up front | §10 |

---

## 1. The algorithm and why it fits

**What it is.** Continue-train the SFT policy on its *own* successful rollouts, using the
**same flow-matching regression loss as SFT**, each sample weighted by a function of its
return, with a regularizer anchoring back to the frozen SFT policy. This is the
**advantage-weighted / reward-weighted regression** family (RWR, Peters & Schaal 2007; AWR,
Peng et al. 2019, arXiv:1910.00177) applied to a flow-matching policy: **ORW-CFM**
(Online Reward-Weighted Conditional Flow Matching, arXiv:2502.06061), the most directly
on-point cite. The kickoff also lists energy-weighted flow matching for offline RL
(arXiv:2503.04975) as supporting.

**Why it needs no action log-probabilities.** The flow-matching training loss is a
**supervised regression**: on interpolated samples between noise ε and a demonstrated action
chunk *a*, regress the predicted denoising velocity field `v_θ(x_t, t | s)` toward the
noise→action target velocity. It is an MSE in (state, noised-action, time), with **no normalizing
constant and no marginal likelihood of *a* anywhere in it.** Reweighting that loss by a
trajectory's return therefore needs only:

1. the ability to **sample** action chunks from the policy (forward ODE, i.e. just
   inference), and
2. the **environment reward** of the resulting trajectory (the sim hands us a binary success),

and then a regression step toward the high-return actions the policy itself produced. At no
point do we evaluate `π_θ(a | s)`.

**Contrast with PPO (the Q4 answer).** PPO's objective is built on the probability
ratio `r_t = π_θ(a|s) / π_old(a|s)`. Forming it requires the **marginal likelihood of the
sampled action chunk** under a flow-matching policy, which means marginalizing over the noise
initialization and the entire ODE path that produced *a*, which is not cheaply or stably tractable.
*(Caveat to volunteer: an exact likelihood does exist via the
probability-flow ODE / instantaneous change-of-variables, a continuous-normalizing-flow
log-det-Jacobian with a Hutchinson trace estimate. It is just **expensive and high-variance**,
so nobody forms PPO ratios that way in practice. This is why the flow-RL literature
instead **injects noise** to recover a cheap per-step Gaussian log-prob (Flow-SDE/Flow-Noise
in πRL arXiv:2510.25889, the learnable noise in ReinFlow arXiv:2505.22094, the SDE conversion
in Flow-GRPO arXiv:2505.05470), or treats the denoising chain as an inner MDP and does PPO
through it, DPPO arXiv:2409.00588.)*

**Where our method sits in that landscape.** Those noise-injection / two-layer-MDP methods
*recover* a tractable likelihood so they can run full on-policy policy gradient. We deliberately
**don't recover one**; we pick the objective (reward-weighted regression) that never needed it.
That is the point of Tier A: it sidesteps the intractable-likelihood problem rather than
engineering around it, at the cost of being "RL-lite" (offline/advantage-weighted, not on-policy
PPO). DPPO/πRL/ReinFlow/Flow-GRPO are the **Tier-C stretch / related work** we cite as the
frontier and explicitly scope out; SimpleVLA-RL (arXiv:2509.09674) and "RFT of flow-matching
VLAs" (arXiv:2510.09976) are the supporting evidence that RL *does* move flow/VLA policies,
especially on generalization.

---

## 2. The rollout → filter → fine-tune loop

One **outer iteration** (start: `π_0` = the 16k SFT checkpoint):

1. **Rollout collection (sim-bound → L4).** For each of the 10 goal tasks, run *N* rollouts of
   the current policy `π_k` in LIBERO under the **exact eval env config** (relative control,
   `n_action_steps=10`, `MUJOCO_GL=egl`, the camera `rename_map`). Record each trajectory's
   (obs, action-chunk) sequence and its terminal **binary success** `r ∈ {0,1}` (and the
   step-count, free from the sim).
2. **Filter / weight.** Keep the (obs, action) pairs from **successful** trajectories; discard
   failures. Assign each kept pair a weight `w` (see the collapse note below). This gives dataset `D_k`
   of self-generated successful demonstrations.
3. **Fine-tune (GPU-bound, but short).** Continue-train `π_k` on `D_k` with the flow-matching
   loss × `w`, **plus the anchor term (§3)**, at a **small LR** (§4), for a **small** number of
   inner steps (refinement, not a fresh 20k run). This gives `π_{k+1}`.
4. **Eval gate.** Eval `π_{k+1}` (proxy eval between iterations; full 100-/500-ep at the end).
   Keep if it improves; **early-stop on regression** (drift guard).
5. Repeat for a few outer iterations (§4).

**The sparse-binary collapse.** With reward `r ∈ {0,1}` and no
other signal, the per-trajectory return is 0 or 1, so the weight `w` is binary and
**"reward-weighting" is mathematically identical to filtering**: keep successes, drop failures,
BC on what's kept. Under sparse binary reward our "reward-weighted BC" *is*
**rejection-sampling fine-tuning** (the robotics analogue of best-of-N → SFT-on-the-passes /
STaR / RFT in the LLM world). Say this plainly in the writeup. It is the precise, defensible
answer to "isn't this just more SFT?" (Q10): the *signal source* is the environment's reward on
the policy's **own** rollouts, not human demos, which makes it a legitimate
(offline/advantage-weighted) RL step, though with a pure 0/1 reward the *weighting* itself is
degenerate.

**What makes the weighting non-trivial (and whether it's worth it):**

| Richer return | Cost | Worth it? |
|---|---|---|
| **Time-to-success / efficiency**: weight successes by `w = exp(-steps/τ_t)` (or rank) | **Free**, step count already logged | **Yes, adopt.** Cheapest way to make weighting non-degenerate; expresses an efficiency preference the binary predicate ignores (don't dawdle). |
| **Learned advantage baseline**: per-task success-rate baseline, or a small critic; weight by `exp(A/τ)` | Per-task baseline ~free; learned critic = real work, drifts toward Tier B | Per-task baseline: optional (normalizes task difficulty so easy-task successes don't dominate). Learned critic: **out of scope.** |
| **Dense/shaped reward** (distance-to-goal, sub-goal predicates) | LIBERO gives none natively; building it is real work + reward-hacking surface | **No.** |

**Recommendation:** ship **pure success-filtering** as the sparse-reward baseline, and
add **time-to-success weighting** as the one cheap step that makes it genuinely *reward*-weighted
rather than just *filtered*. State the collapse explicitly; don't let "reward-weighted" oversell
a binary-reward filter.

---

## 3. How "KL-to-SFT" is actually implemented (for a flow policy)

The exact marginal `KL(π_θ ‖ π_SFT)` over action chunks needs both marginal likelihoods, so it is
**intractable for the same reason PPO is** (§1). The anchor must therefore be a **tractable surrogate**
that never evaluates a marginal likelihood. In preference order:

> **Implementation decision (2026-06-16): a custom training loop.** We are NOT reusing
> `lerobot-train` on a rollout dataset; we write our own fine-tune step over the SmolVLA
> flow-matching loss. Two consequences: (i) the time-to-success reward weight (§2) is applied
> **per-sample inside the loss** (`reduction='none'` × weights → true reward-weighted
> regression), not approximated by sampling frequency; (ii) the **vector-field distillation
> penalty (b) below becomes the primary explicit anchor**, since the custom loss is its
> natural host. SFT-demo mixing (a) stays as a complementary data-level anchor. **Key build
> task:** locate where lerobot 0.5.1's `SmolVLAPolicy` computes/reduces the flow-matching MSE,
> to inject the per-sample weights and the frozen-`π_SFT` velocity-field reference.

> **Anchor decision: vf-distillation (β) is the PRIMARY, swept; demo-mix (λ) is a single
> end-of-loop robustness run, not the per-iteration default.**
> Rationale: (1) the per-sample reward-weighted FM loss + vf anchor are **validated in a smoke
> fine-tune**: on tasks 0,9 it gave an FM-loss drop 1.3→0.09 (β=0) and the anchor
> restrained param-drift 2.15 (β=0) → 0.98 (β=50), with `r_vf` captured via a forward hook on
> `action_out_proj` (its output *is* `v_t`; no monkey-patching, no reimplementation). (2)
> **Compute-frugality finding:** demo-mix needs the 428 SFT demos on disk, but the broken
> file-index (runbook gotcha #2) forces a **full ~35 GB re-download per L4 runtime**, the
> project's largest single data cost, repeated each runtime, which undercuts the "$20–60 VLA
> research" thesis. vf reuses the already-pinned frozen `π_SFT`: **zero extra data**. (3) vf
> follows the §3 impl-note above (the most recent design decision). The β sweep is **{0, 1, 10,
> 30}** (smoke showed β=50 over-restrains, l_fm stuck at 0.33; β=0 = pure-filtering baseline).
> demo-mix's one genuine edge (re-anchoring to human-demo *diversity*, guarding mode-collapse)
> is captured by running it **once** on the best β config as a documented robustness check,
> paying the 35 GB a single time, to show the headline is stable across the anchor family.
> **The (a)/(b) "primary/optional" labels below are superseded by this note** (they reflect the
> pre-implementation plan); the algebra and tractability arguments still hold for both.

**(a) SFT-demo mixing, the primary anchor (recommend).** Interleave a fraction **λ** of the
original 428 goal SFT demonstrations into every fine-tuning batch. The demo term continually
re-fits the policy toward `π_SFT`'s training distribution while the rollout term pushes toward
self-generated successes. This is a **data-level trust region**: the policy can't drift far from
`π_SFT` because it's perpetually re-anchored to `π_SFT`'s data. It is simple to implement (two losses
summed, or one interleaving dataloader), fully tractable, and directly interpretable. **This is
the surrogate to lead with.**

**(b) Vector-field (denoising-velocity) distillation penalty, optional add.** Keep a frozen
copy of `π_SFT`. On the *same* noised inputs `(x_t, t, s)`, penalize the squared distance between
the two velocity fields:

```
R_vf(θ) = E_{s∼rollout, t, ε} ‖ v_θ(x_t, t | s) − v_SFT(x_t, t | s) ‖²
```

This is a well-defined pointwise (Fisher-divergence-flavored) distance between the two **flows**:
it anchors the *mechanism* (the learned velocity field) rather than the data. It is the tractable
analogue of the per-denoising-step KL that DPPO/ReinFlow get "for free" by working in
noise-injected step-space. Costs one frozen reference forward pass per step (a distillation loss).
Add it, weighted by **β**, only if the smoke/early iterations show drift that demo-mixing alone
doesn't contain.

**(c) Sampled-action distance** (MSE between action chunks sampled from `π_θ` and `π_SFT` on the
same states): weakest, given sampling variance plus extra `π_SFT` rollout cost. Mention, don't recommend.

**Where the regularization "really" lives (state this, it shows you understand AWR).** AWR's
KL-to-reference is **implicit in the weighting itself**: `max_π E[A] − (1/τ)·KL(π‖π_ref)` has the
closed form `π*(a|s) ∝ π_ref(a|s)·exp(A(s,a)/τ)`, which you fit by *weighted regression on samples
drawn from `π_ref`*. Because our rollouts come from a policy that *starts as* `π_SFT` (and stays
near it under small LR), the reference pull is **partly baked in by where the data comes from**.
The explicit term (a), and optionally (b), is belt-and-suspenders to hold the anchor across
multiple outer iterations, where the implicit pull weakens as `π_k` drifts from `π_0`.

**Do not** describe this as "we add a KL term." It is a *surrogate* trust region (data-mixing
and/or vector-field distance), and the exact marginal KL is deliberately not computed.

---

## 4. Key hyperparameters

| Hyperparameter | Value | Why |
|---|---|---|
| Rollouts per iteration, *N* | 50/task × 10 = **500** | At ~89% success that's ~445 successful trajectories, i.e. tens of thousands of (obs, chunk) pairs, plenty for filtered BC. Dominant sim-cost driver. |
| Filter threshold | `success == 1` (binary) | Sparse reward; keep successes, drop failures (§2). |
| Return weight | `w = exp(−steps/τ_t)` on successes | Time-to-success weighting (§2); `τ_t` ≈ median success length. Set to `w=1` to recover pure filtering. |
| **Anchor strength** (THE sweep) | demo-mix **λ ∈ {0.25, 0.5, 0.75}** (or vector-field **β ∈ {0, 0.1, 1, 10}** if (b) used) | The collapse-vs-improvement knob, the one HP we sweep. Too weak → drift/collapse; too strong → no movement (≈ frozen `π_SFT`). |
| Learning rate | **1e-5 – 3e-5** (3–10× below SFT's ~1e-4) | Starting from a good policy; small LR is itself a trust region. |
| Outer iterations | **3–5**, early-stop by eval | Filtered BC converges fast; more iterations invite drift on a near-saturated suite. |
| Inner grad steps / iteration | **~500–2000** (≈1–3 epochs over `D_k`) | Refinement, **not** a fresh 20k run. |
| Batch size | **64** | Same as SFT; fits L4 (≈22 GB at batch 64, per the SFT recipe) and A100. |
| Denoising / flow steps | **SFT default** (unchanged) | Changing it changes the policy's behavior *and* the eval protocol; hold fixed. |
| `n_action_steps` | **10** | Match SFT/eval exactly. |
| Optimizer | AdamW (SFT default), fresh state | New objective; low/no warmup. |

---

## 5. Compute plan (the L4-vs-A100 split)

**The tension.** Each outer iteration has a **sim-bound** phase (rollouts, L4-optimal,
CPU/MuJoCo) *and* a **GPU-bound** phase (fine-tune, A100-optimal). Colab gives one runtime/GPU
at a time. Two ways to handle it:

- **(a) Single-card loop on L4 (recommend).** Run everything on one L4. Rollouts are cheap there;
  the fine-tune step is **short** (~500–2000 steps, not 20k) and fits L4 (the SFT recipe already
  ran batch 64 in ~22 GB on an L4-class card). The A100's speed only pays off for *long*
  GPU-bound runs, which RL iterations are not, so paying the A100 unit rate (~4–5× L4) to idle
  it through sim is exactly the anti-pattern the project already calls out ("match hardware to the
  bottleneck"; the bottleneck here is **sim**). Avoids per-iteration runtime juggling.
- **(b) Drive handoff (fallback).** Rollouts on L4 → write filtered `D_k` to Drive → fine-tune on
  A100 → write checkpoint to Drive → next iteration's rollouts on L4. Matches each phase to its
  ideal card but adds manual runtime switching **per iteration** and Drive-handoff friction.
  Use **only** if L4 fine-tune throughput becomes the bottleneck (it shouldn't, given short
  inner runs).

**Budget: RL collects rollouts, so it costs more sim-hours than SFT.** At the measured L4 eval
rate (~80 s/episode, batch 4):

| Per outer iteration | Episodes | L4 wall time |
|---|---|---|
| Rollout collection (500, batch 4) | 500 | ~2.8 h |
| Fine-tune (short, on L4) | n/a | ~0.5–1 h |
| Proxy eval (n=100) | 100 | ~2.2 h |
| **Per iteration** | | **~5.5–6 h** |
| **× 3–5 iterations** | | **~17–30 h** |
| Anchor sweep (re-run fine-tune+eval for each λ) | | + a few × (1 + 2.2) h |
| Final claim eval (n=500, ×3 seeds) | 1500 | ~33 h (or split across runtimes) |

This is **meaningfully more than SFT** (~7–13 h total) and is the real Phase-4 cost. Two levers
to contain it: **batch-parallel envs** during collection (the 80 s/ep already assumes batch 4, so
push higher if RAM allows), and a **cheap proxy eval** (fewer episodes, e.g. n=50, or the
weak-task subset) *between* iterations, reserving the full n=500 protocol for the **final** claim
only. Checkpoint every iteration to Drive (preemption insurance, same as SFT). Fits the Pro+
$200–500 budget but is the largest single compute line in the project; say so.

---

## 6. Validation ladder (gated; don't skip the smoke run)

Mirror Phase 3: a 1–2-task smoke that must pass **every** gate before the full 10-task loop.

**Smoke (tasks 0 and 9, the strong one and the persistent 60% one):**

- **Rollouts collect + score**: ~10 rollouts each on tasks 0, 9; trajectories record; the binary
  success flag is read correctly from the env.
- **Success detection agrees with eval**: the rollout success flag matches `lerobot-eval`'s
  `pc_success` on the *same* task/env (same predicate, same config). Rules out a
  success-parsing bug that would poison the filter.
- **Filter keeps the right trajectories**: kept set = successes, discarded = failures; eyeball
  one kept video (succeeds) and one discarded (fails).
- **Fine-tune loss runs**: one fine-tune pass on the filtered set; loss is finite and decreases.
- **The anchor term works**: high anchor strength → policy barely moves (≈ frozen `π_SFT`);
  anchor off → it moves more. Confirms the term is wired and directionally correct.
- **No one-iteration collapse**: after one tiny iteration, eval on tasks 0/9 doesn't crater
  (e.g. not → 0%, which would signal a wiring bug or catastrophic drift).
- **Checkpoint save / resume / eval-load**: the RL checkpoint saves to Drive, reloads, and runs
  in `lerobot-eval` **with the SAME `rename_map`** (image→camera1, image2→camera2). The RL
  checkpoint inherits the SFT checkpoint's `camera1/2/3` keys, so this is Phase-3 gotcha #4 all
  over again. Pre-create `~/.libero/config.yaml` (gotcha #5) before any eval subprocess.

**Full (all 10 goal tasks):** run the loop with Drive checkpointing each iteration; eval each
iteration on the **standard 100-ep protocol** (relative control, `n_action_steps=10`, the
`rename_map`, `~/.libero/config.yaml` preinit) **vs the 89.0% SFT checkpoint**; early-stop on
regression. Final claim eval at n=500+, ≥3 seeds (§10).

---

## 7. The open decision: start from 16k vs resume SFT to 20k first

**Recommendation: start RL directly from the 16k checkpoint.** Do not spend a fresh A100 + 35 GB
re-download to resume SFT to 20k first.

**Why it's defensible:** the RL headline is the **SFT→RL delta measured against
whatever you start from.** If RL fine-tunes the 16k checkpoint and you eval *both* the 16k SFT
base **and** the RL result under the identical protocol, the delta is clean **regardless** of
whether the base was 16k or 20k; the starting point is held fixed on both sides of the
comparison. The **16k-vs-20k confound only contaminates the *SFT-vs-official-baseline* claim**
(the +4, which `FAQ.md` already caveats as within noise and explicitly *not* the
headline). It does **not** contaminate the RL-vs-SFT delta.

The cost of resuming first: a fresh A100 + ~35 GB re-download for ~4k steps on already-flat loss
(~0.10), likely **<2 pts**, itself inside the n=100 CI. You'd spend the project's scarcest
resource buying precision on a number that isn't the headline.

**Downside to volunteer (don't skip it):** starting from an under-trained base means the result is
"RL on top of a **16k** checkpoint," not "RL on top of a *fully-trained* SFT." If a reviewer
specifically wants the latter, you can't show it without the 20k resume. The delta itself is still
valid. **Treat "resume to 20k" as optional polish for the SFT-vs-baseline sub-claim, not a Phase-4
blocker.**

**Operational note:** whatever checkpoint you start from becomes the **fixed `π_SFT` reference**
for the anchor (§3) *and* the comparison baseline. **Pin it:** copy `checkpoints/last` to an
immutable `checkpoints/rl_base_16k/` on Drive so a later SFT resume (or a stray re-run) can't
overwrite the thing every RL number is measured against.

---

## 8. The headroom problem (do not skip this)

**The risk, stated.** Goal is **~saturated at 89%** with a CI of roughly **±7–10 pts** at n=100
(the naive binomial is ≈±6, but episodes cluster within 10 tasks, and within-task correlation widens
the honest interval). So an RL gain of, say, 89→92 is **invisible** at n=100, and Goal-*aggregate*
is a poor place to demonstrate an RL effect. Design around it three ways:

1. **Report per-task, not just aggregate.** The SFT per-task
   floor is **task 9 @ 60% (6/10)**, then **task 6 @ 70%**, **task 3 @ 80%**. Task 9 stayed at 6/10
   through SFT (it was the baseline's weak task too), so it is the natural RL target. If RL lifts
   task 9 to 9/10, that's **visible even if the aggregate barely moves**. On a near-saturated
   suite, per-task is the measurement, not a supplement.
2. **Add a harder / OOD headroom probe where RL gains are visible:**
   - **LIBERO-Long (`libero_10`)**: reproduced **~52–71%**, far from saturated, lots of room.
     Caveat to volunteer: our checkpoint is goal-only, so on `libero_10` it's OOD/near-zero-shot,
     and **filtered BC needs *some* successes to bootstrap**. If the success rate is very low,
     the filter starves (sparse reward bites hardest exactly where there's the most headroom).
     So `libero_10` is a **probe**, not the main loop; if successes are too scarce, fall back to
     reporting it descriptively or seed it with a few demos.
   - **Held-out goal variants** (secondary): hold out 1–2 goal tasks from rollout/fine-tune,
     eval on them. Does RL on the other 8 help or hurt the held-out 2? Tests overfitting to the
     trained tasks. Or perturb initial states for an in-suite OOD signal.
3. **Frame a flat result as a finding.** The headline is "SFT→RL delta **wherever it's
   measurable**", primarily **per-task Goal (tasks 9, 6, 3) + the `libero_10` probe**. A flat
   result on saturated in-distribution Goal, reported honestly with the CI, **is itself a
   result** ("RL-lite doesn't move a near-saturated in-distribution suite; here's the per-task and
   OOD evidence"), and is consistent with the project's risk framing that an RL
   null/negative is still a respectable deliverable.

**Recommendation:** primary target = **per-task Goal + LIBERO-Long (`libero_10`) probe**;
held-out-goal as a secondary OOD probe if time allows.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Collapse / drift** (self-imitation amplifies a narrow mode) | Anchor (§3: demo-mix + optional vector-field) + small LR + **early-stop by eval each iteration** + keep outer iterations few. |
| **Reward sparsity** | On **Goal**, the 89% policy yields ~445/500 successful rollouts, so no starvation for filtered BC. **(Starvation IS real on `libero_10`** where success is low, see §8; that's why it's a probe, possibly demo-seeded.) |
| **Reward hacking** | The binary predicate is the **sim's ground-truth goal state**, far harder to hack than a learned reward. The *time-to-success* weight could be gamed (fast-but-reckless), so the anchor (§3) keeps it near demonstrated behavior. |
| **Self-imitation feedback loop** (mode collapse onto current successes) | Keep the demo mix (anchors to human-demo diversity); cap outer iterations; watch per-task spread, not just the mean. |
| **Compute blow-up** | Sim-bound rollout collection is the wall-clock sink (§5): batch-parallel envs, cheap proxy eval between iterations, full n=500 only at the end. |
| **Early-stop on eval noise** (selecting on n=100 noise) | Use a **held-out eval seed/episode set** for the stop decision, distinct from the final-report set; report the *trajectory* across iterations, not just the best point. |
| **RL ≤ SFT (no gain)** | Not a project failure; it's the finding (§8, §10), and the methodology + per-task/OOD analysis is the contribution (the project's risk framing). |

---

## 10. Honesty / rigor standards

The headline is the **SFT→RL delta on the 100-ep protocol**. To make it a *real* claim and not
noise:

- **n=500+ eval episodes** (50/task × 10) for the final claim. CI shrinks from ≈±6 (naive, n=100),
  honestly ±7–10 with task clustering, to **≈±3 pts at n=500**. The SFT +4 sits *inside* the
  n=100 interval; an RL gain must clear the **tighter** n=500 interval to count. Use n=100 (or
  cheaper) only as the **per-iteration proxy**; n=500 is the **claim**.
- **Multiple seeds (≥3)**: vary both the RL training seed (rollout sampling + fine-tune) *and*
  the eval seed; report **mean ± std across seeds**. A single-seed RL delta is not a result. This
  is the single biggest rigor lever (`FAQ.md` limitation #2).
- **State the confounds up front:** RL starts from a **16k under-trained base** (§7); single suite
  (mitigated by the `libero_10` probe, §8); sim-only, outcome-only reward; few outer iterations;
  early-stop selection (mitigated, §9); and the delta is vs **our 89% SFT**, not the official
  baseline.
- **Volunteer the degenerate case:** under binary reward, "reward-weighted" BC **=** filtered BC
  (§2); the time-to-success weight is a mild, free signal that makes the weighting non-trivial, so
  don't oversell it.
- **Volunteer the method's altitude (Q10):** this is **offline/advantage-weighted "RL-lite,"** not
  on-policy policy gradient. It is legitimate RL (signal = environment reward on the policy's own
  rollouts) that **dodges** the intractable-likelihood problem; full DPPO/πRL on-policy is the
  acknowledged stretch, not what's claimed here.

Match `FAQ.md`: for every claim, finish "…and the cost/confound is ___"; answer
"is it controlled?" with the **design** first (the matched comparison: same `π_SFT` base, identical
protocol, RL-on vs RL-off), then the statistics (episodes, seeds); never conflate **rollout/train
episodes** with **eval episodes**.

---

## What "done" looks like (Phase 4 deliverables)

1. A validated rollout→filter→fine-tune loop (smoke-gated, §6) with Drive checkpointing.
2. An RL checkpoint, pinned `π_SFT` reference, and the **SFT→RL delta** reported **per-task on
   Goal + the `libero_10` probe**, at **n=500+, ≥3 seeds**, with confounds stated.
3. The one anchor-strength sweep (§4) and a failure-mode look at rollout videos (§9).
4. `results/rl_libero_goal.json` (same schema as `sft_libero_goal.json`) + a writeup section whose
   framing, including a flat/negative result if that's what happens, matches
   `FAQ.md`.

## Budget estimate (Pro+ units; calibrate in the smoke run)

| Item | Hardware | Wall time |
|---|---|---|
| Smoke (tasks 0, 9; 1 tiny iteration + gates) | L4 | ~1–2 h |
| Full loop (3–5 iterations, collect+train+proxy-eval) | L4 | ~17–30 h |
| Anchor-strength sweep | L4 | + a few × ~3 h |
| Final claim eval (n=500, ×3 seeds) | L4 | ~33 h (splittable) |

Rollout collection is the dominant line; RL costs more sim-hours than SFT by design (§5). Fits
the $200–500 Pro+ budget but is the project's largest compute item.

## Implementation note

The RL checkpoint path mirrors the SFT layout (`<Drive>/VLATune/rl/<run>/checkpoints/...`), and
the SAME `rename_map` (image→camera1, image2→camera2) is required at eval (§6). The deterministic
list of the 428 goal episode indices the SFT trained on (`dataset_goal_episodes.json`) lives on
Drive and should be committed alongside the results for full reproducibility.
