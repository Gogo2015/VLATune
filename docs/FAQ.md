# VLATune: Design Rationale & Anticipated-Questions FAQ

A defense-oriented companion to the paper: the reasoning behind each decision in the
SmolVLA × LIBERO (SFT→RL) project, and answers to the hardest questions a skeptical
reviewer might ask. Numbers here are the as-run record. Cross-check against `results/*.json`.

> **The one thing to never get wrong:** your SFT scored **89.0%** vs an **85.0%**
> baseline, **+4.0 pts**, but at **n=100** the 95% binomial CI is ~**±7–10 pts**, so
> +4 is **inside the noise**. It is *not* a claimed improvement. The SFT phase's real
> deliverables are (a) a validated end-to-end pipeline and (b) a checkpoint
> to launch RL from. The intended headline is the **Phase-4 SFT→RL delta**, plus the
> methodology itself. Lead with that, and it reads as rigor, not weakness.

---

## Key numbers cheat-sheet

| Item | Value |
|---|---|
| Model | SmolVLA, ~450M params (~100M = action expert; VLM backbone frozen in fine-tune) |
| Baseline | `HuggingFaceVLA/smolvla_libero` (official, multi-suite), **85.0%**, n=100, on L4 |
| SFT | `smolvla_base` → 428 `libero_goal` eps, **89.0%**, n=100, **16k of 20k steps** (preempted) |
| Delta | **+4.0 pts** (within ±7–10 pt noise at n=100) |
| Baseline per-task | 6,10,8,9,9,10,8,10,9,6 |
| SFT per-task | 10,9,9,8,10,10,7,10,10,6 (biggest move: task 0, +4) |
| SFT loss | 1.41 (step 200) → 0.37 (2k) → 0.102 (9.2k), monotone |
| Paper vs reproductions | paper Goal **92%**; public-codebase reproductions **~81–84%**; ours 85% (top of band) |
| Eval protocol | relative control, `n_action_steps=10`, n=100 (10 tasks × 10 ep), batch 4, `MUJOCO_GL=egl` |
| Hardware | **A100** for SFT (GPU-bound), **L4** for eval (CPU/sim-bound) |
| Stack | lerobot 0.5.1, torch 2.10.0+cu128, transformers 5.3.0, python 3.12.13 |
| Cost thesis | real VLA research on ~$20–60 (consumer GPU plus preemptible Colab Pro+) |

---

## 1. Research framing

**Problem.** Generalist robot manipulation: one policy mapping (camera image +
natural-language instruction) → motor actions, across many tasks, no per-task coding.
**Vision-Language-Action (VLA)** models are the dominant approach: a pretrained
vision-language model plus an action output, trained on robot demonstrations. Two catches:
(a) strong VLAs are *big* (OpenVLA = 7B), so they need cluster-scale compute; (b) improving a VLA
with **RL** is unsolved for the modern ones whose action head is a flow-matching/diffusion
model.

**Two frontiers (why it's interesting):**
1. **Efficiency.** SmolVLA (450M) is ~15× smaller than OpenVLA (7B) yet competitive on
   LIBERO. That collapses the whole VLA research loop onto a consumer GPU plus a few dollars
   of Colab. Making the experiment *affordable* is the point.
2. **RL on flow-matching policies.** SmolVLA emits action chunks via iterative denoising,
   not discrete tokens. Vanilla PPO needs a tractable action log-probability; flow matching
   doesn't provide one, so PPO can't be bolted on. This is live research (LeRobot flags
   VLA-RL as future work; DPPO/πRL/ReinFlow/Flow-GRPO attack it in 2025).

**One-sentence contribution.** *An end-to-end, compute-frugal pipeline for fine-tuning a
flow-matching VLA (SmolVLA) on LIBERO-Goal: a reproduced baseline, a goal-specialized SFT
checkpoint, and a tractable reward-weighted / KL-regularized RL stage that sidesteps the
intractable-likelihood problem that breaks PPO on flow-matching policies, executed and
analyzed honestly on a ~$20–60 budget.*

---

## 2. SmolVLA architecture

Two pieces bolted together (450M total):

1. **Backbone: a compact vision-language model (VLM)** (SmolVLM-family). Eyes and ears:
   it ingests camera images + text instruction + robot joint state → a rich feature summary.
   It already knows how to see and read from pretraining. SmolVLA shrinks it (uses only the
   lower ~half of layers; caps visual tokens per image) for speed.
2. **Action expert: a smaller transformer (~100M)** that does **flow-matching denoising**
   to turn those features into an **action chunk** (several future actions at once).

**Flow matching / iterative denoising (plain version).** The model does *not* pick an
action from a list. It starts from random static (noise shaped like an action) and
cleans it up over several small steps until it's a sensible action, like a blurry photo
developing into a sharp one. "Flow matching" is the recipe it learned for *which direction
to nudge* the static at each step so it reliably flows noise → good action. **Training**
learns those cleanup directions from demo examples; **running** applies them to fresh noise
(no example present at inference).

**The efficiency move.** During fine-tuning the **backbone is frozen**: only the ~100M
action expert is trained. Why that makes it cheap: training overhead (gradients + Adam
optimizer state ≈ 3–4× the param's own size, plus stored activations) lives **only on the
*trainable* params**. Freeze 350M of 450M and you delete all of that for the backbone, so
batch 64 fits on the A100. (The backbone is still *run* forward; it just isn't *trained*.)
Frozen pretrained perception also means you need surprisingly little robot data: a few
hundred episodes (you used 428) teaches the action mapping.

---

## 3. LIBERO-Goal (the benchmark)

**LIBERO** = a robot-manipulation benchmark in **simulation** (MuJoCo physics + robosuite,
rendered camera frames). Sub-suites: Spatial, Object, **Goal**, Long (`libero_10`), `libero_90`.

**Why "Goal" is Goal.** Its 10 tasks share largely the **same objects/scene** but differ in
the **goal** ("open the top drawer," "put the bowl on the plate," "turn on the stove"). Fixing
the scene and varying the instruction isolates *goal understanding*.

**Success = outcome-only, binary, per episode.** The simulator checks a **goal predicate** on
its own ground-truth state ("is the bowl on the plate?"); the episode scores **1 if satisfied
within the step limit, else 0** (timeout = fail). It does **not** care about path, competence,
or whether the "right" objects were touched: ugly-but-achieved beats graceful-but-missed.
The policy sees pixels; the judge sees true state. Reward is **sparse** (you score only
when the goal flips true; note `avg_sum_reward = avg_max_reward = success rate` in the JSONs).
**"85%" = fraction of the 100 episodes (10 tasks × 10) whose predicate came true**, a mean of
100 zero/one outcomes, which is exactly why it's a binomial proportion with that ±7–10 pt CI.

**Episode count: 428, not 500.** Textbook LIBERO-Goal = 50 demos × 10 tasks = 500, but the
processed `HuggingFaceVLA/libero` goal suite yields **428** usable episodes (some demos dropped
in processing). Quote **428**. *(Open item: confirm the exact drop reason.)*

---

## 4. Baseline (85.0%) and the eval protocol

**What it is:** the *official* `HuggingFaceVLA/smolvla_libero` checkpoint (not yours), run on
LIBERO-Goal under a fixed protocol for **85.0%, n=100**. The reference the SFT is measured against.

**The two knobs that carry meaning:**
- **`control_mode=relative`** defines what action numbers *mean*. Relative = deltas applied
  to the arm's current **pose**; absolute = target poses. Must match how the checkpoint was
  trained, or every action is interpreted in the wrong frame, giving **~0% on all tasks**
  ("failure mode #2").
- **`n_action_steps=10`** executes 10 actions of the predicted chunk before re-planning.
  Mainly the **speed knob**: at `=1` it re-infers every sim step (~400 s/ep, an ~11 h run that
  invites preemption); at `=10` it's ~90 s/ep. Behavioral cost if set too high: it acts "blind" longer
  between re-plans.

**Plumbing knobs:** `n=100` (sample size, which sets the CI); `batch_size=4` (parallel envs, throughput
only); `max_parallel_tasks=1` (official reproducible setting); `MUJOCO_GL=egl` (headless GPU
render, since the policy needs rendered frames even at eval); **no `rename_map`** (the official
checkpoint natively uses LIBERO's `image`/`image2` keys).

**85 vs 92 vs 78–84 (the most-challenged number).** Paper Goal = **92%**; independent
reproductions in the public LeRobot codebase = **~81–84%** (documented gap, multiple open
GitHub issues); you got **85%**, top of the reproduction band. 92% is the aspirational cite;
~84% is the expectation. Don't chase 92%: it's not reproducible in the public codebase, and
the baseline only needs to be a fair fixed reference.

**Sanity check on the protocol:** **no task scored 0%** (worst = 60%). The killer config bugs
(wrong control mode, wrong camera keys) crater *every* task uniformly, so a healthy task rules
them out. (Note: this proves the *harness is correct*, NOT that the robot "understands.")

---

## 5. SFT (89.0%): recipe, fixes, honest reading

**SFT = Supervised Fine-Tuning** = **behavior cloning**: continue-train a pretrained model on
(images + instruction + state → demonstrated action) pairs, minimizing the gap to the demo
action. *Supervised* = you have the target action; *fine-tuning* = adjust a pretrained model.
Contrast with **RL** (Phase 4): no target action, so it learns from a reward on its own rollouts;
in principle it can exceed the demos, whereas SFT is capped by them.

**Recipe:** start from `lerobot/smolvla_base` (the *generic* base, **not** the official LIBERO
checkpoint), 428 goal episodes, a 20k-step target (16k actual, preempted), batch 64 on A100,
`save_freq=2000` to Drive, VLM backbone frozen (defaults). Loss 1.41 → 0.10 is the action
expert learning to reproduce demonstrated goal-task actions.

**The 5 non-obvious fixes (engineering story):**
1. **`--policy.push_to_hub=false`.** `lerobot-train` defaults to wanting a Hub upload and
   aborts demanding a `repo_id`; you keep checkpoints on Drive.
2. **Broken dataset index.** `HuggingFaceVLA/libero` metadata says ep 379 → file-025,
   but file-025 actually holds eps 61–63: the **file-level map is wrong on every revision.**
   Selectively downloading "just goal episodes" fetches the wrong files, so the filter matches **zero
   rows**. **Fix: download the full 35 GB once**, then filter on the data's **correct in-row
   `episode_index` column** for the right 428 episodes. (Bad file map vs correct row-level index.)
3. **Camera-key contract (train).** `smolvla_base` expects `camera1/2/3`; LIBERO provides
   `image`/`image2`. The base ships input_features populated, so `make_policy` keeps expecting
   `camera1/2/3`. **Fix: `--rename_map` image→camera1, image2→camera2** (3rd camera auto-skipped).
4. **Eval needs the SAME `rename_map`.** Your SFT checkpoint *inherits* the `camera1/2/3` keys,
   so eval must rename too, unlike the official baseline (natively `image`/`image2`).
5. **Pre-create `~/.libero/config.yaml`.** LIBERO's first import calls `input()` for a dataset
   path; in an eval subprocess with no stdin that's `EOFError`. Write the config first.

**The experimental contrast and its confounds.** Baseline = base + *all* suites, giving 85% on Goal;
SFT = base + *goal-only*, giving 89% on Goal. So it's **generalist-on-Goal vs specialist-on-Goal**.
But +4 is **not a controlled measurement of specialization**, because beyond goal-vs-all-suites:
- the **baseline is a black box** (unknown recipe/steps/data/hyperparameters);
- **data *scale* differs** (all-suites ≫ 428), tangled with composition;
- your checkpoint is **under-trained** (16k vs 20k).

**The clean experiment:** from the same base, train all-suites vs goal-only under an *identical*
recipe (same steps, schedule, seed), eval both on Goal. **Then** add more eval episodes (tighter
CI) and multiple seeds. Design first, statistics second.

---

## 6. The engineering story (as contribution)

**Hardware matched to bottleneck:** **A100 for SFT** (GPU-bound: the bottleneck is the
forward/backward matrix math, so a fast GPU pays off). **L4 for eval** (CPU/sim-bound: the
bottleneck is MuJoCo physics + rendering on the CPU; a bigger GPU just idles, burning ~4–5×
the units for no speedup). Match the hardware to wherever the bottleneck actually is.

**Preemption-resilient checkpointing.** Colab wipes the VM on disconnect and preempts the A100
on demand. So save the model **+ `training_state` (optimizer/scheduler/RNG)** to Drive **every
2000 steps**. That bounds worst-case loss to ~2k steps and makes resume a continuation
rather than a cold restart. Background execution lets a long run survive a
disconnected monitoring session.

**The 16k preemption and the decision.** A100 reclaimed at **step 16k of 20k**; `last` +
`training_state` intact (resumable). You chose to **eval 16k** rather than burn a fresh A100 plus a
35 GB re-download for 4k steps with already-flat loss. **The cost you pay:** no clean standard
20k number; the +4 gains a further "16k vs fully-trained" confound; you can argue but **can't
prove** the last 4k wouldn't move it. (And it forces the Phase-4 question: resume to 20k first,
or launch RL from 16k?)

**Why it's contribution, not plumbing.** The thesis is *affordable VLA research.* Preemptible
consumer cloud **is** what "affordable" means, so surviving preemption, matching hardware to
bottleneck, and staying reproducible through flaky tooling is the engineering that makes the
affordability claim credible. Reframe "you just ran notebooks" as: the constraint forced a
disciplined, cost-aware, checkpoint/resume workflow, and I can account for every compute
decision and its cost.

---

## 7. Limitations & threats to validity

**The discipline: volunteer these before you're asked. State it, size it, fix it.**

| # | Limitation | Size / why it bites | What you'd do |
|---|---|---|---|
| 1 | **n=100 noise** | ±7–10 pt CI makes +4 indistinguishable from 0 | n=500+ **eval** episodes |
| 2 | **Single seed** (train & eval) | no run-to-run variance estimate | 3+ seeds |
| 3 | **16k vs 20k** | under-trained headline checkpoint | resume to 20k |
| 4 | **Black-box baseline** | uncontrolled comparison | matched-recipe ablation |
| 5 | **Sim-only + outcome-only success** | no real-robot evidence; ignores safety/smoothness/efficiency | real-robot eval; trajectory metrics |
| 6 | **Goal-suite only** | no generalization/OOD evidence | eval other suites / held-out goals |
| 7 | **Imperfectly-reproducible baseline** | built on the 92%-vs-84% gap | document gap; report the band |

---

## The 10 hardest questions (with answers)

**Q1. "Your SFT got +4 over baseline, is that your result?"**
No, it's not statistically distinguishable from zero (±7–10 pt CI at n=100). Not a regression
either, and directionally consistent with specialization, but I don't claim it. The SFT
deliverables are a validated pipeline plus an RL-launch checkpoint; to claim the gain I'd
run n=500+ and multiple seeds.

**Q2. "Is +4 a controlled measurement of 'goal specialization'?"**
No. The baseline is a black-box checkpoint (unknown recipe/data/hyperparameters), uses all
suites vs my goal-only, and my model is under-trained at 16k. The clean ablation: same base,
identical recipe/steps/seed, all-suites vs goal-only, both eval'd on Goal. I didn't run it, so
+4 is suggestive, not controlled.

**Q3. "The paper reports 92% on Goal; you got 85%. Did you eval wrong?"**
92% isn't reproducible in the public LeRobot codebase: documented gap, multiple open issues;
reproductions land ~81–84%. My 85% is top-of-band, no task scored 0% (protocol is correct), and
the gap is upstream (released checkpoint/codebase vs the paper's internal setup), not my eval.

**Q4. "Why not just PPO to RL-finetune SmolVLA?"**
PPO needs a tractable action log-probability for its objective. SmolVLA's flow-matching expert
builds actions by denoising random noise over many steps, so the action's marginal likelihood
requires integrating over all noise/paths, which is intractable. With no log-prob, PPO's ratio term can't
be formed. Hence my Phase-4 reward-weighted/KL approach, which needs no such likelihood.

**Q5. "You evaluated 16k, not your planned 20k. Doesn't that invalidate it?"**
It weakens precision, not validity. Loss had flattened (~0.10) and the marginal 4k cost a fresh
A100 plus a 35 GB re-download. The cost: no clean 20k number, a further "16k vs full" confound on +4,
and I can't *prove* the last 4k wouldn't move it. I caveat it as a 16k preempted checkpoint.

**Q6. "It's all in simulation, why believe it transfers to a real robot?"**
I don't claim real-robot transfer; it's a sim-only study, and LIBERO success is outcome-only
(ignores safety/smoothness/efficiency). The contribution is a compute-efficient SFT→RL
methodology plus analysis on a standard sim benchmark; real-robot validation is explicit
future work.

**Q7. "What do you contribute beyond the SmolVLA paper / LeRobot?"**
(1) An end-to-end, reproducible, *affordable* pipeline (consumer GPU + preemptible Colab) with
the real gotchas solved and documented (broken dataset index, camera-key contract,
preemption-resilient checkpointing). (2) Engagement with the open RL-on-flow-matching frontier
via a tractable reward-weighted/KL method, the part LeRobot flags as future work.

**Q8. "If SFT already hits 89% on Goal, where's the headroom for RL? Is Goal even a good RL
testbed?"**
Fair, and I tested it **three ways**, ending in a *general* null. (1) On **saturated Goal**,
RL-lite was a within-noise *redistribution* (π_0 89.0 → π_1 91.0 → π_2 89.4; compounding falsified,
since iter-2 erased iter-1's gains). (2) On unsaturated **LIBERO-Long (`libero_10`)**, my goal-only policy
scored **~0% zero-shot**, so filtered BC had zero successful rollouts and **starved**: the §8
sparse-reward risk realized (the filter starves exactly where there's the most headroom). (3) So I
applied the §8 remedy, **demo-seeded `libero_10`**: SFT'd a `libero_10` specialist (π_0^L10) that
reaches **~44–50%**, removing the starvation, then ran the *same* RL loop on genuine headroom. **It
still produced no durable gain:** π_0 44.0 → π_1 47.5 → π_2 42.5 (paired n=20/task, same seeds). The
iter-1 +3.5 (a marginal-task rescue, concentrated on tasks π_0 had seeded with a few successes) was
**noise**: iter-2 reversed the very tasks that drove it (task3 60→75→55, task7 25→40→25), the same
up-then-down signature as Goal. So across **both** a saturated and an unsaturated-but-bootstrappable
regime, **RL-lite (filtered/reward-weighted BC + vector-field anchor) does not durably move this
flow-matching VLA; it redistributes within noise.** That's a *stronger, more general* null than Goal
alone, and it's the finding, not something to hide: I've mapped where this method can and can't move
the policy, including the marginal-task rescue-then-reversal mechanism and the pure-starvation floor
(task8: 0 seed successes, stays 0 across all iters). Claim-grade rigor would be n=500 / ≥3 seeds (tens
of hours on `libero_10`'s slow render); the up-then-down trajectory and per-task reversal already read as
noise at the same standard that falsified Goal. I report per-task throughout. (Results:
`results/rl_libero10_rl.json`; SFT anchor: `results/sft_libero10.json`; zero-shot probe:
`results/rl_libero10_probe.json`.)

**Q9. "How do you know your eval protocol is correct?"**
No task scored 0% (the key-mismatch / control-mode failure modes crater every task uniformly),
and my baseline lands in the documented reproduction band. I also set it explicitly: relative
control to match training, the camera `rename_map` for my checkpoint's keys, `n_action_steps=10`,
and pre-initialized the LIBERO config to avoid subprocess EOF.

**Q10. "Reward-weighted/KL BC, isn't that just more SFT? Where's the 'reinforcement'?"**
The training signal is the **environment's reward from the policy's own rollouts**, not human
demos: the model generates trajectories, they're scored by task success, and it's fine-tuned
to up-weight high-return behavior with a KL anchor to the SFT policy to prevent collapse. It's
not policy-gradient PPO, but it's a legitimate (advantage-weighted/offline) RL family that
dodges the intractable-likelihood problem; full on-policy DPPO/πRL is the stretch.

**Bonus, Q11. "Why SmolVLA and not OpenVLA-7B or π0?"**
Compute and the research question. SmolVLA (450M, frozen backbone, ~100M trainable) trains on one
consumer-class GPU for ~$20–60; 7B models need a cluster. It's competitive on LIBERO, and its
flow-matching action expert is exactly where RL is open: affordable *and* on-frontier.

**Bonus, Q12. "What's the reward, and why does sparse reward matter for your RL plan?"**
Sparse binary task success (1 at goal, else 0). Sparse reward is hard for RL (little signal),
which is *why* the de-risked plan is reward-weighted filtering of successful rollouts plus a KL
anchor, rather than from-scratch policy gradient.

---

## A note on rigor

For every claim, finish the sentence "…and the cost/confound is ___." When asked "is it
controlled?", answer with the **design** (the matched comparison) first, then the statistics
(episodes, seeds). And never conflate **training episodes** (what the policy learns from) with
**eval episodes** (how precisely it was measured); they answer different questions.
