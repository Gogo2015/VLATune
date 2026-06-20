#!/usr/bin/env python
"""
Regenerate the paper figures from the result JSONs.

Reads results/*.json (the as-run data) and writes vector PDF (for LaTeX) + PNG
(for the README) into results/figures/. Every number plotted is pulled from the
JSON files, so the figures stay in sync with the data.

    python src/make_figures.py            # writes results/figures/*.{pdf,png}

Figures:
  figure1_trajectories  -- overall success across RL iterations (pi_0->pi_1->pi_2),
                           Goal and libero_10: the up-then-down "compounding falsified" signature.
  figure2_pertask       -- per-task success for pi_0/pi_1/pi_2 on both suites:
                           the rescue-then-reversal mechanism + the libero_10 t8 starvation floor.
  figure3_sft           -- the SFT contribution: Goal baseline->SFT per task, and the
                           libero_10 demo-seed lift off the 0% zero-shot floor.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")

# Consistent colors across all figures.
C = {"pi0": "#4C72B0", "pi1": "#DD8452", "pi2": "#55A868",
     "baseline": "#8C8C8C", "sft": "#C44E52"}
TASKS = list(range(10))

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
})


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def pertask(d, key):
    """Return a length-10 list from a {"0":v,...,"9":v} per-task dict."""
    return [d[key][str(t)] for t in TASKS]


def save(fig, stem):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


def fig1_trajectories(goal, l10):
    g = goal["results"]["overall_pc_success"]            # {"pi_0":89.0,...}
    gy = [g["pi_0"], g["pi_1"], g["pi_2"]]
    r = l10["results_paired_n20"]
    ly = [r["pi0_l10_iter0"]["overall"], r["pi1_iter1"]["overall"], r["pi2_iter2"]["overall"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for ax, ys, title, n in [
        (axes[0], gy, "LIBERO-Goal (saturated, in-distribution)", "paired n=500"),
        (axes[1], ly, "libero_10 (unsaturated, demo-seeded)", "paired n=20/task"),
    ]:
        x = [0, 1, 2]
        ax.plot(x, ys, "-o", color=C["pi1"], lw=2, ms=7, zorder=3)
        ax.axhline(ys[0], color=C["baseline"], ls="--", lw=1, zorder=1)
        for xi, yi in zip(x, ys):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([r"$\pi_0$ (SFT)", r"$\pi_1$", r"$\pi_2$"])
        ax.set_title(title)
        ax.set_xlim(-0.3, 2.3)
        pad = max(2.0, (max(ys) - min(ys)))
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_xlabel(f"RL iteration  ({n})")
    axes[0].set_ylabel("overall success (%)")
    fig.tight_layout()
    save(fig, "figure1_trajectories")


def _grouped_pertask(ax, series, labels, colors, title, ylabel=None):
    import numpy as np
    x = np.arange(len(TASKS))
    w = 0.26
    for i, (ys, lab, col) in enumerate(zip(series, labels, colors)):
        ax.bar(x + (i - 1) * w, ys, w, label=lab, color=col, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"t{t}" for t in TASKS])
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.set_xlabel("task")
    if ylabel:
        ax.set_ylabel(ylabel)


def fig2_pertask(goal, l10):
    gp = goal["results"]["per_task_pc_success"]
    g0, g1, g2 = pertask(gp, "pi_0"), pertask(gp, "pi_1"), pertask(gp, "pi_2")
    r = l10["results_paired_n20"]
    l0 = pertask(r["pi0_l10_iter0"], "per_task")
    l1 = pertask(r["pi1_iter1"], "per_task")
    l2 = pertask(r["pi2_iter2"], "per_task")

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.0))
    labels = [r"$\pi_0$", r"$\pi_1$", r"$\pi_2$"]
    cols = [C["pi0"], C["pi1"], C["pi2"]]
    _grouped_pertask(axes[0], [g0, g1, g2], labels, cols,
                     "LIBERO-Goal per-task success (paired n=500)", "success (%)")
    _grouped_pertask(axes[1], [l0, l1, l2], labels, cols,
                     "libero_10 per-task success (paired n=20/task)", "success (%)")
    # Mark the starvation floor (libero_10 t8: 0 seed successes).
    axes[1].annotate("starved\n(0 seed successes)", (8, 14), ha="center", fontsize=7.5,
                     color=C["baseline"],
                     arrowprops=dict(arrowstyle="->", color=C["baseline"], lw=0.8),
                     xytext=(8, 45))
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, ncol=3, loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "figure2_pertask")


def fig3_sft(baseline, sft_goal, sft_l10):
    import numpy as np
    # Goal: baseline (successes/10 -> %) vs SFT (%).
    b = baseline["results"]["per_task"]
    bg = [b[str(t)]["successes"] * 100.0 / b[str(t)]["n"] for t in TASKS]
    sg_raw = sft_goal["results"]["per_task"]
    sg = [sg_raw[str(t)]["successes"] * 100.0 / sg_raw[str(t)]["n"] for t in TASKS]
    # libero_10 SFT per-task (n=10) and the ~0% zero-shot reference.
    sl = pertask(sft_l10["results"], "per_task_pc_success")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    x = np.arange(len(TASKS))
    w = 0.4
    axes[0].bar(x - w / 2, bg, w, label=f"baseline ({baseline['results']['pc_success']:.0f}%)",
                color=C["baseline"], edgecolor="white", linewidth=0.4)
    axes[0].bar(x + w / 2, sg, w, label=f"SFT ({sft_goal['results']['pc_success']:.0f}%)",
                color=C["sft"], edgecolor="white", linewidth=0.4)
    axes[0].set_title("LIBERO-Goal: official baseline vs our SFT")
    axes[0].set_ylabel("success (%)")

    axes[1].axhline(0, color=C["pi0"], ls="--", lw=1.2, label="goal-only zero-shot (~0%)")
    axes[1].bar(x, sl, 0.6, color=C["sft"], edgecolor="white", linewidth=0.4,
                label=f"demo-seed SFT ({sft_l10['results']['overall_pc_success']:.0f}%)")
    axes[1].set_title("libero_10: demo-seed lifts off the 0% floor")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"t{t}" for t in TASKS])
        ax.set_ylim(-4, 105)
        ax.set_xlabel("task")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "figure3_sft")


def main():
    goal = load("rl_libero_goal.json")
    l10 = load("rl_libero10_rl.json")
    baseline = load("baseline_libero_goal.json")
    sft_goal = load("sft_libero_goal.json")
    sft_l10 = load("sft_libero10.json")

    fig1_trajectories(goal, l10)
    fig2_pertask(goal, l10)
    fig3_sft(baseline, sft_goal, sft_l10)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
