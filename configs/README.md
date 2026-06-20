# configs/

- **`eval_libero_goal.json`** — baseline eval parameters (policy, suite,
  control_mode, n_episodes, rename_map placeholder, expected-results range).
  Consumed by the Colab notebook / any eval wrapper. Flag *names* are documented
  in `../docs/flags.md`; the installed `lerobot-eval --help` is the source of truth.

Training has no standalone config file — SFT was driven by `lerobot-train` CLI flags
(see `docs/SFT_RUNBOOK.md` for the exact validated train + eval commands), and the RL
fine-tuning loop is configured via the `src/rl_*.py` command-line arguments.
