# Pinned versions

> **Auto-generated section below is filled by `scripts/setup_env.sh`** (it appends
> the actual resolved versions on first install). The reproducible lock is
> `docs/pip_freeze.txt` (full `pip freeze`), generated at the same time.

## Targets (what we intend to install)

| Component | Target | Why |
|---|---|---|
| OS | Linux (WSL2 Ubuntu) or Colab Linux | LIBERO requires `sys_platform == 'linux'`. |
| Python | 3.12+ | Required by LeRobot v0.5.x. |
| LeRobot | 0.5.x | v0.5 uses Transformers v5 + `--policy.path` flag style. Do NOT upgrade in-place from 0.4.x. |
| Transformers | v5.x | Pulled in by lerobot 0.5; the 0.4→0.5 jump breaks on this. |
| PyTorch | torchcodec-compatible build w/ CUDA | Video decode = TorchCodec + ffmpeg. |
| ffmpeg | conda-forge | Install **before** pip-installing lerobot. |
| extras | `[smolvla,libero]` | smolvla policy + libero env. |

## Observed versions (filled by setup_env.sh)

<!-- VERSIONS_AUTOFILL_START -->
_Not yet captured for the lerobot env. Run `bash scripts/setup_env.sh` inside the WSL2 conda env._

**Reference data point (native Windows, 2026-06-09)** confirms the GPU/driver,
not the target lerobot env:
```
gpu: NVIDIA GeForce RTX 4060 | VRAM 8.0 GiB | capability 8.9
torch: 2.11.0+cu126 | torch.version.cuda: 12.6 | cuda.is_available: True
```
<!-- VERSIONS_AUTOFILL_END -->

## How these were captured

`setup_env.sh` runs, then executes:
```bash
python - <<'PY'
import torch, transformers, lerobot, platform
print("python", platform.python_version())
print("lerobot", lerobot.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
```
and `pip freeze > docs/pip_freeze.txt`.
