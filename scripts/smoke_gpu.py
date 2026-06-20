#!/usr/bin/env python
"""smoke_gpu.py — confirm the GPU is visible (inside WSL2 / Colab).

PASS criteria: torch imports and torch.cuda.is_available() is True.
Prints device name + VRAM. Exits 1 on FAIL so run_smoke_tests.sh can gate.
"""
import sys


def main() -> int:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"[smoke_gpu] FAIL: could not import torch: {e}")
        return 1

    print(f"[smoke_gpu] torch {torch.__version__}  (CUDA build: {torch.version.cuda})")
    avail = torch.cuda.is_available()
    print(f"[smoke_gpu] torch.cuda.is_available(): {avail}")
    if not avail:
        print("[smoke_gpu] FAIL: CUDA not available.")
        print("            WSL2: install the NVIDIA WSL CUDA driver on Windows; "
              "check `nvidia-smi` works inside WSL2.")
        return 1

    n = torch.cuda.device_count()
    print(f"[smoke_gpu] device_count: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"[smoke_gpu]   cuda:{i} = {props.name}  | VRAM {vram_gb:.1f} GiB "
              f"| capability {props.major}.{props.minor}")
        if vram_gb < 9:
            print(f"[smoke_gpu]   note: {vram_gb:.0f}GiB is inference/dev-only "
                  "(full SmolVLA fine-tune wants ~22GB -> train on Colab/L4).")

    # Tiny real allocation to prove the runtime works, not just the query.
    try:
        x = torch.randn(1024, 1024, device="cuda")
        _ = (x @ x).sum().item()
        torch.cuda.synchronize()
        print("[smoke_gpu] matmul on cuda:0 OK")
    except Exception as e:  # noqa: BLE001
        print(f"[smoke_gpu] FAIL: cuda matmul errored: {e}")
        return 1

    print("[smoke_gpu] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
