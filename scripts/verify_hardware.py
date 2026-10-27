#!/usr/bin/env python3
"""Hardware and compute profiling for ML pipeline."""

import os
import platform
import sys
import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def main():
    print("=" * 60)
    print("Enveda CASMI 2026 - Hardware & Execution Environment Profile")
    print("=" * 60)
    print(f"OS Platform        : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version     : {sys.version.split()[0]}")
    cpu_count = os.cpu_count()
    print(f"CPU Cores Available: {cpu_count}")

    if _HAS_PSUTIL:
        mem = psutil.virtual_memory()
        print(f"Total System Memory: {mem.total / (1024**3):.2f} GB (Available: {mem.available / (1024**3):.2f} GB)")
    else:
        print("Total System Memory: Profiled via OS")

    # PyTorch acceleration check
    print("-" * 60)
    print(f"PyTorch Version    : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if cuda_avail:
        print(f"CUDA Available     : True ({torch.cuda.get_device_name(0)})")
        print(f"CUDA Device Count  : {torch.cuda.device_count()}")
    elif mps_avail:
        print("Apple Silicon MPS  : Available (Metal Performance Shaders)")
    else:
        print("Hardware Engine    : Standard CPU")

    print("=" * 60)


if __name__ == "__main__":
    main()
