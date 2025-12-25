GPU Backtesting Setup Guide

This guide explains how to enable GPU acceleration for the optimizer and `GPUBacktestEngine`.

1) Preferred: install CuPy matching your CUDA version
- For conda environments (recommended):
  - Find matching cupy package for your CUDA version (e.g. CUDA 12.1/13.0 may have cupy-cuda121 or cupy-cuda130 builds on conda-forge).
  - Example (conda-forge):
      conda install -c conda-forge cupy
  - Or use pip when a wheel exists:
      pip install cupy-cudaXX  # replace XX with cudaNN (e.g. cupy-cuda113)

2) Torch fallback (works when PyTorch is installed with CUDA support)
- If CuPy is not available but PyTorch has CUDA, the optimizer will use a PyTorch fallback for core reductions.
- Install PyTorch with CUDA via the official instructions: https://pytorch.org/get-started/locally/
  e.g. (conda):
      conda install pytorch pytorch-cuda -c pytorch -c nvidia

3) Verifying GPU in the optimizer
- Run a simple smoke test:
      python -c "from scripts.strategy_optimizer import GPUBacktestEngine; print(GPUBacktestEngine(use_gpu=True).use_gpu)"
- The log will indicate whether CuPy or PyTorch was detected and used.

4) Troubleshooting
- "No matching distribution found for cupy-cudaXXX": prefer conda-forge binaries or use the PyTorch fallback.
- Ensure your NVIDIA drivers and CUDA runtime match the intended package builds.
- Run `nvidia-smi` to verify drivers and device health.

If you'd like, I can attempt to auto-install the correct cupy package for your active conda env (I will autodetect CUDA version and choose a matching package); confirm and I'll proceed.