# NVIDIA Integration (Research & Backtesting)

## Purpose

Use NVIDIA acceleration where it matters: heavy model training, feature generation, and large backtest sweeps.

## Common Workloads

- Model training/tuning: tree/boosting/NN models; hyperparameter sweeps.
- Feature generation: vectorized TA/ML features on large universes.
- Simulation/backtests: GPU-accelerated computation of signals and metrics; parallel scenario runs.

## Architecture Notes

- Keep data local to GPU (cuDF/Arrow) to avoid PCIe thrash.
- Batch operations; avoid chatty host-device transfers.
- Prefer vectorized kernels over Python loops; use RAPIDS or PyTorch/TensorFlow backends.
- Containerize with fixed driver/CUDA/toolkit versions; record image digest.

## Minimal Example (cuDF + RAPIDS)

```python
import cudf
import cupy as cp

df = cudf.read_parquet("prices.parquet")
returns = df['close'].pct_change()
rolling_mean = returns.rolling(window=20).mean()
vol = returns.rolling(window=20).std()
zscore = (returns - rolling_mean) / vol
```

## Backtesting Acceleration

- Partition by symbol/date; run scenarios in parallel batches.
- Use GPU-accelerated stats (cuML) for factor regressions and clustering.
- Profile: ensure GPU utilization is high; watch for CPU-bound steps.

## Ops & Constraints

- Hardware: document GPU type, memory, driver/CUDA version.
- Determinism: fix seeds; note any nondeterministic ops.
- Resource limits: cap GPU memory per job; handle OOM gracefully.
- Fallback: ensure CPU path exists; feature flags to toggle GPU use.

## When Not to Use GPU

- Small datasets, low arithmetic intensity, or IO-bound tasks.
- Latency-sensitive live paths (keep live execution on CPU unless profiled otherwise).
