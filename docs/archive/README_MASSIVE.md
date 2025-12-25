# Massive Scale Optimization with GPU

This module provides a high-performance optimization pipeline for the ATR-RSI strategy, designed to handle large datasets ("Massive" flat files) using GPU acceleration.

## Features

- **Batch Ingestion**: Downloads data from S3-compatible storage (e.g., Polygon/Massive flat files) in configurable batches.
- **GPU Acceleration**: Uses `cudf` for fast CSV loading and dataframe operations, and `numba` (JIT) for high-speed backtesting logic.
- **Bayesian Optimization**: Utilizes Optuna to intelligently search the parameter space, following the mathematical foundations in `advanced-optimization.md`.
- **Edge Case Testing**: Parameter search spaces are configured to cover extreme values ("edge cases") to ensure robustness.
- **Resource Management**: Automatically cleans up downloaded files after processing to manage disk usage.

## Prerequisites

- NVIDIA GPU with CUDA drivers.
- Conda environment with `cudf`, `cupy`, `numba`, `boto3`, `optuna`.

## Usage

Run the optimization script:

```bash
python scripts/optimization/run_massive_gpu_optimization.py --days 30 --trials 100
```

### Configuration

- **S3 Bucket**: Set `MASSIVE_S3_BUCKET` env var (default: `flatfiles`).
- **S3 Prefix**: Set `MASSIVE_S3_PREFIX` env var (default: `us_stocks_sip/minute_aggs_v1`).
- **AWS Credentials**: Ensure standard AWS credentials (or `~/.aws/credentials`) are set if the bucket is private.

## Visualization

After running an optimization, generate interactive plots (HTML) to analyze parameter importance and convergence:

```bash
python scripts/optimization/visualize_results.py
```

This will produce the following in `artifacts/massive_optimization/`:
- `viz_optimization_history.html`: Objective value vs. Trial.
- `viz_parallel_coordinates.html`: High-dimensional view of parameter interactions.
- `viz_slice_plot.html`: Individual parameter sensitivity.
- `viz_contour_top2.html`: Heatmap of the two most important parameters.

## Architecture

1.  **Data Loader**: Fetches minute-aggregate CSVs from S3.
2.  **GPU Pipeline**: Loads CSVs into GPU memory using `cudf`.
3.  **Optimizer**: 
    -   Calculates indicators (RSI, ATR) on-the-fly using optimized Numba kernels.
    -   Simulates trading logic for thousands of iterations.
    -   Uses TPE (Tree-structured Parzen Estimator) to converge on optimal parameters.
4.  **Reporting**: Saves the best parameter set to `artifacts/massive_optimization/`.

## Mathematical Foundations

This implementation aligns with the **Bayesian Optimization** techniques described in the knowledge base, specifically using Gaussian Processes (via TPE proxy) to optimize non-convex, noisy objective functions (trading strategies) efficiently.

The optimizer maximizes **Sortino Ratio** (to penalize downside volatility) while tracking **Max Drawdown**, **Sharpe Ratio**, and **Total Return** for comprehensive performance evaluation.
