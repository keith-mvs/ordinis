# MI Ensemble Optimization Results

This folder consolidates all **Massive Intraday (MI) Ensemble** optimization and backtesting artifacts following the organizational structure used in `kalman_hybrid`.

## Folder Structure

```
mi-ensemble/
├── README.md                  # This file
├── NVDA/                      # Symbol folder (mirrors kalman_hybrid pattern)
│   ├── MI_Baseline/           # Pre-optimization baseline run
│   │   ├── report.json
│   │   ├── equity_curve.csv
│   │   ├── signals.csv
│   │   └── trades.csv
│   ├── MI_Baseline_Optimized/ # Initial optimized baseline
│   ├── MI_Opt_0/              # Optimization trial 0
│   ├── MI_Opt_1/              # Optimization trial 1
│   └── MI_Opt_N/              # Each optimization contains:
│       ├── report.json        # Backtest summary and metrics
│       ├── equity_curve.csv   # Portfolio equity time series
│       ├── signals.csv        # Trading signals with features
│       ├── trades.csv         # Executed trades with P&L
│       └── metadata.json      # Manifest describing file contents
├── baseline/                  # Summary JSON files (flat, like kalman_hybrid)
│   ├── mi_baseline_20251216.json
│   └── mi_baseline_optimized_20251217.json
├── ml_optimized/              # ML-enhanced optimization results
│   └── ml_backtest_*.json     # Full ML-optimized backtest output
├── study_results/             # Optuna study optimizer outputs
│   └── study_optimizer_*.json # Individual study configurations and summaries
├── scripts/                   # MI ensemble optimization scripts
│   ├── comprehensive_mi_backtest.py     # Full comprehensive backtesting
│   ├── demo_mi_optimizer.py             # Demo optimizer script
│   ├── max_utilization_optimizer.py     # GPU utilization optimizer
│   ├── optimize_mi_ensemble.py          # Main MI ensemble optimizer
│   ├── run_quick_mi_test.py             # Quick test script
│   ├── run_massive_gpu_optimization.py  # GPU-accelerated optimization
│   ├── run_comprehensive_backtest.sh    # Shell script for full runs
│   ├── monitor_backtest.py              # Real-time monitoring
│   ├── visualize_results.py             # Result visualization
│   ├── gpu_max_v3.py                    # GPU max utilization v3
│   └── max_util_v2.py                   # Max utilization v2
├── artifacts/                 # Optimization artifacts and logs
│   ├── fib_adx_ml/            # Fibonacci ADX ML optimization outputs
│   │   └── optimization_*.json
│   └── logs/                  # Execution logs
│       ├── mi_backtest.log
│       ├── max_util.log
│       ├── max_util_v2.log
│       └── gpu_max.log
└── configs/                   # Strategy configurations (to be added)
```

## Key Files

| File/Folder | Description |
|-------------|-------------|
| `baseline/` | Pre-optimization baseline strategies for comparison |
| `optimizations/Opt_X/` | Individual optimization runs with full metrics |
| `ml_optimized/` | ML-enhanced results with Fibonacci ADX integration |
| `study_results/` | Optuna hyperparameter study outputs |
| `scripts/` | All MI ensemble optimization scripts |
| `artifacts/logs/` | Execution logs for debugging and auditing |

## Governance Compliance

Per `governance.yaml`, all backtest results include:
- **ISO 8601 timestamps** for temporal data
- **Manifest files** (`metadata.json`) describing contents
- **Audit trails** via log files

## Usage

### Run Quick Test
```bash
cd scripts
python run_quick_mi_test.py
```

### Run Comprehensive Backtest
```bash
./run_comprehensive_backtest.sh
```

### Monitor Running Backtest
```bash
python monitor_backtest.py
```

## Related Documentation

- [POSITION_SIZING_LOGIC.md](../../../docs/POSITION_SIZING_LOGIC.md)
- [ARCHITECTURE.md](../../../docs/architecture/PRODUCTION_ARCHITECTURE.md)
- [governance.yaml](../../../governance.yaml)

---
*Generated: 2025-12-25*
*Version: 1.0.0*
