"""Full test of the integrated sprint module with AI optimization and sensitivity analysis."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

from ordinis.engines.sprint import (
    SprintConfig,
    generate_all_visualizations,
    run_sprint,
)

# Full test with more symbols
config = SprintConfig(
    symbols=["SPY", "QQQ", "IWM", "TLT", "GLD"],
    start_date="2019-01-01",
    end_date="2024-01-01",
    use_gpu=True,
    use_ai=True,  # Enable AI optimization
    walk_forward=True,
    max_ai_iterations=3,
)

print("=" * 70)
print("ORDINIS SPRINT MODULE - FULL INTEGRATION TEST")
print("=" * 70)

runner = run_sprint(config)

# =====================================================================
# AI OPTIMIZATION OF UNDERPERFORMERS
# =====================================================================
print(f"\n{'=' * 70}")
print("AI OPTIMIZATION OF UNDERPERFORMING STRATEGIES")
print("=" * 70)

# Identify and optimize strategies with Sharpe < 1.0
optimized = runner.optimize_underperformers(sharpe_threshold=1.0, max_iterations=3)

if optimized:
    print(f"\n✓ Successfully optimized {len(optimized)} strategies:")
    for name, result in optimized.items():
        print(f"  - {name}: Sharpe = {result.sharpe_ratio:.2f}")
else:
    print("\n→ No strategies required optimization or AI unavailable")

# =====================================================================
# PARAMETER SENSITIVITY ANALYSIS
# =====================================================================
print(f"\n{'=' * 70}")
print("PARAMETER SENSITIVITY ANALYSIS")
print("=" * 70)

# Run sensitivity analysis on top performers
sensitivity_results = {}

# GARCH - most important parameters
print("\n[GARCH Breakout]")
for param in ["breakout_threshold", "garch_lookback"]:
    df = runner.run_parameter_sensitivity("garch_breakout", param, n_samples=5)
    if not df.empty:
        sensitivity_results[f"garch_{param}"] = df

# MTF - alignment threshold sensitivity
print("\n[MTF Momentum]")
for param in ["alignment_threshold", "long_period"]:
    df = runner.run_parameter_sensitivity("mtf_momentum", param, n_samples=5)
    if not df.empty:
        sensitivity_results[f"mtf_{param}"] = df

# EVT - threshold and holding period
print("\n[EVT Tail Risk]")
for param in ["threshold_percentile", "holding_period"]:
    df = runner.run_parameter_sensitivity("evt_tail", param, n_samples=5)
    if not df.empty:
        sensitivity_results[f"evt_{param}"] = df

# Print sensitivity summary
if sensitivity_results:
    print(f"\n{'─' * 50}")
    print("Sensitivity Analysis Summary:")
    print(f"{'─' * 50}")
    for key, df in sensitivity_results.items():
        best_idx = df["sharpe_ratio"].idxmax()
        worst_idx = df["sharpe_ratio"].idxmin()
        param_name = key.split("_", 1)[1]
        print(f"  {key}:")
        print(
            f"    Best:  {param_name}={df.loc[best_idx, param_name]:.3f} -> Sharpe={df.loc[best_idx, 'sharpe_ratio']:.2f}"
        )
        print(
            f"    Worst: {param_name}={df.loc[worst_idx, param_name]:.3f} -> Sharpe={df.loc[worst_idx, 'sharpe_ratio']:.2f}"
        )

# =====================================================================
# FINAL RESULTS
# =====================================================================
print(f"\n{'=' * 70}")
print("FINAL RESULTS AFTER OPTIMIZATION")
print("=" * 70)

runner.print_summary()

# =====================================================================
# VISUALIZATIONS
# =====================================================================
print(f"\n{'=' * 70}")
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Generate all visualizations
viz_paths = generate_all_visualizations(
    runner.results,
    output_dir="artifacts/sprint/viz",
)

print("\nGenerated files:")
for name, path in viz_paths.items():
    if path:
        print(f"  {name}: {path}")

print("\n" + "=" * 70)
print("INTEGRATION TEST COMPLETE")
print("=" * 70)
