#!/bin/bash
# Quick start script for MI Ensemble comprehensive backtesting
# Usage: ./scripts/optimization/run_comprehensive_backtest.sh [mode]

set -e

# Configuration
MODE=${1:-quick}  # full, quick, or test
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/optimization/comprehensive_mi_backtest.py"

echo "=========================================="
echo "MI Ensemble Comprehensive Backtesting"
echo "=========================================="
echo ""
echo "Mode: $MODE"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Activate conda environment
if [ -f "/home/kjfle/miniconda3/etc/profile.d/conda.sh" ]; then
    source /home/kjfle/miniconda3/etc/profile.d/conda.sh
    conda activate ordinis-gpu
    echo "✓ Conda environment activated: ordinis-gpu"
else
    echo "⚠ Conda not found, using system Python"
fi

# Install required packages if missing
echo ""
echo "Checking dependencies..."
pip install -q yfinance optuna plotly kaleido tqdm 2>/dev/null || true

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/artifacts/optimization/mi_ensemble_comprehensive"
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory: $OUTPUT_DIR"

echo ""
echo "Starting optimization..."
echo "=========================================="
echo ""

# Run the optimization
python "$SCRIPT_PATH" \
    --mode "$MODE" \
    --timeframes 1D 1W \
    --min-symbols 50 \
    --use-nvidia \
    --start-date 2020-01-01 \
    --output-dir "$OUTPUT_DIR" \
    --cache-dir "$PROJECT_ROOT/data/cache"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ OPTIMIZATION COMPLETE"
    echo ""
    echo "Results available at:"
    echo "  $OUTPUT_DIR"
    echo ""
    echo "View report:"
    echo "  cat $OUTPUT_DIR/OPTIMIZATION_REPORT.md"
    echo ""
    echo "View best parameters:"
    echo "  cat $OUTPUT_DIR/all_timeframes_results.json"
else
    echo "✗ OPTIMIZATION FAILED (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
