#!/usr/bin/env python3
"""
Monitor comprehensive MI Ensemble backtest progress.

Displays real-time updates on optimization status, trials completed,
and best parameters found so far.

Usage:
    python scripts/optimization/monitor_backtest.py
"""

import json
import sys
import time
from pathlib import Path

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def monitor_study(study_path: Path) -> None:
    """Monitor Optuna study progress."""
    if not HAS_OPTUNA:
        print("⚠ Optuna not installed. Install with: pip install optuna")
        return
    
    storage = f"sqlite:///{study_path}"
    
    try:
        study_name = optuna.study.get_all_study_names(storage)[0]
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        trials = study.trials
        best_trial = study.best_trial
        
        print(f"\n📊 Study: {study_name}")
        print(f"   Trials: {len(trials)}")
        print(f"   Best Score: {study.best_value:.4f}")
        print(f"\n🏆 Best Parameters:")
        for param, value in best_trial.params.items():
            print(f"   {param:20s}: {value}")
        
        # Show recent trials
        if len(trials) > 0:
            print(f"\n📈 Recent Trials (last 5):")
            for trial in trials[-5:]:
                status = "✓" if trial.state == optuna.trial.TrialState.COMPLETE else "⏳"
                value = f"{trial.value:.4f}" if trial.value is not None else "N/A"
                print(f"   {status} Trial {trial.number}: {value}")
    
    except Exception as e:
        print(f"⚠ Could not load study: {e}")


def main():
    base_dir = Path("artifacts/optimization/mi_ensemble_comprehensive")
    
    if not base_dir.exists():
        print(f"⚠ Output directory not found: {base_dir}")
        print(f"   Backtest may not have started yet.")
        return
    
    # Check log file
    log_file = Path("artifacts/optimization/mi_backtest.log")
    if log_file.exists():
        print("\n📝 Recent Log Entries:")
        print("-" * 80)
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(line.rstrip())
        print("-" * 80)
    
    # Monitor timeframe studies
    print("\n\n🔍 Monitoring Optimization Studies...")
    print("=" * 80)
    
    for tf_dir in base_dir.glob("timeframe_*"):
        tf_name = tf_dir.name.replace("timeframe_", "")
        print(f"\n⏱️  Timeframe: {tf_name}")
        print("-" * 40)
        
        study_path = tf_dir / "optuna_study.db"
        if study_path.exists():
            monitor_study(study_path)
        else:
            print("   ⏳ Not started yet")
    
    # Check for results
    results_file = base_dir / "all_timeframes_results.json"
    if results_file.exists():
        print("\n\n✅ OPTIMIZATION COMPLETE!")
        print("=" * 80)
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print("\n📊 Results Summary:")
        for tf, data in results.items():
            print(f"\n   {tf:10s}: Score {data['best_value']:.4f}")
        
        best_tf = max(results.items(), key=lambda x: x[1]['best_value'])
        print(f"\n🏆 Best Overall: {best_tf[0]} ({best_tf[1]['best_value']:.4f})")
        
        report_file = base_dir / "OPTIMIZATION_REPORT.md"
        if report_file.exists():
            print(f"\n📄 Full report: {report_file}")
    else:
        print("\n\n⏳ Optimization in progress...")
        print("   Run this script again to check status")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)
