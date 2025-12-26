#!/usr/bin/env python3
"""
Optimize MI Ensemble strategy parameters for maximum profit.

Usage:
    python scripts/optimization/optimize_mi_ensemble.py --symbols AAPL MSFT GOOGL --trials 100
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ordinis.engines.signalcore.models.mi_ensemble_optimizer import (
    MIEnsembleOptimizer,
    OptimizationObjective,
    ParameterSpace,
    ValidationStrategy,
    run_optimization_pipeline,
)
from ordinis.data.loaders import CSVDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(symbols: list[str], data_dir: Path) -> pd.DataFrame:
    """Load historical data for symbols.
    
    Args:
        symbols: List of ticker symbols
        data_dir: Directory containing CSV files
        
    Returns:
        Combined DataFrame with multi-symbol data
    """
    dfs = []
    
    for symbol in symbols:
        csv_path = data_dir / f"{symbol}.csv"
        
        if not csv_path.exists():
            logger.warning(f"Data file not found: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df.set_index("date", inplace=True)
        df["symbol"] = symbol
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No data files found")
    
    combined = pd.concat(dfs)
    logger.info(f"Loaded {len(combined)} rows for {len(symbols)} symbols")
    
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Optimize MI Ensemble for profit maximization"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL"],
        help="Symbols to optimize on",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/historical"),
        help="Directory with historical CSV files",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/optimization/mi_ensemble"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--objective",
        choices=["profit", "sharpe", "sortino"],
        default="profit",
        help="Primary optimization objective",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of time-series CV splits",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=126,
        help="Test period size in days",
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading data for {len(args.symbols)} symbols...")
        data = load_data(args.symbols, args.data_dir)
        
        # Define objective based on user choice
        objective_metrics = {
            "profit": "total_return",
            "sharpe": "sharpe_ratio",
            "sortino": "sortino_ratio",
        }
        
        objective = OptimizationObjective(
            primary_metric=objective_metrics[args.objective],
            constraints={
                "sharpe_ratio": (">=", 1.0),
                "max_drawdown": ("<=", 0.25),
                "win_rate": (">=", 0.45),
                "profit_factor": (">=", 1.2),
            },
            penalty_weight=100.0,
        )
        
        # Setup validation
        validation = ValidationStrategy(
            n_splits=args.n_splits,
            test_size_days=args.test_days,
            gap_days=0,
        )
        
        # Initialize optimizer
        logger.info(f"Initializing optimizer (objective: {args.objective})")
        optimizer = MIEnsembleOptimizer(
            data=data,
            symbols=args.symbols,
            objective=objective,
            validation=validation,
        )
        
        # Run optimization
        logger.info(f"Starting optimization with {args.trials} trials...")
        storage_path = args.output_dir / "optuna_study.db"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        study = optimizer.optimize(
            n_trials=args.trials,
            timeout=args.timeout,
            study_name=f"mi_ensemble_{args.objective}",
            storage=f"sqlite:///{storage_path}",
        )
        
        # Save results
        logger.info("Saving results...")
        
        # Export trial data
        trials_df = study.trials_dataframe()
        trials_df.to_csv(args.output_dir / "trials.csv", index=False)
        
        # Export best parameters
        best_params_file = args.output_dir / "best_parameters.json"
        import json
        from datetime import datetime
        
        with open(best_params_file, "w") as f:
            json.dump(
                {
                    "best_params": study.best_params,
                    "best_value": study.best_value,
                    "objective": args.objective,
                    "symbols": args.symbols,
                    "n_trials": len(study.trials),
                    "optimization_date": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        optimizer.plot_optimization_history(study, save_path=args.output_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"\nObjective: {args.objective}")
        print(f"Best Score: {study.best_value:.4f}")
        print(f"\nBest Parameters:")
        for param, value in study.best_params.items():
            print(f"  {param:20s}: {value}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
