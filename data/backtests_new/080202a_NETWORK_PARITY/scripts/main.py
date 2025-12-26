#!/usr/bin/env python3
"""
Network Parity Portfolio Optimization - Main Entry Point

A comprehensive quantitative trading system that optimizes equity portfolios
through iterative refinement using NVIDIA Nemo models.

Features:
- Network-based portfolio construction using correlation centrality
- Mean reversion signal generation with z-score triggers
- GPU-accelerated backtesting with transaction cost modeling
- NVIDIA Nemo integration for intelligent parameter optimization
- Full JSON traceability for reproducibility

Usage:
    python main.py [options]

    Options:
        --max-iterations N     Maximum optimization iterations (default: 50)
        --seed N               Random seed for reproducibility (default: 42)
        --no-nemo              Disable Nemo integration (use random perturbation)
        --quick                Run quick test with 10 iterations
        --dry-run              Test configuration without running optimization

Author: Ordinis Quantitative Research
Version: 1.0.0
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import (
    OptimizationConfig,
    OptimizationHyperparams,
    NemoConfig,
    OUTPUT_DIR,
)
from optimization import OptimizationController

# Configure logging
def setup_logging(log_file: Path | None = None) -> logging.Logger:
    """Configure logging with file and console handlers."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)

    return logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Network Parity Portfolio Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full optimization with default settings
    python -m scripts.optimization.network_parity.main

    # Quick test run
    python -m scripts.optimization.network_parity.main --quick

    # Run with specific seed and iterations
    python -m scripts.optimization.network_parity.main --seed 123 --max-iterations 30

    # Dry run to test configuration
    python -m scripts.optimization.network_parity.main --dry-run
        """
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum optimization iterations (default: 50)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)"
    )

    parser.add_argument(
        "--no-nemo",
        action="store_true",
        help="Disable Nemo integration (use random perturbation)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with 10 iterations"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test configuration without running optimization"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def create_config(args: argparse.Namespace) -> OptimizationConfig:
    """Create optimization configuration from arguments."""
    # Override hyperparams based on args
    hyperparams = OptimizationHyperparams(
        max_iterations=10 if args.quick else args.max_iterations,
        early_stopping_patience=3 if args.quick else args.patience,
    )

    # Nemo config
    nemo_config = NemoConfig()
    if args.no_nemo:
        nemo_config.confidence_threshold = 2.0  # Effectively disable

    config = OptimizationConfig(
        random_seed=args.seed,
        hyperparams=hyperparams,
        nemo=nemo_config,
    )

    return config


def print_banner() -> None:
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     NETWORK PARITY PORTFOLIO OPTIMIZATION                        ║
║                                                                  ║
║     Ordinis Quantitative Research                                ║
║     Version 1.0.0                                                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_config_summary(config: OptimizationConfig) -> None:
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Configuration ID: {config.config_id}")
    print(f"Random Seed: {config.random_seed}")
    print(f"Max Iterations: {config.hyperparams.max_iterations}")
    print(f"Early Stopping Patience: {config.hyperparams.early_stopping_patience}")
    print()
    print("Universe:")
    print(f"  Sectors: {len(config.universe.sectors)}")
    print(f"  Total Stocks: {config.universe.total_stocks}")
    print()
    print("Baseline Parameters:")
    for key, value in list(config.baseline_params.to_dict().items())[:8]:
        print(f"  {key}: {value}")
    print("  ...")
    print()
    print("Nemo Integration:")
    print(f"  Model: {config.nemo.model}")
    print(f"  Confidence Threshold: {config.nemo.confidence_threshold}")
    print("=" * 60)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = args.output_dir / f"optimization_{timestamp}.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print banner
    print_banner()

    # Create configuration
    config = create_config(args)
    print_config_summary(config)

    # Dry run - just validate configuration
    if args.dry_run:
        print("\n[DRY RUN] Configuration validated successfully.")
        print(f"Output would be saved to: {args.output_dir}")

        # Validate parameters
        violations = config.baseline_params.validate()
        if violations:
            print(f"\nParameter violations: {violations}")
            return 1

        # Save config for inspection
        config_path = config.save(args.output_dir / "configs" / f"{config.config_id}.json")
        print(f"Configuration saved: {config_path}")

        return 0

    # Run optimization
    try:
        print("\n" + "=" * 60)
        print("STARTING OPTIMIZATION")
        print("=" * 60 + "\n")

        controller = OptimizationController(config)
        result = controller.run()

        # Print final summary
        print("\n" + result.summary())

        # Check if optimization was successful
        if result.final_result is None:
            logger.error("Optimization did not produce a final result")
            return 1

        if result.improvement_vs_baseline < 0:
            logger.warning("Optimization did not improve over baseline")
            # Still return success - the baseline result is valid

        print(f"\nResults saved to: {args.output_dir}")
        print(f"Log file: {log_file}")

        return 0

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        return 130

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
