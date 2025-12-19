import argparse
import json
import pathlib

from .optimizer import PortfolioOptimizer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ordinis-portfolio-opt",
        description="GPU‑accelerated portfolio optimisation (mean‑variance / mean‑CVaR).",
    )
    p.add_argument(
        "--returns",
        type=pathlib.Path,
        required=True,
        help="Path to a CSV/Parquet file with historical returns (shape: T×N).",
    )
    p.add_argument(
        "--constraints",
        type=pathlib.Path,
        required=False,
        help="YAML file describing weight bounds, sector caps, etc.",
    )
    p.add_argument(
        "--method",
        choices=["mean_variance", "mean_cvar"],
        default="mean_variance",
        help="Optimisation objective.",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("opt_result.json"),
        help="File to write the optimisation result.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load returns (pandas handles CSV or Parquet automatically)
    import pandas as pd

    returns = (
        pd.read_parquet(args.returns)
        if args.returns.suffix == ".parquet"
        else pd.read_csv(args.returns)
    )

    # Load optional constraints
    constraints = {}
    if args.constraints:
        import yaml

        with open(args.constraints, encoding="utf-8") as f:
            constraints = yaml.safe_load(f)

    optimizer = PortfolioOptimizer(method=args.method)
    result = optimizer.optimize(returns, constraints)

    # Write a compact JSON report
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"✅ Optimisation finished – result written to {args.output}")


if __name__ == "__main__":
    main()
