"""
Confidence Distribution Analysis

Analyzes confidence scores against realized outcomes, with optional ML
calibration and plot exports.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from phase1_real_market_backtest import (
    UNIVERSE,
    download_market_data,
    generate_trades_from_historical_data,
)

from ordinis.optimizations.confidence_calibrator import ConfidenceCalibrator


def _summarize_bins(df: pd.DataFrame, bins: list[float]) -> list[dict]:
    df = df.copy()
    df["bin"] = pd.cut(df["confidence"], bins=bins, include_lowest=True)
    stats: list[dict] = []
    for bin_label in df["bin"].cat.categories:
        bin_data = df[df["bin"] == bin_label]
        if bin_data.empty:
            continue
        stats.append(
            {
                "bin": str(bin_label),
                "count": int(len(bin_data)),
                "pct": float(len(bin_data) / len(df) * 100),
                "win_rate": float(bin_data["win"].mean()),
                "avg_return": float(bin_data["return"].mean()),
            }
        )
    return stats


def _summarize_deciles(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["decile"] = pd.qcut(df["confidence"], q=10, labels=False, duplicates="drop")
    stats: list[dict] = []
    for decile in sorted(df["decile"].dropna().unique()):
        decile_data = df[df["decile"] == decile]
        stats.append(
            {
                "decile": int(decile) + 1,
                "count": int(len(decile_data)),
                "win_rate": float(decile_data["win"].mean()),
                "avg_return": float(decile_data["return"].mean()),
            }
        )
    return stats


def _export_reports(
    summary: dict,
    bin_stats: list[dict],
    decile_stats: list[dict],
    correlations: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    payload = {
        "summary": summary,
        "correlations": correlations,
        "bins": bin_stats,
        "deciles": decile_stats,
    }

    json_path = out_dir / f"confidence_analysis_{timestamp}.json"
    bins_csv = out_dir / f"confidence_bins_{timestamp}.csv"
    deciles_csv = out_dir / f"confidence_deciles_{timestamp}.csv"

    json_path.write_text(json.dumps(payload, indent=2))
    pd.DataFrame(bin_stats).to_csv(bins_csv, index=False)
    pd.DataFrame(decile_stats).to_csv(deciles_csv, index=False)

    print(f"Reports written: {json_path} | {bins_csv} | {deciles_csv}")


def analyze_distribution(trades: list[dict], confidence_col: str) -> pd.DataFrame:
    confidences = [t.get(confidence_col, 0.0) for t in trades]
    returns = [t["return_pct"] for t in trades]
    wins = [t["win"] for t in trades]

    df = pd.DataFrame(
        {
            "confidence": confidences,
            "return": returns,
            "win": wins,
        }
    )

    summary = {
        "total_trades": int(len(df)),
        "min": float(df["confidence"].min()),
        "p25": float(df["confidence"].quantile(0.25)),
        "median": float(df["confidence"].median()),
        "p75": float(df["confidence"].quantile(0.75)),
        "max": float(df["confidence"].max()),
        "mean": float(df["confidence"].mean()),
        "std": float(df["confidence"].std()),
    }

    print("=" * 80)
    print("CONFIDENCE SCORE DISTRIBUTION")
    print("=" * 80)
    print()
    print(f"Total trades: {summary['total_trades']}")
    print()
    print("Distribution Statistics:")
    print(f"  Min:    {summary['min']:.4f}")
    print(f"  25th %: {summary['p25']:.4f}")
    print(f"  Median: {summary['median']:.4f}")
    print(f"  75th %: {summary['p75']:.4f}")
    print(f"  Max:    {summary['max']:.4f}")
    print(f"  Mean:   {summary['mean']:.4f}")
    print(f"  Std:    {summary['std']:.4f}")
    print()

    bins = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    bin_stats = _summarize_bins(df, bins)

    print("=" * 80)
    print("CONFIDENCE BINS")
    print("=" * 80)
    print()
    for stat in bin_stats:
        print(
            f"{stat['bin']:15} {stat['count']:5} trades ({stat['pct']:5.1f}%) | "
            f"Win Rate: {stat['win_rate']*100:5.1f}% | "
            f"Avg Return: {stat['avg_return']:6.2f}%"
        )
    print()

    corr_return = df["confidence"].corr(df["return"])
    corr_win = df["confidence"].corr(df["win"])

    print("=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    print()
    print(f"Confidence vs Return:   {corr_return:7.4f}")
    print(f"Confidence vs Win Rate: {corr_win:7.4f}")
    if corr_return < -0.05:
        print("WARNING: Negative correlation detected!")
        print("High confidence scores are associated with WORSE returns.")
    elif corr_return < 0.05:
        print("WARNING: Near-zero correlation detected!")
        print("Confidence scores have no predictive power.")
    else:
        print("Positive correlation detected.")
        print("Higher confidence is associated with better returns.")
    print()

    decile_stats = _summarize_deciles(df)
    print("=" * 80)
    print("DECILE ANALYSIS")
    print("=" * 80)
    print()
    print("Decile  Count  Win Rate  Avg Return")
    print("-" * 45)
    for stat in decile_stats:
        print(
            f"{stat['decile']:3}     {stat['count']:5}    "
            f"{stat['win_rate']*100:5.1f}%     {stat['avg_return']:6.2f}%"
        )
    print()

    _export_reports(
        summary=summary,
        bin_stats=bin_stats,
        decile_stats=decile_stats,
        correlations={
            "confidence_vs_return": float(corr_return),
            "confidence_vs_win": float(corr_win),
        },
        out_dir=Path("reports"),
    )

    return df


def generate_plots(df: pd.DataFrame, output_dir: Path, label: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; skipping plot generation")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    conf_col = "confidence"

    plt.figure(figsize=(8, 4))
    plt.hist(df[conf_col], bins=30, color="steelblue", edgecolor="black")
    plt.title(f"Confidence Distribution ({label})")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / f"confidence_hist_{label}.png")
    plt.close()

    bins = np.linspace(0, 1, 11)
    df["bin"] = pd.cut(df[conf_col], bins=bins, include_lowest=True)
    grouped = df.groupby("bin")
    centers = [interval.mid for interval in grouped.size().index]
    actual = grouped["win"].mean().values

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(centers, actual, marker="o", label="Actual win rate")
    plt.title(f"Reliability Curve ({label})")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical win rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"confidence_reliability_{label}.png")
    plt.close()

    df["decile"] = pd.qcut(df[conf_col], q=10, labels=False, duplicates="drop")
    decile_win = df.groupby("decile")["win"].mean()
    plt.figure(figsize=(8, 4))
    plt.bar(decile_win.index, decile_win.values, color="seagreen")
    plt.title(f"Win Rate by Confidence Decile ({label})")
    plt.xlabel("Decile (low -> high)")
    plt.ylabel("Win rate")
    plt.tight_layout()
    plt.savefig(output_dir / f"confidence_deciles_{label}.png")
    plt.close()

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence diagnostics")
    parser.add_argument(
        "--use-calibration", action="store_true", help="Fit ML calibrator before analysis"
    )
    parser.add_argument(
        "--risk-tolerance",
        type=float,
        default=0.50,
        help="Risk slider 0.0=conservative, 1.0=aggressive",
    )
    parser.add_argument("--plots", action="store_true", help="Generate PNG visualizations")
    args = parser.parse_args()

    market_data = download_market_data(
        list(UNIVERSE.keys()),
        "2019-01-01",
        "2024-12-01",
    )
    trades = generate_trades_from_historical_data(market_data)

    confidence_col = "confidence_score"
    label = "raw"

    if args.use_calibration:
        calibrator = ConfidenceCalibrator()
        metrics = calibrator.fit(trades)
        trades = calibrator.calibrate_trades(trades)
        confidence_col = "calibrated_probability"
        label = "calibrated"
        probs = [t[confidence_col] for t in trades]
        suggested = calibrator.threshold_for_risk_tolerance(
            probs,
            args.risk_tolerance,
        )
        print(
            f"Calibrated using ML: Brier {metrics.brier_score:.4f}, "
            f"LogLoss {metrics.log_loss:.4f}, Suggested threshold {suggested:.2f}"
        )
        print()

    df = analyze_distribution(trades, confidence_col)

    if args.plots:
        generate_plots(df, Path("reports"), label)
