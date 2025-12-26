#!/usr/bin/env python3
"""
Network Parity Strategy - Trade-Level Analysis
Analyzes entry/exit signals, holding periods, and trade statistics.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BASE_DIR = Path("/home/kjfle/projects/ordinis/data/backtests_new")
OUTPUT_DIR = BASE_DIR / "080202a_NETWORK_PARITY/analysis_reports"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_trades_from_iterations():
    """Load all trade data from iteration files."""
    all_trades = []

    # Load from 080202 iterations (has detailed trade data)
    iter_dir = BASE_DIR / "080202_NETWORK_PARITY/iterations"

    for iter_folder in sorted(iter_dir.glob("Iteration_*")):
        seq_file = iter_folder / "sequence.json"
        if seq_file.exists():
            try:
                with open(seq_file) as f:
                    content = f.read()
                    content = content.replace(': -Infinity', ': -1e308')
                    content = content.replace(': Infinity', ': 1e308')
                    content = content.replace(': NaN', ': null')
                    data = json.loads(content)

                trades = data.get("backtest_results", {}).get("trades_sample", [])
                iteration = data.get("iteration_number", 0)
                params = data.get("tested_parameters", {})

                for t in trades:
                    t["iteration"] = iteration
                    t["params"] = params
                    all_trades.append(t)
            except Exception as e:
                print(f"Error loading {seq_file}: {e}")

    return all_trades


def parse_date(date_str):
    """Parse ISO date string."""
    if not date_str:
        return None
    try:
        # Handle various formats
        if "T" in date_str:
            return datetime.fromisoformat(date_str.replace("+00:00", "").replace("Z", ""))
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None


def calculate_holding_period(entry_date, exit_date):
    """Calculate holding period in days."""
    entry = parse_date(entry_date)
    exit_dt = parse_date(exit_date)
    if entry and exit_dt:
        return (exit_dt - entry).days
    return None


def analyze_trades(trades):
    """Comprehensive trade analysis."""

    if not trades:
        return None

    # Basic stats
    stats = {
        "total_trades": len(trades),
        "symbols": set(),
        "directions": {"long": 0, "short": 0},
        "exit_reasons": defaultdict(int),
        "holding_periods": [],
        "pnl_by_exit": defaultdict(list),
        "pnl_by_direction": {"long": [], "short": []},
        "pnl_by_symbol": defaultdict(list),
        "winners": [],
        "losers": [],
    }

    for t in trades:
        symbol = t.get("symbol", "UNK")
        stats["symbols"].add(symbol)

        # Direction
        direction = t.get("direction", 1)
        if direction == 1:
            stats["directions"]["long"] += 1
            dir_key = "long"
        else:
            stats["directions"]["short"] += 1
            dir_key = "short"

        # Exit reason
        exit_reason = t.get("exit_reason", "unknown")
        stats["exit_reasons"][exit_reason] += 1

        # Holding period
        hp = calculate_holding_period(t.get("entry_date"), t.get("exit_date"))
        if hp is not None:
            stats["holding_periods"].append(hp)
            t["holding_days"] = hp

        # PnL analysis
        pnl_pct = t.get("pnl_pct", 0)
        stats["pnl_by_exit"][exit_reason].append(pnl_pct)
        stats["pnl_by_direction"][dir_key].append(pnl_pct)
        stats["pnl_by_symbol"][symbol].append(pnl_pct)

        if pnl_pct > 0:
            stats["winners"].append(t)
        else:
            stats["losers"].append(t)

    # Calculate derived metrics
    stats["symbols"] = sorted(stats["symbols"])
    stats["win_rate"] = len(stats["winners"]) / len(trades) if trades else 0

    if stats["holding_periods"]:
        stats["avg_holding_days"] = np.mean(stats["holding_periods"])
        stats["median_holding_days"] = np.median(stats["holding_periods"])
        stats["min_holding_days"] = min(stats["holding_periods"])
        stats["max_holding_days"] = max(stats["holding_periods"])

    # Average PnL by exit reason
    stats["avg_pnl_by_exit"] = {
        k: np.mean(v) if v else 0 for k, v in stats["pnl_by_exit"].items()
    }

    # Average PnL by direction
    stats["avg_pnl_by_direction"] = {
        k: np.mean(v) if v else 0 for k, v in stats["pnl_by_direction"].items()
    }

    return stats


def generate_trade_report(trades, stats):
    """Generate detailed trade report."""

    report = []
    report.append("=" * 100)
    report.append("NETWORK PARITY STRATEGY - TRADE-LEVEL ANALYSIS")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 100)
    report.append("")

    # Overview
    report.append("-" * 100)
    report.append("TRADE OVERVIEW")
    report.append("-" * 100)
    report.append(f"Total Trades Analyzed: {stats['total_trades']}")
    report.append(f"Unique Symbols: {len(stats['symbols'])} ({', '.join(stats['symbols'][:10])}...)")
    report.append(f"Win Rate: {stats['win_rate']*100:.1f}%")
    report.append(f"Winners: {len(stats['winners'])} | Losers: {len(stats['losers'])}")
    report.append("")

    # Direction breakdown
    report.append("-" * 100)
    report.append("DIRECTION ANALYSIS")
    report.append("-" * 100)
    report.append(f"Long Trades:  {stats['directions']['long']:>5} ({stats['directions']['long']/stats['total_trades']*100:.1f}%)")
    report.append(f"Short Trades: {stats['directions']['short']:>5} ({stats['directions']['short']/stats['total_trades']*100:.1f}%)")
    report.append("")
    report.append("Average P&L by Direction:")
    report.append(f"  Long:  {stats['avg_pnl_by_direction']['long']*100:>+8.2f}%")
    report.append(f"  Short: {stats['avg_pnl_by_direction']['short']*100:>+8.2f}%")
    report.append("")

    # Holding Period Analysis
    report.append("-" * 100)
    report.append("HOLDING PERIOD ANALYSIS")
    report.append("-" * 100)
    if stats.get("avg_holding_days"):
        report.append(f"Average Holding Period: {stats['avg_holding_days']:.1f} days")
        report.append(f"Median Holding Period:  {stats['median_holding_days']:.1f} days")
        report.append(f"Min Holding Period:     {stats['min_holding_days']} days")
        report.append(f"Max Holding Period:     {stats['max_holding_days']} days")

        # Distribution
        hp = stats["holding_periods"]
        report.append("")
        report.append("Holding Period Distribution:")
        for bucket, label in [(1, "1 day"), (2, "2 days"), (3, "3 days"),
                              (5, "4-5 days"), (10, "6-10 days"), (float('inf'), ">10 days")]:
            if bucket == 1:
                count = sum(1 for h in hp if h == 1)
            elif bucket == 2:
                count = sum(1 for h in hp if h == 2)
            elif bucket == 3:
                count = sum(1 for h in hp if h == 3)
            elif bucket == 5:
                count = sum(1 for h in hp if 4 <= h <= 5)
            elif bucket == 10:
                count = sum(1 for h in hp if 6 <= h <= 10)
            else:
                count = sum(1 for h in hp if h > 10)
            pct = count / len(hp) * 100 if hp else 0
            bar = "#" * int(pct / 2)
            report.append(f"  {label:>10}: {count:>4} ({pct:>5.1f}%) {bar}")
    report.append("")

    # Exit Reason Analysis
    report.append("-" * 100)
    report.append("EXIT SIGNAL ANALYSIS")
    report.append("-" * 100)
    report.append("")
    report.append(f"{'Exit Reason':<20} | {'Count':>8} | {'Pct':>8} | {'Avg P&L':>10} | {'Assessment':<15}")
    report.append("-" * 80)

    exit_assessments = {
        "take_profit": "TARGET HIT",
        "stop_loss": "RISK MGMT",
        "trailing_stop": "LOCK PROFIT",
        "z_score_exit": "MEAN REVERT",
        "timeout": "TIME EXIT",
        "unknown": "UNCLASSIFIED"
    }

    for reason, count in sorted(stats["exit_reasons"].items(), key=lambda x: -x[1]):
        pct = count / stats["total_trades"] * 100
        avg_pnl = stats["avg_pnl_by_exit"].get(reason, 0)
        assessment = exit_assessments.get(reason, reason.upper())
        report.append(f"{reason:<20} | {count:>8} | {pct:>7.1f}% | {avg_pnl*100:>+9.2f}% | {assessment:<15}")

    report.append("")
    report.append("")

    # Entry Signal Logic
    report.append("-" * 100)
    report.append("ENTRY SIGNAL LOGIC (from optimized v5 parameters)")
    report.append("-" * 100)
    report.append("")
    report.append("LONG ENTRY CONDITIONS:")
    report.append("  1. Momentum Signal: price_change > momentum_threshold (1.06%)")
    report.append("  2. Z-Score Entry: z_score < -zscore_entry (-1.28) [oversold]")
    report.append("  3. Bull Regime: market_return > bull_threshold (1.05%)")
    report.append("  4. Volatility Filter: vol < vol_threshold (5.03%)")
    report.append("  5. Network Weight: inverse_centrality_weight > min_weight")
    report.append("")
    report.append("SHORT ENTRY CONDITIONS:")
    report.append("  1. Momentum Signal: price_change < -momentum_threshold (-1.06%)")
    report.append("  2. Z-Score Entry: z_score > zscore_entry (1.28) [overbought]")
    report.append("  3. Bear Regime: market_return < bear_threshold (-1.77%)")
    report.append("  4. Volatility Filter: vol < vol_threshold (5.03%)")
    report.append("  5. Network Weight: inverse_centrality_weight > min_weight")
    report.append("")

    # Exit Signal Logic
    report.append("-" * 100)
    report.append("EXIT SIGNAL LOGIC")
    report.append("-" * 100)
    report.append("")
    report.append("EXIT TRIGGERS (in priority order):")
    report.append("  1. STOP LOSS:     Triggered at -8.0% from entry")
    report.append("  2. TAKE PROFIT:   Triggered at +41.2% from entry")
    report.append("  3. TRAILING STOP: Dynamic stop follows price (3% trail)")
    report.append("  4. Z-SCORE EXIT:  Mean reversion complete (z-score crosses 0.13)")
    report.append("  5. TIMEOUT:       Max holding period exceeded")
    report.append("")

    # Sample Trades
    report.append("-" * 100)
    report.append("SAMPLE TRADES (Last 10)")
    report.append("-" * 100)
    report.append("")
    report.append(f"{'Symbol':<8} | {'Direction':<6} | {'Entry':<12} | {'Exit':<12} | {'Hold':>5} | {'P&L':>10} | {'Exit Reason':<15}")
    report.append("-" * 90)

    for t in trades[-10:]:
        symbol = t.get("symbol", "UNK")
        direction = "LONG" if t.get("direction", 1) == 1 else "SHORT"
        entry = t.get("entry_date", "")[:10] if t.get("entry_date") else "N/A"
        exit_dt = t.get("exit_date", "")[:10] if t.get("exit_date") else "N/A"
        hold = t.get("holding_days", "?")
        pnl = t.get("pnl_pct", 0)
        reason = t.get("exit_reason", "unknown")

        report.append(f"{symbol:<8} | {direction:<6} | {entry:<12} | {exit_dt:<12} | {hold:>5} | {pnl*100:>+9.2f}% | {reason:<15}")

    report.append("")
    report.append("")

    # Symbol Performance
    report.append("-" * 100)
    report.append("TOP/BOTTOM SYMBOLS BY AVERAGE P&L")
    report.append("-" * 100)

    symbol_stats = []
    for sym, pnls in stats["pnl_by_symbol"].items():
        symbol_stats.append({
            "symbol": sym,
            "trades": len(pnls),
            "avg_pnl": np.mean(pnls),
            "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
        })

    symbol_stats.sort(key=lambda x: -x["avg_pnl"])

    report.append("")
    report.append("TOP 5 PERFORMERS:")
    report.append(f"{'Symbol':<8} | {'Trades':>6} | {'Avg P&L':>10} | {'Win Rate':>10}")
    report.append("-" * 45)
    for s in symbol_stats[:5]:
        report.append(f"{s['symbol']:<8} | {s['trades']:>6} | {s['avg_pnl']*100:>+9.2f}% | {s['win_rate']*100:>9.1f}%")

    report.append("")
    report.append("BOTTOM 5 PERFORMERS:")
    report.append(f"{'Symbol':<8} | {'Trades':>6} | {'Avg P&L':>10} | {'Win Rate':>10}")
    report.append("-" * 45)
    for s in symbol_stats[-5:]:
        report.append(f"{s['symbol']:<8} | {s['trades']:>6} | {s['avg_pnl']*100:>+9.2f}% | {s['win_rate']*100:>9.1f}%")

    report.append("")
    report.append("=" * 100)
    report.append("END OF TRADE ANALYSIS")
    report.append("=" * 100)

    return "\n".join(report)


def generate_trade_visualizations(trades, stats):
    """Generate trade analysis visualizations."""

    if not HAS_MATPLOTLIB:
        print("Skipping visualizations - matplotlib not available")
        return None

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Exit Reason Distribution (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    reasons = list(stats["exit_reasons"].keys())
    counts = [stats["exit_reasons"][r] for r in reasons]
    colors = {'take_profit': '#2ecc71', 'stop_loss': '#e74c3c',
              'trailing_stop': '#f39c12', 'z_score_exit': '#3498db', 'unknown': '#95a5a6'}
    pie_colors = [colors.get(r, '#95a5a6') for r in reasons]

    wedges, texts, autotexts = ax1.pie(counts, labels=reasons, autopct='%1.1f%%',
                                        colors=pie_colors, explode=[0.02]*len(reasons))
    ax1.set_title('Exit Reason Distribution', fontsize=14, fontweight='bold')

    # 2. Holding Period Histogram
    ax2 = fig.add_subplot(gs[0, 1])
    if stats.get("holding_periods"):
        hp = stats["holding_periods"]
        ax2.hist(hp, bins=range(0, max(hp)+2), color='#3498db', edgecolor='black', alpha=0.7)
        ax2.axvline(stats["avg_holding_days"], color='red', linestyle='--',
                   label=f'Mean: {stats["avg_holding_days"]:.1f}d', linewidth=2)
        ax2.axvline(stats["median_holding_days"], color='orange', linestyle='--',
                   label=f'Median: {stats["median_holding_days"]:.1f}d', linewidth=2)
        ax2.set_xlabel('Holding Period (Days)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Holding Period Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. P&L by Exit Reason (bar chart)
    ax3 = fig.add_subplot(gs[0, 2])
    exit_pnl = stats["avg_pnl_by_exit"]
    reasons_sorted = sorted(exit_pnl.keys(), key=lambda x: -exit_pnl[x])
    pnls = [exit_pnl[r] * 100 for r in reasons_sorted]
    bar_colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in pnls]

    bars = ax3.bar(reasons_sorted, pnls, color=bar_colors, edgecolor='black')
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_ylabel('Average P&L (%)')
    ax3.set_title('Average P&L by Exit Reason', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for bar, pnl in zip(bars, pnls):
        ax3.annotate(f'{pnl:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, pnl),
                    xytext=(0, 3 if pnl > 0 else -12), textcoords='offset points',
                    ha='center', fontweight='bold', fontsize=9)

    # 4. Direction Performance
    ax4 = fig.add_subplot(gs[1, 0])
    directions = ['Long', 'Short']
    dir_counts = [stats['directions']['long'], stats['directions']['short']]
    dir_pnl = [stats['avg_pnl_by_direction']['long']*100,
               stats['avg_pnl_by_direction']['short']*100]

    x = np.arange(len(directions))
    width = 0.35

    bars1 = ax4.bar(x - width/2, dir_counts, width, label='Trade Count', color='#3498db')
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, dir_pnl, width, label='Avg P&L %',
                         color=['#2ecc71' if p > 0 else '#e74c3c' for p in dir_pnl])

    ax4.set_xticks(x)
    ax4.set_xticklabels(directions)
    ax4.set_ylabel('Trade Count', color='#3498db')
    ax4_twin.set_ylabel('Average P&L (%)', color='green')
    ax4.set_title('Performance by Direction', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')

    # 5. P&L Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    all_pnl = [t.get("pnl_pct", 0) * 100 for t in trades]
    ax5.hist(all_pnl, bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax5.axvline(0, color='black', linewidth=2)
    ax5.axvline(np.mean(all_pnl), color='red', linestyle='--',
               label=f'Mean: {np.mean(all_pnl):.1f}%', linewidth=2)
    ax5.set_xlabel('P&L (%)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Win Rate by Holding Period
    ax6 = fig.add_subplot(gs[1, 2])
    hp_buckets = {'1d': [], '2d': [], '3d': [], '4-5d': [], '6+d': []}
    for t in trades:
        hp = t.get("holding_days")
        pnl = t.get("pnl_pct", 0)
        if hp is None:
            continue
        if hp == 1:
            hp_buckets['1d'].append(pnl)
        elif hp == 2:
            hp_buckets['2d'].append(pnl)
        elif hp == 3:
            hp_buckets['3d'].append(pnl)
        elif hp <= 5:
            hp_buckets['4-5d'].append(pnl)
        else:
            hp_buckets['6+d'].append(pnl)

    bucket_names = list(hp_buckets.keys())
    win_rates = [sum(1 for p in v if p > 0)/len(v)*100 if v else 0 for v in hp_buckets.values()]
    trade_counts = [len(v) for v in hp_buckets.values()]

    bars = ax6.bar(bucket_names, win_rates, color='#1abc9c', edgecolor='black')
    ax6.axhline(50, color='gray', linestyle='--', label='50% breakeven')
    ax6.set_ylabel('Win Rate (%)')
    ax6.set_xlabel('Holding Period')
    ax6.set_title('Win Rate by Holding Period', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # Add trade count labels
    for bar, count in zip(bars, trade_counts):
        ax6.annotate(f'n={count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    # Title
    fig.suptitle(f'Network Parity Strategy - Trade Analysis\n'
                 f'Total Trades: {stats["total_trades"]} | Win Rate: {stats["win_rate"]*100:.1f}% | '
                 f'Avg Hold: {stats.get("avg_holding_days", 0):.1f} days',
                 fontsize=16, fontweight='bold', y=0.98)

    output_path = OUTPUT_DIR / "network_parity_trade_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def main():
    print("Loading trade data from iterations...")
    trades = load_trades_from_iterations()
    print(f"Loaded {len(trades)} trades")

    if not trades:
        print("No trades found!")
        return

    print("\nAnalyzing trades...")
    stats = analyze_trades(trades)

    print("\nGenerating trade report...")
    report = generate_trade_report(trades, stats)

    # Save report
    report_path = OUTPUT_DIR / "network_parity_trade_analysis.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved report to: {report_path}")

    # Print report
    print("\n" + report)

    # Generate visualizations
    print("\nGenerating visualizations...")
    viz_path = generate_trade_visualizations(trades, stats)
    if viz_path:
        print(f"Saved visualization to: {viz_path}")

    print("\nTrade analysis complete!")


if __name__ == "__main__":
    main()
