#!/usr/bin/env python3
"""
Generate visualization charts from Fibonacci ADX backtest results.
Produces PNG charts for analysis.
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Find most recent backtest results
results_dir = Path("data/backtest_results")
result_files = sorted(results_dir.glob("daily_backtest_async_*.json"), reverse=True)

if not result_files:
    print("No backtest results found!")
    exit(1)

latest_file = result_files[0]
print(f"Loading: {latest_file}")

with open(latest_file) as f:
    results = json.load(f)

# Output directory
output_dir = Path("artifacts/backtest_charts")
output_dir.mkdir(parents=True, exist_ok=True)

# Extract data
categories = ["small_cap", "mid_cap", "large_cap"]
cat_names = ["Small Cap", "Mid Cap", "Large Cap"]
colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]

all_stocks = []
for cat in categories:
    for r in results.get(cat, []):
        if r.get("status") == "success":
            r["category"] = cat
            all_stocks.append(r)

# =============================================================================
# Chart 1: Returns by Stock (Bar Chart)
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

symbols = [s["symbol"] for s in all_stocks]
returns = [s["total_return"] * 100 for s in all_stocks]
bar_colors = [colors[categories.index(s["category"])] for s in all_stocks]

bars = ax.bar(symbols, returns, color=bar_colors, edgecolor='white', linewidth=0.5)

# Add zero line
ax.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.5)

# Color bars by positive/negative
for bar, ret in zip(bars, returns):
    if ret < 0:
        bar.set_alpha(0.7)

ax.set_xlabel('Stock Symbol')
ax.set_ylabel('Total Return (%)')
ax.set_title('Fibonacci ADX Strategy - 3-Year Returns by Stock (Daily Timeframe)')
ax.tick_params(axis='x', rotation=45)

# Legend
patches = [mpatches.Patch(color=c, label=n) for c, n in zip(colors, cat_names)]
ax.legend(handles=patches, loc='upper right')

plt.tight_layout()
plt.savefig(output_dir / "01_returns_by_stock.png", dpi=150, facecolor='#1a1a2e')
print(f"Saved: {output_dir / '01_returns_by_stock.png'}")
plt.close()

# =============================================================================
# Chart 2: Category Performance Comparison
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Aggregate by category
cat_data = {}
for cat, name in zip(categories, cat_names):
    cat_stocks = [s for s in all_stocks if s["category"] == cat]
    if cat_stocks:
        cat_data[name] = {
            "avg_return": np.mean([s["total_return"] for s in cat_stocks]) * 100,
            "avg_sharpe": np.mean([s["sharpe"] for s in cat_stocks]),
            "avg_maxdd": np.mean([s["max_drawdown"] for s in cat_stocks]) * 100,
            "win_rate": sum(1 for s in cat_stocks if s["total_return"] > 0) / len(cat_stocks) * 100,
            "total_trades": sum(s["num_trades"] for s in cat_stocks),
        }

# Average Return
ax1 = axes[0, 0]
names = list(cat_data.keys())
avg_returns = [cat_data[n]["avg_return"] for n in names]
bars = ax1.bar(names, avg_returns, color=colors, edgecolor='white')
ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5)
ax1.set_ylabel('Average Return (%)')
ax1.set_title('Average Return by Category')
for bar, val in zip(bars, avg_returns):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

# Average Sharpe
ax2 = axes[0, 1]
avg_sharpes = [cat_data[n]["avg_sharpe"] for n in names]
bars = ax2.bar(names, avg_sharpes, color=colors, edgecolor='white')
ax2.axhline(y=0, color='white', linestyle='--', alpha=0.5)
ax2.set_ylabel('Average Sharpe Ratio')
ax2.set_title('Average Sharpe Ratio by Category')
for bar, val in zip(bars, avg_sharpes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.2f}', ha='center', va='bottom', fontsize=10)

# Max Drawdown
ax3 = axes[1, 0]
avg_maxdds = [cat_data[n]["avg_maxdd"] for n in names]
bars = ax3.bar(names, avg_maxdds, color=colors, edgecolor='white')
ax3.set_ylabel('Average Max Drawdown (%)')
ax3.set_title('Average Max Drawdown by Category')
for bar, val in zip(bars, avg_maxdds):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

# Win Rate
ax4 = axes[1, 1]
win_rates = [cat_data[n]["win_rate"] for n in names]
bars = ax4.bar(names, win_rates, color=colors, edgecolor='white')
ax4.axhline(y=50, color='yellow', linestyle='--', alpha=0.5, label='50% threshold')
ax4.set_ylabel('Win Rate (%)')
ax4.set_title('Win Rate by Category (Stocks with Positive Return)')
ax4.set_ylim(0, 100)
for bar, val in zip(bars, win_rates):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.0f}%', ha='center', va='bottom', fontsize=10)

plt.suptitle('Fibonacci ADX Strategy - Category Performance Comparison', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "02_category_comparison.png", dpi=150, facecolor='#1a1a2e')
print(f"Saved: {output_dir / '02_category_comparison.png'}")
plt.close()

# =============================================================================
# Chart 3: Risk-Reward Scatter Plot
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

for cat, color, name in zip(categories, colors, cat_names):
    cat_stocks = [s for s in all_stocks if s["category"] == cat]
    sharpes = [s["sharpe"] for s in cat_stocks]
    returns = [s["total_return"] * 100 for s in cat_stocks]
    symbols = [s["symbol"] for s in cat_stocks]
    
    ax.scatter(sharpes, returns, c=color, s=100, alpha=0.8, label=name, edgecolors='white')
    
    # Label points
    for x, y, sym in zip(sharpes, returns, symbols):
        ax.annotate(sym, (x, y), textcoords="offset points", xytext=(5, 5), 
                   fontsize=8, alpha=0.9)

ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='white', linestyle='--', alpha=0.3)

# Highlight quadrants
ax.fill_between([-2, 0], [-10, -10], [10, 10], alpha=0.1, color='red')
ax.fill_between([0, 2], [0, 0], [10, 10], alpha=0.1, color='green')

ax.set_xlabel('Sharpe Ratio')
ax.set_ylabel('Total Return (%)')
ax.set_title('Risk-Reward Analysis: Sharpe Ratio vs Return')
ax.legend(loc='lower right')
ax.set_xlim(-2, 1.5)
ax.set_ylim(-7, 6)

plt.tight_layout()
plt.savefig(output_dir / "03_risk_reward_scatter.png", dpi=150, facecolor='#1a1a2e')
print(f"Saved: {output_dir / '03_risk_reward_scatter.png'}")
plt.close()

# =============================================================================
# Chart 4: Top/Bottom Performers
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sort by return
sorted_stocks = sorted(all_stocks, key=lambda x: x["total_return"], reverse=True)
top_5 = sorted_stocks[:5]
bottom_5 = sorted_stocks[-5:]

# Top 5
ax1 = axes[0]
symbols = [s["symbol"] for s in top_5]
returns = [s["total_return"] * 100 for s in top_5]
sharpes = [s["sharpe"] for s in top_5]
bar_colors = [colors[categories.index(s["category"])] for s in top_5]

bars = ax1.barh(symbols, returns, color=bar_colors, edgecolor='white')
ax1.set_xlabel('Total Return (%)')
ax1.set_title('Top 5 Performers')
ax1.invert_yaxis()

for bar, ret, sharpe in zip(bars, returns, sharpes):
    ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f'{ret:.2f}% (Sharpe: {sharpe:.2f})', va='center', fontsize=10)

# Bottom 5
ax2 = axes[1]
symbols = [s["symbol"] for s in bottom_5]
returns = [s["total_return"] * 100 for s in bottom_5]
sharpes = [s["sharpe"] for s in bottom_5]
bar_colors = [colors[categories.index(s["category"])] for s in bottom_5]

bars = ax2.barh(symbols, returns, color=bar_colors, edgecolor='white')
ax2.set_xlabel('Total Return (%)')
ax2.set_title('Bottom 5 Performers')
ax2.invert_yaxis()

for bar, ret, sharpe in zip(bars, returns, sharpes):
    ax2.text(bar.get_width() - 0.1, bar.get_y() + bar.get_height()/2,
             f'{ret:.2f}% (Sharpe: {sharpe:.2f})', va='center', ha='right', fontsize=10)

plt.suptitle('Fibonacci ADX Strategy - Best and Worst Performers', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "04_top_bottom_performers.png", dpi=150, facecolor='#1a1a2e')
print(f"Saved: {output_dir / '04_top_bottom_performers.png'}")
plt.close()

# =============================================================================
# Chart 5: Timeframe Comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

timeframes = ['1-Minute', '12-Hour', '24-Hour', 'Daily']
avg_returns = [-65, -60, -55, -0.35]  # Approximate from backtests
win_rates = [2, 0, 0, 40]

x = np.arange(len(timeframes))
width = 0.35

bars1 = ax.bar(x - width/2, avg_returns, width, label='Avg Return (%)', color='#ff6b6b', edgecolor='white')
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, win_rates, width, label='Win Rate (%)', color='#4ecdc4', edgecolor='white')

ax.set_xlabel('Timeframe')
ax.set_ylabel('Average Return (%)', color='#ff6b6b')
ax2.set_ylabel('Win Rate (%)', color='#4ecdc4')
ax.set_xticks(x)
ax.set_xticklabels(timeframes)
ax.set_title('Fibonacci ADX Strategy - Timeframe Suitability Analysis')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height - 5,
            f'{height:.1f}%', ha='center', va='top', fontsize=10, color='white')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height:.0f}%', ha='center', va='bottom', fontsize=10, color='white')

# Add recommendation arrow
ax.annotate('✓ RECOMMENDED', xy=(3, 0), xytext=(3, -30),
            fontsize=12, ha='center', color='#4ecdc4',
            arrowprops=dict(arrowstyle='->', color='#4ecdc4'))

ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
ax2.set_ylim(0, 100)

fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.tight_layout()
plt.savefig(output_dir / "05_timeframe_comparison.png", dpi=150, facecolor='#1a1a2e')
print(f"Saved: {output_dir / '05_timeframe_comparison.png'}")
plt.close()

# =============================================================================
# Chart 6: Trade Statistics
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trades by category
ax1 = axes[0]
total_trades = [cat_data[n]["total_trades"] for n in names]
bars = ax1.bar(names, total_trades, color=colors, edgecolor='white')
ax1.set_ylabel('Total Trades')
ax1.set_title('Number of Trades by Category (3-Year Period)')
for bar, val in zip(bars, total_trades):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{val}', ha='center', va='bottom', fontsize=12)

# Win rate distribution
ax2 = axes[1]
win_rates_all = [s["win_rate"] * 100 for s in all_stocks]
ax2.hist(win_rates_all, bins=10, color='#45b7d1', edgecolor='white', alpha=0.8)
ax2.axvline(x=50, color='yellow', linestyle='--', label='50% Threshold')
ax2.set_xlabel('Trade Win Rate (%)')
ax2.set_ylabel('Number of Stocks')
ax2.set_title('Distribution of Trade Win Rates Across Stocks')
ax2.legend()

plt.suptitle('Fibonacci ADX Strategy - Trade Statistics', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "06_trade_statistics.png", dpi=150, facecolor='#1a1a2e')
print(f"Saved: {output_dir / '06_trade_statistics.png'}")
plt.close()

print(f"\n✓ All charts saved to: {output_dir}")
print(f"  - 01_returns_by_stock.png")
print(f"  - 02_category_comparison.png")
print(f"  - 03_risk_reward_scatter.png")
print(f"  - 04_top_bottom_performers.png")
print(f"  - 05_timeframe_comparison.png")
print(f"  - 06_trade_statistics.png")
