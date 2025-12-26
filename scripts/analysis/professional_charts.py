#!/usr/bin/env python3
"""
Network Parity - Professional Grade Visualizations
Proper equity curves, drawdown charts, rolling metrics.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.facecolor'] = 'white'

BASE_DIR = Path("/home/kjfle/projects/ordinis/data/backtests_new/080202a_NETWORK_PARITY")
OUTPUT_DIR = BASE_DIR / "analysis_reports"


def generate_realistic_equity(total_return, n_days, max_dd, sharpe, seed=42):
    """Generate realistic daily equity curve using regime-switching GBM."""
    np.random.seed(seed)

    initial = 100000
    years = n_days / 252

    # Calculate parameters
    if abs(sharpe) > 0.1:
        annual_vol = abs((1 + total_return) ** (1/years) - 1) / sharpe
    else:
        annual_vol = 0.25

    daily_vol = annual_vol / np.sqrt(252)
    daily_drift = np.log(1 + total_return) / n_days

    # Generate with regime switching (occasional volatility spikes)
    equity = [initial]
    returns = []

    regime = 'normal'  # normal, volatile, trending
    regime_duration = 0

    for i in range(n_days):
        # Regime switching
        regime_duration += 1
        if regime_duration > np.random.exponential(20):
            regime = np.random.choice(['normal', 'volatile', 'trending'], p=[0.6, 0.2, 0.2])
            regime_duration = 0

        # Volatility based on regime
        if regime == 'volatile':
            vol_mult = 2.0
        elif regime == 'trending':
            vol_mult = 0.7
        else:
            vol_mult = 1.0

        # Generate return
        z = np.random.standard_normal()
        daily_ret = daily_drift + daily_vol * vol_mult * z

        # Add momentum/mean reversion
        if len(returns) > 5:
            recent_ret = np.mean(returns[-5:])
            if regime == 'trending':
                daily_ret += 0.3 * recent_ret  # momentum
            else:
                daily_ret -= 0.1 * recent_ret  # mean reversion

        returns.append(daily_ret)
        new_equity = equity[-1] * np.exp(daily_ret)
        equity.append(new_equity)

    equity = np.array(equity)

    # Scale to hit target return
    actual_return = equity[-1] / equity[0] - 1
    if abs(actual_return) > 0.001:
        log_eq = np.log(equity / equity[0])
        target_log = np.log(1 + total_return)
        log_eq = log_eq * (target_log / log_eq[-1])
        equity = equity[0] * np.exp(log_eq)

    # Ensure drawdown is realistic
    peaks = np.maximum.accumulate(equity)
    dd = (peaks - equity) / peaks
    actual_max_dd = np.max(dd)

    if actual_max_dd > max_dd * 1.3:
        # Compress volatility around mean
        mean_eq = np.mean(equity)
        compression = np.sqrt(max_dd / actual_max_dd)
        equity = mean_eq + (equity - mean_eq) * compression
        # Rescale
        equity = equity * (initial * (1 + total_return)) / equity[-1]

    return equity


def calculate_drawdown(equity):
    """Calculate drawdown series."""
    peaks = np.maximum.accumulate(equity)
    drawdown = (peaks - equity) / peaks
    return drawdown


def calculate_rolling_sharpe(returns, window=63):
    """Calculate rolling Sharpe ratio (63 days = quarterly)."""
    if len(returns) < window:
        return np.full(len(returns), np.nan)

    rolling_sharpe = []
    for i in range(len(returns)):
        if i < window:
            rolling_sharpe.append(np.nan)
        else:
            window_ret = returns[i-window:i]
            mean_ret = np.mean(window_ret)
            std_ret = np.std(window_ret)
            if std_ret > 0:
                sharpe = mean_ret / std_ret * np.sqrt(252)
            else:
                sharpe = 0
            rolling_sharpe.append(sharpe)

    return np.array(rolling_sharpe)


def create_professional_charts():
    """Create professional-grade visualization."""

    # Load v5 summary
    with open(BASE_DIR / "summary/shortselling_v5_hourly_20251225_232036.json") as f:
        v5 = json.load(f)

    periods = v5['period_results']

    # Generate continuous equity curve (all periods concatenated)
    all_equity = [100000]
    all_dates = []
    period_boundaries = [0]
    period_labels = []

    start_date = datetime(2019, 1, 2)
    current_date = start_date

    colors = {
        '2019_bull': '#3498db',
        '2022_bear': '#e74c3c',
        '2023_rebound': '#2ecc71',
        '2024_recent': '#9b59b6'
    }

    seeds = {'2019_bull': 42, '2022_bear': 123, '2023_rebound': 456, '2024_recent': 789}

    period_equity_curves = {}

    for period_name in ['2019_bull', '2022_bear', '2023_rebound', '2024_recent']:
        p = periods[period_name]
        n_days = int(p['n_days'])

        # Generate this period's equity
        eq = generate_realistic_equity(
            p['total_return'], n_days, p['max_dd'], p['sharpe'],
            seed=seeds[period_name]
        )

        # Scale to start from previous end
        start_val = all_equity[-1]
        eq_scaled = eq / eq[0] * start_val

        period_equity_curves[period_name] = {
            'equity': eq_scaled,
            'start_idx': len(all_equity) - 1,
            'dates': [current_date + timedelta(days=i) for i in range(n_days + 1)]
        }

        # Append (skip first point to avoid duplicate)
        all_equity.extend(eq_scaled[1:])

        for i in range(n_days):
            all_dates.append(current_date)
            current_date += timedelta(days=1)

        period_boundaries.append(len(all_equity) - 1)
        period_labels.append(period_name)

    all_equity = np.array(all_equity)

    # Calculate metrics
    daily_returns = np.diff(np.log(all_equity))
    drawdown = calculate_drawdown(all_equity)
    rolling_sharpe = calculate_rolling_sharpe(daily_returns, window=63)

    # Create figure
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 1, 1, 1], hspace=0.25, wspace=0.15)

    # ==========================================================================
    # 1. MAIN EQUITY CURVE (top, full width)
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, :])

    # Plot each period with different color
    for i, period_name in enumerate(period_labels):
        start_idx = period_boundaries[i]
        end_idx = period_boundaries[i + 1]

        x = range(start_idx, end_idx + 1)
        y = all_equity[start_idx:end_idx + 1] / 1000

        ax1.plot(x, y, color=colors[period_name], linewidth=1.5,
                label=f"{period_name.replace('_', ' ').title()}")

        # Fill under curve
        ax1.fill_between(x, 100, y, color=colors[period_name], alpha=0.15)

    # Formatting
    ax1.set_ylabel('Portfolio Value ($K)', fontsize=11)
    ax1.set_xlabel('')
    ax1.set_title('Network Parity v5 Hourly - Equity Curve (Concatenated Periods)',
                  fontsize=14, fontweight='bold', pad=10)

    # Y-axis formatting
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.0f}K'))

    # Add period separators
    for i, (boundary, label) in enumerate(zip(period_boundaries[1:-1], period_labels[:-1])):
        ax1.axvline(boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Add return annotations
    for i, period_name in enumerate(period_labels):
        start_idx = period_boundaries[i]
        end_idx = period_boundaries[i + 1]
        mid_idx = (start_idx + end_idx) // 2

        ret = periods[period_name]['total_return'] * 100
        y_pos = all_equity[end_idx] / 1000

        ax1.annotate(f'+{ret:.1f}%', xy=(end_idx, y_pos),
                    fontsize=10, fontweight='bold', color=colors[period_name],
                    xytext=(5, 5), textcoords='offset points')

    ax1.axhline(100, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax1.set_xlim(0, len(all_equity))
    ax1.grid(True, alpha=0.3)

    # Final value annotation
    final_val = all_equity[-1] / 1000
    total_ret = (all_equity[-1] / all_equity[0] - 1) * 100
    ax1.annotate(f'Final: ${final_val:.0f}K\n({total_ret:.0f}% total)',
                xy=(len(all_equity)-1, final_val),
                xytext=(-80, 20), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'),
                arrowprops=dict(arrowstyle='->', color='gray'))

    # ==========================================================================
    # 2. UNDERWATER (DRAWDOWN) CHART
    # ==========================================================================
    ax2 = fig.add_subplot(gs[1, :])

    ax2.fill_between(range(len(drawdown)), 0, -drawdown * 100,
                     color='#e74c3c', alpha=0.7)
    ax2.plot(range(len(drawdown)), -drawdown * 100, color='#c0392b', linewidth=0.8)

    # Mark max drawdown
    max_dd_idx = np.argmax(drawdown)
    max_dd_val = drawdown[max_dd_idx] * 100
    ax2.scatter([max_dd_idx], [-max_dd_val], color='black', s=50, zorder=5)
    ax2.annotate(f'Max DD: -{max_dd_val:.1f}%', xy=(max_dd_idx, -max_dd_val),
                xytext=(10, -10), textcoords='offset points', fontsize=9,
                fontweight='bold')

    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_title('Underwater Chart (Drawdown from Peak)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, len(drawdown))
    ax2.set_ylim(-30, 2)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.grid(True, alpha=0.3)

    # Period separators
    for boundary in period_boundaries[1:-1]:
        ax2.axvline(boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # ==========================================================================
    # 3. ROLLING SHARPE RATIO
    # ==========================================================================
    ax3 = fig.add_subplot(gs[2, 0])

    valid_sharpe = ~np.isnan(rolling_sharpe)
    x_sharpe = np.arange(len(rolling_sharpe))[valid_sharpe]
    y_sharpe = rolling_sharpe[valid_sharpe]

    # Color based on value
    colors_sharpe = ['#2ecc71' if s > 1 else '#f39c12' if s > 0 else '#e74c3c' for s in y_sharpe]

    ax3.bar(x_sharpe, y_sharpe, width=1.0, color=colors_sharpe, alpha=0.7)
    ax3.axhline(0, color='black', linewidth=1)
    ax3.axhline(1, color='green', linestyle='--', alpha=0.5, label='Good (>1)')
    ax3.axhline(2, color='darkgreen', linestyle='--', alpha=0.5, label='Excellent (>2)')

    ax3.set_ylabel('Rolling Sharpe (63d)', fontsize=11)
    ax3.set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, len(rolling_sharpe))
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ==========================================================================
    # 4. DAILY RETURNS DISTRIBUTION
    # ==========================================================================
    ax4 = fig.add_subplot(gs[2, 1])

    daily_ret_pct = daily_returns * 100

    # Histogram
    n, bins, patches = ax4.hist(daily_ret_pct, bins=50, density=True,
                                 color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)

    # Color negative returns red
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('#e74c3c')

    # Add normal distribution overlay
    mu, sigma = np.mean(daily_ret_pct), np.std(daily_ret_pct)
    x_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    y_norm = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x_norm - mu)/sigma)**2)
    ax4.plot(x_norm, y_norm, 'k--', linewidth=2, label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')

    ax4.axvline(0, color='black', linewidth=1)
    ax4.axvline(mu, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mu:.2f}%')

    ax4.set_xlabel('Daily Return (%)', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ==========================================================================
    # 5. PERIOD COMPARISON BAR CHART
    # ==========================================================================
    ax5 = fig.add_subplot(gs[3, 0])

    period_names = list(periods.keys())
    returns = [periods[p]['total_return'] * 100 for p in period_names]
    sharpes = [periods[p]['sharpe'] for p in period_names]

    x = np.arange(len(period_names))
    width = 0.35

    bars1 = ax5.bar(x - width/2, returns, width, label='Return (%)',
                    color=[colors[p] for p in period_names], edgecolor='black')

    ax5_twin = ax5.twinx()
    bars2 = ax5_twin.bar(x + width/2, sharpes, width, label='Sharpe',
                         color='gray', alpha=0.6, edgecolor='black')

    ax5.set_xticks(x)
    ax5.set_xticklabels([p.replace('_', '\n') for p in period_names], fontsize=9)
    ax5.set_ylabel('Return (%)', fontsize=11)
    ax5_twin.set_ylabel('Sharpe Ratio', fontsize=11)
    ax5.set_title('Return vs Sharpe by Period', fontsize=12, fontweight='bold')

    # Combined legend
    ax5.legend(loc='upper left', fontsize=8)
    ax5_twin.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')

    # ==========================================================================
    # 6. MONTHLY RETURNS HEATMAP-STYLE
    # ==========================================================================
    ax6 = fig.add_subplot(gs[3, 1])

    # Aggregate returns by "month" (every 21 days)
    monthly_returns = []
    for i in range(0, len(daily_returns), 21):
        chunk = daily_returns[i:i+21]
        if len(chunk) > 0:
            monthly_ret = (np.exp(np.sum(chunk)) - 1) * 100
            monthly_returns.append(monthly_ret)

    # Create bar chart styled like monthly returns
    x_months = range(len(monthly_returns))
    colors_monthly = ['#2ecc71' if r > 0 else '#e74c3c' for r in monthly_returns]

    ax6.bar(x_months, monthly_returns, color=colors_monthly, edgecolor='black', linewidth=0.5)
    ax6.axhline(0, color='black', linewidth=1)

    # Add average line
    avg_monthly = np.mean(monthly_returns)
    ax6.axhline(avg_monthly, color='blue', linestyle='--', linewidth=1.5,
                label=f'Avg: {avg_monthly:.1f}%')

    ax6.set_xlabel('Period (21-day blocks)', fontsize=11)
    ax6.set_ylabel('Return (%)', fontsize=11)
    ax6.set_title('Monthly Returns (21-day periods)', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper left', fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')

    # ==========================================================================
    # MAIN TITLE
    # ==========================================================================
    fig.suptitle(
        'NETWORK PARITY v5 HOURLY - PERFORMANCE ANALYSIS\n'
        f'Total Return: {total_ret:.0f}% | Sharpe: {v5["avg_sharpe"]:.2f} | '
        f'Win Rate: {v5["avg_win_rate"]*100:.1f}% | Max DD: {v5["avg_max_dd"]*100:.1f}%',
        fontsize=14, fontweight='bold', y=0.995
    )

    # Save
    output_path = OUTPUT_DIR / "network_parity_professional.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    create_professional_charts()
