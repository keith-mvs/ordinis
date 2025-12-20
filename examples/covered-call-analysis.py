"""
Covered Call Strategy Analysis Example

Demonstrates covered call P&L scenarios and opportunity analysis.

Shows:
- Covered call payoff diagrams
- Opportunity metrics calculation
- Different strike/premium scenarios
- ROI and yield analysis

Usage:
    python examples/covered_call_analysis.py

Author: Ordinis Project
"""

from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.options.covered_call import CoveredCallStrategy


def print_header(title):
    """Print section header."""
    print()
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)
    print()


def print_analysis(analysis):
    """Print opportunity analysis."""
    print("Opportunity Analysis:")
    print("-" * 80)
    print(f"Underlying Price:          ${analysis['underlying_price']:.2f}")
    print(f"Call Strike:               ${analysis['call_strike']:.2f}")
    print(f"Call Premium:              ${analysis['call_premium']:.2f}")
    print(f"Call Delta:                {analysis['call_delta']:.2f}")
    print(f"Days to Expiration:        {analysis['days_to_expiration']}")
    print()
    print("Profit/Loss Metrics:")
    print(f"  Max Profit:              ${analysis['max_profit']:.2f}")
    print(f"  Max Loss:                ${analysis['max_loss']:.2f}")
    print(f"  Breakeven:               ${analysis['breakeven']:.2f}")
    print()
    print("Return Metrics:")
    print(f"  Premium Yield:           {analysis['premium_yield']*100:.2f}%")
    print(f"  Annualized Yield:        {analysis['annualized_yield']*100:.2f}%")
    print(f"  Downside Protection:     {analysis['downside_protection_pct']*100:.2f}%")
    print(
        f"  Upside Potential:        ${analysis['upside_potential']:.2f} ({analysis['upside_potential_pct']*100:.1f}%)"
    )
    print()
    print("Risk Metrics:")
    print(f"  Probability of Profit:   {analysis['prob_profit']*100:.1f}%")
    print(f"  Return on Risk:          {analysis['return_on_risk']*100:.2f}%")
    print()
    print(f"Meets Strategy Criteria:   {'[YES]' if analysis['meets_criteria'] else '[NO]'}")
    print()


def print_payoff_table(strategy, stock_entry, call_strike, call_premium):
    """Print P&L table across different stock prices."""
    print("Payoff Diagram:")
    print("-" * 80)
    print(f"{'Stock Price':<15} {'Stock P&L':<15} {'Call P&L':<15} {'Total P&L':<15} {'ROI':<10}")
    print("-" * 80)

    # Generate price points from -20% to +30%
    prices = [stock_entry * (1 + pct / 100) for pct in range(-20, 35, 5)]

    for price in prices:
        payoff = strategy.calculate_payoff(price, stock_entry, call_strike, call_premium)

        stock_pnl = payoff["stock_pnl"]
        call_pnl = payoff["call_pnl"]
        total_pnl = payoff["total_pnl"]
        roi = payoff["roi"]
        called = " (Called)" if payoff["stock_called_away"] else ""

        print(
            f"${price:<14.2f} ${stock_pnl:<14.2f} ${call_pnl:<14.2f} ${total_pnl:<14.2f} {roi*100:<9.1f}%{called}"
        )

    print()


def scenario_conservative():
    """Conservative covered call scenario."""
    print_header("Scenario 1: Conservative Covered Call")

    print("Setup:")
    print("  Buy 100 shares AAPL at $150.00")
    print("  Sell 1x AAPL Jan 2026 $160 Call for $3.50")
    print("  45 days to expiration")
    print("  Delta: 0.30 (30% probability ITM)")
    print()

    strategy = CoveredCallStrategy(name="Conservative CC")

    # Analyze opportunity
    analysis = strategy.analyze_opportunity(
        underlying_price=150.0,
        call_strike=160.0,
        call_premium=3.50,
        call_delta=0.30,
        days_to_expiration=45,
    )

    print_analysis(analysis)

    # Show payoff table
    print_payoff_table(strategy, stock_entry=150.0, call_strike=160.0, call_premium=3.50)

    print("Strategy Assessment:")
    print("  + Good annualized yield (28.4%)")
    print("  + Low probability of assignment (30%)")
    print("  + Significant upside capture ($10 stock gain + $3.50 premium)")
    print("  - Lower premium collection")
    print()


def scenario_aggressive():
    """Aggressive covered call scenario."""
    print_header("Scenario 2: Aggressive Covered Call")

    print("Setup:")
    print("  Buy 100 shares AAPL at $150.00")
    print("  Sell 1x AAPL Jan 2026 $152 Call for $5.00")
    print("  45 days to expiration")
    print("  Delta: 0.55 (55% probability ITM)")
    print()

    strategy = CoveredCallStrategy(name="Aggressive CC")

    # Analyze opportunity
    analysis = strategy.analyze_opportunity(
        underlying_price=150.0,
        call_strike=152.0,
        call_premium=5.00,
        call_delta=0.55,
        days_to_expiration=45,
    )

    print_analysis(analysis)

    # Show payoff table
    print_payoff_table(strategy, stock_entry=150.0, call_strike=152.0, call_premium=5.00)

    print("Strategy Assessment:")
    print("  + High premium collection ($5.00)")
    print("  + Excellent annualized yield (40.6%)")
    print("  - High probability of assignment (55%)")
    print("  - Limited upside capture ($2 stock gain + $5 premium = $7 total)")
    print("  [X] Delta too high - DOES NOT meet default criteria (max delta 0.40)")
    print()


def scenario_monthly_income():
    """Monthly income covered call scenario."""
    print_header("Scenario 3: Monthly Income Strategy")

    print("Setup:")
    print("  Buy 100 shares AAPL at $150.00")
    print("  Sell 1x AAPL 30-day $155 Call for $2.00")
    print("  30 days to expiration")
    print("  Delta: 0.35 (35% probability ITM)")
    print()

    strategy = CoveredCallStrategy(name="Monthly Income CC")

    # Analyze opportunity
    analysis = strategy.analyze_opportunity(
        underlying_price=150.0,
        call_strike=155.0,
        call_premium=2.00,
        call_delta=0.35,
        days_to_expiration=30,
    )

    print_analysis(analysis)

    # Show payoff table
    print_payoff_table(strategy, stock_entry=150.0, call_strike=155.0, call_premium=2.00)

    print("Strategy Assessment:")
    print("  + Monthly premium collection strategy")
    print("  + Annualized yield (24.3%) meets minimum (12%)")
    print("  + Delta within target range (0.25-0.40)")
    print("  + Can roll monthly for consistent income")
    print("  [OK] Meets all strategy criteria")
    print()


def scenario_comparison():
    """Compare different covered call approaches."""
    print_header("Scenario Comparison: Conservative vs Aggressive vs Monthly")

    strategy = CoveredCallStrategy(name="Comparison")

    scenarios = [
        {
            "name": "Conservative (45 DTE, $160 strike)",
            "strike": 160.0,
            "premium": 3.50,
            "delta": 0.30,
            "dte": 45,
        },
        {
            "name": "Aggressive (45 DTE, $152 strike)",
            "strike": 152.0,
            "premium": 5.00,
            "delta": 0.55,
            "dte": 45,
        },
        {
            "name": "Monthly (30 DTE, $155 strike)",
            "strike": 155.0,
            "premium": 2.00,
            "delta": 0.35,
            "dte": 30,
        },
    ]

    print(f"{'Metric':<35} {'Conservative':<15} {'Aggressive':<15} {'Monthly':<15}")
    print("-" * 80)

    analyses = [
        strategy.analyze_opportunity(150.0, s["strike"], s["premium"], s["delta"], s["dte"])
        for s in scenarios
    ]

    metrics = [
        ("Max Profit", "max_profit", "${:.2f}"),
        ("Breakeven", "breakeven", "${:.2f}"),
        ("Annualized Yield", "annualized_yield", "{:.1f}%", 100),
        ("Probability of Profit", "prob_profit", "{:.1f}%", 100),
        ("Upside Potential", "upside_potential", "${:.2f}"),
        ("Downside Protection", "downside_protection_pct", "{:.1f}%", 100),
        ("Meets Criteria", "meets_criteria", "{}"),
    ]

    for metric_name, key, fmt, *multiplier in metrics:
        mult = multiplier[0] if multiplier else 1
        values = []
        for analysis in analyses:
            val = analysis[key]
            if isinstance(val, bool):
                values.append("[YES]" if val else "[NO]")
            else:
                values.append(fmt.format(val * mult))

        print(f"{metric_name:<35} {values[0]:<15} {values[1]:<15} {values[2]:<15}")

    print()
    print("Conclusion:")
    print("  • Conservative: Best for growth stocks, low assignment risk")
    print(
        "  • Aggressive: Highest income, but frequent assignment (avoid unless criteria adjusted)"
    )
    print("  • Monthly: Balanced approach, repeatable monthly income")
    print()


def main():
    """Run covered call analysis examples."""
    print()
    print("*" * 80)
    print("Covered Call Strategy Analysis".center(80))
    print("*" * 80)

    # Run scenarios
    scenario_conservative()
    input("Press Enter to continue...")

    scenario_aggressive()
    input("Press Enter to continue...")

    scenario_monthly_income()
    input("Press Enter to continue...")

    scenario_comparison()

    print()
    print("=" * 80)
    print("Analysis Complete!".center(80))
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
