"""
Quick Covered Call Demo (Non-Interactive)

Demonstrates covered call analysis without user input.
"""

from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.options.covered_call import CoveredCallStrategy


def main():
    """Run quick demo."""
    print("\n" + "=" * 80)
    print("Covered Call Strategy - Quick Demo".center(80))
    print("=" * 80 + "\n")

    strategy = CoveredCallStrategy(name="Demo CC")

    # Analyze a conservative covered call
    print("Conservative Covered Call Scenario:")
    print("-" * 80)
    print("Stock: $150, Sell $160 Call for $3.50, 45 DTE, Delta 0.30\n")

    analysis = strategy.analyze_opportunity(
        underlying_price=150.0,
        call_strike=160.0,
        call_premium=3.50,
        call_delta=0.30,
        days_to_expiration=45,
    )

    print(f"Max Profit:        ${analysis['max_profit']:.2f}")
    print(f"Breakeven:         ${analysis['breakeven']:.2f}")
    print(f"Annualized Yield:  {analysis['annualized_yield']*100:.1f}%")
    print(f"Prob of Profit:    {analysis['prob_profit']*100:.0f}%")
    print(f"Meets Criteria:    {'[YES]' if analysis['meets_criteria'] else '[NO]'}")
    print()

    # Show P&L at key price points
    print("Payoff at Key Prices:")
    print("-" * 80)
    print(f"{'Stock Price':<15} {'Total P&L':<15} {'ROI':<10} {'Status'}")
    print("-" * 80)

    for price in [140, 145, 150, 155, 160, 165]:
        payoff = strategy.calculate_payoff(price, 150.0, 160.0, 3.50)
        status = "(Called)" if payoff["stock_called_away"] else ""
        print(f"${price:<14.2f} ${payoff['total_pnl']:<14.2f} {payoff['roi']*100:<9.1f}% {status}")

    print("\n" + "=" * 80)
    print("Demo Complete!".center(80))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
