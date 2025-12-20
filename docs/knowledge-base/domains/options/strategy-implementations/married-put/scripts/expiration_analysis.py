"""
Expiration Cycle Analysis

Compares 30-day, 60-day, and 90-day put options for married-put strategies.
Analyzes monthly costs, rolling frequency, and optimal expiration selection.

Author: Ordinis-1 Project
Version: 1.0.0
Python: 3.8+
"""

from married_put_calculator import MarriedPut
import numpy as np
import pandas as pd


def compare_expirations(
    stock_price: float,
    shares: int,
    put_strike: float,
    premiums_30d: float,
    premiums_60d: float,
    premiums_90d: float,
    transaction_cost: float = 0.65,
) -> dict:
    """
    Compare 30-day, 60-day, and 90-day put options.

    Args:
        stock_price: Current stock price
        shares: Number of shares to protect
        put_strike: Strike price (same for all expirations)
        premiums_30d: Premium for 30-day put
        premiums_60d: Premium for 60-day put
        premiums_90d: Premium for 90-day put
        transaction_cost: Transaction cost per contract

    Returns:
        Dictionary with comprehensive expiration comparison

    Example:
        >>> analysis = compare_expirations(
        ...     stock_price=50.00,
        ...     shares=100,
        ...     put_strike=48.00,
        ...     premiums_30d=1.85,
        ...     premiums_60d=2.95,
        ...     premiums_90d=3.85
        ... )
        >>> print(f"Best monthly cost: {analysis['recommendation']}")
    """
    # Create positions for each expiration
    pos_30d = MarriedPut(
        stock_price, shares, put_strike, premiums_30d, transaction_cost, days_to_expiration=30
    )
    pos_60d = MarriedPut(
        stock_price, shares, put_strike, premiums_60d, transaction_cost, days_to_expiration=60
    )
    pos_90d = MarriedPut(
        stock_price, shares, put_strike, premiums_90d, transaction_cost, days_to_expiration=90
    )

    # Calculate monthly costs (annualized then divided by 12)
    monthly_cost_30d = (premiums_30d * shares + transaction_cost) * (365 / 30) / 12
    monthly_cost_60d = (premiums_60d * shares + transaction_cost) * (365 / 60) / 12
    monthly_cost_90d = (premiums_90d * shares + transaction_cost) * (365 / 90) / 12

    # Annual costs (rolling strategy)
    rolls_per_year_30d = 365 / 30
    rolls_per_year_60d = 365 / 60
    rolls_per_year_90d = 365 / 90

    annual_cost_30d = (premiums_30d * shares + transaction_cost) * rolls_per_year_30d
    annual_cost_60d = (premiums_60d * shares + transaction_cost) * rolls_per_year_60d
    annual_cost_90d = (premiums_90d * shares + transaction_cost) * rolls_per_year_90d

    # Transaction frequency
    annual_transactions_30d = rolls_per_year_30d * 2  # Buy and sell each roll
    annual_transactions_60d = rolls_per_year_60d * 2
    annual_transactions_90d = rolls_per_year_90d * 2

    annual_txn_costs_30d = annual_transactions_30d * transaction_cost
    annual_txn_costs_60d = annual_transactions_60d * transaction_cost
    annual_txn_costs_90d = annual_transactions_90d * transaction_cost

    analysis = {
        "30d": {
            "premium": premiums_30d,
            "total_cost": pos_30d.total_cost,
            "monthly_cost": monthly_cost_30d,
            "annual_cost": annual_cost_30d,
            "annual_cost_pct": pos_30d.annualized_protection_cost,
            "rolls_per_year": rolls_per_year_30d,
            "annual_transactions": annual_transactions_30d,
            "annual_txn_costs": annual_txn_costs_30d,
            "protection_pct": pos_30d.protection_percentage,
            "max_loss": pos_30d.max_loss,
        },
        "60d": {
            "premium": premiums_60d,
            "total_cost": pos_60d.total_cost,
            "monthly_cost": monthly_cost_60d,
            "annual_cost": annual_cost_60d,
            "annual_cost_pct": pos_60d.annualized_protection_cost,
            "rolls_per_year": rolls_per_year_60d,
            "annual_transactions": annual_transactions_60d,
            "annual_txn_costs": annual_txn_costs_60d,
            "protection_pct": pos_60d.protection_percentage,
            "max_loss": pos_60d.max_loss,
        },
        "90d": {
            "premium": premiums_90d,
            "total_cost": pos_90d.total_cost,
            "monthly_cost": monthly_cost_90d,
            "annual_cost": annual_cost_90d,
            "annual_cost_pct": pos_90d.annualized_protection_cost,
            "rolls_per_year": rolls_per_year_90d,
            "annual_transactions": annual_transactions_90d,
            "annual_txn_costs": annual_txn_costs_90d,
            "protection_pct": pos_90d.protection_percentage,
            "max_loss": pos_90d.max_loss,
        },
    }

    # Determine best value
    monthly_costs = {"30d": monthly_cost_30d, "60d": monthly_cost_60d, "90d": monthly_cost_90d}

    best_monthly = min(monthly_costs, key=monthly_costs.get)

    # Calculate savings vs worst option
    worst_monthly = max(monthly_costs.values())
    best_cost = monthly_costs[best_monthly]
    annual_savings = (worst_monthly - best_cost) * 12

    analysis["recommendation"] = best_monthly
    analysis["best_monthly_cost"] = best_cost
    analysis["annual_savings_vs_worst"] = annual_savings
    analysis["stock_price"] = stock_price
    analysis["put_strike"] = put_strike
    analysis["shares"] = shares

    return analysis


def create_expiration_comparison_table(analysis: dict) -> pd.DataFrame:
    """
    Create formatted DataFrame from expiration analysis.

    Args:
        analysis: Output from compare_expirations()

    Returns:
        Formatted DataFrame for display
    """
    data = []

    for period in ["30d", "60d", "90d"]:
        data.append(
            {
                "Expiration": period,
                "Premium": f"${analysis[period]['premium']:.2f}",
                "Total Cost": f"${analysis[period]['total_cost']:.2f}",
                "Monthly Cost": f"${analysis[period]['monthly_cost']:.2f}",
                "Annual Cost": f"${analysis[period]['annual_cost']:.2f}",
                "Annual Cost %": f"{analysis[period]['annual_cost_pct']:.1f}%",
                "Rolls/Year": f"{analysis[period]['rolls_per_year']:.1f}",
                "Annual Txns": int(analysis[period]["annual_transactions"]),
                "Txn Costs": f"${analysis[period]['annual_txn_costs']:.2f}",
            }
        )

    return pd.DataFrame(data)


def calculate_roll_timing(current_days_remaining: int, expiration_cycle: str = "60d") -> dict:
    """
    Determine optimal timing to roll options forward.

    Args:
        current_days_remaining: Days left until current put expires
        expiration_cycle: Target cycle ('30d', '60d', or '90d')

    Returns:
        Dictionary with roll timing analysis

    Guidelines:
        - Roll at 21-30 days remaining to capture theta decay
        - Earlier rolling preserves more time value
        - Later rolling minimizes transaction frequency
    """
    target_days = {"30d": 30, "60d": 60, "90d": 90}
    target = target_days.get(expiration_cycle, 60)

    # Standard roll window: 21-30 days remaining
    optimal_roll_min = 21
    optimal_roll_max = 30

    if current_days_remaining > optimal_roll_max:
        recommendation = "HOLD"
        reason = f"Too early to roll. Wait until {optimal_roll_max} days remaining."
        action_date = None
    elif optimal_roll_min <= current_days_remaining <= optimal_roll_max:
        recommendation = "ROLL NOW"
        reason = f"Optimal roll window. Roll to new {expiration_cycle} put."
        action_date = "Today"
    else:  # < 21 days
        recommendation = "ROLL IMMEDIATELY"
        reason = f"Late for optimal roll. Roll to new {expiration_cycle} put urgently."
        action_date = "Today"

    return {
        "current_days_remaining": current_days_remaining,
        "target_cycle": expiration_cycle,
        "target_days": target,
        "recommendation": recommendation,
        "reason": reason,
        "action_date": action_date,
        "optimal_window": (optimal_roll_min, optimal_roll_max),
    }


def estimate_premium_by_expiration(
    base_premium: float, base_days: int, target_days: int, theta_decay_rate: float = 0.40
) -> float:
    """
    Estimate premium for different expiration using square-root-of-time rule.

    Args:
        base_premium: Known premium for base_days expiration
        base_days: Days to expiration for known premium
        target_days: Target days to expiration
        theta_decay_rate: Adjustment factor for theta decay (0.0-1.0)

    Returns:
        Estimated premium for target expiration

    Note:
        Uses simplified square-root approximation:
        Premium ∠√(Days to Expiration)

        Actual premiums affected by:
        - Implied volatility changes
        - Interest rates
        - Dividend expectations
    """
    if base_days <= 0 or target_days <= 0:
        raise ValueError("Days must be positive")

    # Square-root-of-time approximation
    time_ratio = np.sqrt(target_days / base_days)

    # Adjust for theta decay characteristics
    # Shorter expirations decay faster (higher theta)
    if target_days < base_days:
        # Going shorter: increase premium decay rate
        adjusted_ratio = time_ratio * (1 - theta_decay_rate * 0.1)
    else:
        # Going longer: use standard ratio
        adjusted_ratio = time_ratio

    estimated_premium = base_premium * adjusted_ratio

    return estimated_premium


if __name__ == "__main__":
    print("Expiration Cycle Analysis\n")
    print("=" * 80)

    # Example 1: Standard comparison
    print("\nExample 1: Standard Expiration Comparison")
    print("-" * 80)

    analysis = compare_expirations(
        stock_price=52.75,
        shares=100,
        put_strike=50.00,
        premiums_30d=1.85,
        premiums_60d=2.95,
        premiums_90d=3.85,
    )

    print(f"\nStock: ${analysis['stock_price']:.2f}")
    print(f"Strike: ${analysis['put_strike']:.2f}")
    print(f"Shares: {analysis['shares']}")

    print("\n" + create_expiration_comparison_table(analysis).to_string(index=False))

    print(f"\nRecommendation: {analysis['recommendation']} expiration")
    print(f"Best monthly cost: ${analysis['best_monthly_cost']:.2f}")
    print(f"Annual savings vs worst option: ${analysis['annual_savings_vs_worst']:.2f}")

    # Example 2: High-volatility stock
    print("\n" + "=" * 80)
    print("\nExample 2: High-Volatility Stock (Elevated Premiums)")
    print("-" * 80)

    high_vol_analysis = compare_expirations(
        stock_price=23.80,
        shares=1800,
        put_strike=22.00,
        premiums_30d=2.15,
        premiums_60d=3.45,
        premiums_90d=4.50,
    )

    print(f"\nStock: ${high_vol_analysis['stock_price']:.2f} (High IV)")
    print(f"Position: {high_vol_analysis['shares']} shares")

    hv_table = create_expiration_comparison_table(high_vol_analysis)
    print("\n" + hv_table.to_string(index=False))

    print("\nNote: High volatility increases premiums across all expirations")
    print(f"Recommended: {high_vol_analysis['recommendation']} expiration")
    print(f"Monthly cost: ${high_vol_analysis['best_monthly_cost']:.2f}")

    # Example 3: Roll timing
    print("\n" + "=" * 80)
    print("\nExample 3: Roll Timing Analysis")
    print("-" * 80)

    roll_scenarios = [45, 28, 21, 15, 7]

    print("\nCurrent Position: 60-day put option")
    print("\nDays Remaining | Recommendation | Reason")
    print("-" * 80)

    for days in roll_scenarios:
        timing = calculate_roll_timing(days, "60d")
        print(f"{days:13d} | {timing['recommendation']:14s} | {timing['reason']}")

    # Example 4: Premium estimation
    print("\n" + "=" * 80)
    print("\nExample 4: Estimating Premiums Across Expirations")
    print("-" * 80)

    base_premium = 2.95  # Known 60-day premium
    base_days = 60

    print(f"\nKnown: ${base_premium:.2f} premium for {base_days}-day put")
    print("\nEstimated premiums for other expirations:")

    for target_days in [30, 45, 90, 120]:
        estimated = estimate_premium_by_expiration(base_premium, base_days, target_days)
        print(f"  {target_days:3d} days: ${estimated:.2f}")

    print("\nNote: Estimates use square-root-of-time approximation.")
    print("Actual premiums may vary based on IV, rates, and dividends.")

    # Example 5: Cost comparison over 1 year
    print("\n" + "=" * 80)
    print("\nExample 5: Total Annual Cost Comparison")
    print("-" * 80)

    annual_analysis = compare_expirations(
        stock_price=175.50,
        shares=100,
        put_strike=170.00,
        premiums_30d=3.20,
        premiums_60d=5.10,
        premiums_90d=6.65,
    )

    print(f"\nStock: ${annual_analysis['stock_price']:.2f}")
    print(f"Strike: ${annual_analysis['put_strike']:.2f}")
    print("\nAnnual Cost Analysis:")

    for period in ["30d", "60d", "90d"]:
        data = annual_analysis[period]
        print(f"\n{period} Rolling Strategy:")
        print(f"  Premium per contract: ${data['premium']:.2f}")
        print(f"  Rolls per year: {data['rolls_per_year']:.1f}")
        print(f"  Annual premium cost: ${data['annual_cost']:.2f}")
        print(f"  Annual transaction costs: ${data['annual_txn_costs']:.2f}")
        print(f"  Total annual cost: ${data['annual_cost'] + data['annual_txn_costs']:.2f}")
        print(f"  As % of stock price: {data['annual_cost_pct']:.1f}%")

    savings = annual_analysis["annual_savings_vs_worst"]
    print(f"\nOptimal Choice: {annual_analysis['recommendation']}")
    print(f"Annual savings: ${savings:.2f} vs worst option")

    print("\n" + "=" * 80)
    print("Analysis complete!")
