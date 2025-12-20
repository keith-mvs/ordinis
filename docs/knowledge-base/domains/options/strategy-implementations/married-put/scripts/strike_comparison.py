"""
Strike Price Comparison Analysis

Compares different strike prices (ITM, ATM, OTM) for married-put strategies.
Helps traders evaluate the cost-benefit tradeoff of various protection levels.

Author: Ordinis-1 Project
Version: 1.0.0
Python: 3.8+
"""

from married_put_calculator import MarriedPut
import numpy as np
import pandas as pd


class StrikeComparison:
    """
    Analyzes and compares multiple strike prices for a married-put position.

    Attributes:
        stock_price (float): Current stock price
        shares (int): Number of shares to protect
        strike_data (List[Dict]): List of strike/premium combinations
    """

    def __init__(self, stock_price: float, shares: int = 100):
        """
        Initialize strike comparison analysis.

        Args:
            stock_price: Current stock price
            shares: Number of shares (default: 100)
        """
        self.stock_price = stock_price
        self.shares = shares
        self.strike_data: list[dict] = []
        self.positions: list[MarriedPut] = []

    def add_strike(
        self,
        strike_price: float,
        premium: float,
        days_to_expiration: int | None = None,
        label: str | None = None,
    ) -> None:
        """
        Add a strike price to compare.

        Args:
            strike_price: Put option strike price
            premium: Put option premium
            days_to_expiration: Days until expiration (optional)
            label: Custom label (e.g., "Deep OTM", "ATM")
        """
        # Determine moneyness if no label provided
        if label is None:
            pct_diff = ((strike_price - self.stock_price) / self.stock_price) * 100
            if pct_diff > 2:
                label = "ITM"
            elif pct_diff < -2:
                label = "OTM"
            else:
                label = "ATM"

        # Create position
        position = MarriedPut(
            stock_price=self.stock_price,
            shares=self.shares,
            put_strike=strike_price,
            put_premium=premium,
            days_to_expiration=days_to_expiration,
        )

        self.positions.append(position)
        self.strike_data.append(
            {
                "label": label,
                "strike": strike_price,
                "premium": premium,
                "days_to_expiration": days_to_expiration,
            }
        )

    def calculate_metrics(self) -> pd.DataFrame:
        """
        Calculate comparison metrics for all strikes.

        Returns:
            DataFrame with comprehensive comparison metrics
        """
        if not self.positions:
            raise ValueError("No strikes added. Use add_strike() first.")

        results = []

        for data, position in zip(self.strike_data, self.positions, strict=False):
            metrics = {
                "Label": data["label"],
                "Strike": data["strike"],
                "Premium": data["premium"],
                "Total Cost": position.total_cost,
                "Breakeven": position.breakeven_price,
                "Max Loss": position.max_loss,
                "Max Loss %": position.max_loss_percentage,
                "Protection %": position.protection_percentage,
                "Premium % of Stock": position.protection_cost_percentage,
            }

            # Add annualized cost if available
            if position.annualized_protection_cost is not None:
                metrics["Annual Cost %"] = position.annualized_protection_cost

            # Calculate P/L at key price points
            price_points = [
                self.stock_price * 0.80,  # -20%
                self.stock_price * 0.90,  # -10%
                self.stock_price * 1.00,  # unchanged
                self.stock_price * 1.10,  # +10%
                self.stock_price * 1.20,  # +20%
            ]

            for pct, price in zip([-20, -10, 0, 10, 20], price_points, strict=False):
                pl = position.calculate_pl_at_price(price)
                metrics[f"P/L at {pct:+d}%"] = pl

            results.append(metrics)

        return pd.DataFrame(results)

    def find_best_value(self) -> dict:
        """
        Identify the best value strike based on protection per dollar spent.

        Returns:
            Dictionary with best value analysis
        """
        df = self.calculate_metrics()

        # Calculate value score: Protection % / Premium % of Stock
        df["Value Score"] = df["Protection %"] / df["Premium % of Stock"]

        best_idx = df["Value Score"].idxmax()
        best_row = df.iloc[best_idx]

        return {
            "best_strike": best_row["Strike"],
            "label": best_row["Label"],
            "premium": best_row["Premium"],
            "value_score": best_row["Value Score"],
            "protection_pct": best_row["Protection %"],
            "cost_pct": best_row["Premium % of Stock"],
            "recommendation": f"{best_row['Label']} strike at ${best_row['Strike']:.2f} "
            f"offers best value with {best_row['Protection %']:.1f}% "
            f"protection for {best_row['Premium % of Stock']:.1f}% cost",
        }


def compare_strike_prices(
    stock_price: float,
    shares: int,
    strike_prices: list[float],
    premiums: list[float],
    days_to_expiration: int | None = None,
    labels: list[str] | None = None,
) -> StrikeComparison:
    """
    Quick comparison of multiple strike prices.

    Args:
        stock_price: Current stock price
        shares: Number of shares to protect
        strike_prices: List of strike prices to compare
        premiums: Corresponding premiums for each strike
        days_to_expiration: Days until expiration (optional)
        labels: Custom labels for each strike (optional)

    Returns:
        StrikeComparison object with analysis

    Example:
        >>> results = compare_strike_prices(
        ...     stock_price=50.00,
        ...     shares=100,
        ...     strike_prices=[45, 48, 50, 52],
        ...     premiums=[0.85, 1.75, 2.90, 4.30],
        ...     days_to_expiration=45
        ... )
        >>> df = results.calculate_metrics()
        >>> print(df[['Label', 'Strike', 'Premium', 'Protection %']])
    """
    if len(strike_prices) != len(premiums):
        raise ValueError("strike_prices and premiums must have same length")

    if labels is not None and len(labels) != len(strike_prices):
        raise ValueError("labels must match length of strike_prices")

    comparison = StrikeComparison(stock_price, shares)

    for i, (strike, premium) in enumerate(zip(strike_prices, premiums, strict=False)):
        label = labels[i] if labels else None
        comparison.add_strike(strike, premium, days_to_expiration, label)

    return comparison


def generate_strike_ladder(
    stock_price: float, num_strikes: int = 5, otm_pct: float = 10.0, itm_pct: float = 5.0
) -> list[float]:
    """
    Generate evenly spaced strike prices from OTM to ITM.

    Args:
        stock_price: Current stock price
        num_strikes: Number of strikes to generate
        otm_pct: Percentage below stock price for lowest strike
        itm_pct: Percentage above stock price for highest strike

    Returns:
        List of strike prices rounded to nearest $0.50

    Example:
        >>> strikes = generate_strike_ladder(50.00, num_strikes=5)
        >>> print(strikes)
        [45.0, 47.5, 50.0, 51.5, 52.5]
    """
    otm_strike = stock_price * (1 - otm_pct / 100)
    itm_strike = stock_price * (1 + itm_pct / 100)

    strikes = np.linspace(otm_strike, itm_strike, num_strikes)

    # Round to nearest $0.50 (common strike intervals)
    strikes = np.round(strikes * 2) / 2

    return strikes.tolist()


if __name__ == "__main__":
    print("Strike Price Comparison Analysis\n")
    print("=" * 70)

    # Example 1: Basic comparison
    print("\nExample 1: Comparing ITM, ATM, and OTM Puts")
    print("-" * 70)

    comparison = compare_strike_prices(
        stock_price=50.00,
        shares=100,
        strike_prices=[45, 48, 50, 52],
        premiums=[0.85, 1.75, 2.90, 4.30],
        days_to_expiration=45,
        labels=["Deep OTM", "OTM", "ATM", "ITM"],
    )

    df = comparison.calculate_metrics()

    print("\nKey Metrics Comparison:")
    print(
        df[
            ["Label", "Strike", "Premium", "Protection %", "Premium % of Stock", "Max Loss"]
        ].to_string(index=False)
    )

    print("\n\nProfit/Loss at Various Stock Prices:")
    print(
        df[
            [
                "Label",
                "Strike",
                "P/L at -20%",
                "P/L at -10%",
                "P/L at 0%",
                "P/L at +10%",
                "P/L at +20%",
            ]
        ].to_string(index=False)
    )

    # Example 2: Find best value
    print("\n" + "=" * 70)
    print("\nExample 2: Best Value Analysis")
    print("-" * 70)

    best = comparison.find_best_value()
    print(f"\nRecommendation: {best['recommendation']}")
    print(f"Value Score: {best['value_score']:.2f}")
    print(
        f"This strike provides {best['protection_pct']:.1f}% protection "
        f"for {best['cost_pct']:.1f}% of stock price"
    )

    # Example 3: Small-cap high volatility example
    print("\n" + "=" * 70)
    print("\nExample 3: High-Volatility Small-Cap Stock")
    print("-" * 70)
    print("Stock: $23.80, IV: 68%")

    high_vol_comparison = compare_strike_prices(
        stock_price=23.80,
        shares=1800,  # 18 contracts
        strike_prices=[20, 22, 24, 26],
        premiums=[1.25, 2.65, 4.25, 6.10],
        days_to_expiration=60,
        labels=["Deep OTM", "OTM", "ATM", "ITM"],
    )

    hv_df = high_vol_comparison.calculate_metrics()

    print("\nHigh-volatility strikes (note elevated premiums):")
    print(
        hv_df[["Label", "Strike", "Premium", "Total Cost", "Protection %", "Max Loss"]].to_string(
            index=False
        )
    )

    # Example 4: Generate strike ladder
    print("\n" + "=" * 70)
    print("\nExample 4: Auto-Generated Strike Ladder")
    print("-" * 70)

    stock_price = 175.50
    strikes = generate_strike_ladder(stock_price, num_strikes=7, otm_pct=10, itm_pct=5)

    print(f"\nStock Price: ${stock_price:.2f}")
    print("Generated Strikes:")
    for i, strike in enumerate(strikes, 1):
        pct_diff = ((strike - stock_price) / stock_price) * 100
        moneyness = "ITM" if strike > stock_price else ("ATM" if abs(pct_diff) < 0.5 else "OTM")
        print(f"  {i}. ${strike:6.2f} ({pct_diff:+5.1f}%) - {moneyness}")

    print("\n" + "=" * 70)
    print("\nAnalysis complete!")
