#!/usr/bin/env python3
"""
Options Strategy Calculator Template

Template for creating strategy-specific calculators with comprehensive analysis.

Usage:
    python strategy_calculator.py --underlying SPY --price 450 [strategy-args]

Customize this template by:
1. Rename class StrategyTemplate to your strategy name (e.g., BullCallSpread, IronCondor)
2. Add strategy-specific parameters to __init__ and dataclass fields
3. Implement max_profit, max_loss, breakeven calculations
4. Add strategy-specific CLI arguments in main()
5. Update docstrings and help text
6. Implement comparison and optimization functions

Author: Ordinis-1 Project
Version: 1.0.0
Python: 3.11+
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None
    print("Warning: pandas not installed. Some features may be limited.")


@dataclass
class StrategyTemplate:
    """
    Template for options strategy calculator.

    Customize this class for your specific strategy by:
    - Adding strategy-specific attributes
    - Implementing property methods for calculations
    - Adding validation in __post_init__

    Attributes:
        underlying_symbol: Ticker symbol (e.g., 'SPY', 'AAPL')
        underlying_price: Current stock price
        expiration_date: Option expiration date
        contracts: Number of contracts (default 1)
        volatility: Implied volatility as decimal (default 0.20 = 20%)
        risk_free_rate: Risk-free interest rate as decimal (default 0.05 = 5%)

        # TODO: Add your strategy-specific parameters here
        # Examples:
        # - For vertical spreads: long_strike, short_strike, long_premium, short_premium
        # - For stock+option: stock_shares, stock_price, option_strike, option_premium
        # - For multi-leg: strike1, strike2, strike3, premium1, premium2, premium3
        # - For straddles/strangles: call_strike, put_strike, call_premium, put_premium

    Example:
        >>> position = StrategyTemplate(
        ...     underlying_symbol="SPY",
        ...     underlying_price=450.00,
        ...     expiration_date=datetime.now() + timedelta(days=45),
        ...     contracts=1
        ... )
        >>> print(f"Total Cost: ${position.total_cost:.2f}")
    """

    # Core parameters (required for all strategies)
    underlying_symbol: str
    underlying_price: float
    expiration_date: datetime

    # TODO: Add strategy-specific parameters here
    # Example for vertical spread:
    # long_strike: float
    # short_strike: float
    # long_premium: float
    # short_premium: float

    # Example for stock+option:
    # stock_shares: int
    # option_strike: float
    # option_premium: float

    # Optional parameters (common across strategies)
    contracts: int = 1
    volatility: float = 0.20
    risk_free_rate: float = 0.05
    transaction_cost: float = 0.65  # Per contract

    def __post_init__(self):
        """Validate position parameters after initialization."""
        # Core validations
        if self.underlying_price <= 0:
            raise ValueError(f"Underlying price must be positive, got {self.underlying_price}")

        if self.contracts <= 0:
            raise ValueError(f"Contracts must be positive, got {self.contracts}")

        if self.volatility < 0:
            raise ValueError(f"Volatility cannot be negative, got {self.volatility}")

        if self.expiration_date < datetime.now():
            raise ValueError(f"Expiration date {self.expiration_date} is in the past")

        # TODO: Add strategy-specific validation here
        # Examples:
        # - Verify strikes are in correct order (long < short for call spread)
        # - Verify premiums are consistent (long > short for debit spread)
        # - Verify shares are multiple of 100 for stock+option strategies
        # - Verify strikes are properly spaced for butterflies/condors

    @property
    def days_to_expiration(self) -> int:
        """Calculate days until expiration."""
        delta = self.expiration_date - datetime.now()
        return max(0, delta.days)

    @property
    def time_to_expiration(self) -> float:
        """Calculate time to expiration in years."""
        return self.days_to_expiration / 365.0

    @property
    def total_transaction_cost(self) -> float:
        """
        Calculate total transaction costs for all legs.

        TODO: Adjust multiplier based on number of legs in your strategy
        - Vertical spreads: 2 legs
        - Stock+option: 1 leg (no transaction cost on stock)
        - Straddles/Strangles: 2 legs
        - Butterflies: 4 legs
        - Iron Condors: 4 legs
        """
        # TODO: Update the multiplier for your strategy
        num_legs = 2  # Replace with actual number of option legs
        return self.transaction_cost * num_legs * self.contracts

    @property
    def total_cost(self) -> float:
        """
        Calculate total investment or credit received.

        TODO: Implement for your strategy

        Examples:
        - Debit spread: (long_premium - short_premium) × 100 × contracts + transaction_cost
        - Credit spread: (short_premium - long_premium) × 100 × contracts (negative = credit)
        - Stock+option: (stock_price × shares) + (option_premium × 100 × contracts) + transaction_cost
        - Straddle: (call_premium + put_premium) × 100 × contracts + transaction_cost
        """
        # TODO: Replace with actual calculation
        raise NotImplementedError("Implement total_cost for your strategy")

    @property
    def max_profit(self) -> float:
        """
        Calculate maximum profit for this strategy.

        TODO: Implement strategy-specific max profit calculation.

        Examples:
        - Bull call spread: (Short strike - Long strike - Net debit) × 100 × Contracts
        - Credit spread: Net credit × 100 × Contracts
        - Married put: Unlimited (stock can rise indefinitely)
        - Iron condor: Net credit × 100 × Contracts
        - Long straddle: Unlimited
        - Short straddle: Net credit × 100 × Contracts
        """
        # TODO: Replace with actual calculation
        raise NotImplementedError("Implement max_profit for your strategy")

    @property
    def max_loss(self) -> float:
        """
        Calculate maximum loss for this strategy.

        TODO: Implement strategy-specific max loss calculation.

        Examples:
        - Bull call spread: Net debit × 100 × Contracts
        - Credit spread: (Spread width - Net credit) × 100 × Contracts
        - Married put: (Stock cost + Put premium - Put strike) × Shares
        - Iron condor: (Wing width - Net credit) × 100 × Contracts
        - Long straddle: Total premium paid × 100 × Contracts
        - Short straddle: Unlimited
        """
        # TODO: Replace with actual calculation
        raise NotImplementedError("Implement max_loss for your strategy")

    @property
    def breakeven_price(self) -> float:
        """
        Calculate breakeven stock price at expiration.

        TODO: Implement strategy-specific breakeven calculation.
        Note: Some strategies have TWO breakevens (straddles, strangles, butterflies)

        Examples:
        - Bull call spread: Long strike + Net debit
        - Bull put spread: Short put strike - Net credit
        - Married put: Stock price + Put premium + (Transaction cost / Shares)
        - Straddle: Strike ± Total premium (TWO breakevens)
        - Iron condor: Short strikes ± Net credit (TWO breakevens)
        """
        # TODO: Replace with actual calculation
        raise NotImplementedError("Implement breakeven_price for your strategy")

    @property
    def breakeven_prices(self) -> list[float]:
        """
        Calculate ALL breakeven prices (for strategies with multiple breakevens).

        TODO: Implement for strategies with 2 breakevens (straddles, strangles, etc.)

        Returns:
            List of breakeven prices (lower to higher)
        """
        # For simple strategies with one breakeven, return single-item list
        try:
            return [self.breakeven_price]
        except NotImplementedError:
            # For complex strategies, calculate multiple breakevens
            raise NotImplementedError(
                "Implement breakeven_prices for strategies with multiple breakevens"
            )

    @property
    def risk_reward_ratio(self) -> float:
        """
        Calculate risk/reward ratio.

        Returns:
            Ratio of max_profit to max_loss
            Returns None if max_loss is unlimited
        """
        try:
            if self.max_loss == 0:
                return float("inf")
            return self.max_profit / self.max_loss
        except NotImplementedError:
            return None

    @property
    def max_loss_percentage(self) -> float:
        """
        Maximum loss as percentage of total investment.

        Returns:
            Percentage of capital at risk
        """
        try:
            return (self.max_loss / abs(self.total_cost)) * 100
        except (NotImplementedError, ZeroDivisionError):
            return None

    def calculate_pl_at_price(self, final_stock_price: float) -> float:
        """
        Calculate profit/loss at a specific stock price at expiration.

        TODO: Implement strategy-specific P/L calculation

        Args:
            final_stock_price: Stock price to evaluate

        Returns:
            Total profit (positive) or loss (negative) in dollars

        Example Implementation (Bull Call Spread):
            if final_stock_price <= self.long_strike:
                # Both options expire worthless
                return -self.total_cost
            elif final_stock_price >= self.short_strike:
                # Max profit reached
                return self.max_profit
            else:
                # In between strikes
                intrinsic = (final_stock_price - self.long_strike) * 100 * self.contracts
                return intrinsic - self.total_cost
        """
        # TODO: Implement P/L calculation
        raise NotImplementedError("Implement calculate_pl_at_price for your strategy")

    def calculate_pl_table(
        self, price_range: tuple[float, float] | None = None, num_points: int = 20
    ) -> list[dict[str, float]]:
        """
        Generate profit/loss table across range of stock prices.

        Args:
            price_range: (min_price, max_price) tuple, defaults to ±30% of stock price
            num_points: Number of price points to calculate

        Returns:
            List of dicts with 'stock_price', 'pl_dollars', 'pl_percent', 'pl_per_contract'
        """
        if price_range is None:
            min_price = self.underlying_price * 0.70
            max_price = self.underlying_price * 1.30
        else:
            min_price, max_price = price_range

        prices = np.linspace(min_price, max_price, num_points)

        table = []
        for price in prices:
            try:
                pl_dollars = self.calculate_pl_at_price(price)
                pl_percent = (pl_dollars / abs(self.total_cost)) * 100
                pl_per_contract = pl_dollars / self.contracts

                table.append(
                    {
                        "stock_price": float(price),
                        "pl_dollars": float(pl_dollars),
                        "pl_percent": float(pl_percent),
                        "pl_per_contract": float(pl_per_contract),
                    }
                )
            except NotImplementedError:
                # If calculate_pl_at_price not implemented, skip
                break

        return table

    def get_metrics_summary(self) -> dict[str, float]:
        """
        Get comprehensive summary of position metrics.

        Returns:
            Dictionary containing all key metrics
        """
        metrics = {
            "underlying_symbol": self.underlying_symbol,
            "underlying_price": self.underlying_price,
            "contracts": self.contracts,
            "days_to_expiration": self.days_to_expiration,
            "volatility": self.volatility,
        }

        # Add calculated metrics (with error handling for NotImplementedError)
        try:
            metrics["total_cost"] = self.total_cost
        except NotImplementedError:
            pass

        try:
            metrics["max_profit"] = self.max_profit
        except NotImplementedError:
            pass

        try:
            metrics["max_loss"] = self.max_loss
        except NotImplementedError:
            pass

        try:
            metrics["breakeven_price"] = self.breakeven_price
        except NotImplementedError:
            pass

        try:
            metrics["breakeven_prices"] = self.breakeven_prices
        except NotImplementedError:
            pass

        if self.risk_reward_ratio is not None:
            metrics["risk_reward_ratio"] = self.risk_reward_ratio

        if self.max_loss_percentage is not None:
            metrics["max_loss_percentage"] = self.max_loss_percentage

        # TODO: Add strategy-specific metrics here
        # Examples:
        # - protection_percentage (for married put)
        # - spread_width (for vertical spreads)
        # - profit_zone_width (for butterflies/condors)
        # - implied_move (for straddles/strangles)

        return metrics

    def print_analysis(self):
        """Print formatted analysis to console."""
        metrics = self.get_metrics_summary()

        print(f"\n{'=' * 60}")
        print(f"  {self.underlying_symbol} {self.__class__.__name__} Analysis")
        print(f"{'=' * 60}\n")

        # Position details
        print("Position Details:")
        print(f"  Symbol: {metrics.get('underlying_symbol', 'N/A')}")
        print(f"  Current Price: ${metrics.get('underlying_price', 0):.2f}")
        print(f"  Contracts: {metrics.get('contracts', 0)}")
        print(f"  Days to Expiration: {metrics.get('days_to_expiration', 0)}")
        print()

        # Metrics
        print("Metrics:")
        if "total_cost" in metrics:
            cost_label = "Net Debit" if metrics["total_cost"] > 0 else "Net Credit"
            print(f"  {cost_label}: ${abs(metrics['total_cost']):.2f}")

        if "max_profit" in metrics:
            print(f"  Max Profit: ${metrics['max_profit']:.2f}")

        if "max_loss" in metrics:
            print(f"  Max Loss: ${metrics['max_loss']:.2f}")

        if "breakeven_price" in metrics:
            print(f"  Breakeven: ${metrics['breakeven_price']:.2f}")
        elif "breakeven_prices" in metrics:
            breakevens = metrics["breakeven_prices"]
            print(f"  Breakevens: ${breakevens[0]:.2f} - ${breakevens[-1]:.2f}")

        if "risk_reward_ratio" in metrics:
            print(f"  Risk/Reward Ratio: {metrics['risk_reward_ratio']:.2f}")

        print()

    def __str__(self) -> str:
        """String representation of position."""
        return (
            f"{self.__class__.__name__}("
            f"symbol={self.underlying_symbol}, "
            f"price=${self.underlying_price:.2f}, "
            f"contracts={self.contracts})"
        )

    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()


def compare_positions(positions: list[StrategyTemplate]) -> dict:
    """
    Compare multiple strategy positions side by side.

    Args:
        positions: List of StrategyTemplate instances to compare

    Returns:
        Dictionary with comparison metrics

    Example:
        >>> pos1 = StrategyTemplate(...)
        >>> pos2 = StrategyTemplate(...)
        >>> comparison = compare_positions([pos1, pos2])
    """
    if not positions:
        raise ValueError("Must provide at least one position")

    comparison = {"positions": len(positions), "metrics": []}

    for i, pos in enumerate(positions, 1):
        metrics = pos.get_metrics_summary()
        metrics["position_number"] = i
        comparison["metrics"].append(metrics)

    return comparison


def main():
    """CLI entry point for calculator."""
    parser = argparse.ArgumentParser(
        description="Options Strategy Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python strategy_calculator.py --underlying SPY --price 450
  python strategy_calculator.py --underlying AAPL --price 175 --contracts 2

TODO: Add strategy-specific examples
        """,
    )

    # Required arguments
    parser.add_argument("--underlying", required=True, help="Underlying ticker symbol (e.g., SPY)")
    parser.add_argument(
        "--price",
        type=float,
        required=True,
        help="Current underlying stock price",
    )

    # Optional common arguments
    parser.add_argument("--contracts", type=int, default=1, help="Number of contracts (default: 1)")
    parser.add_argument(
        "--dte",
        type=int,
        default=45,
        help="Days to expiration (default: 45)",
    )
    parser.add_argument(
        "--volatility",
        type=float,
        default=0.20,
        help="Implied volatility as decimal (default: 0.20 = 20%%)",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.05,
        help="Risk-free rate as decimal (default: 0.05 = 5%%)",
    )

    # TODO: Add strategy-specific arguments here
    # Examples:
    # parser.add_argument('--long-strike', type=float, help='Long strike price')
    # parser.add_argument('--short-strike', type=float, help='Short strike price')
    # parser.add_argument('--long-premium', type=float, help='Long option premium')
    # parser.add_argument('--short-premium', type=float, help='Short option premium')
    # parser.add_argument('--put-strike', type=float, help='Put strike price')
    # parser.add_argument('--call-strike', type=float, help='Call strike price')
    # parser.add_argument('--shares', type=int, help='Number of shares')

    args = parser.parse_args()

    # Calculate expiration date
    expiration_date = datetime.now() + timedelta(days=args.dte)

    try:
        # Create position
        # TODO: Update with strategy-specific parameters
        position = StrategyTemplate(
            underlying_symbol=args.underlying,
            underlying_price=args.price,
            expiration_date=expiration_date,
            contracts=args.contracts,
            volatility=args.volatility,
            risk_free_rate=args.risk_free_rate,
        )

        # Print analysis
        position.print_analysis()

        # Generate P/L table if available
        pl_table = position.calculate_pl_table()
        if pl_table:
            print("\nProfit/Loss at Various Prices:")
            print(f"{'Price':<10} {'P/L ($)':<12} {'P/L (%)':<10}")
            print("-" * 32)
            for row in pl_table[::4]:  # Show every 4th row
                print(
                    f"${row['stock_price']:<9.2f} "
                    f"${row['pl_dollars']:<11.2f} "
                    f"{row['pl_percent']:<9.1f}%"
                )

    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
        print(
            f"\nError: {e}\n"
            "This is a template - implement the required methods for your strategy.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
