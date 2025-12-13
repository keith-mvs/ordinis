#!/usr/bin/env python3
"""Options Strategy Calculator Template

Template for creating strategy-specific calculators.

Usage:
    python template_calculator.py --underlying SPY --price 450 [strategy-args]

Customize this template by:
1. Rename class StrategyTemplate to your strategy name (e.g., BullCallSpread)
2. Add strategy-specific parameters to __init__
3. Implement max_profit, max_loss, breakeven calculations
4. Add strategy-specific CLI arguments
5. Update docstrings and help text
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install dependencies: pip install numpy pandas scipy")
    sys.exit(1)


@dataclass
class StrategyTemplate:
    """Template for options strategy calculator.

    Customize this class for your specific strategy.

    Attributes:
        underlying_symbol: Ticker symbol (e.g., 'SPY', 'AAPL')
        underlying_price: Current stock price
        expiration_date: Option expiration date
        contracts: Number of contracts (default 1)
        volatility: Implied volatility as decimal (default 0.20 = 20%)
        risk_free_rate: Risk-free interest rate as decimal (default 0.05 = 5%)

    Example:
        >>> position = StrategyTemplate(
        ...     underlying_symbol="SPY",
        ...     underlying_price=450.00,
        ...     expiration_date=datetime.now() + timedelta(days=45)
        ... )
        >>> print(f"Max Profit: ${position.max_profit:.2f}")
    """

    # Core parameters
    underlying_symbol: str
    underlying_price: float
    expiration_date: datetime

    # Strategy-specific parameters
    # TODO: Add your strategy parameters here
    # Examples:
    # - For spreads: long_strike, short_strike, long_premium, short_premium
    # - For stock+option: stock_shares, put_strike, put_premium
    # - For straddles: call_strike, put_strike, call_premium, put_premium

    # Optional parameters
    contracts: int = 1
    volatility: float = 0.20
    risk_free_rate: float = 0.05

    def __post_init__(self):
        """Validate position parameters after initialization."""
        if self.underlying_price <= 0:
            raise ValueError(f"Underlying price must be positive, got {self.underlying_price}")

        if self.contracts <= 0:
            raise ValueError(f"Contracts must be positive, got {self.contracts}")

        if self.volatility < 0:
            raise ValueError(f"Volatility cannot be negative, got {self.volatility}")

        if self.expiration_date < datetime.now():
            raise ValueError(f"Expiration date {self.expiration_date} is in the past")

        # Add strategy-specific validation
        # TODO: Validate strategy parameters (e.g., strikes in correct order)

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
    def max_profit(self) -> float:
        """Calculate maximum profit for this strategy.

        TODO: Implement strategy-specific max profit calculation.

        Examples:
        - Bull call spread: (Short strike - Long strike - Net debit) × 100 × Contracts
        - Credit spread: Net credit × 100 × Contracts
        - Married put: Unlimited (stock can rise indefinitely)
        """
        # TODO: Replace with actual calculation
        raise NotImplementedError("Implement max_profit for your strategy")

    @property
    def max_loss(self) -> float:
        """Calculate maximum loss for this strategy.

        TODO: Implement strategy-specific max loss calculation.

        Examples:
        - Bull call spread: Net debit × 100 × Contracts
        - Credit spread: (Spread width - Net credit) × 100 × Contracts
        - Married put: (Stock cost + Put premium - Put strike) × Shares
        """
        # TODO: Replace with actual calculation
        raise NotImplementedError("Implement max_loss for your strategy")

    @property
    def breakeven_price(self) -> float:
        """Calculate breakeven stock price at expiration.

        TODO: Implement strategy-specific breakeven calculation.

        Examples:
        - Bull call spread: Long strike + Net debit
        - Bull put spread: Short put strike - Net credit
        - Straddle: Call strike ± Total premium
        """
        # TODO: Replace with actual calculation
        raise NotImplementedError("Implement breakeven_price for your strategy")

    def calculate_greeks(self) -> dict[str, float]:
        """Calculate Greeks for this position.

        Returns:
            Dictionary with delta, gamma, theta, vega, rho

        TODO: Implement Greeks calculations or import from black_scholes.py
        """
        # Placeholder - implement actual Greeks calculations
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }

    def get_analysis(self) -> dict:
        """Generate comprehensive position analysis.

        Returns:
            Dictionary containing position details, metrics, and Greeks
        """
        return {
            "position": {
                "symbol": self.underlying_symbol,
                "current_price": self.underlying_price,
                "contracts": self.contracts,
                "expiration": self.expiration_date.strftime("%Y-%m-%d"),
                "days_to_expiration": self.days_to_expiration,
            },
            "metrics": {
                "max_profit": self.max_profit,
                "max_loss": self.max_loss,
                "breakeven": self.breakeven_price,
                "risk_reward_ratio": (self.max_profit / self.max_loss if self.max_loss != 0 else 0),
            },
            "greeks": self.calculate_greeks(),
        }

    def print_analysis(self):
        """Print formatted analysis to console."""
        analysis = self.get_analysis()

        print(f"\n{'=' * 60}")
        print(f"  {self.underlying_symbol} {self.__class__.__name__} Analysis")
        print(f"{'=' * 60}\n")

        # Position details
        print("Position Details:")
        print(f"  Symbol: {analysis['position']['symbol']}")
        print(f"  Current Price: ${analysis['position']['current_price']:.2f}")
        print(f"  Contracts: {analysis['position']['contracts']}")
        print(f"  Expiration: {analysis['position']['expiration']}")
        print(f"  Days to Expiration: {analysis['position']['days_to_expiration']}")
        print()

        # Metrics
        print("Metrics:")
        print(f"  Max Profit: ${analysis['metrics']['max_profit']:.2f}")
        print(f"  Max Loss: ${analysis['metrics']['max_loss']:.2f}")
        print(f"  Breakeven: ${analysis['metrics']['breakeven']:.2f}")
        print(f"  Risk/Reward Ratio: {analysis['metrics']['risk_reward_ratio']:.2f}")
        print()

        # Greeks
        print("Greeks:")
        for greek, value in analysis["greeks"].items():
            print(f"  {greek.capitalize()}: {value:.4f}")
        print()


def main():
    """CLI entry point for calculator."""
    parser = argparse.ArgumentParser(
        description="Options Strategy Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python template_calculator.py --underlying SPY --price 450
  python template_calculator.py --underlying AAPL --price 175 --contracts 2
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
        help="Implied volatility (default: 0.20)",
    )

    # TODO: Add strategy-specific arguments
    # Examples:
    # parser.add_argument('--long-strike', type=float, help='Long strike price')
    # parser.add_argument('--short-strike', type=float, help='Short strike price')
    # parser.add_argument('--long-premium', type=float, help='Long option premium')
    # parser.add_argument('--short-premium', type=float, help='Short option premium')

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
        )

        # Print analysis
        position.print_analysis()

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
