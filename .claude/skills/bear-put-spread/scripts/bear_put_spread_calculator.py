#!/usr/bin/env python3
"""Bear Put Spread Strategy Calculator

Complete implementation for analyzing bear put spread options strategies.
A bear put spread is a bearish vertical spread using put options.

Structure:
- Buy 1 put at higher strike (more expensive)
- Sell 1 put at lower strike (cheaper)
- Net debit spread with defined risk and limited profit

Usage:
    python bear_put_spread_calculator.py --underlying SPY --price 450 \
        --long-strike 450 --short-strike 440 \
        --long-premium 8.50 --short-premium 4.20
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install dependencies: pip install numpy pandas scipy")
    sys.exit(1)


@dataclass
class BearPutSpread:
    """Bear put spread position with complete analysis capabilities.

    A bearish vertical spread that profits from downward price movement.
    Maximum profit occurs when underlying falls below short put strike.

    Attributes:
        underlying_symbol: Ticker symbol
        underlying_price: Current stock price
        long_strike: Strike price of long put (higher)
        short_strike: Strike price of short put (lower)
        long_premium: Premium paid for long put
        short_premium: Premium received for short put
        expiration_date: Option expiration date
        contracts: Number of contracts (default 1)
        volatility: Implied volatility (default 0.20)
        risk_free_rate: Risk-free interest rate (default 0.05)

    Example:
        >>> position = BearPutSpread(
        ...     underlying_symbol="SPY",
        ...     underlying_price=450.00,
        ...     long_strike=450.00,
        ...     short_strike=440.00,
        ...     long_premium=8.50,
        ...     short_premium=4.20,
        ...     expiration_date=datetime.now() + timedelta(days=45)
        ... )
        >>> print(f"Max Profit: ${position.max_profit:.2f}")
    """

    underlying_symbol: str
    underlying_price: float
    long_strike: float
    short_strike: float
    long_premium: float
    short_premium: float
    expiration_date: datetime
    contracts: int = 1
    volatility: float = 0.20
    risk_free_rate: float = 0.05

    def __post_init__(self):
        """Validate position parameters."""
        if self.long_strike <= self.short_strike:
            raise ValueError(
                f"Long strike ({self.long_strike}) must be greater than "
                f"short strike ({self.short_strike}) for bear put spread"
            )
        if self.long_premium <= self.short_premium:
            raise ValueError(
                "Long premium should exceed short premium for debit spread"
            )
        if self.underlying_price <= 0:
            raise ValueError(
                f"Underlying price must be positive, got {self.underlying_price}"
            )
        if self.contracts <= 0:
            raise ValueError(f"Contracts must be positive, got {self.contracts}")
        if not (0 < self.volatility <= 2.0):
            raise ValueError(f"Volatility must be between 0 and 2.0, got {self.volatility}")
        if self.expiration_date < datetime.now():
            raise ValueError(f"Expiration date {self.expiration_date} is in the past")

    @property
    def net_debit(self) -> float:
        """Net debit per share paid to enter position."""
        return self.long_premium - self.short_premium

    @property
    def spread_width(self) -> float:
        """Spread width (difference between strikes)."""
        return self.long_strike - self.short_strike

    @property
    def position_cost(self) -> float:
        """Total position cost."""
        return self.net_debit * 100 * self.contracts

    @property
    def days_to_expiration(self) -> int:
        """Days until expiration."""
        delta = self.expiration_date - datetime.now()
        return max(0, delta.days)

    @property
    def time_to_expiration(self) -> float:
        """Time to expiration in years."""
        return self.days_to_expiration / 365.0

    @property
    def max_profit(self) -> float:
        """Calculate maximum profit.

        Max profit = (Spread width - Net debit) × 100 × Contracts
        Occurs when underlying is at or below short strike at expiration.
        """
        profit_per_share = self.spread_width - self.net_debit
        return profit_per_share * 100 * self.contracts

    @property
    def max_loss(self) -> float:
        """Calculate maximum loss.

        Max loss = Net debit × 100 × Contracts
        Occurs when underlying is at or above long strike at expiration.
        """
        return self.position_cost

    @property
    def breakeven_price(self) -> float:
        """Calculate breakeven stock price at expiration.

        Breakeven = Long strike - Net debit
        """
        return self.long_strike - self.net_debit

    @property
    def max_roi(self) -> float:
        """Maximum return on investment as percentage."""
        return (self.max_profit / self.max_loss) * 100 if self.max_loss > 0 else 0

    def calculate_pnl(self, stock_price: float) -> Dict[str, float]:
        """Calculate P&L at given stock price (at expiration).

        Args:
            stock_price: Stock price at expiration

        Returns:
            Dictionary with P&L metrics

        Example:
            >>> position = BearPutSpread(...)
            >>> pnl = position.calculate_pnl(445.00)
            >>> print(f"P&L: ${pnl['pnl_total']:.2f}")
        """
        # Long put value (higher strike)
        long_value = max(self.long_strike - stock_price, 0)

        # Short put value (lower strike)
        short_value = max(self.short_strike - stock_price, 0)

        # Net position value
        position_value = long_value - short_value

        # P&L calculation
        pnl_per_share = position_value - self.net_debit
        pnl_total = pnl_per_share * 100 * self.contracts

        # Return percentage
        return_pct = (pnl_per_share / self.net_debit) * 100 if self.net_debit > 0 else 0

        return {
            'stock_price': stock_price,
            'long_put_value': long_value,
            'short_put_value': short_value,
            'position_value': position_value,
            'pnl_per_share': pnl_per_share,
            'pnl_total': pnl_total,
            'return_pct': return_pct
        }

    def calculate_greeks(self) -> Dict[str, float]:
        """Calculate all position Greeks using Black-Scholes.

        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        T = self.time_to_expiration

        if T <= 0:
            return {
                'delta': 0.0, 'gamma': 0.0, 'theta': 0.0,
                'vega': 0.0, 'rho': 0.0
            }

        S = self.underlying_price
        r = self.risk_free_rate
        sigma = self.volatility

        # Calculate Greeks for long put (higher strike)
        long_greeks = self._calculate_put_greeks(
            S, self.long_strike, T, r, sigma
        )

        # Calculate Greeks for short put (lower strike)
        short_greeks = self._calculate_put_greeks(
            S, self.short_strike, T, r, sigma
        )

        # Net position Greeks
        multiplier = 100 * self.contracts

        return {
            'delta': (long_greeks['delta'] - short_greeks['delta']) * multiplier,
            'gamma': (long_greeks['gamma'] - short_greeks['gamma']) * multiplier,
            'theta': (long_greeks['theta'] - short_greeks['theta']) * multiplier,
            'vega': (long_greeks['vega'] - short_greeks['vega']) * multiplier,
            'rho': (long_greeks['rho'] - short_greeks['rho']) * multiplier
        }

    @staticmethod
    def _calculate_put_greeks(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> Dict[str, float]:
        """Calculate Greeks for a single put option.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility

        Returns:
            Dictionary with individual option Greeks
        """
        if T == 0:
            return {
                'delta': -1.0 if S < K else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Standard normal PDF and CDF
        pdf_d1 = norm.pdf(d1)
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_neg_d2 = norm.cdf(-d2)

        # Greeks calculations
        delta = cdf_neg_d1 - 1  # Put delta is negative
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        theta = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * cdf_neg_d2
        ) / 365  # Per day
        vega = S * pdf_d1 * np.sqrt(T) / 100  # Per 1% volatility
        rho = -K * T * np.exp(-r * T) * cdf_neg_d2 / 100  # Per 1% rate

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def get_analysis(self) -> Dict:
        """Generate comprehensive position analysis.

        Returns:
            Dictionary containing position details, metrics, and Greeks
        """
        greeks = self.calculate_greeks()

        return {
            "position": {
                "symbol": self.underlying_symbol,
                "current_price": self.underlying_price,
                "long_strike": self.long_strike,
                "short_strike": self.short_strike,
                "long_premium": self.long_premium,
                "short_premium": self.short_premium,
                "net_debit": self.net_debit,
                "spread_width": self.spread_width,
                "contracts": self.contracts,
                "position_cost": self.position_cost,
                "expiration": self.expiration_date.strftime("%Y-%m-%d"),
                "days_to_expiration": self.days_to_expiration,
                "volatility": self.volatility,
            },
            "metrics": {
                "max_profit": self.max_profit,
                "max_loss": self.max_loss,
                "breakeven": self.breakeven_price,
                "max_roi": self.max_roi,
                "risk_reward_ratio": self.max_profit / self.max_loss if self.max_loss > 0 else 0,
            },
            "greeks": greeks,
        }

    def print_analysis(self):
        """Print formatted analysis to console."""
        analysis = self.get_analysis()

        print(f"\n{'=' * 70}")
        print(f"  {self.underlying_symbol} BEAR PUT SPREAD ANALYSIS")
        print(f"{'=' * 70}\n")

        # Position details
        print("Position Details:")
        print(f"  Symbol:               {analysis['position']['symbol']}")
        print(f"  Current Price:        ${analysis['position']['current_price']:.2f}")
        print(f"  Long Put Strike:      ${analysis['position']['long_strike']:.2f}")
        print(f"  Short Put Strike:     ${analysis['position']['short_strike']:.2f}")
        print(f"  Long Premium:         ${analysis['position']['long_premium']:.2f}")
        print(f"  Short Premium:        ${analysis['position']['short_premium']:.2f}")
        print(f"  Net Debit:            ${analysis['position']['net_debit']:.2f}/share")
        print(f"  Spread Width:         ${analysis['position']['spread_width']:.2f}")
        print(f"  Contracts:            {analysis['position']['contracts']}")
        print(f"  Position Cost:        ${analysis['position']['position_cost']:,.2f}")
        print(f"  Days to Expiration:   {analysis['position']['days_to_expiration']}")
        print(f"  Implied Volatility:   {analysis['position']['volatility']*100:.1f}%")
        print()

        # Metrics
        print("Risk Metrics:")
        print(f"  Max Profit:           ${analysis['metrics']['max_profit']:,.2f} "
              f"(below ${self.short_strike:.2f})")
        print(f"  Max Loss:             ${analysis['metrics']['max_loss']:,.2f} "
              f"(above ${self.long_strike:.2f})")
        print(f"  Breakeven:            ${analysis['metrics']['breakeven']:.2f}")
        print(f"  Max ROI:              {analysis['metrics']['max_roi']:.1f}%")
        print(f"  Risk/Reward Ratio:    1:{analysis['metrics']['risk_reward_ratio']:.2f}")
        print()

        # Greeks
        print("Greeks:")
        print(f"  Delta:    {analysis['greeks']['delta']:>10.2f}  "
              f"(Position move per $1 underlying)")
        print(f"  Gamma:    {analysis['greeks']['gamma']:>10.4f}  "
              f"(Delta change per $1 move)")
        print(f"  Theta:    {analysis['greeks']['theta']:>10.2f}  "
              f"(Daily time decay)")
        print(f"  Vega:     {analysis['greeks']['vega']:>10.2f}  "
              f"(Value per 1% IV change)")
        print(f"  Rho:      {analysis['greeks']['rho']:>10.2f}  "
              f"(Value per 1% rate change)")
        print()

        # Scenario analysis
        print("Scenario Analysis at Expiration:")
        print(f"{'Price':<12} {'Position Value':<16} {'P&L':<16} {'Return %':<10}")
        print("-" * 70)

        key_prices = [
            self.short_strike - 10,
            self.short_strike,
            self.breakeven_price,
            (self.short_strike + self.long_strike) / 2,
            self.underlying_price,
            self.long_strike,
            self.long_strike + 10
        ]

        for price in key_prices:
            result = self.calculate_pnl(price)
            print(f"${price:<11.2f} ${result['position_value']:<15.2f} "
                  f"${result['pnl_total']:<15.2f} {result['return_pct']:<9.1f}%")

        print(f"\n{'=' * 70}\n")


def main():
    """CLI entry point for calculator."""
    parser = argparse.ArgumentParser(
        description="Bear Put Spread Strategy Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic analysis:
    python bear_put_spread_calculator.py --underlying SPY --price 450 \\
        --long-strike 450 --short-strike 440 \\
        --long-premium 8.50 --short-premium 4.20

  Multiple contracts with custom volatility:
    python bear_put_spread_calculator.py --underlying QQQ --price 380 \\
        --long-strike 380 --short-strike 370 \\
        --long-premium 10.20 --short-premium 5.80 \\
        --contracts 5 --volatility 0.25 --dte 60
        """,
    )

    # Required arguments
    parser.add_argument(
        "--underlying", required=True, help="Underlying ticker symbol (e.g., SPY)"
    )
    parser.add_argument(
        "--price", type=float, required=True, help="Current underlying stock price"
    )
    parser.add_argument(
        "--long-strike", type=float, required=True, help="Long put strike (higher)"
    )
    parser.add_argument(
        "--short-strike", type=float, required=True, help="Short put strike (lower)"
    )
    parser.add_argument(
        "--long-premium", type=float, required=True, help="Long put premium"
    )
    parser.add_argument(
        "--short-premium", type=float, required=True, help="Short put premium"
    )

    # Optional arguments
    parser.add_argument(
        "--contracts", type=int, default=1, help="Number of contracts (default: 1)"
    )
    parser.add_argument(
        "--dte", type=int, default=45, help="Days to expiration (default: 45)"
    )
    parser.add_argument(
        "--volatility", type=float, default=0.20,
        help="Implied volatility (default: 0.20)"
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.05,
        help="Risk-free rate (default: 0.05)"
    )

    args = parser.parse_args()

    # Calculate expiration date
    expiration_date = datetime.now() + timedelta(days=args.dte)

    try:
        # Create position
        position = BearPutSpread(
            underlying_symbol=args.underlying,
            underlying_price=args.price,
            long_strike=args.long_strike,
            short_strike=args.short_strike,
            long_premium=args.long_premium,
            short_premium=args.short_premium,
            expiration_date=expiration_date,
            contracts=args.contracts,
            volatility=args.volatility,
            risk_free_rate=args.risk_free_rate,
        )

        # Print analysis
        position.print_analysis()

        return 0

    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
