#!/usr/bin/env python3
"""Bull Call Spread Strategy Calculator

Complete implementation for analyzing bull call spread options strategies
including P&L calculations, Greeks, payoff diagrams, and risk management.

Usage:
    python strategy_calculator.py --underlying SPY --price 450 \
        --long-strike 445 --short-strike 455 \
        --long-premium 8.50 --short-premium 3.20
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


@dataclass
class BullCallSpread:
    """Bull call spread position with complete analysis capabilities.

    Attributes:
        underlying_symbol: Ticker symbol
        underlying_price: Current stock price
        long_strike: Strike price of long call (lower)
        short_strike: Strike price of short call (higher)
        long_premium: Premium paid for long call
        short_premium: Premium received for short call
        expiration_date: Option expiration date
        contracts: Number of contracts (default 1)
        volatility: Implied volatility (default 0.20)
        risk_free_rate: Risk-free interest rate (default 0.05)
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
        if self.long_strike >= self.short_strike:
            raise ValueError("Long strike must be less than short strike")
        if self.long_premium <= self.short_premium:
            raise ValueError("Long premium should exceed short premium for debit spread")
        if self.underlying_price <= 0:
            raise ValueError("Underlying price must be positive")
        if self.contracts <= 0:
            raise ValueError("Contracts must be positive")
        if not (0 < self.volatility <= 2.0):
            raise ValueError("Volatility must be between 0 and 2.0")

    @property
    def net_debit(self) -> float:
        """Net debit per share."""
        return self.long_premium - self.short_premium

    @property
    def spread_width(self) -> float:
        """Spread width (difference in strikes)."""
        return self.short_strike - self.long_strike

    @property
    def position_cost(self) -> float:
        """Total position cost."""
        return self.net_debit * 100 * self.contracts

    @property
    def days_to_expiration(self) -> int:
        """Days until expiration."""
        return max(0, (self.expiration_date - datetime.now()).days)

    @property
    def time_to_expiration(self) -> float:
        """Time to expiration in years."""
        return self.days_to_expiration / 365.0


class StrategyAnalyzer:
    """Comprehensive analysis tools for bull call spread strategy."""

    def __init__(self, position: BullCallSpread):
        """Initialize analyzer with position.

        Args:
            position: BullCallSpread instance
        """
        self.position = position

    def calculate_breakeven(self) -> float:
        """Calculate breakeven price at expiration.

        Returns:
            Breakeven stock price
        """
        return self.position.long_strike + self.position.net_debit

    def calculate_max_profit(self) -> Dict[str, float]:
        """Calculate maximum profit metrics.

        Returns:
            Dictionary with max profit information
        """
        max_profit_per_share = (
            self.position.spread_width - self.position.net_debit
        )
        max_profit_total = max_profit_per_share * 100 * self.position.contracts

        return {
            'max_profit_per_share': max_profit_per_share,
            'max_profit_total': max_profit_total,
            'max_profit_price': self.position.short_strike,
            'max_roi': (max_profit_per_share / self.position.net_debit) * 100
        }

    def calculate_max_loss(self) -> Dict[str, float]:
        """Calculate maximum loss metrics.

        Returns:
            Dictionary with max loss information
        """
        return {
            'max_loss_per_share': self.position.net_debit,
            'max_loss_total': self.position.position_cost,
            'max_loss_price': self.position.long_strike
        }

    def calculate_pnl(self, stock_price: float) -> Dict[str, float]:
        """Calculate P&L at given stock price (at expiration).

        Args:
            stock_price: Stock price at expiration

        Returns:
            Dictionary with P&L metrics
        """
        # Long call value
        long_value = max(stock_price - self.position.long_strike, 0)

        # Short call value
        short_value = max(stock_price - self.position.short_strike, 0)

        # Net position value
        position_value = long_value - short_value

        # P&L calculation
        pnl_per_share = position_value - self.position.net_debit
        pnl_total = pnl_per_share * 100 * self.position.contracts

        # Return percentage
        return_pct = (pnl_per_share / self.position.net_debit) * 100

        return {
            'stock_price': stock_price,
            'long_call_value': long_value,
            'short_call_value': short_value,
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
        T = self.position.time_to_expiration

        if T <= 0:
            return {
                'delta': 0.0, 'gamma': 0.0, 'theta': 0.0,
                'vega': 0.0, 'rho': 0.0
            }

        S = self.position.underlying_price
        r = self.position.risk_free_rate
        sigma = self.position.volatility

        # Calculate Greeks for long call
        long_greeks = self._calculate_call_greeks(
            S, self.position.long_strike, T, r, sigma
        )

        # Calculate Greeks for short call
        short_greeks = self._calculate_call_greeks(
            S, self.position.short_strike, T, r, sigma
        )

        # Net position Greeks
        multiplier = 100 * self.position.contracts

        return {
            'delta': (long_greeks['delta'] - short_greeks['delta']) * multiplier,
            'gamma': (long_greeks['gamma'] - short_greeks['gamma']) * multiplier,
            'theta': (long_greeks['theta'] - short_greeks['theta']) * multiplier,
            'vega': (long_greeks['vega'] - short_greeks['vega']) * multiplier,
            'rho': (long_greeks['rho'] - short_greeks['rho']) * multiplier
        }

    @staticmethod
    def _calculate_call_greeks(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> Dict[str, float]:
        """Calculate Greeks for a single call option.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility

        Returns:
            Dictionary with individual option Greeks
        """
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Standard normal PDF and CDF
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        # Greeks calculations
        delta = cdf_d1
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        theta = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * cdf_d2
        ) / 365  # Per day
        vega = S * pdf_d1 * np.sqrt(T) / 100  # Per 1% volatility
        rho = K * T * np.exp(-r * T) * cdf_d2 / 100  # Per 1% rate

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def print_comprehensive_analysis(self):
        """Print complete position analysis to console."""
        print("\n" + "=" * 70)
        print("BULL CALL SPREAD COMPREHENSIVE ANALYSIS")
        print("=" * 70)

        # Position details
        print(f"\n{'POSITION DETAILS':-^70}")
        print(f"  Symbol:               {self.position.underlying_symbol}")
        print(f"  Current Price:        ${self.position.underlying_price:.2f}")
        print(f"  Long Call Strike:     ${self.position.long_strike:.2f}")
        print(f"  Short Call Strike:    ${self.position.short_strike:.2f}")
        print(f"  Long Premium:         ${self.position.long_premium:.2f}")
        print(f"  Short Premium:        ${self.position.short_premium:.2f}")
        print(f"  Net Debit:            ${self.position.net_debit:.2f}/share")
        print(f"  Spread Width:         ${self.position.spread_width:.2f}")
        print(f"  Contracts:            {self.position.contracts}")
        print(f"  Position Cost:        ${self.position.position_cost:,.2f}")
        print(f"  Days to Expiration:   {self.position.days_to_expiration}")
        print(f"  Implied Volatility:   {self.position.volatility*100:.1f}%")

        # Risk metrics
        print(f"\n{'RISK METRICS':-^70}")
        max_profit = self.calculate_max_profit()
        max_loss = self.calculate_max_loss()
        breakeven = self.calculate_breakeven()

        print(f"  Maximum Profit:       ${max_profit['max_profit_total']:,.2f} "
              f"(at ${max_profit['max_profit_price']:.2f}+)")
        print(f"  Maximum Loss:         ${max_loss['max_loss_total']:,.2f} "
              f"(below ${max_loss['max_loss_price']:.2f})")
        print(f"  Breakeven Price:      ${breakeven:.2f}")
        print(f"  Risk/Reward Ratio:    1:{max_profit['max_roi']/100:.2f}")
        print(f"  Max ROI:              {max_profit['max_roi']:.1f}%")

        # Greeks
        print(f"\n{'GREEKS':-^70}")
        greeks = self.calculate_greeks()
        print(f"  Delta:    {greeks['delta']:>10.2f}  "
              f"(Position move per $1 underlying)")
        print(f"  Gamma:    {greeks['gamma']:>10.4f}  "
              f"(Delta change per $1 move)")
        print(f"  Theta:    {greeks['theta']:>10.2f}  "
              f"(Daily time decay)")
        print(f"  Vega:     {greeks['vega']:>10.2f}  "
              f"(Value per 1% IV change)")
        print(f"  Rho:      {greeks['rho']:>10.2f}  "
              f"(Value per 1% rate change)")

        # Scenario analysis at key prices
        print(f"\n{'SCENARIO ANALYSIS':-^70}")
        print(f"{'Price':<12} {'Position Value':<16} {'P&L':<16} {'Return %':<10}")
        print("-" * 70)

        key_prices = [
            self.position.long_strike - 5,
            self.position.long_strike,
            breakeven,
            self.position.underlying_price,
            (self.position.long_strike + self.position.short_strike) / 2,
            self.position.short_strike,
            self.position.short_strike + 5
        ]

        for price in key_prices:
            result = self.calculate_pnl(price)
            print(f"${price:<11.2f} ${result['position_value']:<15.2f} "
                  f"${result['pnl_total']:<15.2f} {result['return_pct']:<9.1f}%")

        print("\n" + "=" * 70)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Bull Call Spread Strategy Calculator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:

Basic analysis:
  python strategy_calculator.py --underlying SPY --price 450 \\
      --long-strike 445 --short-strike 455 \\
      --long-premium 8.50 --short-premium 3.20

With volatility and custom parameters:
  python strategy_calculator.py --underlying AAPL --price 180 \\
      --long-strike 175 --short-strike 185 \\
      --long-premium 7.20 --short-premium 2.80 \\
      --contracts 5 --volatility 0.25 --dte 60
"""
    )

    parser.add_argument('--underlying', type=str, required=True,
                        help='Underlying symbol (e.g., SPY, AAPL)')
    parser.add_argument('--price', type=float, required=True,
                        help='Current underlying price')
    parser.add_argument('--long-strike', type=float, required=True,
                        help='Long call strike price')
    parser.add_argument('--short-strike', type=float, required=True,
                        help='Short call strike price')
    parser.add_argument('--long-premium', type=float, required=True,
                        help='Long call premium')
    parser.add_argument('--short-premium', type=float, required=True,
                        help='Short call premium')
    parser.add_argument('--dte', type=int, default=45,
                        help='Days to expiration (default: 45)')
    parser.add_argument('--contracts', type=int, default=1,
                        help='Number of contracts (default: 1)')
    parser.add_argument('--volatility', type=float, default=0.20,
                        help='Implied volatility (default: 0.20)')
    parser.add_argument('--risk-free-rate', type=float, default=0.05,
                        help='Risk-free rate (default: 0.05)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip displaying plot')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    try:
        # Create position
        expiration = datetime.now() + timedelta(days=args.dte)

        position = BullCallSpread(
            underlying_symbol=args.underlying,
            underlying_price=args.price,
            long_strike=args.long_strike,
            short_strike=args.short_strike,
            long_premium=args.long_premium,
            short_premium=args.short_premium,
            expiration_date=expiration,
            contracts=args.contracts,
            volatility=args.volatility,
            risk_free_rate=args.risk_free_rate
        )

        # Create analyzer
        analyzer = StrategyAnalyzer(position)

        # Print comprehensive analysis
        analyzer.print_comprehensive_analysis()

        return 0

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
