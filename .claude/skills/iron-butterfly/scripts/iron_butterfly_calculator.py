#!/usr/bin/env python3
"""Iron Butterfly Strategy Calculator

Complete implementation for analyzing iron butterfly options strategies.
An iron butterfly combines a short straddle with protective wings.

Structure:
- Sell 1 ATM call
- Sell 1 ATM put (same strike as call)
- Buy 1 OTM call (higher strike)
- Buy 1 OTM put (lower strike)
- Net credit strategy with defined risk

Usage:
    python iron_butterfly_calculator.py --underlying SPY --price 450 \
        --center-strike 450 --wing-width 10 \
        --call-credit 15.50 --put-credit 15.20 \
        --call-protection 2.10 --put-protection 2.00
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
class IronButterfly:
    """Iron butterfly position with complete analysis capabilities.

    A neutral strategy that profits when underlying stays near center strike.
    Combines short ATM straddle with long protective wings.

    Attributes:
        underlying_symbol: Ticker symbol
        underlying_price: Current stock price
        center_strike: Strike price of short straddle (ATM)
        lower_put_strike: Strike price of long put (OTM protection)
        upper_call_strike: Strike price of long call (OTM protection)
        short_call_premium: Premium received for short ATM call
        short_put_premium: Premium received for short ATM put
        long_call_premium: Premium paid for long OTM call
        long_put_premium: Premium paid for long OTM put
        expiration_date: Option expiration date
        contracts: Number of contracts (default 1)
        volatility: Implied volatility (default 0.20)
        risk_free_rate: Risk-free interest rate (default 0.05)

    Example:
        >>> position = IronButterfly(
        ...     underlying_symbol="SPY",
        ...     underlying_price=450.00,
        ...     center_strike=450.00,
        ...     lower_put_strike=440.00,
        ...     upper_call_strike=460.00,
        ...     short_call_premium=15.50,
        ...     short_put_premium=15.20,
        ...     long_call_premium=2.10,
        ...     long_put_premium=2.00,
        ...     expiration_date=datetime.now() + timedelta(days=45)
        ... )
        >>> print(f"Max Profit: ${position.max_profit:.2f}")
    """

    underlying_symbol: str
    underlying_price: float
    center_strike: float
    lower_put_strike: float
    upper_call_strike: float
    short_call_premium: float
    short_put_premium: float
    long_call_premium: float
    long_put_premium: float
    expiration_date: datetime
    contracts: int = 1
    volatility: float = 0.20
    risk_free_rate: float = 0.05

    def __post_init__(self):
        """Validate position parameters."""
        if not (self.lower_put_strike < self.center_strike < self.upper_call_strike):
            raise ValueError(
                f"Strikes must be ordered: lower_put ({self.lower_put_strike}) < "
                f"center ({self.center_strike}) < upper_call ({self.upper_call_strike})"
            )

        # Check symmetry (recommended but not required)
        lower_width = self.center_strike - self.lower_put_strike
        upper_width = self.upper_call_strike - self.center_strike
        if abs(lower_width - upper_width) > 0.01:
            print(f"Warning: Wings are not symmetric (lower: {lower_width}, upper: {upper_width})")

        if self.underlying_price <= 0:
            raise ValueError(f"Underlying price must be positive, got {self.underlying_price}")
        if self.contracts <= 0:
            raise ValueError(f"Contracts must be positive, got {self.contracts}")
        if not (0 < self.volatility <= 2.0):
            raise ValueError(f"Volatility must be between 0 and 2.0, got {self.volatility}")
        if self.expiration_date < datetime.now():
            raise ValueError(f"Expiration date {self.expiration_date} is in the past")

    @property
    def net_credit(self) -> float:
        """Net credit received per share."""
        credit_received = self.short_call_premium + self.short_put_premium
        cost_paid = self.long_call_premium + self.long_put_premium
        return credit_received - cost_paid

    @property
    def wing_width(self) -> float:
        """Average wing width (should be symmetric)."""
        lower_width = self.center_strike - self.lower_put_strike
        upper_width = self.upper_call_strike - self.center_strike
        return (lower_width + upper_width) / 2

    @property
    def position_credit(self) -> float:
        """Total credit received."""
        return self.net_credit * 100 * self.contracts

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

        Max profit = Net credit × 100 × Contracts
        Occurs when underlying is exactly at center strike at expiration.
        """
        return self.position_credit

    @property
    def max_loss(self) -> float:
        """Calculate maximum loss.

        Max loss = (Wing width - Net credit) × 100 × Contracts
        Occurs when underlying is at or beyond either wing at expiration.
        """
        loss_per_share = self.wing_width - self.net_credit
        return loss_per_share * 100 * self.contracts

    @property
    def breakeven_lower(self) -> float:
        """Lower breakeven stock price at expiration."""
        return self.center_strike - self.net_credit

    @property
    def breakeven_upper(self) -> float:
        """Upper breakeven stock price at expiration."""
        return self.center_strike + self.net_credit

    @property
    def max_roi(self) -> float:
        """Maximum return on risk as percentage."""
        return (self.max_profit / self.max_loss) * 100 if self.max_loss > 0 else 0

    @property
    def probability_of_profit(self) -> float:
        """Estimated probability of profit based on normal distribution.

        Assumes underlying follows normal distribution with given volatility.
        """
        if self.time_to_expiration <= 0:
            return (
                100.0
                if self.breakeven_lower <= self.underlying_price <= self.breakeven_upper
                else 0.0
            )

        # Standard deviation of price movement
        std_dev = self.underlying_price * self.volatility * np.sqrt(self.time_to_expiration)

        # Z-scores for breakeven points
        z_lower = (self.breakeven_lower - self.underlying_price) / std_dev
        z_upper = (self.breakeven_upper - self.underlying_price) / std_dev

        # Probability between breakevens
        prob = norm.cdf(z_upper) - norm.cdf(z_lower)
        return prob * 100

    def calculate_pnl(self, stock_price: float) -> dict[str, float]:
        """Calculate P&L at given stock price (at expiration).

        Args:
            stock_price: Stock price at expiration

        Returns:
            Dictionary with P&L metrics
        """
        # Short call value (negative, we sold it)
        short_call_value = -max(stock_price - self.center_strike, 0)

        # Short put value (negative, we sold it)
        short_put_value = -max(self.center_strike - stock_price, 0)

        # Long call value (positive, we bought it)
        long_call_value = max(stock_price - self.upper_call_strike, 0)

        # Long put value (positive, we bought it)
        long_put_value = max(self.lower_put_strike - stock_price, 0)

        # Net position value at expiration
        position_value = short_call_value + short_put_value + long_call_value + long_put_value

        # P&L calculation (position value + initial credit)
        pnl_per_share = position_value + self.net_credit
        pnl_total = pnl_per_share * 100 * self.contracts

        # Return percentage based on risk
        return_pct = (pnl_per_share / (self.wing_width - self.net_credit)) * 100

        return {
            "stock_price": stock_price,
            "short_call_value": short_call_value,
            "short_put_value": short_put_value,
            "long_call_value": long_call_value,
            "long_put_value": long_put_value,
            "position_value": position_value,
            "pnl_per_share": pnl_per_share,
            "pnl_total": pnl_total,
            "return_pct": return_pct,
        }

    def calculate_greeks(self) -> dict[str, float]:
        """Calculate all position Greeks using Black-Scholes.

        Returns:
            Dictionary with delta, gamma, theta, vega, rho
        """
        T = self.time_to_expiration

        if T <= 0:
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

        S = self.underlying_price
        r = self.risk_free_rate
        sigma = self.volatility

        # Short ATM call
        short_call = self._calculate_call_greeks(S, self.center_strike, T, r, sigma)

        # Short ATM put
        short_put = self._calculate_put_greeks(S, self.center_strike, T, r, sigma)

        # Long OTM call
        long_call = self._calculate_call_greeks(S, self.upper_call_strike, T, r, sigma)

        # Long OTM put
        long_put = self._calculate_put_greeks(S, self.lower_put_strike, T, r, sigma)

        # Net position Greeks (short positions are negative)
        multiplier = 100 * self.contracts

        return {
            "delta": (
                (-short_call["delta"] - short_put["delta"] + long_call["delta"] + long_put["delta"])
                * multiplier
            ),
            "gamma": (
                (-short_call["gamma"] - short_put["gamma"] + long_call["gamma"] + long_put["gamma"])
                * multiplier
            ),
            "theta": (
                (-short_call["theta"] - short_put["theta"] + long_call["theta"] + long_put["theta"])
                * multiplier
            ),
            "vega": (
                (-short_call["vega"] - short_put["vega"] + long_call["vega"] + long_put["vega"])
                * multiplier
            ),
            "rho": (
                (-short_call["rho"] - short_put["rho"] + long_call["rho"] + long_put["rho"])
                * multiplier
            ),
        }

    @staticmethod
    def _calculate_call_greeks(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> dict[str, float]:
        """Calculate Greeks for a single call option."""
        if T == 0:
            return {
                "delta": 1.0 if S > K else 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "rho": 0.0,
            }

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        return {
            "delta": cdf_d1,
            "gamma": pdf_d1 / (S * sigma * np.sqrt(T)),
            "theta": (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2)
            / 365,
            "vega": S * pdf_d1 * np.sqrt(T) / 100,
            "rho": K * T * np.exp(-r * T) * cdf_d2 / 100,
        }

    @staticmethod
    def _calculate_put_greeks(
        S: float, K: float, T: float, r: float, sigma: float
    ) -> dict[str, float]:
        """Calculate Greeks for a single put option."""
        if T == 0:
            return {
                "delta": -1.0 if S < K else 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "rho": 0.0,
            }

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_neg_d2 = norm.cdf(-d2)

        return {
            "delta": cdf_neg_d1 - 1,
            "gamma": pdf_d1 / (S * sigma * np.sqrt(T)),
            "theta": (
                -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * cdf_neg_d2
            )
            / 365,
            "vega": S * pdf_d1 * np.sqrt(T) / 100,
            "rho": -K * T * np.exp(-r * T) * cdf_neg_d2 / 100,
        }

    def get_analysis(self) -> dict:
        """Generate comprehensive position analysis."""
        greeks = self.calculate_greeks()

        return {
            "position": {
                "symbol": self.underlying_symbol,
                "current_price": self.underlying_price,
                "center_strike": self.center_strike,
                "lower_put_strike": self.lower_put_strike,
                "upper_call_strike": self.upper_call_strike,
                "wing_width": self.wing_width,
                "net_credit": self.net_credit,
                "contracts": self.contracts,
                "position_credit": self.position_credit,
                "expiration": self.expiration_date.strftime("%Y-%m-%d"),
                "days_to_expiration": self.days_to_expiration,
                "volatility": self.volatility,
            },
            "metrics": {
                "max_profit": self.max_profit,
                "max_loss": self.max_loss,
                "breakeven_lower": self.breakeven_lower,
                "breakeven_upper": self.breakeven_upper,
                "max_roi": self.max_roi,
                "probability_of_profit": self.probability_of_profit,
            },
            "greeks": greeks,
        }

    def print_analysis(self):
        """Print formatted analysis to console."""
        analysis = self.get_analysis()

        print(f"\n{'=' * 70}")
        print(f"  {self.underlying_symbol} IRON BUTTERFLY ANALYSIS")
        print(f"{'=' * 70}\n")

        # Position details
        print("Position Details:")
        print(f"  Symbol:               {analysis['position']['symbol']}")
        print(f"  Current Price:        ${analysis['position']['current_price']:.2f}")
        print(f"  Center Strike (ATM):  ${analysis['position']['center_strike']:.2f}")
        print(f"  Lower Put Strike:     ${analysis['position']['lower_put_strike']:.2f}")
        print(f"  Upper Call Strike:    ${analysis['position']['upper_call_strike']:.2f}")
        print(f"  Wing Width:           ${analysis['position']['wing_width']:.2f}")
        print(f"  Net Credit:           ${analysis['position']['net_credit']:.2f}/share")
        print(f"  Contracts:            {analysis['position']['contracts']}")
        print(f"  Credit Received:      ${analysis['position']['position_credit']:,.2f}")
        print(f"  Days to Expiration:   {analysis['position']['days_to_expiration']}")
        print(f"  Implied Volatility:   {analysis['position']['volatility']*100:.1f}%")
        print()

        # Metrics
        print("Risk Metrics:")
        print(
            f"  Max Profit:           ${analysis['metrics']['max_profit']:,.2f} "
            f"(at ${self.center_strike:.2f})"
        )
        print(f"  Max Loss:             ${analysis['metrics']['max_loss']:,.2f} " f"(beyond wings)")
        print(f"  Lower Breakeven:      ${analysis['metrics']['breakeven_lower']:.2f}")
        print(f"  Upper Breakeven:      ${analysis['metrics']['breakeven_upper']:.2f}")
        print(f"  Max ROI:              {analysis['metrics']['max_roi']:.1f}%")
        print(f"  Probability of Profit: {analysis['metrics']['probability_of_profit']:.1f}%")
        print()

        # Greeks
        print("Greeks:")
        print(f"  Delta:    {analysis['greeks']['delta']:>10.2f}  (Should be near 0 at ATM)")
        print(f"  Gamma:    {analysis['greeks']['gamma']:>10.4f}  (Negative, short gamma)")
        print(f"  Theta:    {analysis['greeks']['theta']:>10.2f}  (Positive, benefits from decay)")
        print(f"  Vega:     {analysis['greeks']['vega']:>10.2f}  (Negative, hurt by IV increase)")
        print(f"  Rho:      {analysis['greeks']['rho']:>10.2f}  (Interest rate sensitivity)")
        print()

        # Scenario analysis
        print("Scenario Analysis at Expiration:")
        print(f"{'Price':<12} {'Position Value':<16} {'P&L':<16} {'Return %':<10}")
        print("-" * 70)

        key_prices = [
            self.lower_put_strike,
            self.breakeven_lower,
            (self.breakeven_lower + self.center_strike) / 2,
            self.center_strike,
            (self.center_strike + self.breakeven_upper) / 2,
            self.breakeven_upper,
            self.upper_call_strike,
        ]

        for price in key_prices:
            result = self.calculate_pnl(price)
            print(
                f"${price:<11.2f} ${result['position_value']:<15.2f} "
                f"${result['pnl_total']:<15.2f} {result['return_pct']:<9.1f}%"
            )

        print(f"\n{'=' * 70}\n")


def main():
    """CLI entry point for calculator."""
    parser = argparse.ArgumentParser(
        description="Iron Butterfly Strategy Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic analysis:
    python iron_butterfly_calculator.py --underlying SPY --price 450 \\
        --center-strike 450 --wing-width 10 \\
        --call-credit 15.50 --put-credit 15.20 \\
        --call-protection 2.10 --put-protection 2.00

  Asymmetric wings:
    python iron_butterfly_calculator.py --underlying SPY --price 450 \\
        --center-strike 450 --lower-put 440 --upper-call 465 \\
        --call-credit 15.50 --put-credit 15.20 \\
        --call-protection 1.80 --put-protection 2.00
        """,
    )

    # Required arguments
    parser.add_argument("--underlying", required=True, help="Underlying ticker symbol")
    parser.add_argument("--price", type=float, required=True, help="Current stock price")
    parser.add_argument(
        "--center-strike", type=float, required=True, help="Center strike (short straddle)"
    )

    # Wing strikes (either symmetric or explicit)
    wing_group = parser.add_mutually_exclusive_group(required=True)
    wing_group.add_argument("--wing-width", type=float, help="Symmetric wing width from center")
    wing_group.add_argument(
        "--lower-put",
        type=float,
        dest="lower_put_strike",
        help="Lower put strike (use with --upper-call)",
    )

    parser.add_argument(
        "--upper-call",
        type=float,
        dest="upper_call_strike",
        help="Upper call strike (use with --lower-put)",
    )

    # Premiums
    parser.add_argument(
        "--call-credit", type=float, required=True, help="Short call premium received"
    )
    parser.add_argument(
        "--put-credit", type=float, required=True, help="Short put premium received"
    )
    parser.add_argument(
        "--call-protection", type=float, required=True, help="Long call premium paid"
    )
    parser.add_argument("--put-protection", type=float, required=True, help="Long put premium paid")

    # Optional arguments
    parser.add_argument("--contracts", type=int, default=1, help="Number of contracts")
    parser.add_argument("--dte", type=int, default=45, help="Days to expiration")
    parser.add_argument("--volatility", type=float, default=0.20, help="Implied volatility")
    parser.add_argument("--risk-free-rate", type=float, default=0.05, help="Risk-free rate")

    args = parser.parse_args()

    # Determine wing strikes
    if args.wing_width:
        lower_put = args.center_strike - args.wing_width
        upper_call = args.center_strike + args.wing_width
    else:
        if not args.upper_call_strike:
            parser.error("--upper-call required when using --lower-put")
        lower_put = args.lower_put_strike
        upper_call = args.upper_call_strike

    expiration_date = datetime.now() + timedelta(days=args.dte)

    try:
        position = IronButterfly(
            underlying_symbol=args.underlying,
            underlying_price=args.price,
            center_strike=args.center_strike,
            lower_put_strike=lower_put,
            upper_call_strike=upper_call,
            short_call_premium=args.call_credit,
            short_put_premium=args.put_credit,
            long_call_premium=args.call_protection,
            long_put_premium=args.put_protection,
            expiration_date=expiration_date,
            contracts=args.contracts,
            volatility=args.volatility,
            risk_free_rate=args.risk_free_rate,
        )

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
