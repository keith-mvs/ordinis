"""
Married Put Strategy Implementation

A married put combines a long stock position with a long put option for downside
protection while maintaining unlimited upside potential. This is a protective strategy
ideal for risk-averse investors or volatile market conditions.

Strategy Components:
    - Long 100 shares of stock
    - Long 1 put option (100 shares)

Risk Profile:
    - Max Loss: Limited to (Stock Price - Put Strike + Put Premium)
    - Max Gain: Unlimited (stock can rise indefinitely)
    - Breakeven: Stock Price + Put Premium

Use Cases:
    - Protecting newly acquired positions
    - Hedging during uncertain market periods
    - Maintaining upside while limiting downside
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from application.strategies.base import BaseStrategy
from engines.optionscore import OptionsCoreEngine
from engines.signalcore.core.signal import Signal


@dataclass
class MarriedPutConfig:
    """Configuration for married put strategy."""

    min_protection_pct: float = 0.05  # Minimum 5% downside protection
    max_protection_pct: float = 0.15  # Maximum 15% downside protection
    max_premium_pct: float = 0.05  # Max premium as % of stock price (5%)
    min_delta: float = 0.30  # Minimum put delta (OTM)
    max_delta: float = 0.70  # Maximum put delta (ITM)
    days_to_expiration: int = 45  # Target DTE for put options
    dte_tolerance: int = 15  # Allow ±15 days from target
    transaction_cost: float = 0.65  # Per contract transaction cost


class MarriedPutStrategy(BaseStrategy):
    """
    Married put strategy for downside protection.

    Buys (or holds) stock and purchases protective put options to limit
    downside risk while maintaining unlimited upside potential.

    Parameters:
        - min_protection_pct: Minimum downside protection percentage (default 5%)
        - max_protection_pct: Maximum downside protection percentage (default 15%)
        - max_premium_pct: Maximum premium cost as % of stock price (default 5%)
        - min_delta: Minimum put delta for selection (default 0.30)
        - max_delta: Maximum put delta for selection (default 0.70)
        - days_to_expiration: Target DTE for put options (default 45)
        - dte_tolerance: Allowable variation in DTE (default ±15 days)
        - transaction_cost: Per contract transaction cost (default $0.65)
    """

    def configure(self) -> None:
        """Configure strategy parameters with defaults."""
        self.params.setdefault("min_protection_pct", 0.05)  # 5% min protection
        self.params.setdefault("max_protection_pct", 0.15)  # 15% max protection
        self.params.setdefault("max_premium_pct", 0.05)  # 5% max premium cost
        self.params.setdefault("min_delta", 0.30)  # OTM puts
        self.params.setdefault("max_delta", 0.70)  # ITM puts
        self.params.setdefault("days_to_expiration", 45)  # 45-day options
        self.params.setdefault("dte_tolerance", 15)  # ±15 days
        self.params.setdefault("transaction_cost", 0.65)  # $0.65 per contract

    def generate_signal(
        self,
        data: pd.DataFrame,
        timestamp: datetime,
        options_engine: OptionsCoreEngine | None = None,
    ) -> Signal | None:
        """
        Generate married put signal for stock position.

        Args:
            data: OHLCV price data for underlying
            timestamp: Current timestamp
            options_engine: Optional options engine for chain analysis

        Returns:
            Signal with married put recommendation or None if no opportunity

        Note:
            If options_engine is not provided, returns None (requires live options data).
        """
        try:
            # Validate data
            is_valid, _ = self.validate_data(data)
            if not is_valid:
                return None

            # Check if we have options engine
            if options_engine is None:
                return None

            # TODO: Implement options chain analysis using options_engine
            # This would:
            # 1. Fetch put options chain for the underlying
            # 2. Filter puts matching protection and cost criteria
            # 3. Calculate expected P&L and risk metrics
            # 4. Return Signal with recommendations

            return None

        except Exception:
            return None

    def get_description(self) -> str:
        """
        Get strategy description.

        Returns:
            Human-readable description
        """
        return (
            f"Married Put Strategy: Protective put options for downside insurance. "
            f"Target: {self.params['min_protection_pct']*100:.0f}%-{self.params['max_protection_pct']*100:.0f}% protection, "
            f"Max premium: {self.params['max_premium_pct']*100:.0f}% of stock price"
        )

    def analyze_opportunity(
        self,
        stock_price: float,
        put_strike: float,
        put_premium: float,
        put_delta: float,
        days_to_expiration: int,
        shares: int = 100,
    ) -> dict[str, Any]:
        """
        Analyze married put opportunity.

        Args:
            stock_price: Current stock price
            put_strike: Put option strike price
            put_premium: Put option premium (per share)
            put_delta: Put option delta
            days_to_expiration: Days until put expiration
            shares: Number of shares (default 100)

        Returns:
            Dictionary with analysis results including costs, protection,
            breakeven, and whether it meets strategy criteria
        """
        # Cost calculations
        stock_cost = stock_price * shares
        put_cost = put_premium * shares
        transaction_cost = self.params.get("transaction_cost", 0.65)
        total_cost = stock_cost + put_cost + transaction_cost

        # Protection analysis
        protection_amount = stock_price - put_strike
        protection_pct = protection_amount / stock_price

        # P&L calculations
        max_loss = (stock_price - put_strike + put_premium) * shares + transaction_cost
        breakeven = stock_price + put_premium + (transaction_cost / shares)

        # Premium cost as percentage
        premium_pct = put_premium / stock_price

        # Time metrics
        cost_per_day = (put_cost + transaction_cost) / days_to_expiration
        annualized_cost_pct = (premium_pct / days_to_expiration) * 365

        # Check if opportunity meets criteria
        meets_criteria = all(
            [
                protection_pct >= self.params.get("min_protection_pct", 0.05),
                protection_pct <= self.params.get("max_protection_pct", 0.15),
                premium_pct <= self.params.get("max_premium_pct", 0.05),
                abs(put_delta) >= self.params.get("min_delta", 0.30),
                abs(put_delta) <= self.params.get("max_delta", 0.70),
                abs(days_to_expiration - self.params.get("days_to_expiration", 45))
                <= self.params.get("dte_tolerance", 15),
            ]
        )

        return {
            "stock_price": stock_price,
            "put_strike": put_strike,
            "put_premium": put_premium,
            "put_delta": put_delta,
            "shares": shares,
            "stock_cost": stock_cost,
            "put_cost": put_cost,
            "total_cost": total_cost,
            "protection_amount": protection_amount,
            "protection_pct": protection_pct,
            "max_loss": max_loss,
            "max_loss_pct": (max_loss / total_cost),
            "breakeven": breakeven,
            "breakeven_pct_move": ((breakeven - stock_price) / stock_price),
            "premium_pct": premium_pct,
            "cost_per_day": cost_per_day,
            "annualized_cost_pct": annualized_cost_pct,
            "days_to_expiration": days_to_expiration,
            "meets_criteria": meets_criteria,
        }

    def calculate_payoff(
        self,
        stock_entry: float,
        current_stock_price: float,
        put_strike: float,
        put_premium: float,
        shares: int = 100,
    ) -> dict[str, float]:
        """
        Calculate profit/loss for married put position.

        Args:
            stock_entry: Stock entry price
            current_stock_price: Current stock price
            put_strike: Put strike price
            put_premium: Put premium paid (per share)
            shares: Number of shares

        Returns:
            Dictionary with P&L breakdown
        """
        # Stock P&L
        stock_pl = (current_stock_price - stock_entry) * shares

        # Put P&L (long put)
        if current_stock_price < put_strike:
            # Put is ITM - has intrinsic value
            put_value = (put_strike - current_stock_price) * shares
        else:
            # Put is OTM - worthless
            put_value = 0.0

        put_cost = put_premium * shares
        put_pl = put_value - put_cost

        # Total P&L
        total_pl = stock_pl + put_pl
        total_cost = (stock_entry * shares) + put_cost + self.params.get("transaction_cost", 0.65)
        roi = (total_pl / total_cost) * 100 if total_cost > 0 else 0.0

        # Protection activated?
        protection_active = current_stock_price < put_strike

        return {
            "stock_pl": stock_pl,
            "put_pl": put_pl,
            "total_pl": total_pl,
            "roi": roi,
            "protection_active": float(protection_active),
            "put_value": put_value,
        }

    def compare_strikes(
        self,
        stock_price: float,
        strikes_and_premiums: list[tuple[float, float, float]],
        days_to_expiration: int,
        shares: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Compare multiple put strike prices.

        Args:
            stock_price: Current stock price
            strikes_and_premiums: List of (strike, premium, delta) tuples
            days_to_expiration: Days to expiration
            shares: Number of shares

        Returns:
            List of analysis results for each strike
        """
        results = []
        for strike, premium, delta in strikes_and_premiums:
            analysis = self.analyze_opportunity(
                stock_price=stock_price,
                put_strike=strike,
                put_premium=premium,
                put_delta=delta,
                days_to_expiration=days_to_expiration,
                shares=shares,
            )
            results.append(analysis)
        return results
