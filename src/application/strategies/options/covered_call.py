"""
Covered Call Options Strategy

Income-generation strategy that combines long stock position with short call option.
Ideal for neutral-to-slightly-bullish outlook on underlying with high implied volatility.

Strategy Profile:
- Type: Income/Premium Collection
- Market Outlook: Neutral to Slightly Bullish
- Max Profit: Premium Received + (Strike - Stock Price)
- Max Loss: Stock Price - Premium Received (if stock goes to zero)
- Risk Level: Moderate (stock ownership risk, capped upside)

Author: Ordinis Project
License: MIT
"""

from datetime import datetime
from typing import Any

import pandas as pd

from application.strategies.base import BaseStrategy
from engines.optionscore import OptionsCoreEngine
from engines.signalcore.core.signal import Direction, Signal, SignalType


class CoveredCallStrategy(BaseStrategy):
    """
    Covered call strategy implementation.

    Sells call options against long stock position to generate income.

    Entry Criteria:
    - Own or willing to buy 100 shares of underlying
    - Neutral to slightly bullish outlook
    - High implied volatility (premium collection opportunity)
    - Strike selection: OTM (5-10% above current price) for growth + income

    Exit Criteria:
    - Option expires worthless (keep premium)
    - Stock called away at strike (profit capped)
    - Buy back call if stock drops significantly
    - Roll option to next expiration before assignment

    Parameters:
        - min_premium_yield: Minimum annualized premium yield (default 12%)
        - min_delta: Minimum call delta for selection (default 0.25)
        - max_delta: Maximum call delta for selection (default 0.40)
        - days_to_expiration: Target DTE for options (default 30-45)
        - strike_otm_pct: OTM percentage for strike selection (default 5%)

    Example:
        >>> strategy = CoveredCallStrategy(
        ...     name="AAPL Covered Call",
        ...     min_premium_yield=0.15,
        ...     days_to_expiration=45
        ... )
        >>> signal = strategy.generate_signal(data, timestamp)
    """

    def configure(self):
        """Configure covered call parameters with defaults."""
        self.params.setdefault("min_premium_yield", 0.12)  # 12% annualized
        self.params.setdefault("min_delta", 0.25)  # 25 delta
        self.params.setdefault("max_delta", 0.40)  # 40 delta
        self.params.setdefault("days_to_expiration", 45)  # 30-45 DTE ideal
        self.params.setdefault("strike_otm_pct", 0.05)  # 5% OTM
        self.params.setdefault("min_dte", 30)  # Minimum days to expiration
        self.params.setdefault("max_dte", 60)  # Maximum days to expiration

    def generate_signal(
        self, data: pd.DataFrame, timestamp: datetime, options_engine: OptionsCoreEngine = None
    ) -> Signal | None:
        """
        Generate covered call signal.

        Args:
            data: OHLCV price data for underlying
            timestamp: Current timestamp
            options_engine: OptionsCore engine for options pricing (optional)

        Returns:
            Signal object or None if no opportunity

        Note:
            If options_engine is not provided, returns None (requires live options data).
        """
        try:
            # Validate data
            is_valid, message = self.validate_data(data)
            if not is_valid:
                return None

            # Check if we have options engine
            if options_engine is None:
                return None

            # Get current price
            current_price = data["close"].iloc[-1]
            symbol = data.index.name if hasattr(data.index, "name") else "UNKNOWN"

            # Calculate target strike (OTM)
            strike_otm_pct = self.params["strike_otm_pct"]
            target_strike = current_price * (1 + strike_otm_pct)

            # Calculate target expiration range
            min_dte = self.params["min_dte"]
            max_dte = self.params["max_dte"]

            # Generate signal metadata
            metadata = {
                "strategy": self.name,
                "strategy_type": "covered_call",
                "underlying_price": current_price,
                "target_strike": target_strike,
                "strike_otm_pct": strike_otm_pct,
                "min_dte": min_dte,
                "max_dte": max_dte,
                "timestamp": timestamp.isoformat(),
                "selection_criteria": {
                    "min_delta": self.params["min_delta"],
                    "max_delta": self.params["max_delta"],
                    "min_premium_yield": self.params["min_premium_yield"],
                },
            }

            # In a live implementation, you would:
            # 1. Fetch options chain using options_engine
            # 2. Filter for target expiration range
            # 3. Find contracts with delta in target range
            # 4. Calculate annualized premium yield
            # 5. Select best contract meeting criteria

            # For now, generate a signal indicating covered call opportunity exists
            # Real implementation would include specific contract details in metadata

            signal = Signal(
                symbol=symbol,
                timestamp=timestamp,
                signal_type=SignalType.ENTRY,
                direction=Direction.NEUTRAL,  # Covered call is neutral strategy
                probability=0.70,  # Moderate probability of profit
                expected_return=self.params["min_premium_yield"] / 4,  # Quarterly estimate
                confidence_interval=(0.0, target_strike - current_price),
                score=0.65,
                model_id=self.name,
                model_version="1.0.0",
                feature_contributions={},
                metadata=metadata,
            )

            return signal

        except Exception:
            return None

    def get_description(self) -> str:
        """Return strategy description."""
        return (
            f"Covered Call Strategy - Sells OTM call options ({self.params['strike_otm_pct']*100:.0f}% OTM) "
            f"against long stock position to generate income. Target premium yield: "
            f"{self.params['min_premium_yield']*100:.0f}% annualized, DTE: {self.params['days_to_expiration']} days."
        )

    def get_required_bars(self) -> int:
        """Return minimum bars needed."""
        return 20  # Need some price history for volatility assessment

    def analyze_opportunity(
        self,
        underlying_price: float,
        call_strike: float,
        call_premium: float,
        call_delta: float,
        days_to_expiration: int,
    ) -> dict[str, Any]:
        """
        Analyze covered call opportunity metrics.

        Args:
            underlying_price: Current stock price
            call_strike: Call option strike price
            call_premium: Call option premium
            call_delta: Call option delta
            days_to_expiration: Days until expiration

        Returns:
            Dictionary with opportunity analysis

        Example:
            >>> analysis = strategy.analyze_opportunity(
            ...     underlying_price=100.0,
            ...     call_strike=105.0,
            ...     call_premium=2.50,
            ...     call_delta=0.30,
            ...     days_to_expiration=45
            ... )
            >>> print(f"Max profit: ${analysis['max_profit']:.2f}")
            >>> print(f"Annualized yield: {analysis['annualized_yield']*100:.1f}%")
        """
        # Calculate metrics
        max_profit = call_premium + (call_strike - underlying_price)
        max_loss = underlying_price - call_premium  # If stock goes to zero
        breakeven = underlying_price - call_premium

        # Downside protection
        downside_protection_pct = call_premium / underlying_price

        # Upside potential if called away
        upside_potential = call_strike - underlying_price + call_premium
        upside_potential_pct = upside_potential / underlying_price

        # Annualized premium yield
        premium_yield = call_premium / underlying_price
        annualized_yield = (
            premium_yield * (365 / days_to_expiration) if days_to_expiration > 0 else 0
        )

        # Probability of profit (approximated by delta)
        # Delta represents probability ITM, so 1-delta is probability OTM (profit)
        prob_profit = 1 - abs(call_delta)

        # Return on risk (premium / max loss)
        return_on_risk = call_premium / max_loss if max_loss > 0 else 0

        return {
            "underlying_price": underlying_price,
            "call_strike": call_strike,
            "call_premium": call_premium,
            "call_delta": call_delta,
            "days_to_expiration": days_to_expiration,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": breakeven,
            "downside_protection_pct": downside_protection_pct,
            "upside_potential": upside_potential,
            "upside_potential_pct": upside_potential_pct,
            "premium_yield": premium_yield,
            "annualized_yield": annualized_yield,
            "prob_profit": prob_profit,
            "return_on_risk": return_on_risk,
            "meets_criteria": annualized_yield >= self.params["min_premium_yield"]
            and self.params["min_delta"] <= abs(call_delta) <= self.params["max_delta"],
        }

    def calculate_payoff(
        self, underlying_price: float, stock_entry: float, call_strike: float, call_premium: float
    ) -> dict[str, float]:
        """
        Calculate covered call P&L at various underlying prices.

        Args:
            underlying_price: Price point to calculate P&L
            stock_entry: Stock purchase price
            call_strike: Call option strike price
            call_premium: Premium received from selling call

        Returns:
            Dictionary with P&L breakdown

        Example:
            >>> payoff = strategy.calculate_payoff(
            ...     underlying_price=110.0,
            ...     stock_entry=100.0,
            ...     call_strike=105.0,
            ...     call_premium=2.50
            ... )
            >>> print(f"Total P&L: ${payoff['total_pnl']:.2f}")
        """
        # Stock P&L
        stock_pnl = underlying_price - stock_entry

        # Call option P&L (we're short the call)
        # If stock > strike, call is ITM and we lose (stock called away)
        # If stock < strike, call expires worthless and we keep premium
        if underlying_price > call_strike:
            call_pnl = call_premium - (underlying_price - call_strike)
        else:
            call_pnl = call_premium

        # Total P&L
        total_pnl = stock_pnl + call_pnl

        # Return on investment
        total_capital = stock_entry
        roi = total_pnl / total_capital if total_capital > 0 else 0

        return {
            "underlying_price": underlying_price,
            "stock_pnl": stock_pnl,
            "call_pnl": call_pnl,
            "total_pnl": total_pnl,
            "total_capital": total_capital,
            "roi": roi,
            "stock_called_away": underlying_price > call_strike,
        }
