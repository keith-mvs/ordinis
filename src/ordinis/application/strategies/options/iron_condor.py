"""
Iron Condor Options Strategy

Neutral income strategy that profits when underlying stays within a defined range.
Sells OTM put spread AND OTM call spread simultaneously.

Strategy Profile:
- Type: Premium Collection (Delta Neutral)
- Market Outlook: Neutral / Range-bound
- Max Profit: Net Premium Received
- Max Loss: Width of Either Spread - Net Premium
- Risk Level: Defined risk on both sides

Structure:
    - Buy 1 OTM Put (lower strike) - protection
    - Sell 1 OTM Put (higher strike) - income
    - Sell 1 OTM Call (lower strike) - income
    - Buy 1 OTM Call (higher strike) - protection

Ideal Conditions:
    - High IV Rank (>30) for premium collection
    - Low ADX (<25) indicating range-bound market
    - Expected low movement through expiration

Author: Ordinis Project
License: MIT
"""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ordinis.application.strategies.base import BaseStrategy
from ordinis.engines.optionscore import (
    BlackScholesEngine,
    GreeksCalculator,
    OptionType,
    PricingParameters,
)
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType


@dataclass
class IronCondorLegs:
    """Iron condor leg parameters."""

    underlying_price: float
    put_long_strike: float
    put_short_strike: float
    call_short_strike: float
    call_long_strike: float
    days_to_expiration: int
    implied_volatility: float


@dataclass
class IronCondorAnalysis:
    """Complete iron condor analysis."""

    # Structure
    legs: IronCondorLegs
    net_credit: float

    # P&L
    max_profit: float
    max_loss: float
    breakeven_lower: float
    breakeven_upper: float

    # Greeks
    position_delta: float
    position_gamma: float
    position_theta: float
    position_vega: float

    # Risk metrics
    risk_reward_ratio: float
    probability_of_profit: float
    width_put_spread: float
    width_call_spread: float

    # Flags
    meets_criteria: bool
    score: float


class IronCondorStrategy(BaseStrategy):
    """
    Iron Condor strategy implementation.

    Sells both a put spread and call spread to collect premium while
    defining risk on both sides. Profits when underlying stays in range.

    Entry Criteria:
    - IV Rank >= min_iv_rank (premium opportunity)
    - ADX < max_adx (not trending)
    - Sufficient premium (min_credit_per_spread)
    - Acceptable risk/reward ratio

    Exit Criteria:
    - Close at 50% profit target
    - Close at 21 DTE (reduce gamma risk)
    - Close if underlying breaches short strikes
    - Close if loss reaches max_loss_pct

    Parameters:
        - min_iv_rank: Minimum IV rank for entry (default 30)
        - max_adx: Maximum ADX for neutral conditions (default 25)
        - target_delta_short: Target delta for short strikes (default 0.15)
        - spread_width: Width of each spread in dollars (default 5.0)
        - min_credit_ratio: Min credit as % of spread width (default 0.25)
        - min_dte: Minimum days to expiration (default 30)
        - max_dte: Maximum days to expiration (default 60)
        - profit_target_pct: Close at this % of max profit (default 50)
        - exit_dte: Exit when DTE reaches this value (default 21)

    Example:
        >>> strategy = IronCondorStrategy(
        ...     name="SPY Iron Condor",
        ...     min_iv_rank=35,
        ...     spread_width=5.0
        ... )
        >>> analysis = strategy.analyze_opportunity(legs, volatility=0.20)
    """

    def configure(self):
        """Configure iron condor parameters with defaults."""
        # Entry criteria
        self.params.setdefault("min_iv_rank", 30)
        self.params.setdefault("max_adx", 25)

        # Strike selection
        self.params.setdefault("target_delta_short", 0.15)  # ~15 delta OTM
        self.params.setdefault("spread_width", 5.0)  # $5 wide spreads

        # Premium requirements
        self.params.setdefault("min_credit_ratio", 0.25)  # 25% of spread width

        # Expiration
        self.params.setdefault("min_dte", 30)
        self.params.setdefault("max_dte", 60)
        self.params.setdefault("target_dte", 45)

        # Exit parameters
        self.params.setdefault("profit_target_pct", 50)  # Close at 50% profit
        self.params.setdefault("exit_dte", 21)  # Exit at 21 DTE
        self.params.setdefault("max_loss_pct", 200)  # 2x credit received

        # Position sizing
        self.params.setdefault("max_position_risk_pct", 0.02)  # 2% max risk

    async def generate_signal(
        self,
        data: pd.DataFrame,
        timestamp: datetime,
        iv_rank: float | None = None,
        adx: float | None = None,
    ) -> Signal | None:
        """
        Generate iron condor signal.

        Args:
            data: OHLCV price data for underlying
            timestamp: Current timestamp
            iv_rank: Current IV rank (0-100)
            adx: Current ADX value

        Returns:
            Signal object or None if no opportunity
        """
        try:
            is_valid, _message = self.validate_data(data)
            if not is_valid:
                return None

            # Check IV rank requirement
            if iv_rank is not None and iv_rank < self.params["min_iv_rank"]:
                return None

            # Check ADX requirement (want low ADX for range-bound)
            if adx is not None and adx > self.params["max_adx"]:
                return None

            current_price = data["close"].iloc[-1]
            symbol = data.index.name if hasattr(data.index, "name") else "UNKNOWN"

            # Calculate strike positions
            spread_width = self.params["spread_width"]
            # Use ATR for dynamic strike placement
            atr = (data["high"] - data["low"]).rolling(14).mean().iloc[-1]

            # Short strikes: ~1.5-2 ATR from current price
            put_short_strike = round((current_price - (1.5 * atr)) / 5) * 5  # Round to $5
            call_short_strike = round((current_price + (1.5 * atr)) / 5) * 5

            # Long strikes: spread_width away from short strikes
            put_long_strike = put_short_strike - spread_width
            call_long_strike = call_short_strike + spread_width

            metadata = {
                "strategy": self.name,
                "strategy_type": "iron_condor",
                "underlying_price": current_price,
                "put_long_strike": put_long_strike,
                "put_short_strike": put_short_strike,
                "call_short_strike": call_short_strike,
                "call_long_strike": call_long_strike,
                "spread_width": spread_width,
                "target_dte": self.params["target_dte"],
                "iv_rank": iv_rank,
                "adx": adx,
                "profit_target_pct": self.params["profit_target_pct"],
                "exit_dte": self.params["exit_dte"],
                "timestamp": timestamp.isoformat(),
            }

            signal = Signal(
                symbol=symbol,
                timestamp=timestamp,
                signal_type=SignalType.ENTRY,
                direction=Direction.NEUTRAL,
                probability=0.65,  # Typical IC probability
                expected_return=self.params["min_credit_ratio"] * 0.5,  # Conservative
                confidence_interval=(-spread_width, spread_width * self.params["min_credit_ratio"]),
                score=0.60,
                model_id=self.name,
                model_version="1.0.0",
                feature_contributions={},
                metadata=metadata,
            )

            return signal

        except Exception:
            return None

    def analyze_opportunity(
        self,
        legs: IronCondorLegs,
        risk_free_rate: float = 0.05,
    ) -> IronCondorAnalysis:
        """
        Analyze iron condor opportunity with full pricing and Greeks.

        Args:
            legs: Iron condor leg parameters
            risk_free_rate: Risk-free interest rate

        Returns:
            Complete analysis with P&L, Greeks, and risk metrics
        """
        engine = BlackScholesEngine()
        greeks_calc = GreeksCalculator(engine)

        T = legs.days_to_expiration / 365.0

        # Price each leg
        # Long Put (buy protection)
        put_long_params = PricingParameters(
            S=legs.underlying_price,
            K=legs.put_long_strike,
            T=T,
            r=risk_free_rate,
            sigma=legs.implied_volatility,
        )
        put_long_price = engine.price_put(put_long_params)
        put_long_greeks = greeks_calc.all_greeks(put_long_params, OptionType.PUT)

        # Short Put (sell premium)
        put_short_params = PricingParameters(
            S=legs.underlying_price,
            K=legs.put_short_strike,
            T=T,
            r=risk_free_rate,
            sigma=legs.implied_volatility,
        )
        put_short_price = engine.price_put(put_short_params)
        put_short_greeks = greeks_calc.all_greeks(put_short_params, OptionType.PUT)

        # Short Call (sell premium)
        call_short_params = PricingParameters(
            S=legs.underlying_price,
            K=legs.call_short_strike,
            T=T,
            r=risk_free_rate,
            sigma=legs.implied_volatility,
        )
        call_short_price = engine.price_call(call_short_params)
        call_short_greeks = greeks_calc.all_greeks(call_short_params, OptionType.CALL)

        # Long Call (buy protection)
        call_long_params = PricingParameters(
            S=legs.underlying_price,
            K=legs.call_long_strike,
            T=T,
            r=risk_free_rate,
            sigma=legs.implied_volatility,
        )
        call_long_price = engine.price_call(call_long_params)
        call_long_greeks = greeks_calc.all_greeks(call_long_params, OptionType.CALL)

        # Calculate net credit
        # Receive: short put + short call
        # Pay: long put + long call
        net_credit = (put_short_price + call_short_price) - (put_long_price + call_long_price)

        # Calculate spread widths
        width_put_spread = legs.put_short_strike - legs.put_long_strike
        width_call_spread = legs.call_long_strike - legs.call_short_strike

        # P&L calculations
        max_profit = net_credit * 100  # Per contract
        max_loss = (max(width_put_spread, width_call_spread) - net_credit) * 100

        # Breakevens
        breakeven_lower = legs.put_short_strike - net_credit
        breakeven_upper = legs.call_short_strike + net_credit

        # Position Greeks (long positions are positive, short are negative)
        position_delta = (
            put_long_greeks["delta"]
            - put_short_greeks["delta"]
            - call_short_greeks["delta"]
            + call_long_greeks["delta"]
        ) * 100

        position_gamma = (
            put_long_greeks["gamma"]
            - put_short_greeks["gamma"]
            - call_short_greeks["gamma"]
            + call_long_greeks["gamma"]
        ) * 100

        position_theta = (
            put_long_greeks["theta"]
            - put_short_greeks["theta"]
            - call_short_greeks["theta"]
            + call_long_greeks["theta"]
        ) * 100

        position_vega = (
            put_long_greeks["vega"]
            - put_short_greeks["vega"]
            - call_short_greeks["vega"]
            + call_long_greeks["vega"]
        ) * 100

        # Risk metrics
        risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0

        # Approximate probability of profit using delta
        # POP ~ 1 - (short put delta + short call delta)
        prob_profit = 1 - (abs(put_short_greeks["delta"]) + abs(call_short_greeks["delta"]))

        # Score based on criteria
        score = 0.0
        meets_criteria = True

        # Credit ratio check
        credit_ratio = net_credit / max(width_put_spread, width_call_spread)
        if credit_ratio >= self.params["min_credit_ratio"]:
            score += 0.3
        else:
            meets_criteria = False

        # Probability of profit
        if prob_profit >= 0.50:
            score += 0.3
        elif prob_profit >= 0.40:
            score += 0.15

        # Theta positive (time decay in our favor)
        if position_theta > 0:
            score += 0.2

        # Delta neutral
        if abs(position_delta) < 10:
            score += 0.2

        return IronCondorAnalysis(
            legs=legs,
            net_credit=net_credit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_lower=breakeven_lower,
            breakeven_upper=breakeven_upper,
            position_delta=position_delta,
            position_gamma=position_gamma,
            position_theta=position_theta,
            position_vega=position_vega,
            risk_reward_ratio=risk_reward_ratio,
            probability_of_profit=prob_profit,
            width_put_spread=width_put_spread,
            width_call_spread=width_call_spread,
            meets_criteria=meets_criteria,
            score=score,
        )

    def calculate_payoff(
        self,
        underlying_price: float,
        legs: IronCondorLegs,
        net_credit: float,
    ) -> dict[str, float]:
        """
        Calculate iron condor P&L at a given underlying price.

        Args:
            underlying_price: Price point to calculate P&L
            legs: Iron condor leg parameters
            net_credit: Net credit received

        Returns:
            Dictionary with P&L breakdown
        """
        # Put spread P&L (we're short the spread)
        put_long_value = max(legs.put_long_strike - underlying_price, 0)
        put_short_value = max(legs.put_short_strike - underlying_price, 0)
        put_spread_pnl = put_short_value - put_long_value  # We're short

        # Call spread P&L (we're short the spread)
        call_short_value = max(underlying_price - legs.call_short_strike, 0)
        call_long_value = max(underlying_price - legs.call_long_strike, 0)
        call_spread_pnl = call_short_value - call_long_value  # We're short

        # Total P&L
        total_pnl = (net_credit - put_spread_pnl - call_spread_pnl) * 100

        return {
            "underlying_price": underlying_price,
            "put_spread_value": put_spread_pnl,
            "call_spread_value": call_spread_pnl,
            "net_credit": net_credit,
            "total_pnl": total_pnl,
            "in_profit_zone": legs.put_short_strike <= underlying_price <= legs.call_short_strike,
        }

    def get_description(self) -> str:
        """Return strategy description."""
        return (
            f"Iron Condor Strategy - Sells {self.params['target_delta_short']*100:.0f}-delta "
            f"put spread and call spread, ${self.params['spread_width']:.0f} wide. "
            f"Target DTE: {self.params['target_dte']} days. "
            f"Profit target: {self.params['profit_target_pct']}% of max profit."
        )

    def get_required_bars(self) -> int:
        """Return minimum bars needed."""
        return 30


if __name__ == "__main__":
    print("=== Iron Condor Strategy Demo ===\n")

    # Create strategy
    strategy = IronCondorStrategy(
        name="Demo Iron Condor",
        params={
            "min_iv_rank": 30,
            "spread_width": 5.0,
            "min_credit_ratio": 0.30,
        },
    )

    # Create sample legs
    legs = IronCondorLegs(
        underlying_price=450.0,
        put_long_strike=420.0,
        put_short_strike=425.0,
        call_short_strike=475.0,
        call_long_strike=480.0,
        days_to_expiration=45,
        implied_volatility=0.20,
    )

    # Analyze
    analysis = strategy.analyze_opportunity(legs)

    print(f"Underlying Price: ${legs.underlying_price:.2f}")
    print("\nStructure:")
    print(f"  Put Spread: ${legs.put_long_strike}/{legs.put_short_strike}")
    print(f"  Call Spread: ${legs.call_short_strike}/{legs.call_long_strike}")
    print("\nP&L:")
    print(f"  Net Credit: ${analysis.net_credit:.2f}")
    print(f"  Max Profit: ${analysis.max_profit:.2f}")
    print(f"  Max Loss: ${analysis.max_loss:.2f}")
    print(f"  Risk/Reward: {analysis.risk_reward_ratio:.2f}")
    print("\nBreakevens:")
    print(f"  Lower: ${analysis.breakeven_lower:.2f}")
    print(f"  Upper: ${analysis.breakeven_upper:.2f}")
    print("\nGreeks:")
    print(f"  Delta: {analysis.position_delta:.2f}")
    print(f"  Gamma: {analysis.position_gamma:.4f}")
    print(f"  Theta: ${analysis.position_theta:.2f}/day")
    print(f"  Vega: ${analysis.position_vega:.2f}")
    print(f"\nProbability of Profit: {analysis.probability_of_profit*100:.1f}%")
    print(f"Score: {analysis.score:.2f}")
    print(f"Meets Criteria: {analysis.meets_criteria}")
