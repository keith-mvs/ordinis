"""Covered Call Strategy - Position Construction

This module implements covered call strategy construction with parameter
validation, strike selection, and position specification.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class CoveredCallParameters:
    """Configuration for covered call strategy."""

    symbol: str
    stock_quantity: int  # Must be multiple of 100
    strike_percent_otm: float = 0.05  # 5% OTM default
    days_to_expiration: int = 35  # Standard 30-45 day range
    delta_target: float | None = 0.30  # Target delta if using delta selection
    min_premium_yield: float = 0.015  # Minimum 1.5% premium yield
    auto_roll: bool = True  # Automatically roll on expiration


class CoveredCallStrategy:
    """Implement covered call options strategy."""

    def __init__(self, params: CoveredCallParameters):
        self.params = params
        self.validate_parameters()

    def validate_parameters(self) -> None:
        """Ensure parameters meet requirements."""
        if self.params.stock_quantity % 100 != 0:
            raise ValueError("Stock quantity must be multiple of 100")

        if not 0.01 <= self.params.strike_percent_otm <= 0.20:
            raise ValueError("Strike OTM percent must be 1-20%")

        if not 7 <= self.params.days_to_expiration <= 90:
            raise ValueError("Days to expiration must be 7-90")

        if self.params.delta_target:
            if not 0.10 <= self.params.delta_target <= 0.50:
                raise ValueError("Delta target must be 0.10-0.50")

    def construct_position(self, market_data: dict, options_chain: dict) -> dict:
        """Construct covered call position from market data."""
        stock_price = market_data["last_price"]

        # Calculate target strike
        target_strike = self._calculate_target_strike(stock_price, options_chain)

        # Find optimal option contract
        optimal_call = self._select_optimal_call(target_strike, options_chain, stock_price)

        if not optimal_call:
            raise ValueError("No suitable call option found")

        # Construct position specification
        position = {
            "strategy_type": "covered_call",
            "stock_leg": {
                "symbol": self.params.symbol,
                "quantity": self.params.stock_quantity,
                "side": "long",
            },
            "call_leg": {
                "symbol": optimal_call["symbol"],
                "strike": optimal_call["strike"],
                "expiration": optimal_call["expiration"],
                "quantity": self.params.stock_quantity // 100,
                "side": "short",
                "limit_price": optimal_call["bid"],
            },
            "metrics": {
                "stock_price": stock_price,
                "strike_price": optimal_call["strike"],
                "premium": optimal_call["bid"],
                "premium_yield": optimal_call["bid"] / stock_price,
                "max_profit": (optimal_call["strike"] - stock_price) + optimal_call["bid"],
                "break_even": stock_price - optimal_call["bid"],
            },
            "entry_time": datetime.now(),
        }

        return position

    def _calculate_target_strike(self, stock_price: float, options_chain: dict) -> float:
        """Calculate target strike price based on parameters."""
        if self.params.delta_target:
            return self._strike_from_delta(self.params.delta_target, options_chain)
        target_strike = stock_price * (1 + self.params.strike_percent_otm)
        return self._round_to_strike(target_strike, options_chain)

    def _select_optimal_call(
        self, target_strike: float, options_chain: dict, stock_price: float
    ) -> dict | None:
        """Select optimal call option from chain."""
        candidates = []

        for option in options_chain["calls"]:
            # Filter by strike proximity, expiration, liquidity
            if not self._option_meets_criteria(option, target_strike, stock_price):
                continue

            score = self._score_option(option, stock_price, target_strike)
            candidates.append({"option": option, "score": score})

        if not candidates:
            return None

        best = max(candidates, key=lambda x: x["score"])
        return best["option"]

    def _option_meets_criteria(
        self, option: dict, target_strike: float, stock_price: float
    ) -> bool:
        """Check if option meets selection criteria."""
        strike_diff = abs(option["strike"] - target_strike) / target_strike
        premium_yield = option["bid"] / stock_price

        return (
            strike_diff < 0.025  # Within 2.5% of target
            and premium_yield >= self.params.min_premium_yield
            and option["open_interest"] >= 50
            and option["volume"] >= 10
        )

    def _score_option(self, option: dict, stock_price: float, target_strike: float) -> float:
        """Score option contract for selection."""
        score = 0.0

        # Premium yield (40% weight)
        premium_yield = option["bid"] / stock_price
        score += premium_yield * 40

        # Strike proximity (30% weight)
        strike_proximity = 1 - abs(option["strike"] - target_strike) / target_strike
        score += strike_proximity * 30

        # Liquidity (30% weight)
        if option["open_interest"] > 500:
            score += 20
        elif option["open_interest"] > 100:
            score += 10

        spread_pct = (option["ask"] - option["bid"]) / option["bid"]
        if spread_pct < 0.05:
            score += 10
        elif spread_pct < 0.10:
            score += 5

        return score

    def _strike_from_delta(self, target_delta: float, options_chain: dict) -> float:
        """Find strike price closest to target delta."""
        closest_strike = None
        min_delta_diff = float("inf")

        for option in options_chain["calls"]:
            delta_diff = abs(option["delta"] - target_delta)
            if delta_diff < min_delta_diff:
                min_delta_diff = delta_diff
                closest_strike = option["strike"]

        return closest_strike

    def _round_to_strike(self, price: float, options_chain: dict) -> float:
        """Round price to nearest available strike."""
        strikes = [opt["strike"] for opt in options_chain["calls"]]
        return min(strikes, key=lambda x: abs(x - price))
