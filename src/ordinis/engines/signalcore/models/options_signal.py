"""
Options Signal Model - Generates signals for options trading strategies.

Analyzes market conditions and options data to identify opportunities for:
- Covered calls (neutral to bullish, high IV)
- Cash-secured puts (bullish, collecting premium)
- Iron condors (range-bound, high IV)
- Vertical spreads (directional with defined risk)

Integration:
    Extends SignalCore Model interface for compatibility with existing
    strategy loader and trading runtime.

Author: Ordinis Project
License: MIT
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class OptionsStrategyType(Enum):
    """Options strategy classification."""

    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    IRON_CONDOR = "iron_condor"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"


@dataclass
class OptionsOpportunity:
    """Represents an identified options opportunity."""

    strategy_type: OptionsStrategyType
    underlying_symbol: str
    score: float
    confidence: float
    iv_rank: float
    iv_percentile: float
    expected_premium_yield: float
    days_to_expiration: int
    strike_selection: dict
    greeks_profile: dict
    risk_reward_ratio: float
    max_profit: float
    max_loss: float
    breakeven: float | tuple[float, float]


class OptionsSignalModel(Model):
    """
    Options trading signal generator.

    Analyzes underlying price action, volatility regime, and market conditions
    to identify options trading opportunities.

    Strategies Supported:
    - Covered Call: Own stock + sell OTM call (neutral/bullish)
    - Cash-Secured Put: Sell OTM put + cash collateral (bullish)
    - Iron Condor: Sell OTM put spread + call spread (neutral)
    - Vertical Spreads: Directional plays with defined risk

    Entry Criteria:
    - IV Rank > min_iv_rank (default 30)
    - Directional bias from underlying (RSI, trend)
    - ADX for trend strength assessment
    - Volume confirmation

    Parameters:
        min_iv_rank: Minimum IV rank for premium strategies (default 30)
        min_iv_percentile: Minimum IV percentile (default 25)
        min_premium_yield: Minimum annualized premium yield (default 12%)
        target_delta_range: Delta range for strike selection (default 0.25-0.35)
        min_dte: Minimum days to expiration (default 30)
        max_dte: Maximum days to expiration (default 60)
        enable_spreads: Enable spread strategies (default True)
        enable_naked: Enable naked strategies (default False)
        rsi_period: RSI period for direction (default 14)
        adx_period: ADX period for trend strength (default 14)
    """

    def __init__(self, config: ModelConfig):
        """Initialize options signal model."""
        super().__init__(config)

        params = self.config.parameters

        # IV parameters
        self.min_iv_rank = params.get("min_iv_rank", 30)
        self.min_iv_percentile = params.get("min_iv_percentile", 25)

        # Premium parameters
        self.min_premium_yield = params.get("min_premium_yield", 0.12)

        # Strike selection
        self.target_delta_low = params.get("target_delta_low", 0.25)
        self.target_delta_high = params.get("target_delta_high", 0.35)

        # Expiration
        self.min_dte = params.get("min_dte", 30)
        self.max_dte = params.get("max_dte", 60)

        # Strategy flags
        self.enable_spreads = params.get("enable_spreads", True)
        self.enable_naked = params.get("enable_naked", False)
        self.enable_covered_call = params.get("enable_covered_call", True)
        self.enable_csp = params.get("enable_csp", True)
        self.enable_iron_condor = params.get("enable_iron_condor", True)

        # Technical parameters
        self.rsi_period = params.get("rsi_period", 14)
        self.adx_period = params.get("adx_period", 14)
        self.atr_period = params.get("atr_period", 14)
        self.vol_lookback = params.get("vol_lookback", 30)

        # Thresholds
        self.bullish_rsi_threshold = params.get("bullish_rsi_threshold", 55)
        self.bearish_rsi_threshold = params.get("bearish_rsi_threshold", 45)
        self.neutral_adx_threshold = params.get("neutral_adx_threshold", 25)

        # Min data requirement
        max_period = max(self.rsi_period, self.adx_period, self.atr_period, self.vol_lookback) + 10
        self.config.min_data_points = max(self.config.min_data_points, max_period)

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data."""
        if len(data) < self.config.min_data_points:
            return False, f"Insufficient data: {len(data)} < {self.config.min_data_points}"

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            return False, f"Missing columns: {missing}"

        return True, ""

    def _calculate_historical_volatility(self, close: pd.Series, window: int = 30) -> float:
        """Calculate annualized historical volatility."""
        returns = np.log(close / close.shift(1)).dropna()
        if len(returns) < window:
            return 0.0
        return returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)

    def _calculate_iv_rank(
        self,
        current_hv: float,
        hv_series: pd.Series,
        lookback: int = 252,
    ) -> float:
        """
        Calculate IV Rank (approximated from historical volatility).

        IV Rank = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV)

        Note: In production, use actual IV from options data.
        """
        lookback = min(len(hv_series), lookback)

        hv_window = hv_series.iloc[-lookback:]
        hv_min = hv_window.min()
        hv_max = hv_window.max()

        if hv_max == hv_min:
            return 50.0

        iv_rank = (current_hv - hv_min) / (hv_max - hv_min) * 100
        return min(max(iv_rank, 0), 100)

    def _calculate_iv_percentile(
        self,
        current_hv: float,
        hv_series: pd.Series,
        lookback: int = 252,
    ) -> float:
        """
        Calculate IV Percentile.

        IV Percentile = % of days in past year where IV was lower than current.
        """
        lookback = min(len(hv_series), lookback)

        hv_window = hv_series.iloc[-lookback:]
        below_current = (hv_window < current_hv).sum()

        return (below_current / len(hv_window)) * 100

    def _compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Compute ADX for trend strength."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        atr = tr.ewm(span=self.adx_period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=self.adx_period, adjust=False).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.ewm(span=self.adx_period, adjust=False).mean() / (atr + 1e-10)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=self.adx_period, adjust=False).mean()

        return adx

    def _determine_market_regime(
        self,
        rsi: float,
        adx: float,
        trend_direction: int,
    ) -> str:
        """
        Determine market regime for strategy selection.

        Returns:
            'bullish': Strong uptrend or oversold bounce
            'bearish': Strong downtrend or overbought decline
            'neutral': Range-bound or low momentum
        """
        # Strong trend detection
        if adx > 30:
            if trend_direction > 0:
                return "bullish"
            if trend_direction < 0:
                return "bearish"

        # Moderate trend / momentum
        if adx > self.neutral_adx_threshold:
            if rsi > self.bullish_rsi_threshold:
                return "bullish"
            if rsi < self.bearish_rsi_threshold:
                return "bearish"

        # Default to neutral for low ADX or mixed signals
        return "neutral"

    def _select_strategy(
        self,
        regime: str,
        iv_rank: float,
        iv_percentile: float,
    ) -> OptionsStrategyType | None:
        """
        Select appropriate options strategy based on conditions.

        Strategy Selection Matrix:
        - High IV + Neutral: Iron Condor (sell premium)
        - High IV + Bullish: Covered Call or CSP
        - High IV + Bearish: Bear Call Spread
        - Low IV + Bullish: Bull Call Spread (buy cheap premium)
        - Low IV + Bearish: Bear Put Spread
        """
        high_iv = iv_rank >= self.min_iv_rank and iv_percentile >= self.min_iv_percentile

        if high_iv:
            if regime == "neutral" and self.enable_iron_condor:
                return OptionsStrategyType.IRON_CONDOR
            if regime == "bullish":
                if self.enable_covered_call:
                    return OptionsStrategyType.COVERED_CALL
                if self.enable_csp:
                    return OptionsStrategyType.CASH_SECURED_PUT
            elif regime == "bearish" and self.enable_spreads:
                return OptionsStrategyType.BEAR_CALL_SPREAD
        elif self.enable_spreads:
            if regime == "bullish":
                return OptionsStrategyType.BULL_PUT_SPREAD
            if regime == "bearish":
                return OptionsStrategyType.BEAR_CALL_SPREAD

        return None

    async def generate(
        self,
        symbol: str,
        data: pd.DataFrame,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Generate options trading signal.

        Args:
            symbol: Underlying ticker symbol
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with options strategy recommendation or None
        """
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        close = data["close"]
        high = data["high"]
        low = data["low"]
        current_price = close.iloc[-1]

        # Calculate technical indicators
        rsi = TechnicalIndicators.rsi(close, self.rsi_period)
        current_rsi = rsi.iloc[-1]

        adx = self._compute_adx(high, low, close)
        current_adx = adx.iloc[-1]

        # Calculate volatility metrics
        returns = np.log(close / close.shift(1)).dropna()
        hv_series = returns.rolling(window=self.vol_lookback).std() * np.sqrt(252)
        current_hv = hv_series.iloc[-1] if len(hv_series) > 0 else 0.25

        iv_rank = self._calculate_iv_rank(current_hv, hv_series)
        iv_percentile = self._calculate_iv_percentile(current_hv, hv_series)

        # Determine trend direction
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
        trend_direction = (
            1 if current_price > sma_20 > sma_50 else (-1 if current_price < sma_20 < sma_50 else 0)
        )

        # Determine market regime
        regime = self._determine_market_regime(current_rsi, current_adx, trend_direction)

        # Select strategy
        strategy_type = self._select_strategy(regime, iv_rank, iv_percentile)

        if strategy_type is None:
            return None

        # Calculate target strike based on strategy
        atr = (high - low).rolling(self.atr_period).mean().iloc[-1]

        if strategy_type == OptionsStrategyType.COVERED_CALL:
            target_strike = current_price + (2 * atr)  # ~2 ATR OTM
            direction = Direction.NEUTRAL
            signal_type = SignalType.ENTRY
        elif strategy_type == OptionsStrategyType.CASH_SECURED_PUT:
            target_strike = current_price - (2 * atr)
            direction = Direction.LONG
            signal_type = SignalType.ENTRY
        elif strategy_type == OptionsStrategyType.IRON_CONDOR:
            target_strike = current_price  # ATM reference
            direction = Direction.NEUTRAL
            signal_type = SignalType.ENTRY
        elif strategy_type == OptionsStrategyType.BULL_PUT_SPREAD:
            target_strike = current_price - (1.5 * atr)
            direction = Direction.LONG
            signal_type = SignalType.ENTRY
        elif strategy_type == OptionsStrategyType.BEAR_CALL_SPREAD:
            target_strike = current_price + (1.5 * atr)
            direction = Direction.SHORT
            signal_type = SignalType.ENTRY
        else:
            return None

        # Calculate confidence based on IV conditions and trend clarity
        base_confidence = 0.5
        iv_bonus = 0.15 if iv_rank > 50 else 0.05
        trend_bonus = 0.10 if current_adx > 25 else 0.0
        confidence = min(base_confidence + iv_bonus + trend_bonus, 0.85)

        # Create signal
        return Signal(
            symbol=symbol,
            direction=direction,
            signal_type=signal_type,
            timestamp=timestamp,
            price=current_price,
            confidence=confidence,
            metadata={
                "model": "options_signal",
                "strategy_type": strategy_type.value,
                "regime": regime,
                "iv_rank": round(iv_rank, 2),
                "iv_percentile": round(iv_percentile, 2),
                "historical_volatility": round(current_hv * 100, 2),
                "rsi": round(current_rsi, 2),
                "adx": round(current_adx, 2),
                "trend_direction": trend_direction,
                "target_strike": round(target_strike, 2),
                "min_dte": self.min_dte,
                "max_dte": self.max_dte,
                "target_delta_range": [self.target_delta_low, self.target_delta_high],
            },
        )

    def reset_state(self):
        """Reset model state (stateless model, no-op)."""


if __name__ == "__main__":
    import asyncio

    async def main():
        print("=== Options Signal Model Demo ===\n")

        # Create sample data
        np.random.seed(42)
        n_bars = 100
        base_price = 100.0
        returns = np.random.normal(0.0005, 0.02, n_bars)
        prices = base_price * np.cumprod(1 + returns)

        data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.005, n_bars)),
                "high": prices * (1 + abs(np.random.normal(0, 0.01, n_bars))),
                "low": prices * (1 - abs(np.random.normal(0, 0.01, n_bars))),
                "close": prices,
                "volume": np.random.uniform(1e6, 5e6, n_bars).astype(int),
            }
        )
        data.index = pd.date_range(start="2024-01-01", periods=n_bars, freq="D")

        # Create model
        config = ModelConfig(
            model_id="options_demo",
            model_type="options_signal",
            version="1.0.0",
            parameters={
                "min_iv_rank": 30,
                "enable_iron_condor": True,
            },
            min_data_points=50,
        )
        model = OptionsSignalModel(config)

        # Generate signal
        signal = await model.generate("TEST", data, datetime.now())

        if signal:
            print("Signal Generated:")
            print(f"  Strategy: {signal.metadata['strategy_type']}")
            print(f"  Direction: {signal.direction.value}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Regime: {signal.metadata['regime']}")
            print(f"  IV Rank: {signal.metadata['iv_rank']:.1f}")
            print(f"  Target Strike: ${signal.metadata['target_strike']:.2f}")
        else:
            print("No signal generated (conditions not met)")

    asyncio.run(main())
