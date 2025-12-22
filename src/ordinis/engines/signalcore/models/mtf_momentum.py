"""
Multi-Timeframe Momentum Strategy.

Combines daily momentum ranking with intraday stochastic oscillator
for precise entry timing. Only enters when both timeframes align.

Theory:
- Momentum (12-1 month) identifies winning stocks
- Stochastic identifies pullback entries
- Combining gives better entry prices
- Avoid momentum traps via oversold confirmation
"""

from dataclasses import dataclass
from datetime import datetime
import logging

import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class MTFConfig:
    """Multi-Timeframe Momentum configuration."""

    # Momentum parameters
    formation_period: int = 252  # 12 months
    skip_period: int = 21  # Skip most recent month
    momentum_percentile: float = 0.8  # Top 20% = winners

    # Stochastic parameters
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: float = 30.0
    stoch_overbought: float = 70.0

    # Risk parameters
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    atr_tp_mult: float = 3.0


class MTFMomentumModel(Model):
    """
    Multi-Timeframe Momentum Strategy.

    Signal Logic:
    1. Calculate 12-1 month momentum
    2. Classify as winner (top quintile) or loser (bottom quintile)
    3. Calculate stochastic oscillator
    4. LONG: Winner + bullish stoch cross + oversold
    5. SHORT: Loser + bearish stoch cross + overbought

    Example:
        config = ModelConfig(
            model_id="mtf_momentum",
            model_type="momentum",
            parameters={"momentum_percentile": 0.8}
        )
        model = MTFMomentumModel(config)
        signal = await model.generate("AAPL", df, timestamp)
    """

    def __init__(self, config: ModelConfig):
        """Initialize MTF Momentum model."""
        super().__init__(config)
        params = config.parameters or {}

        self.mtf_config = MTFConfig(
            formation_period=params.get("formation_period", 252),
            skip_period=params.get("skip_period", 21),
            momentum_percentile=params.get("momentum_percentile", 0.8),
            stoch_k_period=params.get("stoch_k_period", 14),
            stoch_d_period=params.get("stoch_d_period", 3),
            stoch_oversold=params.get("stoch_oversold", 30.0),
            stoch_overbought=params.get("stoch_overbought", 70.0),
            atr_period=params.get("atr_period", 14),
            atr_stop_mult=params.get("atr_stop_mult", 2.0),
            atr_tp_mult=params.get("atr_tp_mult", 3.0),
        )

        # For multi-stock ranking (set externally)
        self.universe_momentum: pd.Series | None = None

    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """
        Calculate 12-1 month momentum.

        Returns the return from 12 months ago to 1 month ago,
        skipping the most recent month (short-term reversal).
        """
        close = df["close"]

        if len(close) < self.mtf_config.formation_period:
            return 0.0

        # Price 12 months ago
        price_12m = close.iloc[-self.mtf_config.formation_period]
        # Price 1 month ago
        price_1m = close.iloc[-self.mtf_config.skip_period]

        # 12-1 month return
        return (price_1m / price_12m) - 1

    def _calculate_stochastic(self, df: pd.DataFrame) -> tuple[float, float, bool, bool]:
        """
        Calculate Stochastic Oscillator %K and %D.

        Returns:
            Tuple of (current_k, current_d, bullish_cross, bearish_cross)
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        k_period = self.mtf_config.stoch_k_period
        d_period = self.mtf_config.stoch_d_period

        # Raw %K
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()

        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)

        # Slow %K (smoothed)
        slow_k = raw_k.rolling(d_period).mean()

        # %D (signal line)
        slow_d = slow_k.rolling(d_period).mean()

        current_k = slow_k.iloc[-1]
        current_d = slow_d.iloc[-1]
        prev_k = slow_k.iloc[-2]
        prev_d = slow_d.iloc[-2]

        # Crossovers
        bullish_cross = (prev_k <= prev_d) and (current_k > current_d)
        bearish_cross = (prev_k >= prev_d) and (current_k < current_d)

        return current_k, current_d, bullish_cross, bearish_cross

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.mtf_config.atr_period).mean().iloc[-1]

    def set_universe_momentum(self, momentum_scores: pd.Series):
        """
        Set momentum scores for the entire universe.

        Used for cross-sectional ranking when multiple stocks available.

        Args:
            momentum_scores: Series with symbol as index, momentum as value
        """
        self.universe_momentum = momentum_scores

    def _is_winner(self, symbol: str, momentum: float) -> bool:
        """Check if symbol is in top momentum quintile."""
        if self.universe_momentum is not None:
            # Cross-sectional ranking
            threshold = self.universe_momentum.quantile(self.mtf_config.momentum_percentile)
            return momentum >= threshold
        # Absolute threshold (fallback)
        return momentum > 0.15  # 15% return

    def _is_loser(self, symbol: str, momentum: float) -> bool:
        """Check if symbol is in bottom momentum quintile."""
        if self.universe_momentum is not None:
            threshold = self.universe_momentum.quantile(1 - self.mtf_config.momentum_percentile)
            return momentum <= threshold
        return momentum < -0.15

    async def generate(
        self,
        symbol: str,
        data: pd.DataFrame,
        timestamp: datetime,
    ) -> Signal | None:
        """
        Generate multi-timeframe momentum signal.

        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            timestamp: Current timestamp

        Returns:
            Signal if entry conditions met
        """
        min_bars = max(self.mtf_config.formation_period, self.mtf_config.stoch_k_period * 2)

        if len(data) < min_bars:
            return None

        # Calculate momentum (daily timeframe concept)
        momentum = self._calculate_momentum(data)

        # Calculate stochastic (intraday timing)
        stoch_k, stoch_d, bullish_cross, bearish_cross = self._calculate_stochastic(data)

        # Classify momentum
        is_winner = self._is_winner(symbol, momentum)
        is_loser = self._is_loser(symbol, momentum)

        # Stochastic conditions
        is_oversold = stoch_k < self.mtf_config.stoch_oversold
        is_overbought = stoch_k > self.mtf_config.stoch_overbought

        # Determine signal
        signal_type = SignalType.HOLD
        direction = 0

        # LONG: Winner + bullish cross in oversold
        if is_winner and bullish_cross and is_oversold:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG

        # SHORT: Loser + bearish cross in overbought
        elif is_loser and bearish_cross and is_overbought:
            signal_type = SignalType.ENTRY
            direction = Direction.SHORT

        if signal_type == SignalType.HOLD:
            return Signal(
                symbol=symbol,
                timestamp=timestamp,
                signal_type=SignalType.HOLD,
                direction=Direction.NEUTRAL,
                confidence=0.0,
                metadata={
                    "momentum": momentum,
                    "stoch_k": stoch_k,
                    "stoch_d": stoch_d,
                    "is_winner": is_winner,
                    "is_loser": is_loser,
                },
            )

        # Calculate ATR for stops
        atr = self._calculate_atr(data)
        current_price = data["close"].iloc[-1]

        if direction == Direction.LONG:
            stop_loss = current_price - (atr * self.mtf_config.atr_stop_mult)
            take_profit = current_price + (atr * self.mtf_config.atr_tp_mult)
        else:
            stop_loss = current_price + (atr * self.mtf_config.atr_stop_mult)
            take_profit = current_price - (atr * self.mtf_config.atr_tp_mult)

        # Confidence based on momentum strength and stochastic extremity
        momentum_strength = min(1.0, abs(momentum) / 0.3)  # 30% = max
        stoch_extremity = abs(stoch_k - 50) / 50
        confidence = 0.5 + 0.25 * momentum_strength + 0.25 * stoch_extremity

        # Store numeric direction in metadata for backwards compat
        numeric_direction = 1 if direction == Direction.LONG else -1

        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=signal_type,
            direction=direction,
            confidence=confidence,
            metadata={
                "strategy": "mtf_momentum",
                "momentum": momentum,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "bullish_cross": bullish_cross,
                "bearish_cross": bearish_cross,
                "is_winner": is_winner,
                "is_loser": is_loser,
                "direction": numeric_direction,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": atr,
            },
        )


def calculate_universe_momentum(
    prices: pd.DataFrame,
    formation_period: int = 252,
    skip_period: int = 21,
) -> pd.Series:
    """
    Calculate momentum for entire universe.

    Args:
        prices: DataFrame with symbols as columns, dates as index
        formation_period: Lookback for momentum
        skip_period: Recent period to skip

    Returns:
        Series with symbol as index, momentum as value
    """
    if len(prices) < formation_period:
        return pd.Series(dtype=float)

    price_12m = prices.iloc[-formation_period]
    price_1m = prices.iloc[-skip_period]

    momentum = (price_1m / price_12m) - 1
    return momentum.sort_values(ascending=False)


def backtest(
    df: pd.DataFrame,
    symbol: str,
    config: MTFConfig | None = None,
) -> dict:
    """
    Backtest MTF Momentum strategy.
    """
    if config is None:
        config = MTFConfig()

    model_config = ModelConfig(
        model_id=f"mtf_backtest_{symbol}",
        model_type="momentum",
        parameters={
            "formation_period": config.formation_period,
            "skip_period": config.skip_period,
            "stoch_k_period": config.stoch_k_period,
            "stoch_oversold": config.stoch_oversold,
            "stoch_overbought": config.stoch_overbought,
        },
    )

    model = MTFMomentumModel(model_config)

    trades = []
    position = None

    start_idx = config.formation_period + 10

    for i in range(start_idx, len(df)):
        window = df.iloc[: i + 1]
        timestamp = df.index[i]

        import asyncio

        signal = asyncio.get_event_loop().run_until_complete(
            model.generate(symbol, window, timestamp)
        )

        if signal is None:
            continue

        current_price = df["close"].iloc[i]

        # Check exits
        if position is not None:
            exit_reason = None

            if position["direction"] == 1:
                if current_price <= position["stop_loss"]:
                    exit_reason = "stop_loss"
                elif current_price >= position["take_profit"]:
                    exit_reason = "take_profit"
            elif current_price >= position["stop_loss"]:
                exit_reason = "stop_loss"
            elif current_price <= position["take_profit"]:
                exit_reason = "take_profit"

            if exit_reason:
                pnl = (current_price - position["entry"]) * position["direction"]
                pnl_pct = pnl / position["entry"] * 100
                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "entry_price": position["entry"],
                        "exit_price": current_price,
                        "direction": position["direction"],
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                        "momentum": position["momentum"],
                    }
                )
                position = None

        # New entry
        if position is None and signal.signal_type != SignalType.HOLD:
            direction = 1 if signal.direction == Direction.LONG else -1
            position = {
                "entry": current_price,
                "entry_time": timestamp,
                "direction": direction,
                "stop_loss": signal.metadata["stop_loss"],
                "take_profit": signal.metadata["take_profit"],
                "momentum": signal.metadata["momentum"],
            }

    if not trades:
        return {"trades": 0, "total_return": 0, "win_rate": 0}

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df["pnl_pct"] > 0]

    return {
        "trades": len(trades),
        "total_return": trades_df["pnl_pct"].sum(),
        "win_rate": len(winners) / len(trades) * 100,
        "avg_win": winners["pnl_pct"].mean() if len(winners) > 0 else 0,
        "avg_loss": trades_df[trades_df["pnl_pct"] <= 0]["pnl_pct"].mean(),
        "trades_df": trades_df,
    }
