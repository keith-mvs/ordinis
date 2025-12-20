"""
Bollinger RSI Confluence Model.

Multi-indicator confluence model combining Bollinger Bands and RSI
for high-probability mean reversion signals. Uses ordinis.quant
for canonical indicator calculations.
"""

from datetime import datetime, timedelta

import pandas as pd

from ordinis.quant import bollinger_bands, rsi, volatility

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType


class BollingerRSIConfluenceModel(Model):
    """
    Bollinger Bands + RSI Confluence Model.

    Generates signals only when BOTH indicators confirm the same direction,
    reducing false signals and improving win rate.

    Parameters:
        bb_period: Bollinger Bands period (default 20)
        bb_std: Number of standard deviations (default 2.0)
        rsi_period: RSI calculation period (default 14)
        rsi_oversold: RSI oversold threshold (default 30)
        rsi_overbought: RSI overbought threshold (default 70)
        rsi_extreme_oversold: Extreme oversold for higher probability (default 20)
        rsi_extreme_overbought: Extreme overbought for higher probability (default 80)
        min_volatility: Minimum volatility % to generate signals (default 5.0)
        vol_lookback: Volatility calculation lookback (default 22)

    Signals:
        - ENTRY/LONG: Price at/below lower BB AND RSI oversold
        - ENTRY/SHORT: Price at/above upper BB AND RSI overbought
        - HOLD: No confluence or low volatility
    """

    def __init__(self, config: ModelConfig):
        """Initialize Bollinger RSI Confluence model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.bb_period = params.get("bb_period", 20)
        self.bb_std = params.get("bb_std", 2.0)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.rsi_overbought = params.get("rsi_overbought", 70)
        self.rsi_extreme_oversold = params.get("rsi_extreme_oversold", 20)
        self.rsi_extreme_overbought = params.get("rsi_extreme_overbought", 80)
        self.min_volatility = params.get("min_volatility", 5.0)
        self.vol_lookback = params.get("vol_lookback", 22)

        # Update min data points
        self.config.min_data_points = max(self.bb_period, self.rsi_period, self.vol_lookback) + 30

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate trading signal from Bollinger + RSI confluence.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with confluence-based prediction
        """
        # Validate data
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        # Extract symbol
        if "symbol" in data:
            symbol_data = data["symbol"]
            symbol = symbol_data.iloc[0] if hasattr(symbol_data, "iloc") else str(symbol_data)
        else:
            symbol = "UNKNOWN"

        close = data["close"]

        # Calculate indicators using ordinis.quant (canonical implementations)
        bb_df = bollinger_bands(close, w=self.bb_period, k=self.bb_std)
        rsi_series = rsi(close, w=self.rsi_period)
        vol_series = volatility(close, w=self.vol_lookback, returns_type=None)

        # Get current values (reindex to handle trimmed series)
        bb_df_reindexed = bb_df.reindex(close.index)
        rsi_reindexed = rsi_series.reindex(close.index)
        vol_reindexed = vol_series.reindex(close.index)

        current_price = close.iloc[-1]
        lower_band = bb_df_reindexed["lower"].iloc[-1]
        middle_band = bb_df_reindexed["middle"].iloc[-1]
        upper_band = bb_df_reindexed["upper"].iloc[-1]
        current_rsi = rsi_reindexed.iloc[-1]
        current_vol = vol_reindexed.iloc[-1] if not pd.isna(vol_reindexed.iloc[-1]) else 0

        # Handle NaN values
        if pd.isna(lower_band) or pd.isna(current_rsi):
            return self._create_hold_signal(symbol, timestamp, "Insufficient data for indicators")

        # Volatility filter - skip low volatility environments
        if current_vol < self.min_volatility:
            return self._create_hold_signal(
                symbol, timestamp, f"Low volatility: {current_vol:.2f}% < {self.min_volatility}%"
            )

        # Calculate band position (0 = lower, 1 = upper)
        band_range = upper_band - lower_band
        bb_position = (current_price - lower_band) / band_range if band_range > 0 else 0.5

        # Determine confluence conditions
        price_at_lower = current_price <= lower_band * 1.005  # Within 0.5% of lower
        price_at_upper = current_price >= upper_band * 0.995  # Within 0.5% of upper
        rsi_oversold = current_rsi <= self.rsi_oversold
        rsi_overbought = current_rsi >= self.rsi_overbought
        rsi_extreme_low = current_rsi <= self.rsi_extreme_oversold
        rsi_extreme_high = current_rsi >= self.rsi_extreme_overbought

        # Signal determination with confluence
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        probability = 0.5
        score = 0.0
        expected_return = 0.0

        # LONG confluence: Price at lower band AND RSI oversold
        if price_at_lower and rsi_oversold:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG

            # Higher probability for extreme RSI
            if rsi_extreme_low:
                probability = 0.70 + (self.rsi_extreme_oversold - current_rsi) / 100
                score = 0.8
            else:
                probability = 0.60 + (self.rsi_oversold - current_rsi) / 200
                score = 0.6

            expected_return = 0.03 + (score * 0.04)

        # SHORT confluence: Price at upper band AND RSI overbought
        elif price_at_upper and rsi_overbought:
            signal_type = SignalType.ENTRY
            direction = Direction.SHORT

            # Higher probability for extreme RSI
            if rsi_extreme_high:
                probability = 0.70 + (current_rsi - self.rsi_extreme_overbought) / 100
                score = -0.8
            else:
                probability = 0.60 + (current_rsi - self.rsi_overbought) / 200
                score = -0.6

            expected_return = 0.03 + (abs(score) * 0.04)

        # Weak signals - only one indicator triggered
        elif price_at_lower or rsi_oversold:
            signal_type = SignalType.HOLD
            direction = Direction.LONG
            probability = 0.55
            score = 0.3

        elif price_at_upper or rsi_overbought:
            signal_type = SignalType.HOLD
            direction = Direction.SHORT
            probability = 0.55
            score = -0.3

        # Clamp probability
        probability = max(0.5, min(0.85, probability))

        # Calculate confidence interval
        returns = close.pct_change().dropna()
        recent_vol = returns.tail(20).std() if len(returns) >= 20 else 0.02
        confidence_interval = (
            expected_return - 2 * recent_vol,
            expected_return + 2 * recent_vol,
        )

        # Feature contributions for explainability
        feature_contributions = {
            "bb_position": float(bb_position),
            "rsi": float(current_rsi),
            "volatility": float(current_vol),
            "price_at_lower_band": float(price_at_lower),
            "price_at_upper_band": float(price_at_upper),
            "rsi_oversold": float(rsi_oversold),
            "rsi_overbought": float(rsi_overbought),
            "confluence_long": float(price_at_lower and rsi_oversold),
            "confluence_short": float(price_at_upper and rsi_overbought),
        }

        # Regime detection
        if current_vol > 25:
            regime = "high_volatility"
        elif current_vol > 15:
            regime = "moderate_volatility"
        else:
            regime = "low_volatility"

        # Data quality
        recent_close = close.tail(20)
        data_quality = 1.0 - (recent_close.isnull().sum() / len(recent_close))

        # Staleness
        if isinstance(data.index, pd.DatetimeIndex):
            delta = timestamp - data.index[-1]
            staleness = timedelta(seconds=abs(delta.total_seconds()))
        else:
            staleness = timedelta(seconds=0)

        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=signal_type,
            direction=direction,
            probability=probability,
            expected_return=expected_return,
            confidence_interval=confidence_interval,
            score=score,
            model_id=self.config.model_id,
            model_version=self.config.version,
            feature_contributions=feature_contributions,
            regime=regime,
            data_quality=data_quality,
            staleness=staleness,
            metadata={
                "bb_period": self.bb_period,
                "bb_std": self.bb_std,
                "rsi_period": self.rsi_period,
                "current_price": float(current_price),
                "lower_band": float(lower_band),
                "middle_band": float(middle_band),
                "upper_band": float(upper_band),
                "rsi": float(current_rsi),
                "volatility": float(current_vol),
                "bb_position": float(bb_position),
                "confluence": "long"
                if (price_at_lower and rsi_oversold)
                else ("short" if (price_at_upper and rsi_overbought) else "none"),
            },
        )

    def _create_hold_signal(self, symbol: str, timestamp: datetime, reason: str) -> Signal:
        """Create a HOLD signal with reason in metadata."""
        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=SignalType.HOLD,
            direction=Direction.NEUTRAL,
            probability=0.5,
            expected_return=0.0,
            confidence_interval=(0.0, 0.0),
            score=0.0,
            model_id=self.config.model_id,
            model_version=self.config.version,
            feature_contributions={},
            regime="unknown",
            data_quality=1.0,
            staleness=timedelta(seconds=0),
            metadata={"hold_reason": reason},
        )
