"""
Combined Technical Indicators.

Unified interface for all technical indicators, providing
a single entry point for comprehensive technical analysis.
"""

from dataclasses import dataclass

import pandas as pd

from .moving_averages import MovingAverages
from .oscillators import OscillatorCondition, Oscillators, OscillatorSignal
from .trend import IchimokuSignal, TrendIndicators
from .volatility import BandSignal, VolatilityIndicators
from .volume import VolumeIndicators, VolumeSignal


@dataclass
class TechnicalSnapshot:
    """Complete technical analysis snapshot."""

    # Trend
    trend_direction: str  # "bullish", "bearish", "neutral"
    trend_strength: float  # 0-100
    ma_alignment: str  # "bullish", "bearish", "mixed"
    ichimoku: IchimokuSignal

    # Momentum
    rsi: float
    rsi_condition: OscillatorCondition
    stochastic_k: float
    macd_histogram: float

    # Volatility
    atr: float
    atr_percent: float
    bollinger_position: str
    volatility_regime: str

    # Volume
    relative_volume: float
    volume_trend: str
    volume_confirms: bool

    # Signals
    signals: list[str]  # List of active signals
    overall_bias: str  # "strong_buy", "buy", "neutral", "sell", "strong_sell"


class TechnicalIndicators:
    """
    Unified technical analysis interface.

    Combines all indicator types for comprehensive analysis.
    """

    def __init__(self):
        self.ma = MovingAverages()
        self.oscillators = Oscillators()
        self.trend = TrendIndicators()
        self.volatility = VolatilityIndicators()
        self.volume = VolumeIndicators()

    def analyze(self, data: pd.DataFrame) -> TechnicalSnapshot:
        """
        Perform comprehensive technical analysis.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            TechnicalSnapshot with all analysis results
        """
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Trend Analysis
        ma_analysis = self.ma.multi_ma_analysis(data)
        trend_strength = ma_analysis["trend_strength"]
        ma_alignment = ma_analysis["alignment"]

        if trend_strength > 70:
            trend_direction = "bullish"
        elif trend_strength < 30:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"

        # Momentum Analysis
        rsi = self.oscillators.rsi(close).iloc[-1]
        rsi_signal = self.oscillators.rsi_signal(data)

        stoch_k, _ = self.oscillators.stochastic(high, low, close)
        stoch_k_current = stoch_k.iloc[-1]

        # MACD
        _macd_line, _signal_line, histogram = self._calculate_macd(close)

        # Ichimoku Cloud
        _, ichimoku_signal = self.trend.ichimoku(high, low, close)

        # Volatility Analysis
        vol_metrics = self.volatility.volatility_analysis(data)
        bb_signal = self.volatility.bollinger_signal(data)

        # Volume Analysis
        vol_signal = self.volume.volume_confirmation(data)
        rvol = self.volume.volume_relative(volume).iloc[-1]

        # Generate Signals
        signals = self._generate_signals(
            rsi_signal,
            bb_signal,
            vol_signal,
            ma_analysis,
            stoch_k_current,
            histogram,
            ichimoku_signal,
        )

        # Overall Bias
        overall_bias = self._calculate_bias(
            trend_direction,
            rsi,
            stoch_k_current,
            bb_signal.position,
            vol_signal.trend,
            ichimoku_signal,
        )

        return TechnicalSnapshot(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            ma_alignment=ma_alignment,
            ichimoku=ichimoku_signal,
            rsi=rsi,
            rsi_condition=rsi_signal.condition,
            stochastic_k=stoch_k_current,
            macd_histogram=histogram,
            atr=vol_metrics.atr,
            atr_percent=vol_metrics.atr_percent,
            bollinger_position=bb_signal.position,
            volatility_regime=vol_metrics.vol_regime,
            relative_volume=rvol,
            volume_trend=vol_signal.trend,
            volume_confirms=vol_signal.confirmation,
            signals=signals,
            overall_bias=overall_bias,
        )

    def _calculate_macd(
        self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[float, float, float]:
        """Calculate MACD components."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

    def _generate_signals(
        self,
        rsi_signal: OscillatorSignal,
        bb_signal: BandSignal,
        vol_signal: VolumeSignal,
        ma_analysis: dict,
        stoch_k: float,
        macd_hist: float,
        ichimoku_signal: IchimokuSignal,
    ) -> list[str]:
        """Generate list of active trading signals."""
        signals = []

        # RSI signals
        if rsi_signal.signal == "buy":
            signals.append("RSI oversold reversal")
        elif rsi_signal.signal == "sell":
            signals.append("RSI overbought reversal")

        # Bollinger signals
        if bb_signal.position == "below_lower":
            signals.append("BB oversold - potential bounce")
        elif bb_signal.position == "above_upper":
            signals.append("BB overbought - potential pullback")
        if bb_signal.squeeze:
            signals.append("BB squeeze - breakout imminent")

        # MA signals
        if ma_analysis["alignment"] == "bullish":
            signals.append("MA bullish alignment")
        elif ma_analysis["alignment"] == "bearish":
            signals.append("MA bearish alignment")

        # Volume signals
        if vol_signal.divergence == "bullish":
            signals.append("Bullish volume divergence")
        elif vol_signal.divergence == "bearish":
            signals.append("Bearish volume divergence")

        # MACD signals
        if macd_hist > 0 and abs(macd_hist) > 0.1:
            signals.append("MACD bullish momentum")
        elif macd_hist < 0 and abs(macd_hist) > 0.1:
            signals.append("MACD bearish momentum")

        # Stochastic
        if stoch_k < 20:
            signals.append("Stochastic oversold")
        elif stoch_k > 80:
            signals.append("Stochastic overbought")

        # Ichimoku
        if ichimoku_signal.trend == "bullish":
            signals.append("Ichimoku bullish cloud")
        elif ichimoku_signal.trend == "bearish":
            signals.append("Ichimoku bearish cloud")
        if ichimoku_signal.baseline_cross == "bullish":
            signals.append("Kijun bullish cross")
        elif ichimoku_signal.baseline_cross == "bearish":
            signals.append("Kijun bearish cross")

        return signals

    def _calculate_bias(
        self,
        trend: str,
        rsi: float,
        stoch: float,
        bb_pos: str,
        vol_trend: str,
        ichimoku_signal: IchimokuSignal,
    ) -> str:
        """Calculate overall market bias."""
        score = 0

        # Trend component (weight: 2)
        if trend == "bullish":
            score += 2
        elif trend == "bearish":
            score -= 2

        # RSI component (weight: 1)
        if rsi < 30:
            score += 1  # Oversold = potential buy
        elif rsi > 70:
            score -= 1  # Overbought = potential sell

        # Stochastic (weight: 1)
        if stoch < 20:
            score += 1
        elif stoch > 80:
            score -= 1

        # Bollinger (weight: 1)
        if bb_pos == "below_lower":
            score += 1
        elif bb_pos == "above_upper":
            score -= 1

        # Volume (weight: 1)
        if vol_trend == "accumulation":
            score += 1
        elif vol_trend == "distribution":
            score -= 1

        # Ichimoku (weight: 1)
        if ichimoku_signal.trend == "bullish":
            score += 1
        elif ichimoku_signal.trend == "bearish":
            score -= 1
        if ichimoku_signal.position == "above_cloud":
            score += 1
        elif ichimoku_signal.position == "below_cloud":
            score -= 1

        # Map score to bias
        if score >= 4:
            return "strong_buy"
        if score >= 2:
            return "buy"
        if score <= -4:
            return "strong_sell"
        if score <= -2:
            return "sell"
        return "neutral"

    # Convenience methods for individual indicators
    def get_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Get current RSI value."""
        return self.oscillators.rsi(data["close"], period).iloc[-1]

    def get_macd(self, data: pd.DataFrame) -> tuple[float, float, float]:
        """Get MACD line, signal line, histogram."""
        return self._calculate_macd(data["close"])

    def get_bollinger_bands(
        self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> tuple[float, float, float]:
        """Get upper, middle, lower Bollinger bands."""
        upper, middle, lower = self.volatility.bollinger_bands(data["close"], period, std_dev)
        return upper.iloc[-1], middle.iloc[-1], lower.iloc[-1]

    def get_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Get current ATR value."""
        return self.volatility.atr(data["high"], data["low"], data["close"], period).iloc[-1]

    def get_obv(self, data: pd.DataFrame) -> float:
        """Get current OBV value."""
        return self.volume.obv(data["close"], data["volume"]).iloc[-1]

    def get_vwap(self, data: pd.DataFrame) -> float:
        """Get current VWAP value."""
        return self.volume.vwap(data["high"], data["low"], data["close"], data["volume"]).iloc[-1]
