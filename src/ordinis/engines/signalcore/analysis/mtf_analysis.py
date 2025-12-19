"""
Multi-Timeframe Signal Analysis Engine.

Implements advanced signal generation with:
- Multi-timeframe analysis (MTF)
- Cross-timeframe signal confirmation
- Regime detection integration
- Model ensemble coordination

Step 4 of Trade Enhancement Roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Supported timeframes for analysis."""
    
    M1 = "1min"
    M5 = "5min"
    M15 = "15min"
    M30 = "30min"
    H1 = "1hour"
    H4 = "4hour"
    D1 = "1day"
    W1 = "1week"
    
    @property
    def minutes(self) -> int:
        """Get timeframe in minutes."""
        mapping = {
            "1min": 1, "5min": 5, "15min": 15, "30min": 30,
            "1hour": 60, "4hour": 240, "1day": 1440, "1week": 10080,
        }
        return mapping[self.value]
        
    @property
    def pandas_freq(self) -> str:
        """Get pandas frequency string."""
        mapping = {
            "1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min",
            "1hour": "1h", "4hour": "4h", "1day": "1D", "1week": "1W",
        }
        return mapping[self.value]


class SignalDirection(Enum):
    """Signal direction."""
    
    LONG = auto()
    SHORT = auto()
    NEUTRAL = auto()


class SignalStrength(Enum):
    """Signal strength classification."""
    
    WEAK = auto()
    MODERATE = auto()
    STRONG = auto()
    VERY_STRONG = auto()


class MarketRegime(Enum):
    """Market regime classification."""
    
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()
    VOLATILE = auto()
    LOW_VOLATILITY = auto()
    BREAKOUT = auto()


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    
    timeframe: Timeframe
    symbol: str
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    indicators: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_bullish(self) -> bool:
        return self.direction == SignalDirection.LONG
        
    @property
    def is_bearish(self) -> bool:
        return self.direction == SignalDirection.SHORT
        
    @property
    def strength_category(self) -> SignalStrength:
        """Categorize signal strength."""
        if self.strength < 0.3:
            return SignalStrength.WEAK
        if self.strength < 0.6:
            return SignalStrength.MODERATE
        if self.strength < 0.85:
            return SignalStrength.STRONG
        return SignalStrength.VERY_STRONG


@dataclass
class MTFSignal:
    """Multi-timeframe composite signal."""
    
    symbol: str
    primary_timeframe: Timeframe
    direction: SignalDirection
    composite_strength: float
    composite_confidence: float
    alignment_score: float  # 0-1, how aligned timeframes are
    signals: list[TimeframeSignal]
    regime: MarketRegime
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable."""
        return (
            self.direction != SignalDirection.NEUTRAL
            and self.composite_confidence >= 0.6
            and self.alignment_score >= 0.5
        )
        
    @property
    def aligned_timeframes(self) -> list[Timeframe]:
        """Get timeframes aligned with primary direction."""
        return [
            s.timeframe for s in self.signals
            if s.direction == self.direction
        ]
        
    def get_timeframe_signal(self, tf: Timeframe) -> TimeframeSignal | None:
        """Get signal for specific timeframe."""
        for s in self.signals:
            if s.timeframe == tf:
                return s
        return None


@dataclass
class RegimeAnalysis:
    """Market regime analysis result."""
    
    regime: MarketRegime
    confidence: float
    trend_strength: float  # ADX-like measure
    volatility_percentile: float  # Current vol vs historical
    momentum: float  # -1 to 1
    support_level: float | None = None
    resistance_level: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class IndicatorCalculator:
    """Calculate technical indicators for signal generation."""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=period).mean()
        
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()
        
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    @staticmethod
    def macd(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator."""
        fast_ema = data.ewm(span=fast, adjust=False).mean()
        slow_ema = data.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
        
    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
        
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
        
    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = IndicatorCalculator.atr(high, low, close, period=1)
        
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / 
                         tr.ewm(span=period, adjust=False).mean())
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / 
                          tr.ewm(span=period, adjust=False).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx, plus_di, minus_di
        
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        
        return k, d


class RegimeDetector:
    """Detect market regime from price data."""
    
    def __init__(
        self,
        trend_threshold: float = 25.0,  # ADX threshold for trending
        volatility_lookback: int = 100,
        volatility_high_percentile: float = 80.0,
        volatility_low_percentile: float = 20.0,
    ) -> None:
        """Initialize regime detector."""
        self.trend_threshold = trend_threshold
        self.volatility_lookback = volatility_lookback
        self.volatility_high_percentile = volatility_high_percentile
        self.volatility_low_percentile = volatility_low_percentile
        self._calc = IndicatorCalculator()
        
    def detect(self, df: pd.DataFrame) -> RegimeAnalysis:
        """
        Detect market regime from OHLCV data.
        
        Args:
            df: DataFrame with open, high, low, close, volume columns
            
        Returns:
            RegimeAnalysis result
        """
        if len(df) < 50:
            return RegimeAnalysis(
                regime=MarketRegime.RANGING,
                confidence=0.5,
                trend_strength=0.0,
                volatility_percentile=50.0,
                momentum=0.0,
            )
            
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        # Calculate indicators
        adx, plus_di, minus_di = self._calc.adx(high, low, close)
        atr = self._calc.atr(high, low, close)
        rsi = self._calc.rsi(close)
        
        # Current values
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Volatility percentile
        atr_pct = (atr.iloc[-1] / close.iloc[-1]) * 100
        lookback = min(len(df), self.volatility_lookback)
        historical_atr_pct = (atr.tail(lookback) / close.tail(lookback)) * 100
        volatility_percentile = (
            (historical_atr_pct < atr_pct).sum() / len(historical_atr_pct) * 100
        )
        
        # Momentum
        momentum = (current_rsi - 50) / 50  # Normalize to -1 to 1
        
        # Trend strength
        trend_strength = current_adx / 100 if current_adx else 0.0
        
        # Determine regime
        regime = self._classify_regime(
            adx=current_adx,
            plus_di=current_plus_di,
            minus_di=current_minus_di,
            volatility_pct=volatility_percentile,
            momentum=momentum,
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            regime=regime,
            adx=current_adx,
            volatility_pct=volatility_percentile,
        )
        
        # Support/resistance (simple swing highs/lows)
        recent_high = high.tail(20).max()
        recent_low = low.tail(20).min()
        
        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_percentile=volatility_percentile,
            momentum=momentum,
            support_level=float(recent_low),
            resistance_level=float(recent_high),
            metadata={
                "adx": float(current_adx),
                "plus_di": float(current_plus_di),
                "minus_di": float(current_minus_di),
                "rsi": float(current_rsi),
            },
        )
        
    def _classify_regime(
        self,
        adx: float,
        plus_di: float,
        minus_di: float,
        volatility_pct: float,
        momentum: float,
    ) -> MarketRegime:
        """Classify market regime based on indicators."""
        # Strong trend
        if adx > self.trend_threshold:
            if plus_di > minus_di:
                return MarketRegime.TRENDING_UP
            return MarketRegime.TRENDING_DOWN
            
        # Check volatility
        if volatility_pct > self.volatility_high_percentile:
            return MarketRegime.VOLATILE
        if volatility_pct < self.volatility_low_percentile:
            return MarketRegime.LOW_VOLATILITY
            
        # Breakout detection (ADX rising from low levels)
        if adx > 20 and adx < self.trend_threshold:
            return MarketRegime.BREAKOUT
            
        return MarketRegime.RANGING
        
    def _calculate_confidence(
        self,
        regime: MarketRegime,
        adx: float,
        volatility_pct: float,
    ) -> float:
        """Calculate confidence in regime classification."""
        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            # Stronger ADX = higher confidence
            return min(1.0, adx / 50)
            
        if regime == MarketRegime.VOLATILE:
            # Higher volatility = higher confidence
            return min(1.0, volatility_pct / 100)
            
        if regime == MarketRegime.LOW_VOLATILITY:
            # Lower volatility = higher confidence
            return min(1.0, (100 - volatility_pct) / 100)
            
        # Ranging/Breakout - moderate confidence
        return 0.6


class MTFSignalGenerator:
    """
    Multi-Timeframe Signal Generator.
    
    Generates composite signals by analyzing multiple timeframes
    and combining them with proper weighting.
    
    Example:
        >>> generator = MTFSignalGenerator()
        >>> 
        >>> # Add data for each timeframe
        >>> generator.set_data(Timeframe.M5, df_5min)
        >>> generator.set_data(Timeframe.H1, df_1hour)
        >>> generator.set_data(Timeframe.D1, df_daily)
        >>> 
        >>> # Generate composite signal
        >>> signal = generator.generate("AAPL", primary_tf=Timeframe.M5)
    """
    
    # Timeframe weights (higher timeframes get more weight)
    TF_WEIGHTS: dict[Timeframe, float] = {
        Timeframe.M1: 0.5,
        Timeframe.M5: 0.6,
        Timeframe.M15: 0.7,
        Timeframe.M30: 0.8,
        Timeframe.H1: 0.9,
        Timeframe.H4: 1.0,
        Timeframe.D1: 1.1,
        Timeframe.W1: 1.2,
    }
    
    def __init__(
        self,
        timeframes: list[Timeframe] | None = None,
        min_alignment_score: float = 0.5,
        regime_filter_enabled: bool = True,
    ) -> None:
        """
        Initialize MTF signal generator.
        
        Args:
            timeframes: Timeframes to analyze (default: M5, H1, D1)
            min_alignment_score: Minimum alignment for valid signal
            regime_filter_enabled: Whether to filter signals by regime
        """
        self.timeframes = timeframes or [
            Timeframe.M5,
            Timeframe.H1,
            Timeframe.D1,
        ]
        self.min_alignment_score = min_alignment_score
        self.regime_filter_enabled = regime_filter_enabled
        
        self._data: dict[Timeframe, pd.DataFrame] = {}
        self._calc = IndicatorCalculator()
        self._regime_detector = RegimeDetector()
        
    def set_data(self, timeframe: Timeframe, data: pd.DataFrame) -> None:
        """Set data for a timeframe."""
        self._data[timeframe] = data
        
    def clear_data(self) -> None:
        """Clear all data."""
        self._data.clear()
        
    def generate(
        self,
        symbol: str,
        primary_tf: Timeframe | None = None,
    ) -> MTFSignal:
        """
        Generate multi-timeframe signal.
        
        Args:
            symbol: Trading symbol
            primary_tf: Primary timeframe for execution
            
        Returns:
            MTFSignal with composite analysis
        """
        primary_tf = primary_tf or self.timeframes[0]
        
        # Generate signal for each timeframe
        tf_signals: list[TimeframeSignal] = []
        
        for tf in self.timeframes:
            if tf not in self._data:
                continue
                
            data = self._data[tf]
            signal = self._generate_timeframe_signal(symbol, tf, data)
            tf_signals.append(signal)
            
        if not tf_signals:
            return MTFSignal(
                symbol=symbol,
                primary_timeframe=primary_tf,
                direction=SignalDirection.NEUTRAL,
                composite_strength=0.0,
                composite_confidence=0.0,
                alignment_score=0.0,
                signals=[],
                regime=MarketRegime.RANGING,
            )
            
        # Detect regime from highest timeframe
        highest_tf = max(self.timeframes, key=lambda t: t.minutes)
        if highest_tf in self._data:
            regime_analysis = self._regime_detector.detect(self._data[highest_tf])
        else:
            regime_analysis = RegimeAnalysis(
                regime=MarketRegime.RANGING,
                confidence=0.5,
                trend_strength=0.0,
                volatility_percentile=50.0,
                momentum=0.0,
            )
            
        # Calculate composite signal
        direction, strength, confidence = self._calculate_composite(tf_signals)
        alignment_score = self._calculate_alignment(tf_signals, direction)
        
        # Apply regime filter
        if self.regime_filter_enabled:
            direction, strength, confidence = self._apply_regime_filter(
                direction, strength, confidence, regime_analysis
            )
            
        return MTFSignal(
            symbol=symbol,
            primary_timeframe=primary_tf,
            direction=direction,
            composite_strength=strength,
            composite_confidence=confidence,
            alignment_score=alignment_score,
            signals=tf_signals,
            regime=regime_analysis.regime,
            metadata={
                "regime_confidence": regime_analysis.confidence,
                "trend_strength": regime_analysis.trend_strength,
                "volatility_percentile": regime_analysis.volatility_percentile,
            },
        )
        
    def _generate_timeframe_signal(
        self,
        symbol: str,
        timeframe: Timeframe,
        data: pd.DataFrame,
    ) -> TimeframeSignal:
        """Generate signal for a single timeframe."""
        close = data["close"]
        high = data["high"]
        low = data["low"]
        
        # Calculate indicators
        sma20 = self._calc.sma(close, 20)
        sma50 = self._calc.sma(close, 50)
        rsi = self._calc.rsi(close)
        macd_line, signal_line, histogram = self._calc.macd(close)
        upper_bb, middle_bb, lower_bb = self._calc.bollinger_bands(close)
        
        # Current values
        current_close = close.iloc[-1]
        current_sma20 = sma20.iloc[-1]
        current_sma50 = sma50.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # Score components
        scores = []
        
        # Trend (SMA crossover)
        if current_sma20 > current_sma50:
            scores.append(1.0)
        elif current_sma20 < current_sma50:
            scores.append(-1.0)
        else:
            scores.append(0.0)
            
        # Momentum (MACD)
        if current_macd > current_signal:
            scores.append(1.0 if current_histogram > 0 else 0.5)
        else:
            scores.append(-1.0 if current_histogram < 0 else -0.5)
            
        # RSI
        if current_rsi > 70:
            scores.append(-0.5)  # Overbought
        elif current_rsi < 30:
            scores.append(0.5)  # Oversold
        elif current_rsi > 50:
            scores.append(0.3)
        else:
            scores.append(-0.3)
            
        # Price vs SMA
        if current_close > current_sma20:
            scores.append(0.5)
        else:
            scores.append(-0.5)
            
        # Composite score
        avg_score = np.mean(scores)
        
        # Determine direction
        if avg_score > 0.3:
            direction = SignalDirection.LONG
        elif avg_score < -0.3:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL
            
        # Strength and confidence
        strength = min(1.0, abs(avg_score))
        confidence = len([s for s in scores if (s > 0) == (avg_score > 0)]) / len(scores)
        
        return TimeframeSignal(
            timeframe=timeframe,
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            timestamp=datetime.now(UTC),
            indicators={
                "sma20": float(current_sma20),
                "sma50": float(current_sma50),
                "rsi": float(current_rsi),
                "macd": float(current_macd),
                "macd_signal": float(current_signal),
                "macd_histogram": float(current_histogram),
            },
        )
        
    def _calculate_composite(
        self,
        signals: list[TimeframeSignal],
    ) -> tuple[SignalDirection, float, float]:
        """Calculate weighted composite signal."""
        if not signals:
            return SignalDirection.NEUTRAL, 0.0, 0.0
            
        # Weighted direction scores
        weighted_scores = []
        weights = []
        
        for signal in signals:
            weight = self.TF_WEIGHTS.get(signal.timeframe, 1.0)
            
            if signal.direction == SignalDirection.LONG:
                score = signal.strength * weight
            elif signal.direction == SignalDirection.SHORT:
                score = -signal.strength * weight
            else:
                score = 0
                
            weighted_scores.append(score)
            weights.append(weight)
            
        total_weight = sum(weights)
        avg_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0
        
        # Direction
        if avg_score > 0.2:
            direction = SignalDirection.LONG
        elif avg_score < -0.2:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL
            
        # Strength and confidence
        strength = min(1.0, abs(avg_score))
        confidence = np.mean([s.confidence for s in signals])
        
        return direction, strength, confidence
        
    def _calculate_alignment(
        self,
        signals: list[TimeframeSignal],
        direction: SignalDirection,
    ) -> float:
        """Calculate how aligned timeframes are."""
        if not signals or direction == SignalDirection.NEUTRAL:
            return 0.0
            
        aligned = sum(
            1 for s in signals
            if s.direction == direction
        )
        
        return aligned / len(signals)
        
    def _apply_regime_filter(
        self,
        direction: SignalDirection,
        strength: float,
        confidence: float,
        regime: RegimeAnalysis,
    ) -> tuple[SignalDirection, float, float]:
        """Apply regime-based filtering to signals."""
        # In trending markets, boost trend-following signals
        if regime.regime == MarketRegime.TRENDING_UP:
            if direction == SignalDirection.LONG:
                strength *= 1.2
                confidence *= 1.1
            elif direction == SignalDirection.SHORT:
                strength *= 0.7
                confidence *= 0.8
                
        elif regime.regime == MarketRegime.TRENDING_DOWN:
            if direction == SignalDirection.SHORT:
                strength *= 1.2
                confidence *= 1.1
            elif direction == SignalDirection.LONG:
                strength *= 0.7
                confidence *= 0.8
                
        # In ranging markets, reduce signal strength
        elif regime.regime == MarketRegime.RANGING:
            strength *= 0.8
            
        # In volatile markets, require higher confidence
        elif regime.regime == MarketRegime.VOLATILE:
            if confidence < 0.7:
                direction = SignalDirection.NEUTRAL
                
        # Clamp values
        strength = min(1.0, strength)
        confidence = min(1.0, confidence)
        
        return direction, strength, confidence
        
    def resample_data(
        self,
        source_tf: Timeframe,
        target_tf: Timeframe,
    ) -> pd.DataFrame | None:
        """
        Resample data from source to target timeframe.
        
        Args:
            source_tf: Source timeframe (must be smaller)
            target_tf: Target timeframe (must be larger)
            
        Returns:
            Resampled DataFrame or None
        """
        if source_tf not in self._data:
            return None
            
        if source_tf.minutes >= target_tf.minutes:
            logger.warning(f"Cannot resample {source_tf} to {target_tf}")
            return None
            
        df = self._data[source_tf]
        
        resampled = df.resample(target_tf.pandas_freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        
        return resampled
