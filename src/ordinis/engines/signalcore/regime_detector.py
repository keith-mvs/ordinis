"""
Regime Detector - Classifies market conditions to optimize strategy selection.

Based on empirical analysis showing that:
- DKNG (trending, high volatility) = 41.7% profitable configs
- CRWD (choppy, low volatility) = 0% profitable configs

Key metrics:
1. Direction Change Rate: % of bars that reverse from previous bar
2. Big Move Frequency: % of bars with moves > threshold
3. Autocorrelation: Tendency for price to continue (trend) or reverse (mean-revert)
4. Volatility: Average bar range as % of price

Regimes:
- TRENDING: Low direction changes, positive autocorrelation
- MEAN_REVERTING: Medium direction changes, negative autocorrelation
- CHOPPY: High direction changes, weak signals
- VOLATILE_TRENDING: High volatility with trend
- QUIET_CHOPPY: Low volatility with chop (worst regime)
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime classifications."""

    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE_TRENDING = "volatile_trending"
    CHOPPY = "choppy"
    QUIET_CHOPPY = "quiet_choppy"  # Worst regime - avoid
    UNKNOWN = "unknown"


@dataclass
class RegimeMetrics:
    """Quantitative metrics describing market regime."""

    symbol: str
    timeframe: str
    period_return: float  # Total return over period
    direction_change_rate: float  # % of bars that reverse
    big_move_frequency: float  # % of bars with large moves
    autocorrelation: float  # 1-lag return autocorrelation
    avg_range_pct: float  # Average bar range as % of price
    volatility: float  # Std dev of returns
    bounce_after_drop: float  # Avg return after bottom 5% moves
    reversal_after_rally: float  # Avg return after top 5% moves
    # NEW: Industry-standard indicators
    adx: float = 0.0  # Average Directional Index (trend strength)
    plus_di: float = 0.0  # +DI (bullish pressure)
    minus_di: float = 0.0  # -DI (bearish pressure)
    atr_pct: float = 0.0  # ATR as % of price (normalized volatility)

    @property
    def is_trending(self) -> bool:
        """True if stock shows trending behavior (ADX > 25 is industry standard)."""
        # Use ADX if available, fallback to original logic
        if self.adx > 0:
            return self.adx > 25 and self.direction_change_rate < 0.50
        return self.direction_change_rate < 0.48 and abs(self.period_return) > 10

    @property
    def is_strongly_trending(self) -> bool:
        """True if ADX > 40 (very strong trend)."""
        return self.adx > 40

    @property
    def is_mean_reverting(self) -> bool:
        """True if stock shows mean-reversion behavior."""
        # ADX < 20 + negative autocorr = ranging/mean-reverting
        if self.adx > 0:
            return self.adx < 20 and self.autocorrelation < -0.05 and self.bounce_after_drop > 0.03
        return self.autocorrelation < -0.10 and self.bounce_after_drop > 0.05

    @property
    def is_choppy(self) -> bool:
        """True if stock is choppy (high direction changes, low ADX)."""
        if self.adx > 0:
            return self.adx < 20 and self.direction_change_rate >= 0.48
        return self.direction_change_rate >= 0.50

    @property
    def is_volatile(self) -> bool:
        """True if stock has high volatility."""
        if self.atr_pct > 0:
            return self.atr_pct > 0.8  # ATR > 0.8% of price
        return self.big_move_frequency > 0.12 or self.avg_range_pct > 0.45

    @property
    def is_quiet(self) -> bool:
        """True if stock has low volatility."""
        if self.atr_pct > 0:
            return self.atr_pct < 0.3  # ATR < 0.3% of price
        return self.big_move_frequency < 0.10 and self.avg_range_pct < 0.40

    @property
    def has_weak_signals(self) -> bool:
        """True if mean-reversion signals are weak (low bounce after drops)."""
        return abs(self.bounce_after_drop) < 0.05 and abs(self.reversal_after_rally) < 0.05

    @property
    def trend_direction(self) -> str:
        """Returns 'bullish', 'bearish', or 'neutral' based on DI lines."""
        if self.plus_di > self.minus_di + 5:
            return "bullish"
        if self.minus_di > self.plus_di + 5:
            return "bearish"
        return "neutral"


@dataclass
class RegimeAnalysis:
    """Complete regime analysis with recommendations."""

    metrics: RegimeMetrics
    regime: MarketRegime
    confidence: float  # 0-1 confidence in classification
    recommended_strategies: list[str]
    avoid_strategies: list[str]
    trade_recommendation: str  # "TRADE", "CAUTION", "AVOID"
    reasoning: str


class RegimeDetector:
    """
    Detects market regime and recommends appropriate strategies.

    Usage:
        detector = RegimeDetector()
        analysis = detector.analyze(df, symbol="CRWD", timeframe="5min")
        print(analysis.regime)
        print(analysis.recommended_strategies)
    """

    # Thresholds based on empirical analysis
    DIRECTION_CHANGE_TRENDING = 0.48  # Below this = trending
    DIRECTION_CHANGE_CHOPPY = 0.50  # Above this = choppy
    BIG_MOVE_THRESHOLD = 0.005  # 0.5% move considered "big"
    BIG_MOVE_FREQ_HIGH = 0.15  # Above this = volatile
    BIG_MOVE_FREQ_LOW = 0.14  # Below this = quiet
    AUTOCORR_TREND = 0.05  # Above this = trending tendency
    AUTOCORR_MEANREV = -0.10  # Below this = mean-reversion tendency
    RANGE_HIGH = 0.50  # Above this = high volatility
    RANGE_LOW = 0.45  # Below this = low volatility

    def __init__(
        self,
        big_move_threshold: float = 0.005,
        lookback_periods: int = 14,
    ):
        """
        Initialize regime detector.

        Args:
            big_move_threshold: Threshold for "big" moves (default 0.5%)
            lookback_periods: Periods for RSI and other indicators
        """
        self.big_move_threshold = big_move_threshold
        self.lookback_periods = lookback_periods

    def compute_metrics(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "5min",
    ) -> RegimeMetrics:
        """
        Compute regime metrics from OHLCV data.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
            symbol: Stock symbol for labeling
            timeframe: Timeframe string for labeling

        Returns:
            RegimeMetrics with all computed values
        """
        if len(df) < 50:
            raise ValueError(f"Need at least 50 bars for regime detection, got {len(df)}")

        # Ensure we have required columns
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Calculate returns
        returns = df["close"].pct_change().dropna()

        # 1. Direction change rate
        direction = np.sign(returns)
        direction_changes = (direction.shift(1) * direction < 0).sum()
        direction_change_rate = direction_changes / len(returns)

        # 2. Big move frequency
        big_moves = (abs(returns) > self.big_move_threshold).sum()
        big_move_frequency = big_moves / len(returns)

        # 3. Autocorrelation
        autocorrelation = returns.autocorr(lag=1)
        if pd.isna(autocorrelation):
            autocorrelation = 0.0

        # 4. Average range
        avg_range_pct = ((df["high"] - df["low"]) / df["close"]).mean() * 100

        # 5. Volatility
        volatility = returns.std() * 100

        # 6. Period return
        period_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

        # 7. Mean reversion metrics
        bottom_5pct = returns.quantile(0.05)
        top_5pct = returns.quantile(0.95)
        next_return = returns.shift(-1)

        big_down = returns < bottom_5pct
        bounce_after_drop = next_return[big_down].mean() * 100 if big_down.sum() > 0 else 0

        big_up = returns > top_5pct
        reversal_after_rally = next_return[big_up].mean() * 100 if big_up.sum() > 0 else 0

        # 8. NEW: ADX/DMI calculation (industry-standard trend strength)
        adx, plus_di, minus_di = self._compute_adx(df["high"], df["low"], df["close"])

        # 9. NEW: ATR as % of price (normalized volatility)
        atr = self._compute_atr(df["high"], df["low"], df["close"])
        atr_pct = (atr / df["close"]).iloc[-1] * 100 if len(atr) > 0 else 0.0

        return RegimeMetrics(
            symbol=symbol,
            timeframe=timeframe,
            period_return=period_return,
            direction_change_rate=direction_change_rate,
            big_move_frequency=big_move_frequency,
            autocorrelation=autocorrelation,
            avg_range_pct=avg_range_pct,
            volatility=volatility,
            bounce_after_drop=bounce_after_drop,
            reversal_after_rally=reversal_after_rally,
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            atr_pct=atr_pct,
        )

    def _compute_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> tuple[float, float, float]:
        """
        Compute ADX, +DI, and -DI (industry-standard trend strength).

        Returns:
            Tuple of (ADX, +DI, -DI) - current values
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        # Smoothed averages (Wilder's smoothing)
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()

        # Return current values
        return adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]

    def _compute_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Compute Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def classify_regime(self, metrics: RegimeMetrics) -> tuple[MarketRegime, float]:
        """
        Classify regime based on metrics.

        Args:
            metrics: Computed regime metrics

        Returns:
            Tuple of (MarketRegime, confidence)
        """
        confidence = 0.5  # Base confidence

        # Use ADX as primary trend indicator if available
        has_adx = metrics.adx > 0

        # QUIET_CHOPPY: Worst regime - ADX < 20 + high chop + low volatility
        if has_adx and metrics.adx < 20 and metrics.is_choppy and metrics.is_quiet:
            confidence = 0.85  # Very confident with ADX confirmation
            return MarketRegime.QUIET_CHOPPY, confidence

        # Fallback to original logic if no ADX
        if metrics.is_choppy and (metrics.is_quiet or metrics.has_weak_signals):
            confidence = 0.8 if metrics.direction_change_rate > 0.51 else 0.7
            return MarketRegime.QUIET_CHOPPY, confidence

        # VOLATILE_TRENDING: Strong trend (ADX > 25) with high volatility
        if has_adx and metrics.adx > 25 and metrics.is_volatile:
            confidence = 0.85 if metrics.adx > 40 else 0.75
            return MarketRegime.VOLATILE_TRENDING, confidence

        if metrics.is_trending and metrics.is_volatile:
            confidence = 0.8 if abs(metrics.period_return) > 15 else 0.6
            return MarketRegime.VOLATILE_TRENDING, confidence

        # TRENDING: ADX > 25 indicates trend (industry standard)
        if has_adx and metrics.adx > 25:
            confidence = 0.8 if metrics.adx > 35 else 0.65
            return MarketRegime.TRENDING, confidence

        if metrics.is_trending:
            confidence = 0.7 if abs(metrics.period_return) > 10 else 0.5
            return MarketRegime.TRENDING, confidence

        # MEAN_REVERTING: ADX < 20 + negative autocorrelation with bounce tendency
        if has_adx and metrics.adx < 20 and metrics.autocorrelation < -0.05:
            confidence = 0.75 if metrics.bounce_after_drop > 0.05 else 0.6
            return MarketRegime.MEAN_REVERTING, confidence

        if metrics.is_mean_reverting and not metrics.is_choppy:
            confidence = 0.7 if metrics.autocorrelation < -0.12 else 0.5
            return MarketRegime.MEAN_REVERTING, confidence

        # CHOPPY: ADX < 20 with direction changes (no clear trend)
        if has_adx and metrics.adx < 20:
            confidence = 0.7
            return MarketRegime.CHOPPY, confidence

        if metrics.is_choppy:
            confidence = 0.6
            return MarketRegime.CHOPPY, confidence

        # Check for weak signal regime even if not technically choppy
        if metrics.has_weak_signals and not metrics.is_volatile:
            confidence = 0.5
            return MarketRegime.QUIET_CHOPPY, confidence

        return MarketRegime.UNKNOWN, 0.3

    def get_recommendations(
        self,
        regime: MarketRegime,
        metrics: RegimeMetrics,
    ) -> tuple[list[str], list[str], str, str]:
        """
        Get strategy recommendations for regime.

        Args:
            regime: Classified regime
            metrics: Regime metrics

        Returns:
            Tuple of (recommended_strategies, avoid_strategies, trade_rec, reasoning)
        """
        if regime == MarketRegime.QUIET_CHOPPY:
            return (
                ["scalping_tight_stops", "volatility_breakout"],
                ["rsi_reversion", "trend_following", "momentum_breakout"],
                "AVOID",
                f"High chop ({metrics.direction_change_rate:.1%}) with low volatility "
                f"({metrics.avg_range_pct:.2f}% range). Signals have no follow-through. "
                f"Wait for volatility expansion or skip this stock.",
            )

        if regime == MarketRegime.VOLATILE_TRENDING:
            return (
                ["trend_following", "momentum_breakout", "trend_fast"],
                ["rsi_reversion", "mean_reversion"],
                "TRADE",
                f"Strong trend ({metrics.period_return:+.1f}%) with high volatility. "
                f"Ride momentum, use trailing stops. Direction change rate only "
                f"{metrics.direction_change_rate:.1%} indicates persistence.",
            )

        if regime == MarketRegime.TRENDING:
            return (
                ["trend_following", "trend_std", "momentum_breakout"],
                ["rsi_reversion", "scalping"],
                "TRADE",
                f"Trending market ({metrics.period_return:+.1f}%). "
                f"Follow the trend direction. Pullback entries work well.",
            )

        if regime == MarketRegime.MEAN_REVERTING:
            return (
                ["rsi_reversion", "rsi_standard", "rsi_relaxed"],
                ["trend_following", "momentum_breakout"],
                "TRADE",
                f"Mean-reverting behavior (autocorr={metrics.autocorrelation:.3f}). "
                f"RSI signals should have follow-through. "
                f"Bounce after drops: {metrics.bounce_after_drop:+.3f}%",
            )

        if regime == MarketRegime.CHOPPY:
            return (
                ["scalping", "volatility_breakout", "range_trading"],
                ["trend_following", "rsi_reversion"],
                "CAUTION",
                f"Choppy market ({metrics.direction_change_rate:.1%} direction changes). "
                f"Use tight stops and quick exits. Consider reducing position size.",
            )

        return (
            ["rsi_standard"],
            [],
            "CAUTION",
            "Unable to classify regime with confidence. Use standard parameters.",
        )

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "5min",
    ) -> RegimeAnalysis:
        """
        Perform complete regime analysis.

        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            timeframe: Data timeframe

        Returns:
            Complete RegimeAnalysis with recommendations
        """
        metrics = self.compute_metrics(df, symbol, timeframe)
        regime, confidence = self.classify_regime(metrics)
        recommended, avoid, trade_rec, reasoning = self.get_recommendations(regime, metrics)

        return RegimeAnalysis(
            metrics=metrics,
            regime=regime,
            confidence=confidence,
            recommended_strategies=recommended,
            avoid_strategies=avoid,
            trade_recommendation=trade_rec,
            reasoning=reasoning,
        )

    def analyze_multiple(
        self,
        data_dict: dict[str, pd.DataFrame],
        timeframe: str = "5min",
    ) -> dict[str, RegimeAnalysis]:
        """
        Analyze multiple symbols.

        Args:
            data_dict: Dict of symbol -> DataFrame
            timeframe: Data timeframe

        Returns:
            Dict of symbol -> RegimeAnalysis
        """
        results = {}
        for symbol, df in data_dict.items():
            try:
                results[symbol] = self.analyze(df, symbol, timeframe)
            except Exception as e:
                print(f"Warning: Could not analyze {symbol}: {e}")
        return results

    def print_analysis(self, analysis: RegimeAnalysis) -> None:
        """Pretty print regime analysis."""
        m = analysis.metrics

        print(f"\n{'='*60}")
        print(f"📊 REGIME ANALYSIS: {m.symbol} ({m.timeframe})")
        print(f"{'='*60}")

        # Regime badge
        regime_icons = {
            MarketRegime.TRENDING: "📈",
            MarketRegime.VOLATILE_TRENDING: "🚀",
            MarketRegime.MEAN_REVERTING: "🔄",
            MarketRegime.CHOPPY: "🌊",
            MarketRegime.QUIET_CHOPPY: "⚠️",
            MarketRegime.UNKNOWN: "❓",
        }
        icon = regime_icons.get(analysis.regime, "")
        print(f"\n{icon} Regime: {analysis.regime.value.upper()}")
        print(f"   Confidence: {analysis.confidence:.0%}")

        # Trade recommendation
        rec_colors = {"TRADE": "🟢", "CAUTION": "🟡", "AVOID": "🔴"}
        rec_icon = rec_colors.get(analysis.trade_recommendation, "")
        print(f"\n{rec_icon} Recommendation: {analysis.trade_recommendation}")

        # Metrics
        print("\n📉 Metrics:")
        print(f"   Period Return:        {m.period_return:+.1f}%")
        print(f"   Direction Changes:    {m.direction_change_rate:.1%}")
        print(f"   Big Move Frequency:   {m.big_move_frequency:.1%}")
        print(f"   Autocorrelation:      {m.autocorrelation:.4f}")
        print(f"   Avg Range:            {m.avg_range_pct:.3f}%")
        print(f"   Volatility:           {m.volatility:.3f}%")

        # NEW: ADX/DMI metrics (trend strength)
        print("\n📈 Trend Strength (ADX/DMI):")
        if m.adx > 0:
            adx_interpretation = (
                "NO TREND"
                if m.adx < 20
                else "EMERGING"
                if m.adx < 25
                else "TRENDING"
                if m.adx < 40
                else "STRONG TREND"
            )
            print(f"   ADX:                  {m.adx:.1f} ({adx_interpretation})")
            print(f"   +DI:                  {m.plus_di:.1f}")
            print(f"   -DI:                  {m.minus_di:.1f}")
            print(f"   Direction:            {m.trend_direction.upper()}")
        else:
            print("   ADX: Not calculated")

        # NEW: ATR metrics (volatility)
        if m.atr_pct > 0:
            vol_level = "LOW" if m.atr_pct < 0.3 else "NORMAL" if m.atr_pct < 0.8 else "HIGH"
            print("\n📊 Volatility (ATR):")
            print(f"   ATR %:                {m.atr_pct:.3f}% ({vol_level})")

        # Mean reversion stats
        print("\n🔄 Mean Reversion Stats:")
        print(f"   Bounce after drop:    {m.bounce_after_drop:+.3f}%")
        print(f"   Reversal after rally: {m.reversal_after_rally:+.3f}%")

        # Recommendations
        print("\n✅ Recommended Strategies:")
        for s in analysis.recommended_strategies:
            print(f"   • {s}")

        print("\n❌ Avoid Strategies:")
        for s in analysis.avoid_strategies:
            print(f"   • {s}")

        print("\n💡 Reasoning:")
        print(f"   {analysis.reasoning}")
        print()


def regime_filter(
    df: pd.DataFrame,
    strategy_type: str,
    symbol: str = "UNKNOWN",
) -> tuple[bool, str]:
    """
    Quick filter to check if a strategy should trade this stock.

    Args:
        df: OHLCV DataFrame
        strategy_type: One of 'rsi', 'momentum', 'trend'
        symbol: Stock symbol

    Returns:
        Tuple of (should_trade, reason)
    """
    detector = RegimeDetector()

    try:
        analysis = detector.analyze(df, symbol)
    except Exception as e:
        return True, f"Could not analyze regime: {e}"

    # Check if strategy matches regime
    if strategy_type in ["rsi", "rsi_reversion", "mean_reversion"]:
        if analysis.regime == MarketRegime.QUIET_CHOPPY:
            return False, f"SKIP: {symbol} is QUIET_CHOPPY - RSI signals won't follow through"
        if analysis.regime in [MarketRegime.TRENDING, MarketRegime.VOLATILE_TRENDING]:
            return False, f"SKIP: {symbol} is TRENDING - use trend strategies instead"

    if strategy_type in ["momentum", "momentum_breakout"]:
        if analysis.regime == MarketRegime.QUIET_CHOPPY:
            return False, f"SKIP: {symbol} is QUIET_CHOPPY - breakouts will fail"
        if analysis.metrics.big_move_frequency < 0.10:
            return (
                False,
                f"SKIP: {symbol} has low big-move frequency ({analysis.metrics.big_move_frequency:.1%})",
            )

    if strategy_type in ["trend", "trend_following"]:
        if analysis.regime in [MarketRegime.CHOPPY, MarketRegime.QUIET_CHOPPY]:
            return False, f"SKIP: {symbol} is CHOPPY - no trend to follow"
        if analysis.metrics.direction_change_rate > 0.50:
            return (
                False,
                f"SKIP: {symbol} direction changes too high ({analysis.metrics.direction_change_rate:.1%})",
            )

    return True, f"OK: {symbol} regime ({analysis.regime.value}) compatible with {strategy_type}"
