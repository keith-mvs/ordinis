"""
Regime-Adaptive Trading Strategies.

A comprehensive suite of strategies that adapt to market conditions:
- Regime Detection: Identifies bull, bear, sideways, and volatile markets
- Trend-Following: MA crossovers, breakouts for trending markets
- Mean-Reversion: Bollinger fades, RSI reversals for ranging markets
- Volatility Trading: Scalping, options-aware for high-volatility periods
- Adaptive Weighting: Dynamic strategy allocation based on detected regime
"""

from .adaptive_manager import (
    AdaptiveConfig,
    AdaptiveStrategyManager,
    StrategyWeight,
    create_strategy_callback,
)
from .mean_reversion import (
    BollingerFadeStrategy,
    KeltnerChannelStrategy,
    MeanReversionStrategy,
    RSIReversalStrategy,
)
from .regime_detector import (
    MarketRegime,
    RegimeDetector,
    RegimeSignal,
)
from .trend_following import (
    ADXTrendStrategy,
    BreakoutStrategy,
    MACrossoverStrategy,
    TrendFollowingStrategy,
)
from .volatility_trading import (
    ATRTrailingStrategy,
    ScalpingStrategy,
    VolatilityBreakoutStrategy,
    VolatilityStrategy,
)

__all__ = [
    "ADXTrendStrategy",
    "ATRTrailingStrategy",
    "AdaptiveConfig",
    # Adaptive Management
    "AdaptiveStrategyManager",
    "BollingerFadeStrategy",
    "BreakoutStrategy",
    "KeltnerChannelStrategy",
    "MACrossoverStrategy",
    # Regime Detection
    "MarketRegime",
    # Mean Reversion
    "MeanReversionStrategy",
    "RSIReversalStrategy",
    "RegimeDetector",
    "RegimeSignal",
    "ScalpingStrategy",
    "StrategyWeight",
    # Trend Following
    "TrendFollowingStrategy",
    "VolatilityBreakoutStrategy",
    # Volatility Trading
    "VolatilityStrategy",
    "create_strategy_callback",
]
