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
    # Regime Detection
    "MarketRegime",
    "RegimeDetector",
    "RegimeSignal",
    # Trend Following
    "TrendFollowingStrategy",
    "MACrossoverStrategy",
    "BreakoutStrategy",
    "ADXTrendStrategy",
    # Mean Reversion
    "MeanReversionStrategy",
    "BollingerFadeStrategy",
    "RSIReversalStrategy",
    "KeltnerChannelStrategy",
    # Volatility Trading
    "VolatilityStrategy",
    "ScalpingStrategy",
    "VolatilityBreakoutStrategy",
    "ATRTrailingStrategy",
    # Adaptive Management
    "AdaptiveStrategyManager",
    "StrategyWeight",
    "AdaptiveConfig",
    "create_strategy_callback",
]
