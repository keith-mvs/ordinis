"""
Adaptive Strategy Manager.

Dynamically selects and weights strategies based on detected market regime.

Key Features:
- Real-time regime detection and monitoring
- Automatic strategy selection for current conditions
- Dynamic position sizing based on regime confidence
- Smooth transitions between regimes
- Performance tracking per regime

Architecture:
- RegimeDetector: Classifies current market conditions
- Strategy Pools: Trend-following, Mean-reversion, Volatility
- Weighting System: Allocates capital across strategies
- Risk Manager: Adjusts exposure based on regime
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from .mean_reversion import (
    BollingerFadeStrategy,
    MeanReversionEnsemble,
)
from .regime_detector import MarketRegime, RegimeDetector, RegimeSignal
from .trend_following import (
    MACrossoverStrategy,
    SignalType,
    TradingSignal,
    TrendFollowingEnsemble,
)
from .volatility_trading import (
    ATRTrailingStrategy,
    VolatilityTradingEnsemble,
)


@dataclass
class StrategyWeight:
    """Weight allocation for a strategy pool."""

    trend_following: float = 0.0
    mean_reversion: float = 0.0
    volatility: float = 0.0
    cash: float = 0.0  # Risk-off allocation

    def normalize(self):
        """Ensure weights sum to 1.0."""
        total = self.trend_following + self.mean_reversion + self.volatility + self.cash
        if total > 0:
            self.trend_following /= total
            self.mean_reversion /= total
            self.volatility /= total
            self.cash /= total


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive strategy manager."""

    # Regime detection
    regime_lookback: int = 200
    regime_confirm_bars: int = 3

    # Strategy selection
    use_ensemble: bool = True  # Use ensemble strategies within each pool

    # Position sizing
    base_position_size: float = 0.8  # Base allocation when confident
    min_position_size: float = 0.3  # Minimum during transitions
    confidence_scaling: bool = True  # Scale size by regime confidence

    # Risk management
    max_drawdown_threshold: float = 0.10  # Reduce exposure at 10% drawdown
    volatility_scaling: bool = True  # Reduce size in high volatility

    # Regime-specific overrides
    regime_weights: dict = field(default_factory=dict)


# Default regime weights
DEFAULT_REGIME_WEIGHTS = {
    MarketRegime.BULL: StrategyWeight(
        trend_following=0.80,
        mean_reversion=0.10,
        volatility=0.05,
        cash=0.05,
    ),
    MarketRegime.BEAR: StrategyWeight(
        trend_following=0.20,  # Inverse/defensive
        mean_reversion=0.20,
        volatility=0.10,
        cash=0.50,  # High cash allocation
    ),
    MarketRegime.SIDEWAYS: StrategyWeight(
        trend_following=0.10,
        mean_reversion=0.70,
        volatility=0.10,
        cash=0.10,
    ),
    MarketRegime.VOLATILE: StrategyWeight(
        trend_following=0.20,
        mean_reversion=0.20,
        volatility=0.40,
        cash=0.20,
    ),
    MarketRegime.TRANSITIONAL: StrategyWeight(
        trend_following=0.15,
        mean_reversion=0.15,
        volatility=0.10,
        cash=0.60,  # High cash during uncertainty
    ),
}


@dataclass
class ManagerState:
    """Current state of the adaptive manager."""

    current_regime: MarketRegime
    regime_signal: RegimeSignal
    active_weights: StrategyWeight
    position_size_multiplier: float
    days_in_regime: int
    last_signal: TradingSignal | None
    equity_curve: list = field(default_factory=list)


class AdaptiveStrategyManager:
    """
    Unified manager for regime-adaptive trading.

    Automatically:
    1. Detects current market regime
    2. Selects appropriate strategy pool
    3. Generates trading signals
    4. Manages position sizing
    5. Handles regime transitions
    """

    def __init__(self, config: AdaptiveConfig | None = None):
        self.config = config or AdaptiveConfig()

        # Initialize regime detector
        self.detector = RegimeDetector()

        # Initialize strategy pools
        if self.config.use_ensemble:
            self.trend_strategies = TrendFollowingEnsemble()
            self.reversion_strategies = MeanReversionEnsemble()
            self.volatility_strategies = VolatilityTradingEnsemble()
        else:
            # Single best strategy from each pool
            self.trend_strategies = MACrossoverStrategy()
            self.reversion_strategies = BollingerFadeStrategy()
            self.volatility_strategies = ATRTrailingStrategy()

        # Set regime weights
        self.regime_weights = self.config.regime_weights or DEFAULT_REGIME_WEIGHTS

        # State tracking
        self._state = ManagerState(
            current_regime=MarketRegime.SIDEWAYS,
            regime_signal=RegimeSignal(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.5,
                trend_strength=0,
                volatility_percentile=50,
                momentum=50,
                days_in_regime=0,
            ),
            active_weights=self.regime_weights[MarketRegime.SIDEWAYS],
            position_size_multiplier=1.0,
            days_in_regime=0,
            last_signal=None,
        )

        self._position = 0
        self._entry_price = 0.0

    def update(self, data: pd.DataFrame) -> TradingSignal:
        """
        Process new data and generate trading signal.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            TradingSignal with action and parameters
        """
        # 1. Detect regime
        regime_signal = self.detector.detect(data)
        self._update_state(regime_signal)

        # 2. Get strategy weights for current regime
        weights = self._get_regime_weights(regime_signal)

        # 3. Generate signals from each pool
        signals = self._generate_pool_signals(data)

        # 4. Combine signals based on weights
        combined_signal = self._combine_signals(signals, weights)

        # 5. Apply position sizing
        final_signal = self._apply_position_sizing(combined_signal, regime_signal)

        self._state.last_signal = final_signal
        return final_signal

    def _update_state(self, regime_signal: RegimeSignal):
        """Update manager state with new regime info."""
        if regime_signal.regime != self._state.current_regime:
            # Regime change detected
            self._state.days_in_regime = 0
        else:
            self._state.days_in_regime += 1

        self._state.current_regime = regime_signal.regime
        self._state.regime_signal = regime_signal

        # Update active weights based on current regime
        self._state.active_weights = self._get_regime_weights(regime_signal)

    def _get_regime_weights(self, signal: RegimeSignal) -> StrategyWeight:
        """Get strategy weights for current regime."""
        base_weights = self.regime_weights.get(
            signal.regime, self.regime_weights[MarketRegime.SIDEWAYS]
        )

        # Adjust based on confidence
        if self.config.confidence_scaling and signal.confidence < 0.6:
            # Low confidence - increase cash allocation
            adjusted = StrategyWeight(
                trend_following=base_weights.trend_following * signal.confidence,
                mean_reversion=base_weights.mean_reversion * signal.confidence,
                volatility=base_weights.volatility * signal.confidence,
                cash=1.0 - (signal.confidence * (1 - base_weights.cash)),
            )
            adjusted.normalize()
            return adjusted

        return base_weights

    def _generate_pool_signals(self, data: pd.DataFrame) -> dict:
        """Generate signals from each strategy pool."""
        return {
            "trend": self.trend_strategies.generate_signal(data),
            "reversion": self.reversion_strategies.generate_signal(data),
            "volatility": self.volatility_strategies.generate_signal(data),
        }

    def _combine_signals(self, signals: dict, weights: StrategyWeight) -> TradingSignal:
        """Combine signals using regime-based weights."""
        current_price = signals["trend"].price  # All should have same price

        # Score each signal type
        buy_score = 0.0
        sell_score = 0.0

        # Trend-following contribution
        if signals["trend"].signal_type == SignalType.BUY:
            buy_score += weights.trend_following * signals["trend"].strength
        elif signals["trend"].signal_type in [SignalType.SELL, SignalType.EXIT]:
            sell_score += weights.trend_following * signals["trend"].strength

        # Mean-reversion contribution
        if signals["reversion"].signal_type == SignalType.BUY:
            buy_score += weights.mean_reversion * signals["reversion"].strength
        elif signals["reversion"].signal_type in [SignalType.SELL, SignalType.EXIT]:
            sell_score += weights.mean_reversion * signals["reversion"].strength

        # Volatility contribution
        if signals["volatility"].signal_type == SignalType.BUY:
            buy_score += weights.volatility * signals["volatility"].strength
        elif signals["volatility"].signal_type in [SignalType.SELL, SignalType.EXIT]:
            sell_score += weights.volatility * signals["volatility"].strength

        # Determine action
        # Lower threshold to allow trades in transitional/uncertain regimes
        # With 60% cash, max possible score is 0.4, so threshold must be < 0.4
        action_threshold = 0.2  # Minimum score to act

        if buy_score > sell_score and buy_score >= action_threshold:
            # Find best stop loss from contributing strategies
            stop_losses = [
                s.stop_loss
                for s in signals.values()
                if s.signal_type == SignalType.BUY and s.stop_loss
            ]
            stop_loss = min(stop_losses) if stop_losses else None

            return TradingSignal(
                SignalType.BUY,
                strength=buy_score,
                price=current_price,
                stop_loss=stop_loss,
                position_size=1.0 - weights.cash,  # Invest non-cash portion
            )

        if sell_score > buy_score and sell_score >= action_threshold:
            return TradingSignal(
                SignalType.SELL,
                strength=sell_score,
                price=current_price,
            )

        return TradingSignal(SignalType.HOLD, 0.0, current_price)

    def _apply_position_sizing(self, signal: TradingSignal, regime: RegimeSignal) -> TradingSignal:
        """Apply adaptive position sizing."""
        if signal.signal_type != SignalType.BUY:
            return signal

        # Start with base size
        size = self.config.base_position_size

        # Scale by regime confidence
        if self.config.confidence_scaling:
            size *= regime.confidence

        # Scale by volatility (reduce in high vol)
        if self.config.volatility_scaling and regime.volatility_percentile > 75:
            vol_factor = 1.0 - ((regime.volatility_percentile - 75) / 100)
            size *= vol_factor

        # Ensure minimum
        size = max(size, self.config.min_position_size)

        # Apply signal's position size recommendation
        size *= signal.position_size

        return TradingSignal(
            signal.signal_type,
            signal.strength,
            signal.price,
            signal.stop_loss,
            signal.take_profit,
            position_size=size,
        )

    def get_regime_info(self) -> dict:
        """Get current regime information for display."""
        return {
            "regime": self._state.current_regime.value,
            "confidence": self._state.regime_signal.confidence,
            "days_in_regime": self._state.days_in_regime,
            "trend_strength": self._state.regime_signal.trend_strength,
            "volatility_percentile": self._state.regime_signal.volatility_percentile,
            "momentum": self._state.regime_signal.momentum,
            "weights": {
                "trend_following": self._state.active_weights.trend_following,
                "mean_reversion": self._state.active_weights.mean_reversion,
                "volatility": self._state.active_weights.volatility,
                "cash": self._state.active_weights.cash,
            },
        }

    def reset(self):
        """Reset all strategy states."""
        self.detector.reset()

        if hasattr(self.trend_strategies, "reset"):
            self.trend_strategies.reset()
        if hasattr(self.reversion_strategies, "reset"):
            self.reversion_strategies.reset()
        if hasattr(self.volatility_strategies, "reset"):
            self.volatility_strategies.reset()

        self._state = ManagerState(
            current_regime=MarketRegime.SIDEWAYS,
            regime_signal=RegimeSignal(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.5,
                trend_strength=0,
                volatility_percentile=50,
                momentum=50,
                days_in_regime=0,
            ),
            active_weights=self.regime_weights[MarketRegime.SIDEWAYS],
            position_size_multiplier=1.0,
            days_in_regime=0,
            last_signal=None,
        )

        self._position = 0
        self._entry_price = 0.0


def create_strategy_callback(manager: AdaptiveStrategyManager) -> Callable:
    """
    Create a callback function for use with ProofBench simulator.

    Returns a function compatible with the backtesting engine.
    """
    from src.engines.proofbench.core.execution import Order, OrderSide, OrderType

    position = {"shares": 0, "entry": 0.0}

    def strategy_callback(engine, symbol: str, bar):
        """Adaptive strategy callback for simulator."""
        # Get historical data up to current bar
        data = engine.data[symbol].loc[: bar.timestamp]

        if len(data) < 200:
            return  # Not enough data

        # Generate signal
        signal = manager.update(data)

        current_price = bar.close

        # Execute signal
        if signal.signal_type == SignalType.BUY and position["shares"] == 0:
            # Calculate position size
            available_cash = engine.portfolio.cash
            position_value = available_cash * signal.position_size * 0.95

            shares = int(position_value / current_price)
            if shares > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=shares,
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = shares
                position["entry"] = current_price

        elif signal.signal_type in [SignalType.SELL, SignalType.EXIT]:
            if position["shares"] > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position["shares"],
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = 0

        # Check stop loss
        elif position["shares"] > 0 and signal.stop_loss:
            if current_price <= signal.stop_loss:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position["shares"],
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                position["shares"] = 0

    return strategy_callback
