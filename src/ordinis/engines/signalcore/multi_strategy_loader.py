"""
Multi-Strategy Loader for SignalCore.

Supports loading and managing multiple trading strategies with different model types.
Includes signal confluence for combining signals from multiple models.
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ordinis.engines.signalcore.core.model import Model, ModelConfig, ModelRegistry
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType

# Import all available models
from ordinis.engines.signalcore.models import (
    ADXTrendModel,
    ATRBreakoutModel,
    ATROptimizedRSIModel,
    BollingerBandsModel,
    FibonacciRetracementModel,
    GARCHBreakoutModel,
    HMMRegimeModel,
    KalmanHybridModel,
    MACDModel,
    MIEnsembleModel,
    MomentumBreakoutModel,
    MTFMomentumModel,
    OUPairsModel,
    ParabolicSARModel,
    RSIMeanReversionModel,
    RSIVolumeReversionModel,
    SMACrossoverModel,
    StatisticalReversionModel,
    TrendFollowingModel,
    VolumeTrendModel,
)
from ordinis.engines.signalcore.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


# Model type to class mapping
MODEL_REGISTRY: dict[str, type[Model]] = {
    # Mean Reversion
    "atr_optimized_rsi": ATROptimizedRSIModel,
    "rsi_mean_reversion": RSIMeanReversionModel,
    "rsi_volume_reversion": RSIVolumeReversionModel,
    "bollinger_bands": BollingerBandsModel,
    "statistical_reversion": StatisticalReversionModel,
    # Trend Following
    "trend_following": TrendFollowingModel,
    "adx_trend": ADXTrendModel,
    "macd": MACDModel,
    "sma_crossover": SMACrossoverModel,
    "parabolic_sar": ParabolicSARModel,
    # Breakout
    "atr_breakout": ATRBreakoutModel,
    "momentum_breakout": MomentumBreakoutModel,
    "garch_breakout": GARCHBreakoutModel,
    # Advanced Quantitative
    "hmm_regime": HMMRegimeModel,
    "kalman_hybrid": KalmanHybridModel,
    "mi_ensemble": MIEnsembleModel,
    "mtf_momentum": MTFMomentumModel,
    "ou_pairs": OUPairsModel,
    "fibonacci_retracement": FibonacciRetracementModel,
    "volume_trend": VolumeTrendModel,
}


@dataclass
class StrategyWeight:
    """Weight configuration for a strategy in confluence."""

    strategy_name: str
    weight: float = 1.0
    min_confidence: float = 0.5
    enabled: bool = True


@dataclass
class ConfluentSignal:
    """Signal combined from multiple strategies."""

    symbol: str
    timestamp: datetime
    direction: Direction
    signal_type: SignalType
    combined_confidence: float
    contributing_signals: list[Signal]
    strategy_agreement: float  # Percentage of strategies agreeing
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiStrategyLoader:
    """
    Loads and manages multiple trading strategies from YAML configs.

    Supports:
    - Multiple model types (not just ATR-RSI)
    - Strategy weighting for confluence
    - Regime-aware signal generation
    - Signal aggregation across strategies

    Usage:
        loader = MultiStrategyLoader()
        loader.load_strategy("configs/strategies/atr_optimized_rsi.yaml")
        loader.load_strategy("configs/strategies/trend_following.yaml")

        # Get confluent signal for a symbol
        signal = loader.get_confluent_signal("AAPL", df, timestamp)
    """

    def __init__(self):
        """Initialize multi-strategy loader."""
        self.strategies: dict[str, dict] = {}
        self.models: dict[str, dict[str, Model]] = {}  # strategy -> symbol -> model
        self.weights: dict[str, StrategyWeight] = {}
        self.detector = RegimeDetector()
        self.registry = ModelRegistry()

    def load_strategy(
        self,
        config_path: str,
        weight: float = 1.0,
        min_confidence: float = 0.5,
    ) -> bool:
        """
        Load strategy from YAML config.

        Args:
            config_path: Path to strategy YAML file.
            weight: Weight for this strategy in confluence (default 1.0).
            min_confidence: Minimum confidence to include signal (default 0.5).

        Returns:
            True if loaded successfully.
        """
        path = Path(config_path)
        if not path.exists():
            logger.error(f"Strategy config not found: {config_path}")
            return False

        try:
            with open(path) as f:
                config = yaml.safe_load(f)

            strategy_info = config.get("strategy", {})
            strategy_name = strategy_info.get("name", path.stem)
            strategy_type = strategy_info.get("type", "mean_reversion")

            # Get model class for this strategy type
            model_class = MODEL_REGISTRY.get(strategy_type)
            if model_class is None:
                # Try to infer from name
                for key, cls in MODEL_REGISTRY.items():
                    if key in path.stem.lower():
                        model_class = cls
                        strategy_type = key
                        break

            if model_class is None:
                logger.warning(
                    f"Unknown strategy type '{strategy_type}', defaulting to ATROptimizedRSIModel"
                )
                model_class = ATROptimizedRSIModel

            self.strategies[strategy_name] = config
            self.models[strategy_name] = {}
            self.weights[strategy_name] = StrategyWeight(
                strategy_name=strategy_name,
                weight=weight,
                min_confidence=min_confidence,
            )

            # Create models for each symbol
            global_params = config.get("global_params", {})
            symbols = config.get("symbols", {})

            for symbol, symbol_params in symbols.items():
                params = {**global_params, **symbol_params}

                model_config = ModelConfig(
                    model_id=f"{strategy_type}_{symbol}",
                    model_type=strategy_type,
                    version=strategy_info.get("version", "1.0.0"),
                    parameters=params,
                )

                model = model_class(model_config)
                self.models[strategy_name][symbol] = model

                # Also register in global registry
                try:
                    self.registry.register(model)
                except ValueError:
                    pass  # Already registered

            logger.info(
                f"Loaded strategy '{strategy_name}' ({strategy_type}) "
                f"with {len(symbols)} symbols, weight={weight}"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading strategy: {e}")
            return False

    def get_model(self, strategy_name: str, symbol: str) -> Model | None:
        """Get model for a specific strategy and symbol."""
        return self.models.get(strategy_name, {}).get(symbol)

    def get_all_models_for_symbol(self, symbol: str) -> list[tuple[str, Model]]:
        """Get all models that trade a symbol."""
        results = []
        for strategy_name, models in self.models.items():
            if symbol in models:
                results.append((strategy_name, models[symbol]))
        return results

    def get_symbols(self) -> set[str]:
        """Get all configured symbols across all strategies."""
        symbols = set()
        for models in self.models.values():
            symbols.update(models.keys())
        return symbols

    def should_trade(
        self, symbol: str, df: pd.DataFrame, timeframe: str = "5min"
    ) -> tuple[bool, str]:
        """
        Check if symbol should be traded based on regime filter.

        Uses the most restrictive regime filter across all strategies.
        """
        # Check if any strategy trades this symbol
        strategies_with_symbol = [name for name, models in self.models.items() if symbol in models]

        if not strategies_with_symbol:
            return False, f"{symbol} not in any strategy config"

        # Check regime for each strategy
        for strategy_name in strategies_with_symbol:
            config = self.strategies.get(strategy_name, {})
            regime_filter = config.get("regime_filter", {})

            if not regime_filter.get("enabled", True):
                continue

            avoid_regimes = regime_filter.get("avoid_regimes", ["quiet_choppy", "choppy"])

            try:
                analysis = self.detector.analyze(df, symbol=symbol, timeframe=timeframe)

                if analysis.regime.value in avoid_regimes:
                    return False, f"{symbol} regime is {analysis.regime.value}"

            except Exception as e:
                logger.warning(f"Regime detection failed for {symbol}: {e}")

        return True, "OK"

    async def generate_signals(
        self,
        symbol: str,
        df: pd.DataFrame,
        timestamp: datetime,
    ) -> list[tuple[str, Signal]]:
        """
        Generate signals from all strategies for a symbol.

        Returns:
            List of (strategy_name, signal) tuples.
        """
        signals = []

        for strategy_name, models in self.models.items():
            if symbol not in models:
                continue

            weight_config = self.weights.get(strategy_name)
            if weight_config and not weight_config.enabled:
                continue

            model = models[symbol]

            try:
                is_valid, msg = model.validate(df)
                if not is_valid:
                    continue

                signal = await model.generate(symbol, df, timestamp)
                if signal:
                    # Check minimum confidence
                    confidence = getattr(signal, "confidence", None) or signal.probability
                    if weight_config and confidence < weight_config.min_confidence:
                        continue

                    signals.append((strategy_name, signal))

            except Exception as e:
                logger.debug(f"Strategy {strategy_name} failed for {symbol}: {e}")
                continue

        return signals

    async def get_confluent_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        timestamp: datetime,
        min_agreement: float = 0.5,
    ) -> ConfluentSignal | None:
        """
        Get a confluent signal combining all strategies.

        Args:
            symbol: Stock symbol.
            df: Price DataFrame.
            timestamp: Current timestamp.
            min_agreement: Minimum strategy agreement ratio (default 0.5 = 50%).

        Returns:
            ConfluentSignal if strategies agree, None otherwise.
        """
        signals = await self.generate_signals(symbol, df, timestamp)

        if not signals:
            return None

        # Separate by direction
        long_signals = [(name, sig) for name, sig in signals if sig.direction == Direction.LONG]
        short_signals = [(name, sig) for name, sig in signals if sig.direction == Direction.SHORT]

        # Calculate weighted votes
        long_weight = sum(
            self.weights.get(name, StrategyWeight(name)).weight for name, _ in long_signals
        )
        short_weight = sum(
            self.weights.get(name, StrategyWeight(name)).weight for name, _ in short_signals
        )

        total_weight = long_weight + short_weight
        if total_weight == 0:
            return None

        # Determine winning direction
        if long_weight > short_weight:
            winning_signals = long_signals
            direction = Direction.LONG
            agreement = long_weight / total_weight
        else:
            winning_signals = short_signals
            direction = Direction.SHORT
            agreement = short_weight / total_weight

        # Check minimum agreement
        if agreement < min_agreement:
            return None

        # Calculate combined confidence (weighted average)
        combined_confidence = 0.0
        total_conf_weight = 0.0
        for name, sig in winning_signals:
            weight = self.weights.get(name, StrategyWeight(name)).weight
            confidence = getattr(sig, "confidence", None) or sig.probability
            combined_confidence += confidence * weight
            total_conf_weight += weight

        if total_conf_weight > 0:
            combined_confidence /= total_conf_weight

        # Determine signal type (entry if any entry, exit if all exit)
        signal_types = [sig.signal_type for _, sig in winning_signals]
        if SignalType.ENTRY in signal_types:
            signal_type = SignalType.ENTRY
        else:
            signal_type = SignalType.EXIT

        return ConfluentSignal(
            symbol=symbol,
            timestamp=timestamp,
            direction=direction,
            signal_type=signal_type,
            combined_confidence=combined_confidence,
            contributing_signals=[sig for _, sig in winning_signals],
            strategy_agreement=agreement,
            metadata={
                "long_weight": long_weight,
                "short_weight": short_weight,
                "strategies": [name for name, _ in winning_signals],
            },
        )

    def set_strategy_weight(self, strategy_name: str, weight: float) -> None:
        """Update weight for a strategy."""
        if strategy_name in self.weights:
            self.weights[strategy_name].weight = weight

    def enable_strategy(self, strategy_name: str, enabled: bool = True) -> None:
        """Enable or disable a strategy."""
        if strategy_name in self.weights:
            self.weights[strategy_name].enabled = enabled

    def get_risk_params(self, symbol: str) -> dict:
        """Get combined risk parameters for a symbol."""
        # Use most conservative params across strategies
        max_position_pct = 100.0
        max_daily_loss_pct = 100.0
        max_concurrent = 100

        for strategy_name, config in self.strategies.items():
            if symbol not in self.models.get(strategy_name, {}):
                continue

            risk = config.get("risk_management", {})
            max_position_pct = min(max_position_pct, risk.get("max_position_size_pct", 5.0))
            max_daily_loss_pct = min(max_daily_loss_pct, risk.get("max_daily_loss_pct", 2.0))
            max_concurrent = min(max_concurrent, risk.get("max_concurrent_positions", 5))

        return {
            "max_position_pct": max_position_pct,
            "max_daily_loss_pct": max_daily_loss_pct,
            "max_concurrent_positions": max_concurrent,
        }

    def list_strategies(self) -> list[dict]:
        """List all loaded strategies with their status."""
        return [
            {
                "name": name,
                "symbols": len(models),
                "weight": self.weights.get(name, StrategyWeight(name)).weight,
                "enabled": self.weights.get(name, StrategyWeight(name)).enabled,
                "type": self.strategies.get(name, {}).get("strategy", {}).get("type", "unknown"),
            }
            for name, models in self.models.items()
        ]
