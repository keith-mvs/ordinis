"""
Live Trading Strategy Loader.

Integrates the ATR-Optimized RSI strategy with the Ordinis runtime.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models.atr_optimized_rsi import ATROptimizedRSIModel
from ordinis.engines.signalcore.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


class StrategyLoader:
    """
    Loads and manages trading strategies from YAML configs.

    Usage:
        loader = StrategyLoader()
        loader.load_strategy("configs/strategies/atr_optimized_rsi.yaml")

        # Get model for a symbol
        model = loader.get_model("COIN")

        # Check if symbol should be traded
        if loader.should_trade("COIN", df):
            signal = await model.generate("COIN", df, timestamp)
    """

    def __init__(self):
        """Initialize strategy loader."""
        self.strategies: dict[str, dict] = {}
        self.models: dict[str, ATROptimizedRSIModel] = {}
        self.detector = RegimeDetector()

    def load_strategy(self, config_path: str) -> bool:
        """
        Load strategy from YAML config.

        Args:
            config_path: Path to strategy YAML file.

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

            strategy_name = config.get("strategy", {}).get("name", path.stem)
            self.strategies[strategy_name] = config

            # Create models for each symbol
            global_params = config.get("global_params", {})
            symbols = config.get("symbols", {})

            for symbol, symbol_params in symbols.items():
                # Merge global and symbol-specific params
                params = {**global_params, **symbol_params}

                model_config = ModelConfig(
                    model_id=f"atr_rsi_{symbol}",
                    model_type="mean_reversion",
                    version=config.get("strategy", {}).get("version", "1.0.0"),
                    parameters=params,
                )

                self.models[symbol] = ATROptimizedRSIModel(model_config)

            logger.info(f"Loaded strategy '{strategy_name}' with {len(symbols)} symbols")
            return True

        except Exception as e:
            logger.error(f"Error loading strategy: {e}")
            return False

    def get_model(self, symbol: str) -> ATROptimizedRSIModel | None:
        """Get model for a symbol."""
        return self.models.get(symbol)

    def get_symbols(self) -> list[str]:
        """Get all configured symbols."""
        return list(self.models.keys())

    def should_trade(self, symbol: str, df: Any, timeframe: str = "5min") -> tuple[bool, str]:
        """
        Check if symbol should be traded based on regime filter.

        Args:
            symbol: Stock symbol.
            df: Price DataFrame for regime analysis.
            timeframe: Data timeframe.

        Returns:
            Tuple of (should_trade, reason).
        """
        # Check if symbol is configured
        if symbol not in self.models:
            return False, f"{symbol} not in strategy config"

        # Get regime filter settings
        for strategy_config in self.strategies.values():
            regime_filter = strategy_config.get("regime_filter", {})

            if not regime_filter.get("enabled", True):
                return True, "Regime filter disabled"

            avoid_regimes = regime_filter.get("avoid_regimes", ["quiet_choppy", "choppy"])

            try:
                analysis = self.detector.analyze(df, symbol=symbol, timeframe=timeframe)

                if analysis.regime.value in avoid_regimes:
                    return False, f"{symbol} regime is {analysis.regime.value}"

                return True, f"{symbol} regime is {analysis.regime.value} - OK"

            except Exception as e:
                logger.warning(f"Regime detection failed for {symbol}: {e}")
                return True, "Regime detection failed, proceeding with caution"

        return True, "No regime filter configured"

    def get_risk_params(self, symbol: str) -> dict:
        """Get risk management parameters for a symbol."""
        for strategy_config in self.strategies.values():
            risk = strategy_config.get("risk_management", {})
            symbol_config = strategy_config.get("symbols", {}).get(symbol, {})

            return {
                "max_position_pct": risk.get("max_position_size_pct", 5.0),
                "max_daily_loss_pct": risk.get("max_daily_loss_pct", 2.0),
                "max_concurrent_positions": risk.get("max_concurrent_positions", 5),
                "atr_stop_mult": symbol_config.get("atr_stop_mult", 1.5),
                "atr_tp_mult": symbol_config.get("atr_tp_mult", 2.0),
            }

        return {}


def register_atr_rsi_strategy(
    container: Any, config_path: str = "configs/strategies/atr_optimized_rsi.yaml"
) -> StrategyLoader:
    """
    Register ATR-Optimized RSI strategy with the container.

    Args:
        container: Ordinis container instance.
        config_path: Path to strategy config.

    Returns:
        StrategyLoader instance.
    """
    loader = StrategyLoader()
    loader.load_strategy(config_path)

    # Register models with SignalCore engine if available
    if hasattr(container, "signal_engine"):
        for symbol, model in loader.models.items():
            try:
                container.signal_engine.registry.register(model)
                logger.info(f"Registered ATR-RSI model for {symbol}")
            except Exception as e:
                logger.warning(f"Could not register model for {symbol}: {e}")

    # Store loader on container for access
    container.strategy_loader = loader

    return loader
