"""
SignalCore Signal Generation Engine.

Standardized engine extending BaseEngine for ML-based signal generation.
Orchestrates multiple signal models with governance hooks.
"""

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ordinis.engines.base import (
    BaseEngine,
    EngineMetrics,
    GovernanceHook,
    HealthLevel,
    HealthStatus,
)
from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
from ordinis.engines.signalcore.core.ensemble import EnsembleStrategy, SignalEnsemble
from ordinis.engines.signalcore.core.model import Model, ModelRegistry
from ordinis.engines.signalcore.core.signal import Signal, SignalBatch
from ordinis.optimizations.confidence_filter import ConfidenceFilter

if False:  # TYPE_CHECKING
    from ordinis.ai.helix.engine import Helix
    from ordinis.engines.learning.collectors.feedback import FeedbackCollector


class SignalCoreEngine(BaseEngine[SignalCoreEngineConfig]):
    """Unified signal generation engine extending BaseEngine.

    Orchestrates multiple signal models and manages signal generation.
    Provides a unified interface for ML-based trading signal generation.

    Example:
        >>> from ordinis.engines.signalcore import (
        ...     SignalCoreEngine,
        ...     SignalCoreEngineConfig,
        ...     SMACrossoverModel,
        ...     ModelConfig,
        ... )
        >>> config = SignalCoreEngineConfig(
        ...     min_probability=0.6,
        ...     min_score=0.3,
        ...     enable_governance=True,
        ... )
        >>> engine = SignalCoreEngine(config)
        >>> await engine.initialize()
        >>> model_config = ModelConfig(
        ...     model_id="sma_crossover_v1",
        ...     model_type="technical",
        ...     parameters={"short_period": 10, "long_period": 50},
        ... )
        >>> model = SMACrossoverModel(model_config)
        >>> engine.register_model(model)
        >>> signal = await engine.generate_signal(symbol="AAPL", data=df)
    """

    def __init__(
        self,
        config: SignalCoreEngineConfig | None = None,
        governance_hook: GovernanceHook | None = None,
        helix: "Helix | None" = None,
        feedback_collector: "FeedbackCollector | None" = None,
    ) -> None:
        """Initialize the SignalCore engine.

        Args:
            config: Engine configuration (uses defaults if None)
            governance_hook: Optional governance hook for preflight/audit
            helix: Optional Helix engine for LLM features
            feedback_collector: Feedback collector for signal batch recording
        """
        super().__init__(config or SignalCoreEngineConfig(), governance_hook)

        self.helix = helix
        self._feedback = feedback_collector
        self._registry = ModelRegistry()
        self._last_generation: datetime | None = None
        self._signals_generated: int = 0

        # Initialize confidence filter if enabled
        self._confidence_filter: ConfidenceFilter | None = None
        if self.config.enable_confidence_filter:
            self._confidence_filter = ConfidenceFilter(
                min_confidence=self.config.min_confidence,
                min_agreeing_models=self.config.min_agreeing_models,
                apply_volatility_adjustment=self.config.apply_volatility_adjustment,
            )

    async def _do_initialize(self) -> None:
        """Initialize SignalCore engine resources."""
        # Registry is initialized in __init__ and preserved
        self._last_generation = None
        self._signals_generated = 0

    async def _do_shutdown(self) -> None:
        """Shutdown SignalCore engine resources."""
        # Registry doesn't need explicit cleanup

    async def _do_health_check(self) -> HealthStatus:
        """Check SignalCore engine health.

        Returns:
            Current health status
        """
        issues: list[str] = []

        # Check if any models are registered
        enabled_models = self._registry.list_models(enabled_only=True)
        if not enabled_models:
            issues.append("No enabled models registered")

        total_models = self._registry.list_models(enabled_only=False)

        level = HealthLevel.HEALTHY if not issues else HealthLevel.DEGRADED
        return HealthStatus(
            level=level,
            message="SignalCore engine operational" if not issues else "; ".join(issues),
            details={
                "total_models": len(total_models),
                "enabled_models": len(enabled_models),
                "signals_generated": self._signals_generated,
                "last_generation": (
                    self._last_generation.isoformat() if self._last_generation else None
                ),
            },
        )

    def register_model(self, model: Model) -> None:
        """Register a signal model.

        Args:
            model: Model instance to register

        Raises:
            ValueError: If model_id already registered
        """
        self._registry.register(model)
        if model.config.model_id not in self.config.registered_models:
            self.config.registered_models.append(model.config.model_id)

    def unregister_model(self, model_id: str) -> None:
        """Remove a model from the registry.

        Args:
            model_id: Model identifier to remove

        Raises:
            KeyError: If model not found
        """
        self._registry.unregister(model_id)
        if model_id in self.config.registered_models:
            self.config.registered_models.remove(model_id)

    def get_model(self, model_id: str) -> Model:
        """Get a registered model by ID.

        Args:
            model_id: Model identifier

        Returns:
            The registered model instance

        Raises:
            KeyError: If model not found
        """
        return self._registry.get(model_id)

    def list_models(self, enabled_only: bool = False) -> list[str]:
        """List registered model IDs.

        Args:
            enabled_only: Only return enabled models

        Returns:
            List of model identifiers
        """
        return self._registry.list_models(enabled_only=enabled_only)

    async def generate_signals(self, data: Any) -> list[Any]:
        """Generate signals (Protocol implementation)."""
        # Adapt input data if necessary. Assuming data is dict[str, pd.DataFrame]
        # If data is not in expected format, we might need to convert it.
        if isinstance(data, dict):
            # Check if values are DataFrames
            if all(isinstance(v, pd.DataFrame) for v in data.values()):
                batch = await self.generate_batch(data)
                return batch.signals

            # If values are dicts (like from mock data), try to convert or just return empty for now
            # For the demo, we can just return an empty list if data isn't perfect,
            # or try to generate a dummy signal if we want to see flow.
            # Let's return empty list to satisfy protocol without crashing.
            return []

        return []

    async def generate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        model_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> Signal | None:
        """Generate signal for a single symbol.

        Includes governance preflight check if enabled.

        Args:
            symbol: Stock ticker symbol
            data: Historical OHLCV data (indexed by timestamp)
            model_id: Specific model to use (None = first enabled)
            timestamp: Signal timestamp (default: now)

        Returns:
            Signal object or None if generation failed
        """
        timestamp = timestamp or datetime.now(tz=UTC)

        # Governance preflight check
        if self.config.enable_governance and self._governance:
            # BaseEngine.preflight takes (action, inputs, **metadata)
            # It constructs PreflightContext internally
            result = await self.preflight(
                action="generate_signal",
                inputs={
                    "symbol": symbol,
                    "model_id": model_id,
                    "data_points": len(data),
                },
                trace_id=f"signalcore-{symbol}-{timestamp.timestamp()}",
            )
            if not result.allowed:
                await self.audit(
                    action="generate_signal",
                    inputs={"symbol": symbol, "model_id": model_id},
                    outputs={"status": "blocked", "reason": result.reason},
                    latency_ms=0.0,
                )
                return None

        async with self.track_operation("generate_signal"):
            try:
                if model_id:
                    model = self._registry.get(model_id)
                else:
                    enabled_models = self._registry.list_models(enabled_only=True)
                    if not enabled_models:
                        return None
                    model = self._registry.get(enabled_models[0])

                # Validate data
                is_valid, _msg = model.validate(data)
                if not is_valid:
                    return None

                # Generate signal
                signal = await model.generate(symbol, data, timestamp)
                self._signals_generated += 1
                self._last_generation = timestamp

                # Audit success
                if self.config.enable_governance:
                    await self.audit(
                        action="generate_signal",
                        inputs={"symbol": symbol, "model_id": model.config.model_id},
                        outputs={"status": "success", "signal": str(signal)},
                        latency_ms=0.0,
                    )

                # DEBUG print removed for performance
                return signal

            except Exception as e:
                if self.config.enable_governance:
                    await self.audit(
                        action="generate_signal",
                        inputs={"symbol": symbol},
                        outputs={"status": "error", "error": str(e)},
                        latency_ms=0.0,
                    )
                return None

    async def generate_batch(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime | None = None,
    ) -> SignalBatch:
        """Generate signals for multiple symbols using all enabled models.

        Includes governance preflight check if enabled.

        Args:
            data: Dictionary of symbol -> OHLCV data
            timestamp: Batch timestamp (default: now)

        Returns:
            SignalBatch with signals from all models
        """
        timestamp = timestamp or datetime.now(tz=UTC)

        # Circuit breaker check - blocks if any engine has tripped
        # This prevents generating signals when system is in distress
        if self._feedback:
            allowed, _reason = self._feedback.should_allow_signals()
            if not allowed:
                return SignalBatch(
                    timestamp=timestamp,
                    signals=[],
                    universe=list(data.keys()),
                )

        # Governance preflight check
        if self.config.enable_governance and self._governance:
            result = await self.preflight(
                action="generate_batch",
                inputs={
                    "symbols": list(data.keys()),
                    "symbol_count": len(data),
                },
                trace_id=f"signalcore-batch-{timestamp.timestamp()}",
            )
            if not result.allowed:
                await self.audit(
                    action="generate_batch",
                    inputs={"symbols": list(data.keys())},
                    outputs={"status": "blocked", "reason": result.reason},
                    latency_ms=0.0,
                )
                return SignalBatch(
                    timestamp=timestamp,
                    signals=[],
                    universe=list(data.keys()),
                )

        async with self.track_operation("generate_batch"):
            # Enforce batch size limit
            if len(data) > self.config.max_batch_size:
                symbols_to_process = list(data.keys())[: self.config.max_batch_size]
                data = {k: v for k, v in data.items() if k in symbols_to_process}

            batch = await self._registry.generate_all(data, timestamp)

            # Apply ensemble if enabled
            if self.config.enable_ensemble:
                ensemble_signals = []
                # Group signals by symbol
                signals_by_symbol = {}
                for signal in batch.signals:
                    if signal.symbol not in signals_by_symbol:
                        signals_by_symbol[signal.symbol] = []
                    signals_by_symbol[signal.symbol].append(signal)

                # Generate ensemble for each symbol
                strategy = EnsembleStrategy(self.config.ensemble_strategy)
                for symbol, signals in signals_by_symbol.items():
                    if len(signals) > 1:
                        consensus = SignalEnsemble.combine(signals, strategy)
                        if consensus:
                            ensemble_signals.append(consensus)
                    elif signals:
                        ensemble_signals.append(signals[0])

                # Replace batch signals with ensemble signals
                batch.signals = ensemble_signals

            self._signals_generated += len(batch.signals)
            self._last_generation = timestamp

            # Governance audit
            if self.config.enable_governance:
                await self.audit(
                    action="generate_batch",
                    inputs={"symbols_requested": len(data)},
                    outputs={
                        "status": "success",
                        "signals_generated": len(batch.signals),
                    },
                    latency_ms=0.0,
                )

            # Record signal batch for LearningEngine feedback
            if self._feedback and batch.signals:
                avg_prob = sum(s.probability for s in batch.signals) / len(batch.signals)
                avg_score = sum(s.score for s in batch.signals) / len(batch.signals)
                await self._feedback.record_signal_batch(
                    batch_size=len(batch.signals),
                    avg_probability=avg_prob,
                    avg_score=avg_score,
                    strategy="ensemble" if self.config.enable_ensemble else "single",
                    symbols=[s.symbol for s in batch.signals],
                )

            return batch

    def filter_actionable(self, batch: SignalBatch) -> list[Signal]:
        """Filter batch to only actionable signals.

        Applies probability/score thresholds first, then confidence filter
        if enabled. The confidence filter requires 80%+ model agreement.

        Args:
            batch: SignalBatch to filter

        Returns:
            List of actionable signals meeting all thresholds
        """
        # First apply probability/score thresholds
        signals = batch.filter_actionable(
            min_probability=self.config.min_probability,
            min_score=self.config.min_score,
        )

        # Apply confidence filter if enabled
        if self._confidence_filter and signals:
            filtered = []
            for signal in signals:
                signal_dict = {
                    "confidence_score": signal.probability,
                    "num_agreeing_models": signal.metadata.get("agreeing_models", 5),
                    "market_volatility": signal.metadata.get("volatility", 0),
                }
                if self._confidence_filter.should_execute(signal_dict):
                    filtered.append(signal)

            self._logger.debug(
                f"Confidence filter: {len(filtered)}/{len(signals)} signals passed "
                f"({self.config.min_confidence:.0%} threshold)"
            )
            signals = filtered

        return signals

    def get_metrics(self) -> EngineMetrics:
        """Get SignalCore engine metrics.

        Returns:
            Current engine metrics including signal-specific stats
        """
        metrics = super().get_metrics()
        metrics.custom_metrics.update(
            {
                "models_registered": len(self._registry.list_models(enabled_only=False)),
                "models_enabled": len(self._registry.list_models(enabled_only=True)),
                "signals_generated": self._signals_generated,
                "last_generation": (
                    self._last_generation.isoformat() if self._last_generation else None
                ),
            }
        )
        return metrics

    def get_registry_state(self) -> dict[str, Any]:
        """Get model registry state.

        Returns:
            Dictionary with registry information
        """
        return self._registry.to_dict()
