"""
Cortex LLM orchestration engine with NVIDIA AI integration.

Provides research, strategy generation, and code analysis using NVIDIA models via Helix.
Includes governance integration, input sanitization, and model fallback chains.
"""

from contextlib import suppress
import json
import re
from typing import Any
import uuid

from prometheus_client import Counter, Histogram

from ordinis.ai.helix.models import ChatMessage
from ordinis.core.logging import TraceContext
from ordinis.engines.base import (
    BaseEngine,
    EngineState,
    HealthLevel,
    HealthStatus,
)

from .config import CortexConfig, ModelConfig
from .outputs import CortexOutput, OutputType, StrategyHypothesis, StructuredHypothesis
from .confidence import ConfidenceCalibrator, ConfidenceConfig, ConfidenceScore


class CortexEngineError(Exception):
    """Exception raised by CortexEngine operations."""

    def __init__(self, code: str, message: str, engine: str = "CortexEngine", recoverable: bool = True):
        super().__init__(message)
        self.code = code
        self.message = message
        self.engine = engine
        self.recoverable = recoverable


if False:  # TYPE_CHECKING
    from ordinis.ai.helix.engine import Helix
    from ordinis.adapters.storage.repositories.cortex import CortexRepository

# --- Prometheus Metrics ---
AI_REQUESTS = Counter("ai_requests_total", "Total AI model requests", ["model", "operation"])
AI_ERRORS = Counter(
    "ai_errors_total", "Total AI model errors", ["model", "operation", "error_type"]
)
AI_LATENCY = Histogram("ai_request_duration_seconds", "AI request latency", ["model", "operation"])
AI_FALLBACKS = Counter("ai_fallbacks_total", "Total AI model fallbacks", ["from_model", "to_model", "operation"])
AI_GOVERNANCE = Counter("ai_governance_total", "Total governance decisions", ["action", "decision"])

# Optional RAG integration
try:
    from ordinis.engines.cortex.rag.integration import CortexRAGHelper

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    CortexRAGHelper = None  # type: ignore[misc, assignment]


# --- Input Sanitization Patterns ---
PII_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),  # Email
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE_REDACTED]"),  # Phone
    (r"\b\d{3}[-]?\d{2}[-]?\d{4}\b", "[SSN_REDACTED]"),  # SSN
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CC_REDACTED]"),  # Credit card
]

SECRET_PATTERNS = [
    (r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]?[^\s'\"]+", "[PASSWORD_REDACTED]"),
    (r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[=:]\s*['\"]?[\w\-]+", "[SECRET_REDACTED]"),
    (r"(?i)bearer\s+[\w\-\.]+", "[BEARER_REDACTED]"),
    (r"(?i)(aws[_-]?(access|secret)[_-]?key?)\s*[=:]\s*['\"]?[\w\-/]+", "[AWS_REDACTED]"),
    # AWS access key ID format: AKIA followed by 16 alphanumeric chars
    (r"\bAKIA[0-9A-Z]{16}\b", "[AWS_KEY_ID_REDACTED]"),
]


class CortexEngine(BaseEngine[CortexConfig]):
    """
    Cortex LLM orchestration engine.

    Integrates NVIDIA AI models via Helix for:
    - Strategy hypothesis generation (Nemotron-Ultra)
    - Code analysis and generation (Nemotron-Ultra)
    - Research synthesis (Nemotron-Ultra)
    - Market insight analysis

    Features:
    - Governance preflight/audit integration
    - Input sanitization (PII/secrets)
    - Model fallback chains
    - Prometheus metrics
    """

    def __init__(
        self,
        helix: "Helix",
        config: CortexConfig | None = None,
        rag_enabled: bool = False,
        repository: "CortexRepository | None" = None,
    ):
        """
        Initialize Cortex engine.

        Args:
            helix: Helix LLM provider instance
            config: Cortex configuration (uses defaults if None)
            rag_enabled: Enable RAG context retrieval (overrides config)
            repository: Optional persistence repository for outputs
        """
        config = config or CortexConfig()
        if rag_enabled:
            config.rag_enabled = True
        super().__init__(config=config)
        self.helix = helix
        self._rag_helper = None
        self._repository = repository

        # Confidence calibration
        self._confidence_calibrator = ConfidenceCalibrator(
            helix=helix,
            config=ConfidenceConfig(enabled=config.use_confidence_calibration),
        )

        # Output history
        self._outputs: list[CortexOutput] = []
        self._hypotheses: dict[str, StrategyHypothesis] = {}

        # Compile sanitization patterns
        self._pii_patterns = [(re.compile(p), r) for p, r in PII_PATTERNS]
        self._secret_patterns = [(re.compile(p), r) for p, r in SECRET_PATTERNS]
        self._blocked_patterns = [re.compile(p) for p in config.safety.blocked_patterns]

    @property
    def rag_enabled(self) -> bool:
        """Check if RAG is enabled."""
        return self._config.rag_enabled and RAG_AVAILABLE

    async def _do_initialize(self) -> None:
        """Initialize Cortex engine."""
        self._logger.info("Initializing Cortex engine...")

        # Validate config
        errors = self._config.validate()
        if errors:
            raise CortexEngineError(
                code="CONFIG_INVALID",
                message=f"Configuration errors: {errors}",
                engine=self.name,
                recoverable=False,
            )

        # Initialize RAG helper if enabled
        if self._config.rag_enabled and RAG_AVAILABLE:
            try:
                self._rag_helper = CortexRAGHelper()
                self._logger.info("Cortex RAG helper initialized")
            except Exception as e:
                self._logger.warning(f"Failed to initialize RAG helper: {e}")

    async def _do_shutdown(self) -> None:
        """Shutdown Cortex engine."""
        self._logger.info("Shutting down Cortex engine...")

    async def _do_health_check(self) -> HealthStatus:
        """Check engine health."""
        # Check Helix health
        helix_health = await self.helix.health_check()
        if helix_health.level != HealthLevel.HEALTHY:
            return HealthStatus(
                level=HealthLevel.DEGRADED,
                message=f"Helix provider degraded: {helix_health.message}",
            )

        return HealthStatus(
            level=HealthLevel.HEALTHY,
            message="Cortex engine healthy",
        )

    # ─────────────────────────────────────────────────────────────────
    # Persistence Helpers
    # ─────────────────────────────────────────────────────────────────

    @property
    def persistence_enabled(self) -> bool:
        """Check if persistence is enabled and available."""
        return self._config.persist_outputs and self._repository is not None

    async def _persist_output(self, output: CortexOutput) -> None:
        """
        Persist a CortexOutput to storage if enabled.

        Args:
            output: Output to persist
        """
        if not self.persistence_enabled:
            return

        try:
            from ordinis.adapters.storage.repositories.cortex import CortexOutputRow

            row = CortexOutputRow(
                output_id=output.output_id,
                output_type=output.output_type.value,
                content=json.dumps(output.content, default=str),
                confidence=output.confidence,
                reasoning=output.reasoning,
                model_used=output.model_used,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                metadata=json.dumps(output.metadata, default=str) if output.metadata else None,
                created_at=output.created_at.isoformat(),
            )
            await self._repository.save_output(row)
            self._logger.debug("Persisted CortexOutput", data={"output_id": output.output_id})
        except Exception as e:
            self._logger.warning(f"Failed to persist CortexOutput: {e}")

    async def _persist_hypothesis(self, hypothesis: StrategyHypothesis) -> None:
        """
        Persist a StrategyHypothesis to storage if enabled.

        Args:
            hypothesis: Hypothesis to persist
        """
        if not self.persistence_enabled:
            return

        try:
            from ordinis.adapters.storage.repositories.cortex import StrategyHypothesisRow

            row = StrategyHypothesisRow(
                hypothesis_id=hypothesis.hypothesis_id,
                name=hypothesis.name,
                description=hypothesis.description,
                rationale=hypothesis.rationale,
                strategy_type=hypothesis.strategy_type,
                instrument_class=hypothesis.instrument_class,
                time_horizon=hypothesis.time_horizon,
                parameters=json.dumps(hypothesis.parameters, default=str),
                entry_conditions=json.dumps(hypothesis.entry_conditions, default=str),
                exit_conditions=json.dumps(hypothesis.exit_conditions, default=str),
                max_position_size_pct=hypothesis.max_position_size_pct,
                stop_loss_pct=hypothesis.stop_loss_pct,
                take_profit_pct=hypothesis.take_profit_pct,
                expected_sharpe=hypothesis.expected_sharpe,
                expected_win_rate=hypothesis.expected_win_rate,
                confidence=hypothesis.confidence,
                validated=hypothesis.validated,
                validation_notes=hypothesis.validation_notes,
                created_at=hypothesis.created_at.isoformat(),
            )
            await self._repository.save_hypothesis(row)
            self._logger.debug("Persisted StrategyHypothesis", data={"hypothesis_id": hypothesis.hypothesis_id})
        except Exception as e:
            self._logger.warning(f"Failed to persist StrategyHypothesis: {e}")

    # ─────────────────────────────────────────────────────────────────
    # Input Sanitization
    # ─────────────────────────────────────────────────────────────────

    def _sanitize_input(self, text: str, context: str = "input") -> str:
        """
        Sanitize input text by removing PII and secrets.

        Args:
            text: Input text to sanitize
            context: Context for logging (e.g., "prompt", "code")

        Returns:
            Sanitized text
        """
        if not self._config.safety.sanitize_inputs:
            return text

        original_len = len(text)
        sanitized = text

        # Strip PII if enabled
        if self._config.safety.strip_pii:
            for pattern, replacement in self._pii_patterns:
                sanitized = pattern.sub(replacement, sanitized)

        # Strip secrets if enabled
        if self._config.safety.strip_secrets:
            for pattern, replacement in self._secret_patterns:
                sanitized = pattern.sub(replacement, sanitized)

        # Check for blocked patterns
        for pattern in self._blocked_patterns:
            if pattern.search(sanitized):
                self._logger.warning(
                    f"Blocked pattern detected in {context}",
                    data={"pattern": pattern.pattern[:50]},
                )
                sanitized = pattern.sub("[BLOCKED]", sanitized)

        if len(sanitized) != original_len:
            self._logger.debug(f"Sanitized {context}: {original_len} -> {len(sanitized)} chars")

        return sanitized

    def _truncate_input(self, text: str, max_length: int, context: str = "input") -> str:
        """Truncate input to maximum length."""
        if len(text) <= max_length:
            return text

        self._logger.warning(
            f"Truncating {context} from {len(text)} to {max_length} chars"
        )
        return text[:max_length] + "\n\n[TRUNCATED]"

    # ─────────────────────────────────────────────────────────────────
    # Model Execution with Fallback
    # ─────────────────────────────────────────────────────────────────

    async def _generate_with_fallback(
        self,
        messages: list[ChatMessage],
        model_config: ModelConfig,
        operation: str,
    ) -> tuple[str, str, int, int]:
        """
        Generate response with model fallback chain.

        Args:
            messages: Chat messages to send
            model_config: Model configuration with fallback chain
            operation: Operation name for metrics

        Returns:
            Tuple of (response_content, model_used, input_tokens, output_tokens)

        Raises:
            EngineError: If all models fail
        """
        models_to_try = [model_config.primary, *model_config.fallback]
        last_error: Exception | None = None

        for i, model in enumerate(models_to_try):
            try:
                AI_REQUESTS.labels(model=model, operation=operation).inc()

                with AI_LATENCY.labels(model=model, operation=operation).time():
                    response = await self.helix.generate(
                        messages=messages,
                        model=model,
                        temperature=model_config.temperature,
                        max_tokens=model_config.max_tokens,
                    )

                # Extract token counts if available
                input_tokens = getattr(response, "input_tokens", 0)
                output_tokens = getattr(response, "output_tokens", 0)

                # Log fallback if not primary
                if i > 0:
                    AI_FALLBACKS.labels(
                        from_model=models_to_try[i - 1],
                        to_model=model,
                        operation=operation,
                    ).inc()
                    self._logger.info(
                        f"Used fallback model {model} for {operation}",
                        data={"attempt": i + 1, "total_models": len(models_to_try)},
                    )

                return response.content, model, input_tokens, output_tokens

            except Exception as e:
                AI_ERRORS.labels(
                    model=model,
                    operation=operation,
                    error_type=type(e).__name__,
                ).inc()
                self._logger.warning(
                    f"Model {model} failed for {operation}: {e}",
                    data={"attempt": i + 1, "total_models": len(models_to_try)},
                )
                last_error = e
                continue

        # All models failed
        raise CortexEngineError(
            code="ALL_MODELS_FAILED",
            message=f"All models failed for {operation}: {last_error}",
            engine=self.name,
            recoverable=True,
        )

    async def _generate_structured_hypothesis(
        self,
        market_context: dict[str, Any],
        constraints: dict[str, Any],
        model_config: ModelConfig,
    ) -> tuple[StructuredHypothesis | None, str, int, int]:
        """
        Generate structured hypothesis using JSON mode.

        Uses LangChain's with_structured_output for Pydantic validation.

        Args:
            market_context: Market conditions
            constraints: Strategy constraints
            model_config: Model configuration

        Returns:
            Tuple of (structured_hypothesis, model_used, input_tokens, output_tokens)
            Returns None for hypothesis if generation fails.
        """
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            from pydantic import ValidationError
        except ImportError:
            self._logger.warning("LangChain NVIDIA not available for structured output")
            return None, "unavailable", 0, 0

        # Build prompt
        regime = self._sanitize_input(str(market_context.get("regime", "unknown")), "regime")
        volatility = self._sanitize_input(str(market_context.get("volatility", "medium")), "volatility")

        system_prompt = """You are Cortex, an advanced AI trading strategist.
Generate a trading strategy hypothesis based on market conditions.
Respond with a valid JSON object matching the schema exactly."""

        user_prompt = f"""Market Regime: {regime}
Volatility: {volatility}
Constraints: {json.dumps(constraints, default=str)}

Generate a strategy hypothesis with:
- Appropriate strategy_type for the market regime
- Realistic risk parameters (max_position_size_pct, stop_loss_pct)
- Clear entry and exit conditions
- Reasonable expected_sharpe and expected_win_rate if applicable
"""

        models_to_try = [model_config.primary, *model_config.fallback]

        for i, model in enumerate(models_to_try):
            try:
                AI_REQUESTS.labels(model=model, operation="structured_hypothesis").inc()

                # Create client with structured output
                client = ChatNVIDIA(
                    model=model,
                    temperature=model_config.temperature,
                    max_completion_tokens=model_config.max_tokens,
                )
                structured_client = client.with_structured_output(StructuredHypothesis)

                # Invoke with structured output
                with AI_LATENCY.labels(model=model, operation="structured_hypothesis").time():
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    result = structured_client.invoke(messages)

                if i > 0:
                    AI_FALLBACKS.labels(
                        from_model=models_to_try[i - 1],
                        to_model=model,
                        operation="structured_hypothesis",
                    ).inc()

                # Result is already validated StructuredHypothesis
                self._logger.info(
                    "Generated structured hypothesis",
                    data={"model": model, "strategy_type": result.strategy_type},
                )
                return result, model, 0, 0  # Token counts not available from this path

            except ValidationError as e:
                self._logger.warning(
                    f"Structured output validation failed for {model}: {e}",
                    data={"errors": e.errors()},
                )
                AI_ERRORS.labels(
                    model=model,
                    operation="structured_hypothesis",
                    error_type="ValidationError",
                ).inc()
                continue

            except Exception as e:
                AI_ERRORS.labels(
                    model=model,
                    operation="structured_hypothesis",
                    error_type=type(e).__name__,
                ).inc()
                self._logger.warning(f"Model {model} failed for structured output: {e}")
                continue

        return None, "failed", 0, 0

    # ─────────────────────────────────────────────────────────────────
    # Public API Methods
    # ─────────────────────────────────────────────────────────────────

    async def generate_hypothesis(
        self, market_context: dict[str, Any], constraints: dict[str, Any] | None = None
    ) -> StrategyHypothesis:
        """
        Generate trading strategy hypothesis using configured model.

        Args:
            market_context: Current market conditions and data
            constraints: Strategy constraints (risk limits, instruments, etc.)

        Returns:
            Strategy hypothesis for validation

        Raises:
            EngineError: If governance denies or all models fail
        """
        if self.state == EngineState.UNINITIALIZED:
            await self.initialize()

        constraints = constraints or {}
        hypothesis_id = f"hyp-{uuid.uuid4().hex[:12]}"

        with TraceContext() as trace_id:
            # --- Governance Preflight ---
            if self._config.governance_enabled:
                preflight_result = await self.preflight(
                    action="generate_hypothesis",
                    inputs={
                        "regime": market_context.get("regime"),
                        "volatility": market_context.get("volatility"),
                        "constraints": constraints,
                    },
                    trace_id=trace_id,
                )
                AI_GOVERNANCE.labels(action="generate_hypothesis", decision=preflight_result.decision.value).inc()

                if preflight_result.blocked:
                    raise CortexEngineError(
                        code="GOVERNANCE_DENIED",
                        message=f"Hypothesis generation denied: {preflight_result.reason}",
                        engine=self.name,
                        recoverable=False,
                    )

                # Apply any adjustments from governance
                if preflight_result.adjustments:
                    constraints = {**constraints, **preflight_result.adjustments}

            self._logger.info("Generating hypothesis", data={"market_context": market_context})

            # Extract and sanitize market context values
            regime_raw = market_context.get("regime", "unknown")
            volatility_raw = market_context.get("volatility", "medium")
            
            # Sanitize individual values before using in prompt
            regime = self._sanitize_input(str(regime_raw), "regime")
            volatility = self._sanitize_input(str(volatility_raw), "volatility")

            # Get RAG context if enabled
            rag_context = ""
            if self.rag_enabled and self._rag_helper:
                with suppress(Exception):
                    rag_context = self._rag_helper.format_hypothesis_context(
                        market_regime=regime,
                        strategy_type=None,
                    )
                    # Truncate RAG context
                    rag_context = self._truncate_input(rag_context, 4000, "rag_context")

            # Construct prompts
            system_prompt = """You are Cortex, an advanced AI trading strategist.
Your goal is to generate a high-confidence trading hypothesis based on market conditions.
Use your Chain of Thought reasoning to analyze the market regime, volatility, and constraints before formulating a strategy.
Output MUST be valid JSON matching the StrategyHypothesis schema."""

            # Sanitize user inputs
            market_context_str = self._sanitize_input(
                json.dumps(market_context, default=str),
                "market_context",
            )
            constraints_str = self._sanitize_input(
                json.dumps(constraints, default=str),
                "constraints",
            )

            user_prompt = f"""
Market Context:
Regime: {regime}
Volatility: {volatility}
Data: {market_context_str}

Constraints:
{constraints_str}

RAG Context (Historical/Research):
{rag_context}

Generate a detailed trading strategy hypothesis.
Include: name, description, rationale, parameters, entry_conditions, exit_conditions.
"""
            user_prompt = self._truncate_input(
                user_prompt,
                self._config.safety.max_prompt_length,
                "user_prompt",
            )

            # Try LLM generation with fallback
            llm_rationale = ""
            model_used = "rule-based"
            input_tokens = 0
            output_tokens = 0
            structured_result: StructuredHypothesis | None = None

            # Try structured output first if enabled
            if self._config.use_structured_output:
                try:
                    structured_result, model_used, input_tokens, output_tokens = (
                        await self._generate_structured_hypothesis(
                            market_context=market_context,
                            constraints=constraints,
                            model_config=self._config.hypothesis_model,
                        )
                    )
                    if structured_result:
                        self._logger.info(
                            "Structured hypothesis generated",
                            data={"model": model_used, "strategy_type": structured_result.strategy_type},
                        )
                        llm_rationale = structured_result.rationale
                except Exception as e:
                    self._logger.warning(f"Structured output failed, falling back: {e}")
                    structured_result = None

            # If structured output failed or disabled, use regular LLM for rationale
            if not structured_result:
                try:
                    llm_rationale, model_used, input_tokens, output_tokens = await self._generate_with_fallback(
                        messages=[
                            ChatMessage(role="system", content=system_prompt),
                            ChatMessage(role="user", content=user_prompt),
                        ],
                        model_config=self._config.hypothesis_model,
                        operation="generate_hypothesis",
                    )
                    self._logger.info(
                        "Hypothesis generated by LLM",
                        data={"model": model_used, "rationale_preview": llm_rationale[:100]},
                    )
                except CortexEngineError:
                    self._logger.warning("All LLM models failed, using rule-based fallback")
                    llm_rationale = "LLM generation failed, using rule-based logic."

            # Create hypothesis from structured output or fallback to rules
            if structured_result:
                # Use LLM-generated structured hypothesis
                hypothesis = StrategyHypothesis(
                    hypothesis_id=hypothesis_id,
                    name=structured_result.name,
                    description=structured_result.description,
                    rationale=structured_result.rationale,
                    instrument_class=structured_result.instrument_class,
                    time_horizon=structured_result.time_horizon,
                    strategy_type=structured_result.strategy_type,
                    parameters=structured_result.parameters,
                    entry_conditions=structured_result.entry_conditions,
                    exit_conditions=structured_result.exit_conditions,
                    max_position_size_pct=structured_result.max_position_size_pct,
                    stop_loss_pct=structured_result.stop_loss_pct,
                    take_profit_pct=structured_result.take_profit_pct,
                    expected_sharpe=structured_result.expected_sharpe,
                    expected_win_rate=structured_result.expected_win_rate,
                    confidence=structured_result.confidence,
                    metadata={"source": "structured_output", "model": model_used},
                )
            elif regime_raw == "trending" and volatility_raw == "low":
                hypothesis = self._create_trend_following_hypothesis(
                    hypothesis_id, market_context, constraints
                )
            elif regime_raw == "mean_reverting" or volatility_raw == "high":
                hypothesis = self._create_mean_reversion_hypothesis(
                    hypothesis_id, market_context, constraints
                )
            else:
                hypothesis = self._create_balanced_hypothesis(
                    hypothesis_id, market_context, constraints
                )

            # Enhance rule-based hypothesis with LLM insights (skip for structured output)
            if not structured_result and llm_rationale:
                hypothesis.rationale = (
                    f"{hypothesis.rationale}\n\nAI Analysis: {llm_rationale[:500]}..."
                )

            # Enhance confidence if RAG context available
            if rag_context:
                hypothesis.confidence = min(1.0, hypothesis.confidence + 0.05)

            # Calibrate confidence using reward model if enabled
            confidence_score: ConfidenceScore | None = None
            if self._config.use_confidence_calibration:
                try:
                    confidence_score = await self._confidence_calibrator.evaluate_hypothesis(
                        hypothesis.to_dict(),
                        market_context,
                    )
                    # Blend original and calibrated confidence (weighted average)
                    original_conf = hypothesis.confidence
                    calibrated_conf = confidence_score.overall
                    hypothesis.confidence = 0.6 * original_conf + 0.4 * calibrated_conf
                    self._logger.debug(
                        f"Confidence calibrated: {original_conf:.2f} -> {hypothesis.confidence:.2f}"
                    )
                except Exception as e:
                    self._logger.warning(f"Confidence calibration failed (non-fatal): {e}")

            # Store hypothesis
            self._hypotheses[hypothesis_id] = hypothesis

            # Create Cortex output
            generation_method = "Structured JSON" if structured_result else "Hybrid Logic"
            output = CortexOutput(
                output_type=OutputType.HYPOTHESIS,
                content=hypothesis.to_dict(),
                confidence=hypothesis.confidence,
                reasoning=f"Generated via {generation_method} + {model_used}. Rationale: {llm_rationale[:100]}...",
                model_used=model_used,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                metadata={
                    "market_context": market_context,
                    "constraints": constraints,
                    "rag_context_available": bool(rag_context),
                    "structured_output_used": structured_result is not None,
                    "confidence_calibrated": confidence_score is not None,
                    "confidence_score": confidence_score.to_dict() if confidence_score else None,
                    "trace_id": trace_id,
                },
            )
            self._outputs.append(output)

            # Persist to storage if enabled
            await self._persist_output(output)
            await self._persist_hypothesis(hypothesis)

            # --- Governance Audit ---
            if self._config.audit_enabled:
                try:
                    await self.audit(
                        action="generate_hypothesis",
                        inputs={"regime": regime, "volatility": volatility},
                        outputs={"hypothesis_id": hypothesis_id, "strategy_type": hypothesis.strategy_type},
                        model_used=model_used,
                        latency_ms=None,  # Captured by Prometheus
                        trace_id=trace_id,
                    )
                except Exception as e:
                    # Audit failures are non-fatal
                    self._logger.warning(f"Audit failed (non-fatal): {e}")

            return hypothesis

    def _create_trend_following_hypothesis(
        self, hypothesis_id: str, market_context: dict[str, Any], constraints: dict[str, Any]
    ) -> StrategyHypothesis:
        """Create trend-following hypothesis."""
        return StrategyHypothesis(
            hypothesis_id=hypothesis_id,
            name="SMA Crossover Trend Following",
            description="Follow trends using moving average crossovers",
            rationale="Low volatility trending market favors momentum strategies",
            instrument_class=constraints.get("instrument_class", "equity"),
            time_horizon="swing",
            strategy_type="trend_following",
            parameters={
                "fast_period": 50,
                "slow_period": 200,
                "min_trend_strength": 0.6,
            },
            entry_conditions=[
                "Fast SMA crosses above slow SMA",
                "Volume above 20-day average",
                "Price above 200-day SMA",
            ],
            exit_conditions=[
                "Fast SMA crosses below slow SMA",
                "Stop loss hit",
                "Profit target reached",
            ],
            max_position_size_pct=constraints.get("max_position_pct", 0.10),
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            expected_sharpe=1.5,
            expected_win_rate=0.45,
            confidence=0.75,
        )

    def _create_mean_reversion_hypothesis(
        self, hypothesis_id: str, market_context: dict[str, Any], constraints: dict[str, Any]
    ) -> StrategyHypothesis:
        """Create mean reversion hypothesis."""
        return StrategyHypothesis(
            hypothesis_id=hypothesis_id,
            name="RSI Mean Reversion",
            description="Buy oversold, sell overbought using RSI",
            rationale="High volatility mean-reverting market favors contrarian strategies",
            instrument_class=constraints.get("instrument_class", "equity"),
            time_horizon="swing",
            strategy_type="mean_reversion",
            parameters={
                "rsi_period": 14,
                "oversold_threshold": 30,
                "overbought_threshold": 70,
                "extreme_oversold": 20,
            },
            entry_conditions=[
                "RSI crosses above 30 from oversold",
                "Price near support level",
                "Volume spike on reversal",
            ],
            exit_conditions=[
                "RSI crosses above 70",
                "Stop loss hit",
                "Profit target reached",
            ],
            max_position_size_pct=constraints.get("max_position_pct", 0.10),
            stop_loss_pct=0.03,
            take_profit_pct=0.08,
            expected_sharpe=1.2,
            expected_win_rate=0.55,
            confidence=0.70,
        )

    def _create_balanced_hypothesis(
        self, hypothesis_id: str, market_context: dict[str, Any], constraints: dict[str, Any]
    ) -> StrategyHypothesis:
        """Create balanced/adaptive hypothesis."""
        return StrategyHypothesis(
            hypothesis_id=hypothesis_id,
            name="Adaptive Regime-Based Strategy",
            description="Adapt strategy based on detected market regime",
            rationale="Unknown/neutral market conditions require adaptive approach",
            instrument_class=constraints.get("instrument_class", "equity"),
            time_horizon="swing",
            strategy_type="adaptive",
            parameters={
                "regime_lookback": 60,
                "volatility_threshold": 0.02,
                "trend_strength_threshold": 0.5,
            },
            entry_conditions=[
                "Regime detection confirms setup",
                "Risk-reward ratio >= 2:1",
                "Multiple timeframe alignment",
            ],
            exit_conditions=[
                "Regime changes",
                "Stop loss hit",
                "Time-based exit (5 days)",
            ],
            max_position_size_pct=constraints.get("max_position_pct", 0.08),
            stop_loss_pct=0.04,
            take_profit_pct=0.10,
            expected_sharpe=1.0,
            expected_win_rate=0.50,
            confidence=0.60,
        )

    async def analyze_code(self, code: str, analysis_type: str = "review") -> CortexOutput:
        """
        Analyze code using configured model via Helix.

        Args:
            code: Code to analyze
            analysis_type: Type of analysis ("review", "optimize", "explain")

        Returns:
            Code analysis output

        Raises:
            EngineError: If governance denies operation
        """
        if self.state == EngineState.UNINITIALIZED:
            await self.initialize()

        with TraceContext() as trace_id:
            # --- Governance Preflight ---
            if self._config.governance_enabled:
                preflight_result = await self.preflight(
                    action="analyze_code",
                    inputs={"analysis_type": analysis_type, "code_length": len(code)},
                    trace_id=trace_id,
                )
                AI_GOVERNANCE.labels(action="analyze_code", decision=preflight_result.decision.value).inc()

                if preflight_result.blocked:
                    raise CortexEngineError(
                        code="GOVERNANCE_DENIED",
                        message=f"Code analysis denied: {preflight_result.reason}",
                        engine=self.name,
                        recoverable=False,
                    )

            # Sanitize and truncate code
            sanitized_code = self._sanitize_input(code, "code")
            sanitized_code = self._truncate_input(
                sanitized_code,
                self._config.safety.max_code_length,
                "code",
            )

            # Get RAG context if enabled
            rag_context = ""
            if self.rag_enabled and self._rag_helper:
                with suppress(Exception):
                    rag_context = self._rag_helper.format_code_analysis_context(
                        analysis_type=analysis_type,
                        code_snippet=sanitized_code[:100],
                    )

            prompt = f"""Analyze the following code for {analysis_type}.

Provide analysis in this format:
- Code Quality: (good/fair/poor)
- Suggestions: (list 2-3 specific improvements)
- Complexity Score: (0.0-1.0, where 1.0 is most complex)
- Maintainability Index: (0-100, where 100 is most maintainable)

Code to analyze:
```python
{sanitized_code}
```"""

            if rag_context:
                prompt += f"\n\nRelevant best practices:\n{rag_context}"

            # Try LLM generation with fallback
            model_used = "fallback-rule-based"
            confidence = 0.5
            input_tokens = 0
            output_tokens = 0

            try:
                analysis_content, model_used, input_tokens, output_tokens = await self._generate_with_fallback(
                    messages=[ChatMessage(role="user", content=prompt)],
                    model_config=self._config.code_analysis_model,
                    operation="analyze_code",
                )
                confidence = 0.90

                analysis = {
                    "code_quality": "AI-analyzed",
                    "llm_analysis": analysis_content,
                    "suggestions": ["See LLM analysis"],
                    "complexity_score": 0.7,
                    "maintainability_index": 70,
                }
            except CortexEngineError as e:
                self._logger.error(f"Code analysis failed: {e}")
                analysis = {"code_quality": "unknown", "error": str(e)}

            output = CortexOutput(
                output_type=OutputType.CODE_ANALYSIS,
                content={
                    "analysis_type": analysis_type,
                    "analysis": analysis,
                    "code_snippet": sanitized_code[:200],
                },
                confidence=confidence,
                reasoning=f"Code analysis using {model_used}",
                model_used=model_used,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                metadata={"trace_id": trace_id},
            )

            self._outputs.append(output)

            # Persist to storage if enabled
            await self._persist_output(output)

            # --- Governance Audit ---
            if self._config.audit_enabled:
                try:
                    await self.audit(
                        action="analyze_code",
                        inputs={"analysis_type": analysis_type, "code_length": len(code)},
                        outputs={"quality": analysis.get("code_quality")},
                        model_used=model_used,
                        trace_id=trace_id,
                    )
                except Exception as e:
                    # Audit failures are non-fatal
                    self._logger.warning(f"Audit failed (non-fatal): {e}")

            return output

    async def synthesize_research(
        self, query: str, sources: list[str], context: dict[str, Any] | None = None
    ) -> CortexOutput:
        """
        Synthesize research using configured model.

        Args:
            query: Research query
            sources: List of source URLs or documents
            context: Additional context

        Returns:
            Research synthesis output

        Raises:
            EngineError: If governance denies operation
        """
        if self.state == EngineState.UNINITIALIZED:
            await self.initialize()

        context = context or {}

        with TraceContext() as trace_id:
            # --- Governance Preflight ---
            if self._config.governance_enabled:
                preflight_result = await self.preflight(
                    action="synthesize_research",
                    inputs={"query": query[:100], "source_count": len(sources)},
                    trace_id=trace_id,
                )
                AI_GOVERNANCE.labels(action="synthesize_research", decision=preflight_result.decision.value).inc()

                if preflight_result.blocked:
                    raise CortexEngineError(
                        code="GOVERNANCE_DENIED",
                        message=f"Research synthesis denied: {preflight_result.reason}",
                        engine=self.name,
                        recoverable=False,
                    )

            # Sanitize inputs
            sanitized_query = self._sanitize_input(query, "query")

            prompt = f"""Synthesize the following research query based on the provided sources.
Query: {sanitized_query}
Sources: {sources}
Context: {context}

Provide a summary, key points, and confidence assessment.
"""

            # Try LLM generation with fallback
            model_used = "fallback"
            input_tokens = 0
            output_tokens = 0

            try:
                content, model_used, input_tokens, output_tokens = await self._generate_with_fallback(
                    messages=[ChatMessage(role="user", content=prompt)],
                    model_config=self._config.research_model,
                    operation="synthesize_research",
                )
            except CortexEngineError as e:
                content = f"Synthesis failed: {e}"

            research = {
                "summary": content[:200] + "...",
                "full_analysis": content,
                "sources_analyzed": len(sources),
                "confidence_level": "high" if model_used != "fallback" else "low",
            }

            output = CortexOutput(
                output_type=OutputType.RESEARCH,
                content={"query": query, "research": research, "sources": sources},
                confidence=0.85 if model_used != "fallback" else 0.5,
                reasoning=f"Synthesized via {model_used}",
                model_used=model_used,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                metadata={**context, "trace_id": trace_id},
            )

            self._outputs.append(output)

            # Persist to storage if enabled
            await self._persist_output(output)

            # --- Governance Audit ---
            if self._config.audit_enabled:
                try:
                    await self.audit(
                        action="synthesize_research",
                        inputs={"query": query[:100], "source_count": len(sources)},
                        outputs={"sources_analyzed": len(sources)},
                        model_used=model_used,
                        trace_id=trace_id,
                    )
                except Exception as e:
                    # Audit failures are non-fatal
                    self._logger.warning(f"Audit failed (non-fatal): {e}")

            return output

    def review_output(
        self, engine: str, output: dict[str, Any], criteria: dict[str, Any] | None = None
    ) -> CortexOutput:
        """
        Review output from another engine.

        Note: This remains synchronous for now as it's a simple rule check,
        but could be upgraded to async LLM check later.
        """
        criteria = criteria or {}

        # Analyze output based on engine type
        strengths: list[str] = []
        concerns: list[str] = []
        recommendations: list[str] = []
        assessment = "satisfactory"

        if engine == "signalcore":
            if output.get("probability", 0) < 0.6:
                concerns.append("Low probability signal")
                recommendations.append("Consider filtering low-confidence signals")

        elif engine == "riskguard":
            if output.get("halted", False):
                concerns.append("Trading halted - investigate cause")
                assessment = "needs_attention"

        review = {
            "engine": engine,
            "assessment": assessment,
            "strengths": strengths,
            "concerns": concerns,
            "recommendations": recommendations,
        }

        output_obj = CortexOutput(
            output_type=OutputType.REVIEW,
            content=review,
            confidence=0.70,
            reasoning=f"Reviewed {engine} output against criteria",
            metadata={"original_output": output, "criteria": criteria},
        )

        self._outputs.append(output_obj)
        # Note: review_output is sync; persistence requires async upgrade
        # For now, review outputs are stored in-memory only
        return output_obj

    def get_hypothesis(self, hypothesis_id: str) -> StrategyHypothesis | None:
        """Get hypothesis by ID."""
        return self._hypotheses.get(hypothesis_id)

    def list_hypotheses(self, min_confidence: float = 0.0) -> list[StrategyHypothesis]:
        """List all hypotheses above minimum confidence."""
        return [h for h in self._hypotheses.values() if h.confidence >= min_confidence]

    def get_outputs(self, output_type: OutputType | None = None) -> list[CortexOutput]:
        """Get all outputs, optionally filtered by type."""
        if output_type:
            return [o for o in self._outputs if o.output_type == output_type]
        return self._outputs.copy()

    async def calibrate_hypothesis_confidence(
        self, hypothesis_id: str
    ) -> ConfidenceScore | None:
        """
        Calibrate confidence for an existing hypothesis using the reward model.

        Args:
            hypothesis_id: ID of the hypothesis to calibrate

        Returns:
            ConfidenceScore if calibration succeeded, None if hypothesis not found
        """
        hypothesis = self._hypotheses.get(hypothesis_id)
        if not hypothesis:
            self._logger.warning(f"Hypothesis {hypothesis_id} not found for calibration")
            return None

        score = await self._confidence_calibrator.evaluate_hypothesis(
            hypothesis.to_dict(),
            {"strategy_type": hypothesis.strategy_type},
        )

        # Update hypothesis confidence with calibrated value
        original_conf = hypothesis.confidence
        hypothesis.confidence = 0.6 * original_conf + 0.4 * score.overall

        self._logger.info(
            f"Hypothesis {hypothesis_id} confidence calibrated: "
            f"{original_conf:.2f} -> {hypothesis.confidence:.2f}"
        )

        return score

    @property
    def confidence_calibrator(self) -> ConfidenceCalibrator:
        """Get the confidence calibrator instance."""
        return self._confidence_calibrator

    def to_dict(self) -> dict[str, Any]:
        """Get engine state as dictionary."""
        return {
            "rag_enabled": self.rag_enabled,
            "total_outputs": len(self._outputs),
            "total_hypotheses": len(self._hypotheses),
            "outputs_by_type": {
                output_type.value: len([o for o in self._outputs if o.output_type == output_type])
                for output_type in OutputType
            },
            "config": self._config.to_dict(),
        }
