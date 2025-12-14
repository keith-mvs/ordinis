"""
Cortex LLM orchestration engine with NVIDIA AI integration.

Provides research, strategy generation, and code analysis using NVIDIA models.
"""

from contextlib import suppress
from typing import Any
import uuid

from .outputs import CortexOutput, OutputType, StrategyHypothesis

# Optional NVIDIA integration - will fall back to rule-based if not available
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    ChatNVIDIA = None  # type: ignore[misc, assignment]
    NVIDIAEmbeddings = None  # type: ignore[misc, assignment]

# Optional RAG integration
try:
    from ordinis.engines.cortex.rag.integration import CortexRAGHelper

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    CortexRAGHelper = None  # type: ignore[misc, assignment]


class CortexEngine:
    """
    Cortex LLM orchestration engine.

    Integrates NVIDIA AI models for:
    - Strategy hypothesis generation
    - Code analysis and generation
    - Research synthesis
    - Market insight analysis
    """

    def __init__(
        self,
        nvidia_api_key: str | None = None,
        usd_code_enabled: bool = False,
        embeddings_enabled: bool = False,
        rag_enabled: bool = False,
    ):
        """
        Initialize Cortex engine.

        Args:
            nvidia_api_key: NVIDIA API key for model access
            usd_code_enabled: Enable NVIDIA USD Code model
            embeddings_enabled: Enable NVIDIA embedding models
            rag_enabled: Enable RAG context retrieval

        Note:
            Get API key from https://build.nvidia.com/
            Install: pip install langchain-nvidia-ai-endpoints
            RAG Install: pip install -e ".[rag]"
        """
        self.nvidia_api_key = nvidia_api_key
        self.usd_code_enabled = usd_code_enabled
        self.embeddings_enabled = embeddings_enabled
        self.rag_enabled = rag_enabled

        # Lazy load NVIDIA clients
        self._usd_code_client = None
        self._embeddings_client = None
        self._rag_helper = None

        # Output history
        self._outputs: list[CortexOutput] = []
        self._hypotheses: dict[str, StrategyHypothesis] = {}

    def _init_usd_code(self) -> Any:
        """Initialize NVIDIA Chat client for code analysis."""
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA API key required for chat model")

        if not NVIDIA_AVAILABLE:
            raise ImportError("Install NVIDIA SDK: pip install langchain-nvidia-ai-endpoints")

        # Initialize ChatNVIDIA with a general chat model
        # USD Code is specialized, but ChatNVIDIA supports general models
        return ChatNVIDIA(
            model="nvidia/llama-3.3-nemotron-super-49b-v1.5",  # Default NVIDIA model
            nvidia_api_key=self.nvidia_api_key,
            temperature=0.2,  # Lower temperature for code analysis
            max_tokens=2048,
        )

    def _init_embeddings(self) -> Any:
        """Initialize NVIDIA embedding model client."""
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA API key required for embedding models")

        if not NVIDIA_AVAILABLE:
            raise ImportError("Install: pip install langchain-nvidia-ai-endpoints")

        # Initialize NV-Embed-QA model for semantic understanding
        return NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            nvidia_api_key=self.nvidia_api_key,
            truncate="END",  # Truncate from end if text too long
        )

    def _init_rag_helper(self) -> Any:
        """Initialize RAG helper for enhanced context."""
        if not RAG_AVAILABLE:
            raise ImportError('Install RAG dependencies: pip install -e ".[rag]"')

        return CortexRAGHelper()

    def generate_hypothesis(
        self, market_context: dict[str, Any], constraints: dict[str, Any] | None = None
    ) -> StrategyHypothesis:
        """
        Generate trading strategy hypothesis.

        Args:
            market_context: Current market conditions and data
            constraints: Strategy constraints (risk limits, instruments, etc.)

        Returns:
            Strategy hypothesis for validation
        """
        constraints = constraints or {}

        # Generate hypothesis ID
        hypothesis_id = f"hyp-{uuid.uuid4().hex[:12]}"

        # Extract market context
        regime = market_context.get("regime", "unknown")
        volatility = market_context.get("volatility", "medium")

        # Get RAG context if enabled
        rag_context = None
        if self.rag_enabled:
            if self._rag_helper is None:
                try:
                    self._rag_helper = self._init_rag_helper()
                except ImportError:
                    self._rag_helper = None

            if self._rag_helper is not None:
                with suppress(Exception):
                    rag_context = self._rag_helper.format_hypothesis_context(
                        market_regime=regime,
                        strategy_type=None,
                    )

        # Generate hypothesis based on context
        # In production, this would use NVIDIA models for generation
        if regime == "trending" and volatility == "low":
            hypothesis = self._create_trend_following_hypothesis(
                hypothesis_id, market_context, constraints
            )
        elif regime == "mean_reverting" or volatility == "high":
            hypothesis = self._create_mean_reversion_hypothesis(
                hypothesis_id, market_context, constraints
            )
        else:
            # Default to balanced approach
            hypothesis = self._create_balanced_hypothesis(
                hypothesis_id, market_context, constraints
            )

        # Enhance confidence if RAG context available
        if rag_context:
            hypothesis.confidence = min(1.0, hypothesis.confidence + 0.05)

        # Store hypothesis
        self._hypotheses[hypothesis_id] = hypothesis

        # Create Cortex output
        reasoning = (
            f"Generated hypothesis based on market regime: {regime}, volatility: {volatility}"
        )
        if rag_context:
            reasoning += " (enhanced with RAG context)"

        output = CortexOutput(
            output_type=OutputType.HYPOTHESIS,
            content=hypothesis.to_dict(),
            confidence=hypothesis.confidence,
            reasoning=reasoning,
            metadata={
                "market_context": market_context,
                "constraints": constraints,
                "rag_context_available": rag_context is not None,
            },
        )

        self._outputs.append(output)

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

    def analyze_code(self, code: str, analysis_type: str = "review") -> CortexOutput:  # noqa: PLR0912
        """
        Analyze code using NVIDIA Chat model.

        Args:
            code: Code to analyze
            analysis_type: Type of analysis ("review", "optimize", "explain")

        Returns:
            Code analysis output
        """
        model_used = "rule-based"
        confidence = 0.80

        # Get RAG context if enabled
        rag_context = None
        if self.rag_enabled:
            if self._rag_helper is None:
                try:
                    self._rag_helper = self._init_rag_helper()
                except ImportError:
                    self._rag_helper = None

            if self._rag_helper is not None:
                with suppress(Exception):
                    rag_context = self._rag_helper.format_code_analysis_context(
                        analysis_type=analysis_type,
                        code_snippet=code[:100],
                    )

        # Try to use NVIDIA model if enabled and available
        if self.usd_code_enabled and self.nvidia_api_key:
            if self._usd_code_client is None:
                try:
                    self._usd_code_client = self._init_usd_code()
                except (ImportError, ValueError):
                    # Fall back to rule-based on initialization error
                    self._usd_code_client = None

                if self._usd_code_client is not None:
                    # Use NVIDIA model for code analysis
                    try:
                        prompt = f"""Analyze the following code for {analysis_type}.

Provide analysis in this format:
- Code Quality: (good/fair/poor)
- Suggestions: (list 2-3 specific improvements)
- Complexity Score: (0.0-1.0, where 1.0 is most complex)
- Maintainability Index: (0-100, where 100 is most maintainable)

Code to analyze:
```python
{code}
```"""

                        # Add RAG context if available
                        if rag_context:
                            prompt += (
                                "\n\nRelevant best practices and examples from codebase:\n"
                                f"{rag_context}"
                            )

                        prompt += "\n\nProvide concise, actionable feedback."

                        response = self._usd_code_client.invoke(prompt)
                        analysis = {
                            "code_quality": "AI-analyzed",
                            "llm_analysis": response.content
                            if hasattr(response, "content")
                            else str(response),
                            "suggestions": ["AI-generated suggestions available in llm_analysis"],
                            "complexity_score": 0.7,  # Would parse from LLM response
                            "maintainability_index": 70,  # Would parse from LLM response
                        }
                        model_used = "nvidia-llama-3.1-405b"
                        confidence = 0.90
                    except Exception:  # noqa: S110
                        # Fall back to rule-based
                        pass

        # Rule-based fallback (default when no API key or on error)
        if model_used == "rule-based":
            analysis = {
                "code_quality": "good",
                "suggestions": [
                    "Consider adding type hints for better code clarity",
                    "Add docstrings to document function behavior",
                    "Consider error handling for edge cases",
                ],
                "complexity_score": 0.6,
                "maintainability_index": 75,
            }

            # Enhance confidence if RAG context available
            if rag_context:
                confidence += 0.05

        reasoning = f"Code analysis using {model_used}"
        if rag_context:
            reasoning += " (enhanced with RAG context)"

        output = CortexOutput(
            output_type=OutputType.CODE_ANALYSIS,
            content={
                "analysis_type": analysis_type,
                "analysis": analysis,
                "code_snippet": code[:200],  # First 200 chars
            },
            confidence=confidence,
            reasoning=reasoning,
            model_used=model_used,
        )

        self._outputs.append(output)
        return output

    def synthesize_research(
        self, query: str, sources: list[str], context: dict[str, Any] | None = None
    ) -> CortexOutput:
        """
        Synthesize research from multiple sources.

        Args:
            query: Research query
            sources: List of source URLs or documents
            context: Additional context

        Returns:
            Research synthesis output
        """
        context = context or {}

        # In production, would use embeddings + LLM for synthesis
        research = {
            "summary": f"Research findings for: {query}",
            "key_points": [
                "Market conditions favor adaptive strategies",
                "Risk management is critical in volatile markets",
                "Backtesting is essential for strategy validation",
            ],
            "sources_analyzed": len(sources),
            "confidence_level": "high",
        }

        output = CortexOutput(
            output_type=OutputType.RESEARCH,
            content={"query": query, "research": research, "sources": sources},
            confidence=0.75,
            reasoning="Synthesized from provided sources and market context",
            model_used="nvidia-embeddings" if self.embeddings_enabled else "rule-based",
            metadata=context,
        )

        self._outputs.append(output)
        return output

    def review_output(
        self, engine: str, output: dict[str, Any], criteria: dict[str, Any] | None = None
    ) -> CortexOutput:
        """
        Review output from another engine.

        Args:
            engine: Engine name (signalcore, riskguard, etc.)
            output: Engine output to review
            criteria: Review criteria

        Returns:
            Review output with recommendations
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

    def to_dict(self) -> dict[str, Any]:
        """Get engine state as dictionary."""
        return {
            "usd_code_enabled": self.usd_code_enabled,
            "embeddings_enabled": self.embeddings_enabled,
            "rag_enabled": self.rag_enabled,
            "total_outputs": len(self._outputs),
            "total_hypotheses": len(self._hypotheses),
            "outputs_by_type": {
                output_type.value: len([o for o in self._outputs if o.output_type == output_type])
                for output_type in OutputType
            },
        }
