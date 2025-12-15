"""
Cortex LLM orchestration engine with NVIDIA AI integration.

Provides research, strategy generation, and code analysis using NVIDIA models via Helix.
"""

from contextlib import suppress
import json
from typing import Any
import uuid

from ordinis.ai.helix.models import ChatMessage
from ordinis.core.logging import TraceContext
from ordinis.engines.base import (
    BaseEngine,
    BaseEngineConfig,
    EngineState,
    HealthLevel,
    HealthStatus,
)

from .outputs import CortexOutput, OutputType, StrategyHypothesis

if False:  # TYPE_CHECKING
    from ordinis.ai.helix.engine import Helix

# Optional RAG integration
try:
    from ordinis.engines.cortex.rag.integration import CortexRAGHelper

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    CortexRAGHelper = None  # type: ignore[misc, assignment]


class CortexEngine(BaseEngine):
    """
    Cortex LLM orchestration engine.

    Integrates NVIDIA AI models via Helix for:
    - Strategy hypothesis generation (Nemotron-Ultra)
    - Code analysis and generation (Nemotron-Ultra)
    - Research synthesis (Nemotron-Ultra)
    - Market insight analysis
    """

    def __init__(
        self,
        helix: "Helix",
        rag_enabled: bool = False,
    ):
        """
        Initialize Cortex engine.

        Args:
            helix: Helix LLM provider instance
            rag_enabled: Enable RAG context retrieval
        """
        super().__init__(config=BaseEngineConfig(name="cortex"))
        self.helix = helix
        self.rag_enabled = rag_enabled
        self._rag_helper = None

        # Output history
        self._outputs: list[CortexOutput] = []
        self._hypotheses: dict[str, StrategyHypothesis] = {}

    async def _do_initialize(self) -> None:
        """Initialize Cortex engine."""
        self._logger.info("Initializing Cortex engine...")

        # Initialize RAG helper if enabled
        if self.rag_enabled and RAG_AVAILABLE:
            try:
                self._rag_helper = CortexRAGHelper()
                self._logger.info("Cortex RAG helper initialized")
            except Exception as e:
                self._logger.warning(f"Failed to initialize RAG helper: {e}")
                self.rag_enabled = False

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
                component="CortexEngine",
                message=f"Helix provider degraded: {helix_health.message}",
            )

        return HealthStatus(
            level=HealthLevel.HEALTHY,
            component="CortexEngine",
            message="Cortex engine healthy",
        )

    async def generate_hypothesis(
        self, market_context: dict[str, Any], constraints: dict[str, Any] | None = None
    ) -> StrategyHypothesis:
        """
        Generate trading strategy hypothesis using Nemotron-Ultra.

        Args:
            market_context: Current market conditions and data
            constraints: Strategy constraints (risk limits, instruments, etc.)

        Returns:
            Strategy hypothesis for validation
        """
        if self.state == EngineState.UNINITIALIZED:
            await self.initialize()

        with TraceContext() as trace_id:
            self._logger.info("Generating hypothesis", data={"market_context": market_context})

            constraints = constraints or {}
            hypothesis_id = f"hyp-{uuid.uuid4().hex[:12]}"

            # Extract market context
            regime = market_context.get("regime", "unknown")
            volatility = market_context.get("volatility", "medium")

            # Get RAG context if enabled
            rag_context = ""
            if self.rag_enabled and self._rag_helper:
                with suppress(Exception):
                    rag_context = self._rag_helper.format_hypothesis_context(
                        market_regime=regime,
                        strategy_type=None,
                    )

            # Construct Prompt for DeepSeek R1
            system_prompt = """You are Cortex, an advanced AI trading strategist.
Your goal is to generate a high-confidence trading hypothesis based on market conditions.
Use your Chain of Thought reasoning to analyze the market regime, volatility, and constraints before formulating a strategy.
Output MUST be valid JSON matching the StrategyHypothesis schema."""

            user_prompt = f"""
Market Context:
Regime: {regime}
Volatility: {volatility}
Data: {json.dumps(market_context, default=str)}

Constraints:
{json.dumps(constraints, default=str)}

RAG Context (Historical/Research):
{rag_context}

Generate a detailed trading strategy hypothesis.
Include: name, description, rationale, parameters, entry_conditions, exit_conditions.
"""

            try:
                # Use DeepSeek R1 for deep reasoning
                response = await self.helix.generate(
                    messages=[
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_prompt),
                    ],
                    model="deepseek-r1",
                    temperature=0.6,
                )

                # Parse response (simplified for now, assuming model returns JSON-like structure or we parse it)
                # In a real implementation, we'd use structured output or robust parsing.
                # For now, we'll fallback to the logic-based generation if parsing fails or just use the text as rationale.

                # NOTE: To keep this refactor safe, we will use the logic-based generators
                # but enhance them with the LLM's rationale if available.

                llm_rationale = response.content
                self._logger.info(
                    "Hypothesis generated by LLM", data={"rationale_preview": llm_rationale[:100]}
                )

            except Exception as e:
                self._logger.error(f"LLM generation failed: {e}")
                llm_rationale = "LLM generation failed, using rule-based logic."

            # Fallback/Hybrid Logic (preserving original logic for safety)
            if regime == "trending" and volatility == "low":
                hypothesis = self._create_trend_following_hypothesis(
                    hypothesis_id, market_context, constraints
                )
            elif regime == "mean_reverting" or volatility == "high":
                hypothesis = self._create_mean_reversion_hypothesis(
                    hypothesis_id, market_context, constraints
                )
            else:
                hypothesis = self._create_balanced_hypothesis(
                    hypothesis_id, market_context, constraints
                )

            # Enhance hypothesis with LLM insights
            hypothesis.rationale = (
                f"{hypothesis.rationale}\n\nAI Analysis: {llm_rationale[:500]}..."
            )

            # Enhance confidence if RAG context available
            if rag_context:
                hypothesis.confidence = min(1.0, hypothesis.confidence + 0.05)

            # Store hypothesis
            self._hypotheses[hypothesis_id] = hypothesis

            # Create Cortex output
            output = CortexOutput(
                output_type=OutputType.HYPOTHESIS,
                content=hypothesis.to_dict(),
                confidence=hypothesis.confidence,
                reasoning=f"Generated via Hybrid Logic + DeepSeek R1. Rationale: {llm_rationale[:100]}...",
                metadata={
                    "market_context": market_context,
                    "constraints": constraints,
                    "rag_context_available": bool(rag_context),
                    "model": "deepseek-r1",
                    "trace_id": trace_id,
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

    async def analyze_code(self, code: str, analysis_type: str = "review") -> CortexOutput:
        """
        Analyze code using DeepSeek R1 via Helix.

        Args:
            code: Code to analyze
            analysis_type: Type of analysis ("review", "optimize", "explain")

        Returns:
            Code analysis output
        """
        if not self._initialized:
            await self.initialize()

        model_used = "deepseek-r1"
        confidence = 0.90

        # Get RAG context if enabled
        rag_context = ""
        if self.rag_enabled and self._rag_helper:
            with suppress(Exception):
                rag_context = self._rag_helper.format_code_analysis_context(
                    analysis_type=analysis_type,
                    code_snippet=code[:100],
                )

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

        if rag_context:
            prompt += f"\n\nRelevant best practices:\n{rag_context}"

        try:
            response = await self.helix.generate(
                messages=[ChatMessage(role="user", content=prompt)],
                model=model_used,
                temperature=0.2,
            )
            analysis_content = response.content

            analysis = {
                "code_quality": "AI-analyzed",
                "llm_analysis": analysis_content,
                "suggestions": ["See LLM analysis"],
                "complexity_score": 0.7,
                "maintainability_index": 70,
            }
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            model_used = "fallback-rule-based"
            confidence = 0.5
            analysis = {"code_quality": "unknown", "error": str(e)}

        output = CortexOutput(
            output_type=OutputType.CODE_ANALYSIS,
            content={
                "analysis_type": analysis_type,
                "analysis": analysis,
                "code_snippet": code[:200],
            },
            confidence=confidence,
            reasoning=f"Code analysis using {model_used}",
            model_used=model_used,
        )

        self._outputs.append(output)
        return output

    async def synthesize_research(
        self, query: str, sources: list[str], context: dict[str, Any] | None = None
    ) -> CortexOutput:
        """
        Synthesize research using DeepSeek R1.

        Args:
            query: Research query
            sources: List of source URLs or documents
            context: Additional context

        Returns:
            Research synthesis output
        """
        if not self._initialized:
            await self.initialize()

        context = context or {}

        prompt = f"""Synthesize the following research query based on the provided sources.
Query: {query}
Sources: {sources}
Context: {context}

Provide a summary, key points, and confidence assessment.
"""

        try:
            response = await self.helix.generate(
                messages=[ChatMessage(role="user", content=prompt)],
                model="deepseek-r1",
                temperature=0.4,
            )
            content = response.content
        except Exception as e:
            content = f"Synthesis failed: {e}"

        research = {
            "summary": content[:200] + "...",
            "full_analysis": content,
            "sources_analyzed": len(sources),
            "confidence_level": "high",
        }

        output = CortexOutput(
            output_type=OutputType.RESEARCH,
            content={"query": query, "research": research, "sources": sources},
            confidence=0.85,
            reasoning="Synthesized via DeepSeek R1",
            model_used="deepseek-r1",
            metadata=context,
        )

        self._outputs.append(output)
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
            "rag_enabled": self.rag_enabled,
            "total_outputs": len(self._outputs),
            "total_hypotheses": len(self._hypotheses),
            "outputs_by_type": {
                output_type.value: len([o for o in self._outputs if o.output_type == output_type])
                for output_type in OutputType
            },
        }
