"""
LLM-Enhanced RiskGuard with NVIDIA integration.

Uses NVIDIA models for:
- Trade evaluation explanations
- Risk scenario analysis
- Rule optimization suggestions
- Dynamic rule generation
"""

from datetime import datetime
from typing import Any

from ...signalcore.core.signal import Signal
from .engine import PortfolioState, ProposedTrade, RiskGuardEngine
from .rules import RiskCheckResult

# Optional NVIDIA integration
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    ChatNVIDIA = None  # type: ignore[misc, assignment]


class LLMEnhancedRiskGuard(RiskGuardEngine):
    """
    LLM-enhanced RiskGuard that wraps base engine with AI interpretation.

    Adds natural language explanations and insights using NVIDIA LLMs.
    """

    def __init__(
        self,
        base_engine: RiskGuardEngine | None = None,
        nvidia_api_key: str | None = None,
        llm_enabled: bool = False,
    ):
        """
        Initialize LLM-enhanced RiskGuard.

        Args:
            base_engine: Underlying engine to enhance (creates new if None)
            nvidia_api_key: NVIDIA API key for LLM access
            llm_enabled: Enable LLM features
        """
        # Initialize as RiskGuardEngine
        if base_engine:
            super().__init__(rules=base_engine._rules)
            self._halted = base_engine._halted
            self._halt_reason = base_engine._halt_reason
        else:
            super().__init__()

        self.nvidia_api_key = nvidia_api_key
        self.llm_enabled = llm_enabled
        self._llm_client = None

    def _init_llm(self) -> Any:
        """Initialize NVIDIA LLM client."""
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA API key required for LLM features")

        if not NVIDIA_AVAILABLE:
            raise ImportError("Install: pip install langchain-nvidia-ai-endpoints")

        return ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            nvidia_api_key=self.nvidia_api_key,
            temperature=0.3,
            max_completion_tokens=512,
        )

    def evaluate_signal(
        self, signal: Signal, proposed_trade: ProposedTrade, portfolio: PortfolioState
    ) -> tuple[bool, list[RiskCheckResult], Signal | None]:
        """
        Evaluate signal with LLM enhancement.

        Args:
            signal: Trading signal
            proposed_trade: Proposed trade
            portfolio: Portfolio state

        Returns:
            Tuple of (passed, results, adjusted_signal)
        """
        # Get base evaluation
        passed, results, adjusted_signal = super().evaluate_signal(
            signal, proposed_trade, portfolio
        )

        # Add LLM explanation if enabled
        if self.llm_enabled and self.nvidia_api_key:
            explanation = self._generate_explanation(
                signal, proposed_trade, portfolio, passed, results
            )
            if explanation:
                # Add explanation to signal metadata
                if adjusted_signal is None:
                    adjusted_signal = signal
                adjusted_signal.metadata["risk_explanation"] = explanation
                adjusted_signal.metadata["risk_llm_model"] = "meta/llama-3.3-70b-instruct"

        return passed, results, adjusted_signal

    def _generate_explanation(
        self,
        signal: Signal,
        trade: ProposedTrade,
        portfolio: PortfolioState,
        passed: bool,
        results: list[RiskCheckResult],
    ) -> str | None:
        """
        Generate natural language explanation of risk evaluation.

        Args:
            signal: Trading signal
            trade: Proposed trade
            portfolio: Portfolio state
            passed: Whether evaluation passed
            results: Check results

        Returns:
            Explanation string or None
        """
        if self._llm_client is None:
            try:
                self._llm_client = self._init_llm()
            except (ImportError, ValueError):
                return None

        if self._llm_client is None:
            return None

        try:
            # Format results summary
            failed_checks = [r for r in results if not r.passed]
            passed_checks = [r for r in results if r.passed]

            prompt = f"""Explain this risk management evaluation in 2-3 sentences.

Trade Details:
- Symbol: {trade.symbol}
- Direction: {trade.direction}
- Quantity: {trade.quantity:,}
- Entry Price: ${trade.entry_price:.2f}
- Stop Price: ${trade.stop_price:.2f if trade.stop_price else 'None'}

Portfolio State:
- Equity: ${portfolio.equity:,.2f}
- Open Positions: {portfolio.total_positions}
- Daily P&L: ${portfolio.daily_pnl:,.2f}

Risk Evaluation:
- Overall Result: {'PASSED' if passed else 'FAILED'}
- Checks Passed: {len(passed_checks)}/{len(results)}
- Failed Checks: {', '.join(r.rule_name for r in failed_checks) if failed_checks else 'None'}

Provide a concise explanation focusing on:
1. Why the trade passed/failed
2. Key risk factors
3. What action should be taken

Keep it actionable and professional."""

            response = self._llm_client.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        except Exception:
            return None


class LLMRiskAnalyzer:
    """
    LLM-powered risk scenario analysis for RiskGuard.

    Uses NVIDIA models to analyze risk scenarios and suggest optimizations.
    """

    def __init__(self, nvidia_api_key: str | None = None):
        """
        Initialize risk analyzer.

        Args:
            nvidia_api_key: NVIDIA API key
        """
        self.nvidia_api_key = nvidia_api_key
        self._llm_client = None

    def _init_llm(self) -> Any:
        """Initialize NVIDIA LLM client."""
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA API key required")

        if not NVIDIA_AVAILABLE:
            raise ImportError("Install: pip install langchain-nvidia-ai-endpoints")

        return ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            nvidia_api_key=self.nvidia_api_key,
            temperature=0.4,
            max_completion_tokens=1024,
        )

    def analyze_risk_scenario(
        self, scenario_description: str, portfolio: PortfolioState
    ) -> dict[str, Any]:
        """
        Analyze a risk scenario and suggest mitigations.

        Args:
            scenario_description: Description of risk scenario
            portfolio: Current portfolio state

        Returns:
            Analysis results with suggestions
        """
        if not self.nvidia_api_key:
            return self._get_basic_analysis(scenario_description)

        if self._llm_client is None:
            try:
                self._llm_client = self._init_llm()
            except (ImportError, ValueError):
                return self._get_basic_analysis(scenario_description)

        if self._llm_client is None:
            return self._get_basic_analysis(scenario_description)

        try:
            prompt = f"""Analyze this risk scenario for a trading portfolio.

Scenario: {scenario_description}

Portfolio State:
- Equity: ${portfolio.equity:,.2f}
- Cash: ${portfolio.cash:,.2f}
- Open Positions: {portfolio.total_positions}
- Total Exposure: ${portfolio.total_exposure:,.2f}
- Daily P&L: ${portfolio.daily_pnl:,.2f}

Provide analysis in this format:
1. Risk Assessment (High/Medium/Low)
2. Potential Impact (describe in 1 sentence)
3. Recommended Actions (list 2-3 specific actions)
4. Suggested Rule Adjustments (if any)

Be concise and actionable."""

            response = self._llm_client.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            return {
                "scenario": scenario_description,
                "analysis": content,
                "llm_model": "meta/llama-3.3-70b-instruct",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception:
            return self._get_basic_analysis(scenario_description)

    def suggest_rule_optimization(
        self, portfolio: PortfolioState, performance_metrics: dict[str, Any]
    ) -> list[str]:
        """
        Suggest rule optimizations based on portfolio performance.

        Args:
            portfolio: Current portfolio state
            performance_metrics: Performance metrics (sharpe, drawdown, etc.)

        Returns:
            List of optimization suggestions
        """
        if not self.nvidia_api_key:
            return self._get_basic_suggestions()

        if self._llm_client is None:
            try:
                self._llm_client = self._init_llm()
            except (ImportError, ValueError):
                return self._get_basic_suggestions()

        if self._llm_client is None:
            return self._get_basic_suggestions()

        try:
            sharpe = performance_metrics.get("sharpe_ratio", 0.0)
            max_dd = performance_metrics.get("max_drawdown", 0.0)
            win_rate = performance_metrics.get("win_rate", 0.0)

            prompt = f"""Suggest risk rule optimizations for this trading portfolio.

Portfolio State:
- Equity: ${portfolio.equity:,.2f}
- Open Positions: {portfolio.total_positions}
- Drawdown: {(portfolio.equity - portfolio.peak_equity) / portfolio.peak_equity:.1%}

Performance Metrics:
- Sharpe Ratio: {sharpe:.2f}
- Max Drawdown: {max_dd:.1%}
- Win Rate: {win_rate:.1%}

Based on the performance, suggest 3-5 specific rule adjustments to:
1. Improve risk-adjusted returns
2. Reduce drawdown
3. Optimize position sizing

Format each suggestion as:
- Rule Name: Suggested Change (reason)

Be specific with threshold values."""

            response = self._llm_client.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Parse suggestions from response
            suggestions = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and line.strip().startswith("-")
            ]

            return suggestions[:5] if suggestions else self._get_basic_suggestions()

        except Exception:
            return self._get_basic_suggestions()

    def explain_rule(self, rule_name: str, rule_description: str) -> str:
        """
        Explain a risk rule in plain language.

        Args:
            rule_name: Name of rule
            rule_description: Technical description

        Returns:
            Plain language explanation
        """
        if not self.nvidia_api_key:
            return f"{rule_name}: {rule_description}"

        if self._llm_client is None:
            try:
                self._llm_client = self._init_llm()
            except (ImportError, ValueError):
                return f"{rule_name}: {rule_description}"

        if self._llm_client is None:
            return f"{rule_name}: {rule_description}"

        try:
            prompt = f"""Explain this risk management rule in plain language for traders.

Rule: {rule_name}
Technical Description: {rule_description}

Provide a 2-3 sentence explanation covering:
1. What the rule protects against
2. How it works in practice
3. Why it's important

Keep it simple and practical."""

            response = self._llm_client.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        except Exception:
            return f"{rule_name}: {rule_description}"

    def _get_basic_analysis(self, scenario: str) -> dict[str, Any]:
        """Get basic analysis as fallback."""
        return {
            "scenario": scenario,
            "analysis": "Basic risk analysis - enable LLM for detailed insights",
            "llm_model": "rule-based",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _get_basic_suggestions(self) -> list[str]:
        """Get basic suggestions as fallback."""
        return [
            "Review position size limits based on recent volatility",
            "Consider tightening stop loss requirements",
            "Evaluate sector concentration limits",
        ]
