"""
LLM-Enhanced ProofBench Analytics with NVIDIA integration.

Uses NVIDIA models for:
- Performance narration and insights
- Strategy optimization suggestions
- Trade pattern analysis
- Comparative backtest analysis
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ordinis.engines.proofbench.core.simulator import SimulationResults

# Optional NVIDIA integration
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    ChatNVIDIA = None  # type: ignore[misc, assignment]


class LLMPerformanceNarrator:
    """
    LLM-powered performance narration for backtest results.

    Uses NVIDIA models to generate natural language insights and explanations.
    """

    def __init__(self, nvidia_api_key: str | None = None):
        """
        Initialize performance narrator.

        Args:
            nvidia_api_key: NVIDIA API key for LLM access
        """
        self.nvidia_api_key = nvidia_api_key
        self._llm_client = None

    def _init_llm(self) -> Any:
        """Initialize NVIDIA LLM client."""
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA API key required for LLM features")

        if not NVIDIA_AVAILABLE:
            raise ImportError("Install: pip install langchain-nvidia-ai-endpoints")

        return ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            nvidia_api_key=self.nvidia_api_key,
            temperature=0.4,
            max_completion_tokens=1024,
        )

    def narrate_results(self, results: SimulationResults) -> dict[str, Any]:
        """
        Generate natural language narration of backtest results.

        Args:
            results: Simulation results to narrate

        Returns:
            Narration with insights and recommendations
        """
        if not self.nvidia_api_key:
            return self._get_basic_narration(results)

        if self._llm_client is None:
            try:
                self._llm_client = self._init_llm()
            except (ImportError, ValueError):
                return self._get_basic_narration(results)

        if self._llm_client is None:
            return self._get_basic_narration(results)

        try:
            metrics = results.metrics

            prompt = f"""Narrate this backtesting performance in a clear, insightful way.

Backtest Summary:
- Period: {results.start_time.date()} to {results.end_time.date()}
- Initial Capital: ${results.config.initial_capital:,.0f}
- Final Equity: ${metrics.equity_final:,.0f}

Returns:
- Total Return: {metrics.total_return:.2%}
- Annualized Return: {metrics.annualized_return:.2%}
- Volatility: {metrics.volatility:.2%}

Risk Metrics:
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Sortino Ratio: {metrics.sortino_ratio:.2f}
- Max Drawdown: {metrics.max_drawdown:.2%}
- Calmar Ratio: {metrics.calmar_ratio:.2f}

Trade Statistics:
- Number of Trades: {metrics.num_trades}
- Win Rate: {metrics.win_rate:.1%}
- Profit Factor: {metrics.profit_factor:.2f}
- Avg Win: ${metrics.avg_win:,.2f}
- Avg Loss: ${metrics.avg_loss:,.2f}

Provide analysis in this format:
1. Overall Assessment (1-2 sentences on strategy viability)
2. Strengths (2-3 key positives)
3. Weaknesses (2-3 key concerns)
4. Optimization Suggestions (3-4 specific improvements)

Be direct and actionable."""

            response = self._llm_client.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            return {
                "narration": content,
                "llm_model": "nvidia-llama-3.1-70b",
                "timestamp": datetime.utcnow().isoformat(),
                "metrics_summary": {
                    "total_return": metrics.total_return,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "max_drawdown": metrics.max_drawdown,
                    "win_rate": metrics.win_rate,
                },
            }

        except Exception:
            return self._get_basic_narration(results)

    def compare_results(self, results_list: list[tuple[str, SimulationResults]]) -> dict[str, Any]:
        """
        Compare multiple backtest results and identify best strategy.

        Args:
            results_list: List of (name, results) tuples to compare

        Returns:
            Comparative analysis with recommendations
        """
        if not self.nvidia_api_key:
            return self._get_basic_comparison(results_list)

        if self._llm_client is None:
            try:
                self._llm_client = self._init_llm()
            except (ImportError, ValueError):
                return self._get_basic_comparison(results_list)

        if self._llm_client is None:
            return self._get_basic_comparison(results_list)

        try:
            comparison_data = []
            for name, results in results_list:
                m = results.metrics
                comparison_data.append(
                    f"{name}:\n"
                    f"  Return: {m.total_return:.2%}, Sharpe: {m.sharpe_ratio:.2f}, "
                    f"Drawdown: {m.max_drawdown:.2%}, Win Rate: {m.win_rate:.1%}"
                )

            prompt = f"""Compare these backtesting results and recommend the best strategy.

Strategies Tested:
{chr(10).join(comparison_data)}

Provide analysis:
1. Rankings (rank by overall quality)
2. Best For Risk-Adjusted Returns (which has best Sharpe/Sortino)
3. Best For Stability (lowest drawdown)
4. Best For Consistency (highest win rate)
5. Recommendation (which to deploy and why, in 2-3 sentences)

Be specific and practical."""

            response = self._llm_client.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            return {
                "comparison": content,
                "llm_model": "nvidia-llama-3.1-70b",
                "timestamp": datetime.utcnow().isoformat(),
                "strategies_compared": len(results_list),
            }

        except Exception:
            return self._get_basic_comparison(results_list)

    def suggest_optimizations(
        self, results: SimulationResults, focus: str = "general"
    ) -> list[str]:
        """
        Suggest strategy optimizations based on results.

        Args:
            results: Simulation results
            focus: Optimization focus ("returns", "risk", "consistency", "general")

        Returns:
            List of optimization suggestions
        """
        if not self.nvidia_api_key:
            return self._get_basic_suggestions(focus)

        if self._llm_client is None:
            try:
                self._llm_client = self._init_llm()
            except (ImportError, ValueError):
                return self._get_basic_suggestions(focus)

        if self._llm_client is None:
            return self._get_basic_suggestions(focus)

        try:
            metrics = results.metrics

            prompt = f"""Suggest optimizations for this trading strategy focusing on {focus}.

Current Performance:
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Max Drawdown: {metrics.max_drawdown:.2%}
- Win Rate: {metrics.win_rate:.1%}
- Profit Factor: {metrics.profit_factor:.2f}
- Num Trades: {metrics.num_trades}
- Avg Win: ${metrics.avg_win:,.2f}
- Avg Loss: ${metrics.avg_loss:,.2f}

Suggest 5-7 specific optimizations to improve {focus}.
Format each as: "- Optimization: Reason"

Focus on actionable changes to parameters, entry/exit logic, or risk management."""

            response = self._llm_client.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Parse suggestions from response
            suggestions = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and line.strip().startswith("-")
            ]

            return suggestions[:7] if suggestions else self._get_basic_suggestions(focus)

        except Exception:
            return self._get_basic_suggestions(focus)

    def explain_metric(self, metric_name: str, metric_value: float) -> str:
        """
        Explain what a performance metric means in plain language.

        Args:
            metric_name: Name of metric (e.g., "Sharpe Ratio")
            metric_value: Value of metric

        Returns:
            Plain language explanation
        """
        if not self.nvidia_api_key:
            return f"{metric_name}: {metric_value:.2f}"

        if self._llm_client is None:
            try:
                self._llm_client = self._init_llm()
            except (ImportError, ValueError):
                return f"{metric_name}: {metric_value:.2f}"

        if self._llm_client is None:
            return f"{metric_name}: {metric_value:.2f}"

        try:
            prompt = f"""Explain the {metric_name} metric for traders in plain language.

Current Value: {metric_value:.2f}

Provide 2-3 sentences covering:
1. What it measures
2. Whether this value is good/bad/acceptable
3. What it means for the strategy

Keep it practical and non-technical."""

            response = self._llm_client.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        except Exception:
            return f"{metric_name}: {metric_value:.2f}"

    def analyze_trade_patterns(self, results: SimulationResults) -> dict[str, Any]:
        """
        Analyze trade patterns and identify issues.

        Args:
            results: Simulation results

        Returns:
            Trade pattern analysis
        """
        if not self.nvidia_api_key:
            return self._get_basic_pattern_analysis(results)

        if self._llm_client is None:
            try:
                self._llm_client = self._init_llm()
            except (ImportError, ValueError):
                return self._get_basic_pattern_analysis(results)

        if self._llm_client is None:
            return self._get_basic_pattern_analysis(results)

        try:
            metrics = results.metrics

            prompt = f"""Analyze these trading patterns and identify issues.

Trade Statistics:
- Total Trades: {metrics.num_trades}
- Win Rate: {metrics.win_rate:.1%}
- Profit Factor: {metrics.profit_factor:.2f}
- Avg Win: ${metrics.avg_win:,.2f}
- Avg Loss: ${metrics.avg_loss:,.2f}
- Largest Win: ${metrics.largest_win:,.2f}
- Largest Loss: ${metrics.largest_loss:,.2f}
- Avg Trade Duration: {metrics.avg_trade_duration:.1f} days

Identify:
1. Concerning Patterns (what looks problematic)
2. Positive Patterns (what's working well)
3. Risk Flags (potential issues to address)
4. Recommendations (2-3 specific fixes)

Be specific about what to change."""

            response = self._llm_client.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            return {
                "analysis": content,
                "llm_model": "nvidia-llama-3.1-70b",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception:
            return self._get_basic_pattern_analysis(results)

    def _get_basic_narration(self, results: SimulationResults) -> dict[str, Any]:
        """Get basic narration as fallback."""
        metrics = results.metrics

        narration = f"""Backtest completed from {results.start_time.date()} to {results.end_time.date()}.

Overall Assessment: Strategy generated {metrics.total_return:.2%} total return with Sharpe ratio of {metrics.sharpe_ratio:.2f}.

Key Metrics:
- Win Rate: {metrics.win_rate:.1%}
- Max Drawdown: {metrics.max_drawdown:.2%}
- Total Trades: {metrics.num_trades}

Enable LLM for detailed insights and optimization suggestions."""

        return {
            "narration": narration,
            "llm_model": "rule-based",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_summary": {
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
            },
        }

    def _get_basic_comparison(
        self, results_list: list[tuple[str, SimulationResults]]
    ) -> dict[str, Any]:
        """Get basic comparison as fallback."""
        best_sharpe = max(results_list, key=lambda x: x[1].metrics.sharpe_ratio)

        comparison = f"""Compared {len(results_list)} strategies.

Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1].metrics.sharpe_ratio:.2f})

Enable LLM for detailed comparative analysis."""

        return {
            "comparison": comparison,
            "llm_model": "rule-based",
            "timestamp": datetime.utcnow().isoformat(),
            "strategies_compared": len(results_list),
        }

    def _get_basic_suggestions(self, focus: str) -> list[str]:
        """Get basic suggestions as fallback."""
        suggestions = {
            "returns": [
                "Increase position sizes when win rate is high",
                "Adjust entry/exit thresholds for better timing",
                "Consider adding leverage on high-confidence trades",
            ],
            "risk": [
                "Tighten stop losses to reduce max drawdown",
                "Implement position sizing based on volatility",
                "Add portfolio-level risk limits",
            ],
            "consistency": [
                "Filter low-probability signals",
                "Improve entry criteria selectivity",
                "Add confirmation indicators",
            ],
            "general": [
                "Review entry/exit criteria",
                "Optimize position sizing",
                "Analyze losing trades for patterns",
            ],
        }

        return suggestions.get(focus, suggestions["general"])

    def _get_basic_pattern_analysis(self, results: SimulationResults) -> dict[str, Any]:
        """Get basic pattern analysis as fallback."""
        metrics = results.metrics

        analysis = f"""Trade Pattern Analysis:
- Total Trades: {metrics.num_trades}
- Win Rate: {metrics.win_rate:.1%}
- Profit Factor: {metrics.profit_factor:.2f}

Enable LLM for detailed pattern analysis and recommendations."""

        return {
            "analysis": analysis,
            "llm_model": "rule-based",
            "timestamp": datetime.utcnow().isoformat(),
        }
