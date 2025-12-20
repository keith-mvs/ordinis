"""
AI Strategy Optimizer using NVIDIA and Mistral models.

Leverages LLMs for intelligent hyperparameter optimization with
Bayesian reasoning about strategy parameter spaces.
"""

import asyncio
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)


@dataclass
class AIOptimizerConfig:
    """Configuration for AI optimizer."""

    # Model selection (priority order)
    primary_model: str = "mistral-ai/mistral-small-2503"  # Fast, good reasoning
    fallback_model: str = "openai/gpt-4.1-mini"  # Fallback

    # Provider preference
    provider: str = "github"  # github, azure, mistral, nvidia

    # Optimization settings
    max_iterations: int = 5
    samples_per_iteration: int = 8
    exploration_ratio: float = 0.3  # Balance explore vs exploit

    # Performance targets
    min_sharpe: float = 0.5
    max_drawdown: float = 0.25
    min_win_rate: float = 0.45


@dataclass
class StrategyProfile:
    """Profile describing a trading strategy for AI optimization."""

    name: str
    description: str
    param_definitions: dict[str, dict[str, Any]]  # name -> {type, min, max, default, description}
    objective: str = "sharpe"  # Primary metric to optimize
    constraints: dict[str, float] = field(default_factory=dict)  # metric -> threshold


class AIStrategyOptimizer:
    """AI-powered strategy parameter optimizer."""

    # System prompt for strategy optimization
    SYSTEM_PROMPT = """You are an expert quantitative trading strategist and parameter optimizer.
Your role is to suggest optimal parameter combinations for trading strategies based on:
1. The strategy mechanics and what each parameter controls
2. Historical backtest results showing which parameters worked
3. Financial theory about market dynamics and risk management

When suggesting parameters:
- Balance between signal sensitivity and noise filtering
- Consider transaction costs and realistic execution
- Avoid overfitting by preferring simpler parameter values
- Respect risk constraints (max drawdown, position sizing)
- Consider market regime (trending vs mean-reverting)

Always return valid JSON arrays of parameter dictionaries."""

    def __init__(self, config: AIOptimizerConfig | None = None):
        self.config = config or AIOptimizerConfig()
        self._client = None
        self._mistral_client = None
        self._is_mistral = False
        self._model_available = False
        self._history: list[dict] = []

    async def initialize(self) -> bool:
        """Initialize AI client."""
        # Try GitHub Models first (free tier)
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    base_url="https://models.github.ai/inference",
                    api_key=github_token,
                )
                # Test connection
                test = self._client.chat.completions.create(
                    model=self.config.primary_model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5,
                )
                self._model_available = True
                logger.info(
                    f"AI Optimizer initialized with GitHub Models ({self.config.primary_model})"
                )
                return True
            except Exception as e:
                logger.warning(f"GitHub Models failed: {e}")

        # Try Mistral API (new client format)
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if mistral_key:
            try:
                from mistralai import Mistral

                self._mistral_client = Mistral(api_key=mistral_key)
                self._client = self._mistral_client
                self._is_mistral = True
                self._model_available = True
                logger.info("AI Optimizer initialized with Mistral API (new client)")
                return True
            except ImportError:
                # Try old client as fallback
                try:
                    from mistralai.client import MistralClient

                    self._client = MistralClient(api_key=mistral_key)
                    self._is_mistral = True
                    self._model_available = True
                    logger.info("AI Optimizer initialized with Mistral API (legacy client)")
                    return True
                except Exception as e:
                    logger.warning(f"Mistral API (legacy) failed: {e}")
            except Exception as e:
                logger.warning(f"Mistral API failed: {e}")

        # Try NVIDIA API
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        if nvidia_key:
            try:
                from langchain_nvidia_ai_endpoints import ChatNVIDIA

                self._client = ChatNVIDIA(
                    model="mistralai/mistral-7b-instruct-v0.2",
                    nvidia_api_key=nvidia_key,
                )
                self._model_available = True
                logger.info("AI Optimizer initialized with NVIDIA API")
                return True
            except Exception as e:
                logger.warning(f"NVIDIA API failed: {e}")

        logger.warning("No AI provider available, using heuristic optimization")
        return False

    async def suggest_parameters(
        self,
        profile: StrategyProfile,
        history: list[dict[str, Any]],
        n_suggestions: int = 5,
    ) -> list[dict[str, Any]]:
        """Get AI-suggested parameter combinations."""
        if not self._model_available:
            return self._heuristic_suggestions(profile, history, n_suggestions)

        # Build prompt with history context
        prompt = self._build_optimization_prompt(profile, history, n_suggestions)

        try:
            suggestions = await self._query_model(prompt, profile)
            if suggestions:
                return suggestions[:n_suggestions]
        except Exception as e:
            logger.warning(f"AI suggestion failed: {e}")

        return self._heuristic_suggestions(profile, history, n_suggestions)

    def _build_optimization_prompt(
        self,
        profile: StrategyProfile,
        history: list[dict[str, Any]],
        n_suggestions: int,
    ) -> str:
        """Build prompt for parameter optimization."""
        # Format parameter definitions
        param_desc = []
        for name, defn in profile.param_definitions.items():
            param_desc.append(
                f"- {name}: {defn.get('description', '')} "
                f"(range: {defn.get('min', '?')} to {defn.get('max', '?')}, "
                f"default: {defn.get('default', '?')})"
            )

        # Format recent history (best and worst)
        history_text = ""
        if history:
            sorted_hist = sorted(history, key=lambda x: x.get(profile.objective, 0), reverse=True)
            best_3 = sorted_hist[:3]
            worst_3 = sorted_hist[-3:] if len(sorted_hist) > 3 else []

            history_text = "\n\nPrevious Results (sorted by objective):\n"
            history_text += "BEST:\n"
            for h in best_3:
                params = h.get("params", {})
                history_text += (
                    f"  {params} -> {profile.objective}={h.get(profile.objective, 0):.4f}, "
                )
                history_text += (
                    f"win_rate={h.get('win_rate', 0):.1f}%, dd={h.get('max_drawdown', 0):.2f}%\n"
                )

            if worst_3:
                history_text += "WORST:\n"
                for h in worst_3:
                    params = h.get("params", {})
                    history_text += (
                        f"  {params} -> {profile.objective}={h.get(profile.objective, 0):.4f}\n"
                    )

        prompt = f"""Strategy: {profile.name}
Description: {profile.description}

Parameters to optimize:
{chr(10).join(param_desc)}

Objective: Maximize {profile.objective}
Constraints: {profile.constraints}
{history_text}

Suggest {n_suggestions} different parameter combinations to test.
Use the history to focus on promising regions while exploring alternatives.

Return ONLY a JSON array of parameter dictionaries:
[{{"param1": value1, "param2": value2, ...}}, ...]"""

        return prompt

    async def _query_model(
        self,
        prompt: str,
        profile: StrategyProfile,
    ) -> list[dict[str, Any]]:
        """Query the AI model for suggestions."""
        if isinstance(self._client, type(None)):
            return []

        # Mistral new client
        if self._is_mistral and hasattr(self._client, "chat"):
            response = self._client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content
        # OpenAI-compatible client (GitHub Models)
        elif hasattr(self._client, "chat") and hasattr(self._client.chat, "completions"):
            response = self._client.chat.completions.create(
                model=self.config.primary_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            content = response.choices[0].message.content
        else:
            # LangChain-style client
            response = self._client.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

        # Parse JSON
        return self._parse_suggestions(content, profile)

    def _parse_suggestions(
        self,
        content: str,
        profile: StrategyProfile,
    ) -> list[dict[str, Any]]:
        """Parse AI response into parameter suggestions."""
        import re

        try:
            # Clean up common formatting issues
            # Remove markdown code blocks
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```\s*", "", content)

            # Find JSON array in response
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]

                # Fix common JSON issues
                # Remove trailing commas
                json_str = re.sub(r",\s*}", "}", json_str)
                json_str = re.sub(r",\s*]", "]", json_str)
                # Fix single quotes to double quotes
                json_str = json_str.replace("'", '"')
                # Remove comments
                json_str = re.sub(r"//[^\n]*", "", json_str)

                suggestions = json.loads(json_str)

                # Validate and clip to ranges
                valid = []
                for s in suggestions:
                    if not isinstance(s, dict):
                        continue
                    validated = {}
                    for name, defn in profile.param_definitions.items():
                        if name in s:
                            val = s[name]
                            # Handle string numbers
                            if isinstance(val, str):
                                try:
                                    val = float(val)
                                except ValueError:
                                    val = defn.get("default", 0)
                            # Clip to range
                            if "min" in defn and val < defn["min"]:
                                val = defn["min"]
                            if "max" in defn and val > defn["max"]:
                                val = defn["max"]
                            validated[name] = val
                        else:
                            validated[name] = defn.get("default", 0)
                    if validated:  # Only add non-empty
                        valid.append(validated)

                return valid
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI response: {e}")
            # Try to extract individual objects
            try:
                matches = re.findall(r"\{[^{}]+\}", content)
                valid = []
                for m in matches[:5]:  # Limit to 5
                    try:
                        obj = json.loads(m.replace("'", '"'))
                        if isinstance(obj, dict):
                            validated = {}
                            for name, defn in profile.param_definitions.items():
                                val = obj.get(name, defn.get("default", 0))
                                if isinstance(val, str):
                                    try:
                                        val = float(val)
                                    except ValueError:
                                        val = defn.get("default", 0)
                                validated[name] = val
                            valid.append(validated)
                    except:
                        pass
                if valid:
                    return valid
            except:
                pass

        return []

    def _heuristic_suggestions(
        self,
        profile: StrategyProfile,
        history: list[dict[str, Any]],
        n_suggestions: int,
    ) -> list[dict[str, Any]]:
        """Generate suggestions using heuristics when AI unavailable."""
        suggestions = []

        # If we have history, use best performers as anchors
        if history:
            sorted_hist = sorted(history, key=lambda x: x.get(profile.objective, 0), reverse=True)
            best = sorted_hist[0].get("params", {})

            # Perturb best parameters
            for i in range(n_suggestions):
                suggestion = {}
                for name, defn in profile.param_definitions.items():
                    base = best.get(name, defn.get("default", 0))
                    range_size = (defn.get("max", 1) - defn.get("min", 0)) * 0.1

                    if i < n_suggestions // 2:
                        # Exploit: small perturbations
                        noise = np.random.uniform(-range_size, range_size)
                    else:
                        # Explore: larger jumps
                        noise = np.random.uniform(-range_size * 3, range_size * 3)

                    val = base + noise
                    val = max(defn.get("min", val), min(defn.get("max", val), val))
                    suggestion[name] = round(val, 4)

                suggestions.append(suggestion)
        else:
            # Random sampling from parameter space
            for _ in range(n_suggestions):
                suggestion = {}
                for name, defn in profile.param_definitions.items():
                    low = defn.get("min", 0)
                    high = defn.get("max", 1)
                    val = np.random.uniform(low, high)
                    suggestion[name] = round(val, 4)
                suggestions.append(suggestion)

        return suggestions

    async def run_optimization(
        self,
        profile: StrategyProfile,
        backtest_fn: Callable[[dict[str, Any]], dict[str, Any]],
        initial_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run full AI-guided optimization loop."""
        start_time = time.perf_counter()

        all_results = []
        best_params = initial_params or {
            name: defn.get("default", 0) for name, defn in profile.param_definitions.items()
        }
        best_score = float("-inf")

        # Initial backtest
        initial_result = backtest_fn(best_params)
        initial_result["params"] = best_params
        all_results.append(initial_result)
        best_score = initial_result.get(profile.objective, 0)

        logger.info(f"Initial {profile.objective}: {best_score:.4f}")

        for iteration in range(self.config.max_iterations):
            logger.info(f"Optimization iteration {iteration + 1}/{self.config.max_iterations}")

            # Get AI suggestions
            suggestions = await self.suggest_parameters(
                profile,
                all_results,
                self.config.samples_per_iteration,
            )

            # Run backtests
            for params in suggestions:
                try:
                    result = backtest_fn(params)
                    result["params"] = params
                    all_results.append(result)

                    score = result.get(profile.objective, 0)
                    if score > best_score:
                        # Check constraints
                        passes_constraints = True
                        for metric, threshold in profile.constraints.items():
                            if metric == "max_drawdown":
                                if result.get(metric, 1) > threshold:
                                    passes_constraints = False
                            elif result.get(metric, 0) < threshold:
                                passes_constraints = False

                        if passes_constraints:
                            best_score = score
                            best_params = params
                            logger.info(f"  New best: {profile.objective}={score:.4f}")

                except Exception as e:
                    logger.warning(f"Backtest failed: {e}")

            # Early stopping if target reached
            if best_score >= self.config.min_sharpe:
                best_result = next(
                    (r for r in all_results if r.get("params") == best_params),
                    {},
                )
                if best_result.get("max_drawdown", 1) <= self.config.max_drawdown:
                    logger.info("Early stopping: targets reached")
                    break

        optimization_time = time.perf_counter() - start_time

        return {
            "best_params": best_params,
            "best_score": best_score,
            "iterations": iteration + 1,
            "total_backtests": len(all_results),
            "optimization_time": optimization_time,
            "all_results": all_results,
            "ai_available": self._model_available,
        }


# =============================================================================
# Pre-defined Strategy Profiles
# =============================================================================

GARCH_BREAKOUT_PROFILE = StrategyProfile(
    name="GARCH Breakout",
    description="Trades volatility breakouts using GARCH-style vol ratio. "
    "Enters when short-term vol exceeds long-term vol by threshold.",
    param_definitions={
        "breakout_threshold": {
            "type": "float",
            "min": 1.1,
            "max": 3.0,
            "default": 1.5,
            "description": "Vol ratio threshold for entry signal",
        },
        "garch_lookback": {
            "type": "int",
            "min": 20,
            "max": 120,
            "default": 60,
            "description": "EWMA span for short-term volatility",
        },
        "atr_stop_mult": {
            "type": "float",
            "min": 1.0,
            "max": 4.0,
            "default": 2.0,
            "description": "ATR multiplier for stop loss",
        },
        "atr_tp_mult": {
            "type": "float",
            "min": 1.5,
            "max": 6.0,
            "default": 3.0,
            "description": "ATR multiplier for take profit",
        },
    },
    objective="sharpe",
    constraints={"max_drawdown": 0.25, "win_rate": 0.40},
)

KALMAN_TREND_PROFILE = StrategyProfile(
    name="Kalman Trend Filter",
    description="Trend following using Kalman filter for signal smoothing. "
    "Trades breakouts from filtered trend with ATR-based sizing.",
    param_definitions={
        "process_variance": {
            "type": "float",
            "min": 0.0001,
            "max": 0.1,
            "default": 0.01,
            "description": "Kalman filter process noise",
        },
        "measurement_variance": {
            "type": "float",
            "min": 0.01,
            "max": 1.0,
            "default": 0.1,
            "description": "Kalman filter measurement noise",
        },
        "trend_threshold": {
            "type": "float",
            "min": 0.001,
            "max": 0.05,
            "default": 0.01,
            "description": "Trend strength threshold for entry",
        },
        "lookback": {
            "type": "int",
            "min": 10,
            "max": 100,
            "default": 30,
            "description": "Trend calculation lookback",
        },
    },
    objective="sharpe",
    constraints={"max_drawdown": 0.20},
)

HMM_REGIME_PROFILE = StrategyProfile(
    name="HMM Regime Detector",
    description="Hidden Markov Model for market regime detection. "
    "Trades regime transitions and regime-appropriate strategies.",
    param_definitions={
        "n_regimes": {
            "type": "int",
            "min": 2,
            "max": 5,
            "default": 3,
            "description": "Number of market regimes",
        },
        "lookback": {
            "type": "int",
            "min": 60,
            "max": 500,
            "default": 252,
            "description": "Training lookback window",
        },
        "transition_threshold": {
            "type": "float",
            "min": 0.5,
            "max": 0.95,
            "default": 0.7,
            "description": "Probability threshold for regime change",
        },
        "bull_regime_id": {
            "type": "int",
            "min": 0,
            "max": 4,
            "default": 0,
            "description": "Which regime ID represents bullish",
        },
    },
    objective="sharpe",
    constraints={"max_drawdown": 0.30},
)

OU_PAIRS_PROFILE = StrategyProfile(
    name="Ornstein-Uhlenbeck Pairs",
    description="Mean-reversion pairs trading using OU process. "
    "Trades spread when it deviates from equilibrium.",
    param_definitions={
        "zscore_entry": {
            "type": "float",
            "min": 1.0,
            "max": 3.0,
            "default": 2.0,
            "description": "Z-score threshold for entry",
        },
        "zscore_exit": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
            "description": "Z-score threshold for exit",
        },
        "lookback": {
            "type": "int",
            "min": 20,
            "max": 120,
            "default": 60,
            "description": "Spread calculation lookback",
        },
        "half_life_max": {
            "type": "int",
            "min": 5,
            "max": 60,
            "default": 30,
            "description": "Maximum mean-reversion half-life",
        },
    },
    objective="sharpe",
    constraints={"max_drawdown": 0.15},
)

# Extreme Value Theory (EVT) Tail Risk Strategy
EVT_TAIL_PROFILE = StrategyProfile(
    name="EVT Tail Risk",
    description="Extreme Value Theory strategy for tail risk trading. "
    "Detects unusual price moves using GPD distribution and trades reversals.",
    param_definitions={
        "threshold_percentile": {
            "type": "float",
            "min": 90.0,
            "max": 99.0,
            "default": 95.0,
            "description": "Percentile for extreme threshold",
        },
        "lookback": {
            "type": "int",
            "min": 50,
            "max": 500,
            "default": 252,
            "description": "Window for tail distribution estimation",
        },
        "holding_period": {
            "type": "int",
            "min": 1,
            "max": 20,
            "default": 5,
            "description": "Days to hold after extreme event",
        },
        "min_exceedances": {
            "type": "int",
            "min": 5,
            "max": 50,
            "default": 20,
            "description": "Minimum tail events for valid estimation",
        },
    },
    objective="sharpe",
    constraints={"max_drawdown": 0.25},
)

# Multi-Timeframe Momentum Strategy
MTF_MOMENTUM_PROFILE = StrategyProfile(
    name="Multi-Timeframe Momentum",
    description="Combines momentum signals across multiple timeframes. "
    "Aligns short, medium, and long-term trends for higher conviction.",
    param_definitions={
        "short_period": {
            "type": "int",
            "min": 5,
            "max": 20,
            "default": 10,
            "description": "Short-term momentum lookback",
        },
        "medium_period": {
            "type": "int",
            "min": 20,
            "max": 60,
            "default": 30,
            "description": "Medium-term momentum lookback",
        },
        "long_period": {
            "type": "int",
            "min": 60,
            "max": 252,
            "default": 120,
            "description": "Long-term momentum lookback",
        },
        "alignment_threshold": {
            "type": "float",
            "min": 0.5,
            "max": 1.0,
            "default": 0.7,
            "description": "Required timeframe agreement ratio",
        },
    },
    objective="sharpe",
    constraints={"max_drawdown": 0.20},
)

# Mutual Information Strategy
MI_LEAD_LAG_PROFILE = StrategyProfile(
    name="Mutual Information Lead-Lag",
    description="Uses mutual information to detect leading indicators. "
    "Identifies predictive relationships between assets.",
    param_definitions={
        "mi_lookback": {
            "type": "int",
            "min": 20,
            "max": 120,
            "default": 60,
            "description": "Window for MI calculation",
        },
        "mi_threshold": {
            "type": "float",
            "min": 0.05,
            "max": 0.5,
            "default": 0.15,
            "description": "Minimum MI for signal",
        },
        "lag_range": {
            "type": "int",
            "min": 1,
            "max": 10,
            "default": 5,
            "description": "Maximum lag days to check",
        },
        "signal_smoothing": {
            "type": "int",
            "min": 1,
            "max": 10,
            "default": 3,
            "description": "EMA smoothing for signals",
        },
    },
    objective="sharpe",
    constraints={"max_drawdown": 0.20},
)

# Network/Correlation Regime Strategy
NETWORK_REGIME_PROFILE = StrategyProfile(
    name="Network Correlation Regime",
    description="Tracks correlation network structure to detect regime shifts. "
    "High connectivity suggests risk-off, fragmentation suggests opportunities.",
    param_definitions={
        "corr_lookback": {
            "type": "int",
            "min": 20,
            "max": 120,
            "default": 60,
            "description": "Correlation calculation window",
        },
        "edge_threshold": {
            "type": "float",
            "min": 0.3,
            "max": 0.8,
            "default": 0.5,
            "description": "Correlation threshold for network edges",
        },
        "density_high": {
            "type": "float",
            "min": 0.5,
            "max": 0.9,
            "default": 0.7,
            "description": "High density = risk-off regime",
        },
        "density_low": {
            "type": "float",
            "min": 0.1,
            "max": 0.4,
            "default": 0.3,
            "description": "Low density = opportunity regime",
        },
    },
    objective="sharpe",
    constraints={"max_drawdown": 0.25},
)

STRATEGY_PROFILES = {
    "garch_breakout": GARCH_BREAKOUT_PROFILE,
    "kalman_trend": KALMAN_TREND_PROFILE,
    "hmm_regime": HMM_REGIME_PROFILE,
    "ou_pairs": OU_PAIRS_PROFILE,
    "evt_tail": EVT_TAIL_PROFILE,
    "mtf_momentum": MTF_MOMENTUM_PROFILE,
    "mi_lead_lag": MI_LEAD_LAG_PROFILE,
    "network_regime": NETWORK_REGIME_PROFILE,
    "ou_pairs": OU_PAIRS_PROFILE,
}


# =============================================================================
# Demo
# =============================================================================


async def demo():
    """Demonstrate AI optimizer capabilities."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    logger.info("=" * 60)
    logger.info("AI STRATEGY OPTIMIZER DEMO")
    logger.info("=" * 60)

    optimizer = AIStrategyOptimizer()
    available = await optimizer.initialize()

    logger.info(f"AI available: {available}")

    # Mock backtest function
    def mock_backtest(params: dict[str, Any]) -> dict[str, Any]:
        # Simulate backtest with some noise
        base_sharpe = 0.5

        # Threshold sweet spot around 1.5-2.0
        threshold = params.get("breakout_threshold", 1.5)
        threshold_bonus = -abs(threshold - 1.75) * 0.5

        # Lookback sweet spot around 60
        lookback = params.get("garch_lookback", 60)
        lookback_bonus = -abs(lookback - 60) / 100

        sharpe = base_sharpe + threshold_bonus + lookback_bonus + np.random.randn() * 0.1

        return {
            "sharpe": sharpe,
            "total_return": sharpe * 15 + np.random.randn() * 5,
            "win_rate": 45 + sharpe * 10,
            "max_drawdown": 0.15 + (1 - sharpe) * 0.1,
            "trades": 50,
        }

    # Get suggestions for GARCH strategy
    profile = GARCH_BREAKOUT_PROFILE
    suggestions = await optimizer.suggest_parameters(profile, [], 5)

    logger.info(f"\nSuggested parameters for {profile.name}:")
    for i, s in enumerate(suggestions, 1):
        logger.info(f"  {i}. {s}")

    # Run mini optimization
    logger.info("\nRunning optimization loop...")
    result = await optimizer.run_optimization(
        profile,
        mock_backtest,
        {"breakout_threshold": 1.5, "garch_lookback": 60, "atr_stop_mult": 2.0, "atr_tp_mult": 3.0},
    )

    logger.info(f"\nOptimization complete!")
    logger.info(f"Best params: {result['best_params']}")
    logger.info(f"Best {profile.objective}: {result['best_score']:.4f}")
    logger.info(f"Total backtests: {result['total_backtests']}")
    logger.info(f"Time: {result['optimization_time']:.1f}s")

    return result


if __name__ == "__main__":
    asyncio.run(demo())
