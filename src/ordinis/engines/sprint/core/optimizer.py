"""
AI Strategy Optimizer.

Uses LLMs (Mistral, NVIDIA, OpenAI) for intelligent hyperparameter optimization
with Bayesian reasoning about strategy parameter spaces.
"""

from dataclasses import dataclass, field
import json
import logging
import os
import re
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AIOptimizerConfig:
    """Configuration for AI optimizer."""

    primary_model: str = "mistral-ai/mistral-small-2503"
    fallback_model: str = "openai/gpt-4.1-mini"
    provider: str = "github"  # github, azure, mistral, nvidia
    max_iterations: int = 5
    samples_per_iteration: int = 8
    exploration_ratio: float = 0.3
    min_sharpe: float = 0.5
    max_drawdown: float = 0.25
    min_win_rate: float = 0.45


@dataclass
class StrategyProfile:
    """Profile describing a trading strategy for AI optimization."""

    name: str
    description: str
    param_definitions: dict[str, dict[str, Any]]
    objective: str = "sharpe"
    constraints: dict[str, float] = field(default_factory=dict)


class AIStrategyOptimizer:
    """AI-powered strategy parameter optimizer."""

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

        # Try Mistral API (new client)
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if mistral_key:
            try:
                from mistralai import Mistral

                self._mistral_client = Mistral(api_key=mistral_key)
                self._client = self._mistral_client
                self._is_mistral = True
                self._model_available = True
                logger.info("AI Optimizer initialized with Mistral API")
                return True
            except ImportError:
                try:
                    from mistralai.client import MistralClient

                    self._client = MistralClient(api_key=mistral_key)
                    self._is_mistral = True
                    self._model_available = True
                    logger.info("AI Optimizer initialized with Mistral API (legacy)")
                    return True
                except Exception as e:
                    logger.warning(f"Mistral API failed: {e}")
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
        param_desc = []
        for name, defn in profile.param_definitions.items():
            param_desc.append(
                f"- {name}: {defn.get('description', '')} "
                f"(range: {defn.get('min', '?')} to {defn.get('max', '?')}, "
                f"default: {defn.get('default', '?')})"
            )

        history_text = ""
        if history:
            sorted_hist = sorted(history, key=lambda x: x.get(profile.objective, 0), reverse=True)
            best_3 = sorted_hist[:3]
            worst_3 = sorted_hist[-3:] if len(sorted_hist) > 3 else []

            history_text = "\n\nPrevious Results:\nBEST:\n"
            for h in best_3:
                params = h.get("params", {})
                history_text += (
                    f"  {params} -> {profile.objective}={h.get(profile.objective, 0):.4f}\n"
                )

            if worst_3:
                history_text += "WORST:\n"
                for h in worst_3:
                    params = h.get("params", {})
                    history_text += (
                        f"  {params} -> {profile.objective}={h.get(profile.objective, 0):.4f}\n"
                    )

        return f"""Strategy: {profile.name}
Description: {profile.description}

Parameters to optimize:
{chr(10).join(param_desc)}

Objective: Maximize {profile.objective}
Constraints: {profile.constraints}
{history_text}

Suggest {n_suggestions} different parameter combinations.
Return ONLY a JSON array: [{{"param1": value1, ...}}, ...]"""

    async def _query_model(
        self,
        prompt: str,
        profile: StrategyProfile,
    ) -> list[dict[str, Any]]:
        """Query the AI model for suggestions."""
        if self._client is None:
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

        return self._parse_suggestions(content, profile)

    def _parse_suggestions(
        self,
        content: str,
        profile: StrategyProfile,
    ) -> list[dict[str, Any]]:
        """Parse AI response into parameter suggestions."""
        try:
            # Clean up formatting
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```\s*", "", content)

            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                json_str = re.sub(r",\s*}", "}", json_str)
                json_str = re.sub(r",\s*]", "]", json_str)
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r"//[^\n]*", "", json_str)

                suggestions = json.loads(json_str)

                valid = []
                for s in suggestions:
                    if not isinstance(s, dict):
                        continue
                    validated = {}
                    for name, defn in profile.param_definitions.items():
                        if name in s:
                            val = s[name]
                            if isinstance(val, str):
                                try:
                                    val = float(val)
                                except ValueError:
                                    val = defn.get("default", 0)
                            if "min" in defn and val < defn["min"]:
                                val = defn["min"]
                            if "max" in defn and val > defn["max"]:
                                val = defn["max"]
                            validated[name] = val
                        else:
                            validated[name] = defn.get("default", 0)
                    if validated:
                        valid.append(validated)

                return valid
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI response: {e}")
            # Try extracting individual objects
            try:
                matches = re.findall(r"\{[^{}]+\}", content)
                valid = []
                for m in matches[:5]:
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

        if history:
            sorted_hist = sorted(history, key=lambda x: x.get(profile.objective, 0), reverse=True)
            best = sorted_hist[0].get("params", {})

            for i in range(n_suggestions):
                suggestion = {}
                for name, defn in profile.param_definitions.items():
                    base = best.get(name, defn.get("default", 0))
                    range_size = (defn.get("max", 1) - defn.get("min", 0)) * 0.1

                    if i < n_suggestions // 2:
                        noise = np.random.uniform(-range_size, range_size)
                    else:
                        noise = np.random.uniform(-range_size * 3, range_size * 3)

                    val = base + noise
                    val = max(defn.get("min", val), min(defn.get("max", val), val))
                    suggestion[name] = round(val, 4)

                suggestions.append(suggestion)
        else:
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
        max_iterations: int | None = None,
    ) -> dict[str, Any]:
        """Run full optimization loop."""
        max_iter = max_iterations or self.config.max_iterations
        history = []
        best_result = None
        best_params = initial_params or {}

        # Run initial backtest
        if initial_params:
            result = backtest_fn(initial_params)
            result["params"] = initial_params
            history.append(result)
            best_result = result
            logger.info(f"Initial {profile.objective}: {result.get(profile.objective, 0):.4f}")

        for iteration in range(max_iter):
            logger.info(f"Optimization iteration {iteration + 1}/{max_iter}")

            # Get suggestions
            suggestions = await self.suggest_parameters(profile, history, 5)

            # Evaluate suggestions
            for params in suggestions:
                result = backtest_fn(params)
                result["params"] = params
                history.append(result)

                if best_result is None or result.get(profile.objective, 0) > best_result.get(
                    profile.objective, 0
                ):
                    best_result = result
                    best_params = params
                    logger.info(
                        f"  New best: {profile.objective}={result.get(profile.objective, 0):.4f}"
                    )

            # Check early stopping
            if best_result and best_result.get(profile.objective, 0) >= self.config.min_sharpe:
                if best_result.get("max_drawdown", 1) <= self.config.max_drawdown:
                    if best_result.get("win_rate", 0) >= self.config.min_win_rate * 100:
                        logger.info("Early stopping: targets reached")
                        break

        return {
            "best_params": best_params,
            "best_result": best_result,
            "history": history,
            "iterations": iteration + 1,
            "ai_available": self._model_available,
        }
