#!/usr/bin/env python3
"""
NVIDIA Nemo Integration for Network Parity Optimization

Integrates NVIDIA Nemo models to provide intelligent parameter
suggestions during the optimization loop.

Author: Ordinis Quantitative Research
Version: 1.0.0
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from config import (
    NVIDIA_API_KEY_ENV,
    NEMOTRON_MODEL,
    NEMOTRON_ENDPOINT,
    PARAMETER_BOUNDS,
    NemoConfig,
)
from backtesting import BacktestResult

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NemoSuggestion:
    """A single parameter suggestion from Nemo."""

    param_name: str
    current_value: float
    suggested_value: float
    confidence: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "param_name": self.param_name,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }

    def is_valid(self) -> bool:
        """Check if suggestion is within parameter bounds."""
        if self.param_name not in PARAMETER_BOUNDS:
            return True  # Unknown param, allow

        min_val, max_val = PARAMETER_BOUNDS[self.param_name]
        return min_val <= self.suggested_value <= max_val

    def clip_to_bounds(self) -> "NemoSuggestion":
        """Return new suggestion with value clipped to bounds."""
        if self.param_name not in PARAMETER_BOUNDS:
            return self

        min_val, max_val = PARAMETER_BOUNDS[self.param_name]
        clipped_value = max(min_val, min(max_val, self.suggested_value))

        return NemoSuggestion(
            param_name=self.param_name,
            current_value=self.current_value,
            suggested_value=clipped_value,
            confidence=self.confidence,
            reasoning=self.reasoning + " (clipped to bounds)",
        )


@dataclass
class NemoResponse:
    """Response from Nemo API."""

    suggestions: list[NemoSuggestion] = field(default_factory=list)
    analysis: str = ""
    raw_response: str = ""
    success: bool = False
    error: str | None = None


# =============================================================================
# NEMO OPTIMIZER
# =============================================================================

class NemoOptimizer:
    """
    NVIDIA Nemo integration for optimization guidance.

    Uses Nemo models to analyze backtest results and suggest
    parameter refinements.
    """

    def __init__(
        self,
        config: NemoConfig | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize Nemo optimizer.

        Args:
            config: Nemo configuration
            api_key: NVIDIA API key (or uses env var)
        """
        self.config = config or NemoConfig()
        self.api_key = api_key or os.environ.get(NVIDIA_API_KEY_ENV, "")

        if not self.api_key:
            logger.warning(
                f"NVIDIA API key not set. Set {NVIDIA_API_KEY_ENV} environment variable."
            )

    @property
    def is_available(self) -> bool:
        """Check if Nemo API is available."""
        return bool(self.api_key)

    def _build_prompt(
        self,
        current_params: dict[str, Any],
        result: BacktestResult,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Build the prompt for Nemo.

        Args:
            current_params: Current strategy parameters
            result: Latest backtest result
            history: Optional history of previous iterations

        Returns:
            Formatted prompt string
        """
        # Format parameter bounds for context
        bounds_str = "\n".join([
            f"  {p}: [{b[0]}, {b[1]}]"
            for p, b in PARAMETER_BOUNDS.items()
        ])

        # Format history summary
        history_str = "None available"
        if history and len(history) > 0:
            recent = history[-5:]  # Last 5 iterations
            history_lines = []
            for h in recent:
                history_lines.append(
                    f"  Iter {h.get('iteration', '?')}: "
                    f"Return={h.get('total_return', 0):.2%}, "
                    f"Sharpe={h.get('sharpe_ratio', 0):.2f}, "
                    f"Sortino={h.get('sortino_ratio', 0):.2f}"
                )
            history_str = "\n".join(history_lines)

        prompt = f"""You are a quantitative portfolio optimization expert specializing in network-based portfolio construction and mean reversion strategies.

Analyze the following Network Parity strategy performance and suggest parameter refinements to improve risk-adjusted returns.

## Current Parameters
```json
{json.dumps(current_params, indent=2)}
```

## Parameter Bounds
{bounds_str}

## Latest Performance Metrics
- Total Return: {result.total_return:.2%}
- Sharpe Ratio: {result.sharpe_ratio:.2f}
- Sortino Ratio: {result.sortino_ratio:.2f}
- Maximum Drawdown: {result.max_drawdown:.2%}
- Win Rate: {result.win_rate:.1%}
- Profit Factor: {result.profit_factor:.2f}
- Number of Trades: {result.num_trades}
- Network Rebalances: {result.rebalance_count}

## Optimization History (Recent)
{history_str}

## Strategy Context
- Network Parity uses correlation-based asset weighting
- Peripheral assets (low centrality) get higher weights for diversification
- Mean reversion signals based on z-score deviation from rolling mean
- Target: Maximize Sortino ratio while maintaining reasonable trade frequency

## Instructions
Suggest up to {self.config.max_suggestions} parameter changes to improve risk-adjusted returns.

For each suggestion, provide:
1. Parameter name (must match exactly from current params)
2. Current value
3. Suggested value (must be within bounds)
4. Confidence score (0.0 to 1.0)
5. Brief reasoning (1-2 sentences)

Respond ONLY with a JSON array of suggestions:
```json
[
  {{
    "param_name": "parameter_name",
    "current_value": 0.0,
    "suggested_value": 0.0,
    "confidence": 0.0,
    "reasoning": "explanation"
  }}
]
```"""
        return prompt

    def _parse_response(self, response_text: str) -> list[NemoSuggestion]:
        """
        Parse Nemo response into suggestions.

        Args:
            response_text: Raw response from Nemo

        Returns:
            List of NemoSuggestion objects
        """
        suggestions = []

        try:
            # Try to extract JSON from response
            # Look for JSON array pattern
            json_match = re.search(r'\[[\s\S]*?\]', response_text)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                for item in data:
                    if isinstance(item, dict):
                        suggestion = NemoSuggestion(
                            param_name=str(item.get("param_name", "")),
                            current_value=float(item.get("current_value", 0)),
                            suggested_value=float(item.get("suggested_value", 0)),
                            confidence=float(item.get("confidence", 0.5)),
                            reasoning=str(item.get("reasoning", "")),
                        )

                        # Validate and clip to bounds
                        if suggestion.param_name:
                            if not suggestion.is_valid():
                                suggestion = suggestion.clip_to_bounds()
                            suggestions.append(suggestion)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Nemo response JSON: {e}")
        except Exception as e:
            logger.warning(f"Error parsing Nemo response: {e}")

        return suggestions

    async def get_suggestions_async(
        self,
        current_params: dict[str, Any],
        result: BacktestResult,
        history: list[dict[str, Any]] | None = None,
    ) -> NemoResponse:
        """
        Get parameter suggestions from Nemo asynchronously.

        Args:
            current_params: Current strategy parameters
            result: Latest backtest result
            history: Optional history of previous iterations

        Returns:
            NemoResponse with suggestions
        """
        if not self.is_available:
            return NemoResponse(
                success=False,
                error="NVIDIA API key not configured",
            )

        prompt = self._build_prompt(current_params, result, history)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    NEMOTRON_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.config.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    },
                )

                if response.status_code != 200:
                    return NemoResponse(
                        success=False,
                        error=f"API error: {response.status_code} - {response.text}",
                    )

                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                suggestions = self._parse_response(content)

                # Filter by confidence threshold
                suggestions = [
                    s for s in suggestions
                    if s.confidence >= self.config.confidence_threshold
                ]

                return NemoResponse(
                    suggestions=suggestions,
                    raw_response=content,
                    success=True,
                )

        except httpx.TimeoutException:
            return NemoResponse(
                success=False,
                error="Request timed out",
            )
        except Exception as e:
            return NemoResponse(
                success=False,
                error=str(e),
            )

    def get_suggestions(
        self,
        current_params: dict[str, Any],
        result: BacktestResult,
        history: list[dict[str, Any]] | None = None,
    ) -> NemoResponse:
        """
        Get parameter suggestions from Nemo (sync wrapper).

        Args:
            current_params: Current strategy parameters
            result: Latest backtest result
            history: Optional history of previous iterations

        Returns:
            NemoResponse with suggestions
        """
        import asyncio
        return asyncio.run(
            self.get_suggestions_async(current_params, result, history)
        )

    def apply_suggestions(
        self,
        current_params: dict[str, Any],
        suggestions: list[NemoSuggestion],
    ) -> dict[str, Any]:
        """
        Apply suggestions to current parameters.

        Args:
            current_params: Current parameters
            suggestions: Suggestions to apply

        Returns:
            Updated parameters
        """
        new_params = current_params.copy()

        for suggestion in suggestions:
            if suggestion.param_name in new_params:
                logger.info(
                    f"Applying suggestion: {suggestion.param_name} "
                    f"{suggestion.current_value} -> {suggestion.suggested_value} "
                    f"(confidence: {suggestion.confidence:.2f})"
                )
                new_params[suggestion.param_name] = suggestion.suggested_value

        return new_params


# Parameters that should be integers
INTEGER_PARAMS = {"corr_lookback", "recalc_frequency", "momentum_lookback", "max_positions"}


def generate_random_perturbation(
    current_params: dict[str, Any],
    scale: float = 0.1,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Generate random parameter perturbation as fallback.

    Args:
        current_params: Current parameters
        scale: Scale of perturbation (0.1 = ±10%)
        seed: Random seed

    Returns:
        Perturbed parameters
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    new_params = current_params.copy()

    for param, (min_val, max_val) in PARAMETER_BOUNDS.items():
        if param in new_params:
            current = new_params[param]
            range_size = max_val - min_val

            # Random perturbation
            delta = rng.uniform(-scale, scale) * range_size
            new_value = current + delta

            # Clip to bounds
            new_value = max(min_val, min(max_val, new_value))

            # Cast to int if needed
            if param in INTEGER_PARAMS:
                new_value = int(round(new_value))

            new_params[param] = new_value

    return new_params


if __name__ == "__main__":
    import asyncio
    from backtesting import BacktestResult

    # Test with mock data
    config = NemoConfig()
    optimizer = NemoOptimizer(config)

    print(f"Nemo Available: {optimizer.is_available}")

    # Mock backtest result
    result = BacktestResult(
        total_return=0.15,
        sharpe_ratio=1.2,
        sortino_ratio=1.8,
        max_drawdown=0.12,
        win_rate=0.55,
        profit_factor=1.5,
        num_trades=45,
    )

    # Mock params
    params = {
        "corr_lookback": 60,
        "corr_threshold": 0.3,
        "z_score_entry": 2.0,
        "z_score_exit": 0.5,
        "stop_loss_pct": 0.05,
    }

    if optimizer.is_available:
        print("\nGetting suggestions from Nemo...")
        response = optimizer.get_suggestions(params, result)

        if response.success:
            print(f"Received {len(response.suggestions)} suggestions:")
            for s in response.suggestions:
                print(f"  {s.param_name}: {s.current_value} -> {s.suggested_value}")
                print(f"    Confidence: {s.confidence:.2f}")
                print(f"    Reasoning: {s.reasoning}")
        else:
            print(f"Error: {response.error}")
    else:
        print("\nUsing random perturbation (API not available):")
        new_params = generate_random_perturbation(params, scale=0.1, seed=42)
        for p, v in new_params.items():
            if p in params and params[p] != v:
                print(f"  {p}: {params[p]} -> {v:.4f}")
