"""
Confidence calibration using NVIDIA Nemotron Reward model.

This module provides confidence scoring and calibration for Cortex outputs
using the nemotron-reward model to evaluate hypothesis quality.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ordinis.ai.helix import Helix


class QualityDimension(Enum):
    """Dimensions evaluated by the reward model."""

    HELPFULNESS = "helpfulness"
    CORRECTNESS = "correctness"
    COHERENCE = "coherence"
    COMPLEXITY = "complexity"
    VERBOSITY = "verbosity"


@dataclass
class ConfidenceScore:
    """Confidence score result from reward model evaluation."""

    # Overall confidence (0.0 - 1.0)
    overall: float

    # Dimension-specific scores
    helpfulness: float = 0.0
    correctness: float = 0.0
    coherence: float = 0.0
    complexity: float = 0.0
    verbosity: float = 0.0

    # Metadata
    model_used: str = "nemotron-reward"
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0

    # Raw response for debugging
    raw_response: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "overall": self.overall,
            "dimensions": {
                "helpfulness": self.helpfulness,
                "correctness": self.correctness,
                "coherence": self.coherence,
                "complexity": self.complexity,
                "verbosity": self.verbosity,
            },
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
        }

    @property
    def is_high_confidence(self) -> bool:
        """Check if overall confidence exceeds threshold (0.7)."""
        return self.overall >= 0.7

    @property
    def is_acceptable(self) -> bool:
        """Check if confidence is acceptable (>= 0.5)."""
        return self.overall >= 0.5


@dataclass
class ConfidenceConfig:
    """Configuration for confidence calibration."""

    # Model to use for reward scoring
    reward_model: str = "nemotron-reward"

    # Whether calibration is enabled
    enabled: bool = True

    # Minimum confidence threshold for accepting outputs
    min_confidence: float = 0.5

    # Timeout for reward model calls (seconds)
    timeout: float = 30.0

    # Cache scores to avoid redundant API calls
    cache_scores: bool = True


class ConfidenceCalibrator:
    """
    Calibrates confidence scores for Cortex outputs using the reward model.

    The reward model evaluates the quality of LLM outputs across multiple
    dimensions (helpfulness, correctness, coherence, etc.) and provides
    calibrated confidence scores.
    """

    # Prompt template for reward model evaluation
    EVALUATION_PROMPT = """Evaluate the following AI-generated trading hypothesis for quality.

## Original Request Context:
Market Regime: {regime}
Volatility: {volatility}
Additional Context: {context}

## Generated Hypothesis:
Strategy Type: {strategy_type}
Time Horizon: {time_horizon}
Rationale: {rationale}
Entry Conditions: {entry_conditions}
Exit Conditions: {exit_conditions}
Expected Sharpe: {expected_sharpe}
Expected Win Rate: {expected_win_rate}

## Evaluation Criteria:
Rate the hypothesis on these dimensions (0-10 scale):
1. HELPFULNESS: Does this hypothesis provide actionable trading guidance?
2. CORRECTNESS: Is the reasoning financially sound and logically consistent?
3. COHERENCE: Are the entry/exit conditions clearly defined and consistent with the strategy?
4. COMPLEXITY: Is the strategy appropriately complex (not too simple, not over-engineered)?
5. VERBOSITY: Is the rationale appropriately detailed (not too brief, not excessive)?

Provide your evaluation as:
HELPFULNESS: [score]
CORRECTNESS: [score]
COHERENCE: [score]
COMPLEXITY: [score]
VERBOSITY: [score]
OVERALL: [weighted average]"""

    def __init__(self, helix: Helix, config: ConfidenceConfig | None = None):
        """
        Initialize the confidence calibrator.

        Args:
            helix: Helix instance for LLM calls
            config: Configuration options
        """
        self._helix = helix
        self._config = config or ConfidenceConfig()
        self._cache: dict[str, ConfidenceScore] = {}
        self._logger = logger.bind(component="ConfidenceCalibrator")

    async def evaluate_hypothesis(
        self,
        hypothesis: dict,
        market_context: dict | None = None,
    ) -> ConfidenceScore:
        """
        Evaluate a hypothesis and return calibrated confidence scores.

        Args:
            hypothesis: The hypothesis to evaluate (dict with strategy fields)
            market_context: Optional market context used for generation

        Returns:
            ConfidenceScore with calibrated scores across dimensions
        """
        if not self._config.enabled:
            self._logger.debug("Confidence calibration disabled, returning default score")
            return ConfidenceScore(overall=0.5)

        # Check cache
        cache_key = self._make_cache_key(hypothesis)
        if self._config.cache_scores and cache_key in self._cache:
            self._logger.debug("Returning cached confidence score")
            return self._cache[cache_key]

        import time

        start_time = time.perf_counter()

        try:
            # Build evaluation prompt
            prompt = self._build_evaluation_prompt(hypothesis, market_context)

            # Call reward model
            response = await asyncio.wait_for(
                self._helix.generate(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._config.reward_model,
                    temperature=0.1,  # Low temperature for consistent scoring
                ),
                timeout=self._config.timeout,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            content = getattr(response, "content", str(response))

            # Parse scores from response
            score = self._parse_scores(content, latency_ms)

            # Cache result
            if self._config.cache_scores:
                self._cache[cache_key] = score

            self._logger.info(
                f"Hypothesis confidence: {score.overall:.2f} "
                f"(latency: {latency_ms:.0f}ms)"
            )

            return score

        except asyncio.TimeoutError:
            self._logger.warning(
                f"Reward model timeout after {self._config.timeout}s, using default score"
            )
            return ConfidenceScore(overall=0.5, latency_ms=self._config.timeout * 1000)

        except Exception as e:
            self._logger.error(f"Confidence evaluation failed: {e}")
            return ConfidenceScore(overall=0.5)

    def _build_evaluation_prompt(
        self, hypothesis: dict, market_context: dict | None
    ) -> str:
        """Build the evaluation prompt from hypothesis data."""
        context = market_context or {}

        return self.EVALUATION_PROMPT.format(
            regime=context.get("regime", "unknown"),
            volatility=context.get("volatility", "medium"),
            context=str(context.get("additional_context", "")),
            strategy_type=hypothesis.get("strategy_type", "unknown"),
            time_horizon=hypothesis.get("time_horizon", "unknown"),
            rationale=hypothesis.get("rationale", ""),
            entry_conditions=str(hypothesis.get("entry_conditions", [])),
            exit_conditions=str(hypothesis.get("exit_conditions", [])),
            expected_sharpe=hypothesis.get("expected_sharpe", "N/A"),
            expected_win_rate=hypothesis.get("expected_win_rate", "N/A"),
        )

    def _parse_scores(self, response: str, latency_ms: float) -> ConfidenceScore:
        """Parse dimension scores from the reward model response."""
        import re

        scores = {
            "helpfulness": 0.5,
            "correctness": 0.5,
            "coherence": 0.5,
            "complexity": 0.5,
            "verbosity": 0.5,
            "overall": 0.5,
        }

        # Pattern to match "DIMENSION: [score]" format
        for dimension in scores:
            pattern = rf"{dimension.upper()}:\s*\[?(\d+(?:\.\d+)?)\]?"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                raw_score = float(match.group(1))
                # Normalize to 0-1 range (assuming 0-10 input)
                scores[dimension] = min(1.0, raw_score / 10.0)

        return ConfidenceScore(
            overall=scores["overall"],
            helpfulness=scores["helpfulness"],
            correctness=scores["correctness"],
            coherence=scores["coherence"],
            complexity=scores["complexity"],
            verbosity=scores["verbosity"],
            latency_ms=latency_ms,
            raw_response=response[:500],  # Truncate for storage
        )

    def _make_cache_key(self, hypothesis: dict) -> str:
        """Create a cache key from hypothesis content."""
        import hashlib
        import json

        # Use strategy type and rationale as key components
        key_data = {
            "strategy_type": hypothesis.get("strategy_type"),
            "rationale": hypothesis.get("rationale", "")[:200],  # First 200 chars
            "time_horizon": hypothesis.get("time_horizon"),
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def clear_cache(self) -> int:
        """Clear the confidence score cache. Returns number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        self._logger.debug(f"Cleared {count} cached confidence scores")
        return count

    @property
    def cache_size(self) -> int:
        """Return current cache size."""
        return len(self._cache)
