"""
Golden prompt fixtures for Cortex evaluation.

These fixtures define expected prompts and responses for deterministic testing.
Each fixture includes:
- input: The input to the Cortex method
- expected_prompt_patterns: Patterns that must appear in generated prompts
- expected_output_schema: Expected structure of the output
- expected_constraints: Invariants that must hold
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GoldenPromptFixture:
    """A golden fixture for prompt evaluation."""

    name: str
    description: str
    method: str  # generate_hypothesis, analyze_code, synthesize_research
    input_data: dict[str, Any]
    expected_prompt_patterns: list[str]
    expected_output_keys: list[str]
    expected_constraints: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


# =============================================================================
# Hypothesis Generation Golden Fixtures
# =============================================================================

HYPOTHESIS_GOLDEN_FIXTURES: list[GoldenPromptFixture] = [
    GoldenPromptFixture(
        name="trending_low_vol",
        description="Trending market with low volatility should generate trend following",
        method="generate_hypothesis",
        input_data={
            "market_context": {
                "regime": "trending",
                "volatility": "low",
            }
        },
        expected_prompt_patterns=[
            "trending",
            "volatility",
            "strategy",
        ],
        expected_output_keys=[
            "hypothesis_id",
            "name",
            "strategy_type",
            "parameters",
            "entry_conditions",
            "exit_conditions",
        ],
        expected_constraints={
            "strategy_type": "trend_following",
            "confidence_range": (0.5, 1.0),
            "max_position_size_pct_max": 0.10,
        },
        tags=["core", "trending"],
    ),
    GoldenPromptFixture(
        name="mean_reverting_high_vol",
        description="Mean reverting market with high volatility should generate mean reversion",
        method="generate_hypothesis",
        input_data={
            "market_context": {
                "regime": "mean_reverting",
                "volatility": "high",
            }
        },
        expected_prompt_patterns=[
            "mean_revert",
            "volatility",
        ],
        expected_output_keys=[
            "hypothesis_id",
            "strategy_type",
            "stop_loss_pct",
        ],
        expected_constraints={
            "strategy_type": "mean_reversion",
            "confidence_range": (0.5, 1.0),
        },
        tags=["core", "mean_reversion"],
    ),
    GoldenPromptFixture(
        name="unknown_regime",
        description="Unknown regime should generate adaptive/balanced strategy",
        method="generate_hypothesis",
        input_data={
            "market_context": {
                "regime": "unknown",
                "volatility": "moderate",
            }
        },
        expected_prompt_patterns=[],
        expected_output_keys=["hypothesis_id", "strategy_type"],
        expected_constraints={
            "strategy_type_in": ["adaptive", "balanced"],
            "confidence_range": (0.4, 1.0),
        },
        tags=["edge_case"],
    ),
    GoldenPromptFixture(
        name="with_constraints",
        description="Hypothesis with user-defined constraints",
        method="generate_hypothesis",
        input_data={
            "market_context": {
                "regime": "trending",
                "volatility": "low",
            },
            "constraints": {
                "max_drawdown": 0.05,
                "min_sharpe": 1.5,
                "instrument_class": "equity",
            },
        },
        expected_prompt_patterns=[
            "constraint",
            "drawdown",
            "sharpe",
        ],
        expected_output_keys=["hypothesis_id", "strategy_type", "instrument_class"],
        expected_constraints={
            "instrument_class": "equity",
        },
        tags=["constraints"],
    ),
]


# =============================================================================
# Code Analysis Golden Fixtures
# =============================================================================

ANALYSIS_GOLDEN_FIXTURES: list[GoldenPromptFixture] = [
    GoldenPromptFixture(
        name="review_simple_function",
        description="Simple function review should identify structure",
        method="analyze_code",
        input_data={
            "code": """
def calculate_returns(prices: list[float]) -> list[float]:
    '''Calculate simple returns from price series.'''
    if len(prices) < 2:
        return []
    return [(prices[i] - prices[i-1]) / prices[i-1] 
            for i in range(1, len(prices))]
""",
            "analysis_type": "review",
        },
        expected_prompt_patterns=[
            "review",
            "calculate_returns",
            "prices",
        ],
        expected_output_keys=["content", "confidence", "output_type"],
        expected_constraints={
            "confidence_range": (0.5, 1.0),
            "output_type": "code_analysis",
        },
        tags=["core", "review"],
    ),
    GoldenPromptFixture(
        name="security_analysis",
        description="Security analysis should check for vulnerabilities",
        method="analyze_code",
        input_data={
            "code": """
import os
def run_command(user_input: str):
    os.system(user_input)  # Potential command injection
""",
            "analysis_type": "security",
        },
        expected_prompt_patterns=[
            "security",
            "os.system",
        ],
        expected_output_keys=["content", "confidence"],
        expected_constraints={
            "confidence_range": (0.5, 1.0),
        },
        tags=["security"],
    ),
    GoldenPromptFixture(
        name="performance_analysis",
        description="Performance analysis should identify optimization opportunities",
        method="analyze_code",
        input_data={
            "code": """
def slow_search(items: list, target) -> bool:
    for item in items:
        if item == target:
            return True
    return False
""",
            "analysis_type": "performance",
        },
        expected_prompt_patterns=[
            "performance",
        ],
        expected_output_keys=["content", "confidence"],
        expected_constraints={},
        tags=["performance"],
    ),
]


# =============================================================================
# Research Synthesis Golden Fixtures
# =============================================================================

RESEARCH_GOLDEN_FIXTURES: list[GoldenPromptFixture] = [
    GoldenPromptFixture(
        name="momentum_strategy_research",
        description="Research synthesis on momentum strategies",
        method="synthesize_research",
        input_data={
            "query": "What are the key characteristics of momentum trading strategies?",
            "sources": [
                "Momentum strategies capitalize on price trends.",
                "Key factors include lookback period and rebalancing frequency.",
            ],
            "context": {"domain": "quantitative_finance"},
        },
        expected_prompt_patterns=[
            "momentum",
            "strategy",
            "trend",
        ],
        expected_output_keys=["content", "confidence", "output_type"],
        expected_constraints={
            "output_type": "research",
            "confidence_range": (0.5, 1.0),
        },
        tags=["core", "research"],
    ),
    GoldenPromptFixture(
        name="risk_management_research",
        description="Research synthesis on risk management",
        method="synthesize_research",
        input_data={
            "query": "Best practices for portfolio risk management",
            "sources": [
                "VaR and CVaR are common risk measures.",
                "Position sizing based on volatility.",
            ],
            "context": None,
        },
        expected_prompt_patterns=[
            "risk",
            "management",
        ],
        expected_output_keys=["content", "confidence"],
        expected_constraints={},
        tags=["risk"],
    ),
]


# =============================================================================
# Aggregate all fixtures
# =============================================================================

ALL_GOLDEN_FIXTURES = (
    HYPOTHESIS_GOLDEN_FIXTURES
    + ANALYSIS_GOLDEN_FIXTURES
    + RESEARCH_GOLDEN_FIXTURES
)


def get_fixtures_by_method(method: str) -> list[GoldenPromptFixture]:
    """Get all fixtures for a specific method."""
    return [f for f in ALL_GOLDEN_FIXTURES if f.method == method]


def get_fixtures_by_tag(tag: str) -> list[GoldenPromptFixture]:
    """Get all fixtures with a specific tag."""
    return [f for f in ALL_GOLDEN_FIXTURES if tag in f.tags]
