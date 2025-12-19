"""
Type definitions for CortexEngine API contracts.

Provides TypedDict schemas for type-safe communication between engines
and clear API contracts for research, strategy generation, and analysis tasks.
"""

from typing import Literal, NotRequired, TypedDict

# Market Context Types


class MarketContext(TypedDict):
    """
    Market environment and conditions for strategy/analysis requests.

    Used to provide market state context to Cortex for informed decision-making.
    """

    ticker: str
    """Primary ticker symbol being analyzed."""

    sector: NotRequired[str]
    """Market sector (e.g., 'Technology', 'Healthcare')."""

    market_regime: NotRequired[Literal["bull", "bear", "sideways", "volatile"]]
    """Current market regime classification."""

    iv_percentile: NotRequired[float]
    """Implied volatility percentile (0-100)."""

    trend_direction: NotRequired[Literal["up", "down", "neutral"]]
    """Current price trend direction."""

    support_level: NotRequired[float]
    """Key support price level."""

    resistance_level: NotRequired[float]
    """Key resistance price level."""

    current_price: NotRequired[float]
    """Current market price."""

    avg_volume: NotRequired[int]
    """Average daily trading volume."""

    earnings_date: NotRequired[str]
    """Next earnings date (ISO format)."""

    additional_context: NotRequired[dict[str, str | float | int | bool]]
    """Additional market-specific context."""


# Strategy Constraint Types


class StrategyConstraints(TypedDict):
    """
    Trading constraints and risk parameters for strategy generation.

    Defines the boundaries within which Cortex should operate when
    generating strategy hypotheses.
    """

    max_position_size: NotRequired[float]
    """Maximum position size as decimal (e.g., 0.10 = 10% of portfolio)."""

    max_risk_per_trade: NotRequired[float]
    """Maximum risk per trade as decimal (e.g., 0.02 = 2%)."""

    min_profit_target: NotRequired[float]
    """Minimum profit target as decimal (e.g., 0.05 = 5%)."""

    max_holding_period: NotRequired[int]
    """Maximum holding period in days."""

    allowed_instruments: NotRequired[list[Literal["stocks", "options", "futures"]]]
    """Allowed instrument types for trading."""

    leverage_limit: NotRequired[float]
    """Maximum leverage allowed (e.g., 2.0 = 2x)."""

    stop_loss_required: NotRequired[bool]
    """Whether stop-loss is mandatory."""

    min_liquidity: NotRequired[int]
    """Minimum average daily volume required."""

    prohibited_sectors: NotRequired[list[str]]
    """Sectors to exclude from trading."""

    additional_constraints: NotRequired[dict[str, str | float | int | bool]]
    """Additional strategy-specific constraints."""


# Research Context Types


class ResearchContext(TypedDict):
    """
    Context for research and analysis requests.

    Provides scope and focus for Cortex research operations, including
    code review, documentation generation, and market analysis.
    """

    research_type: Literal[
        "code_review", "market_analysis", "strategy_design", "documentation", "general"
    ]
    """Type of research being requested."""

    focus_areas: NotRequired[list[str]]
    """Specific areas to focus on (e.g., ['error_handling', 'type_safety'])."""

    depth_level: NotRequired[Literal["quick", "standard", "comprehensive"]]
    """Desired depth of analysis."""

    output_format: NotRequired[Literal["markdown", "json", "text", "code"]]
    """Preferred output format."""

    include_examples: NotRequired[bool]
    """Whether to include code examples in output."""

    target_audience: NotRequired[Literal["developer", "trader", "executive", "technical"]]
    """Intended audience for the research output."""

    time_horizon: NotRequired[Literal["intraday", "short_term", "medium_term", "long_term"]]
    """Time horizon for analysis (relevant for market research)."""

    context_files: NotRequired[list[str]]
    """File paths to include as context."""

    additional_context: NotRequired[dict[str, str | float | int | bool]]
    """Additional research-specific context."""


# Response Types


class CortexResponse(TypedDict):
    """
    Standard response format from CortexEngine operations.

    Provides consistent structure for all Cortex outputs with
    tracking and metadata.
    """

    output_id: str
    """Unique identifier for this output."""

    output_type: Literal["strategy", "research", "code", "analysis"]
    """Type of output generated."""

    content: str
    """Primary content of the response."""

    confidence: NotRequired[float]
    """Confidence score (0-1) if applicable."""

    metadata: NotRequired[dict[str, str | float | int | bool | list[str]]]
    """Additional metadata about the response."""

    sources: NotRequired[list[str]]
    """Source references used in generating response."""

    timestamp: NotRequired[str]
    """ISO timestamp of response generation."""


# Configuration Types


class NVIDIAConfig(TypedDict):
    """
    Configuration for NVIDIA client initialization.

    Controls behavior of NVIDIA API client adapters.
    """

    chat_model: NotRequired[str]
    """Chat model identifier (default: nemotron-super)."""

    embedding_model: NotRequired[str]
    """Embedding model identifier (default: nv-embedqa-e5-v5)."""

    temperature: NotRequired[float]
    """Generation temperature (0.0-1.0)."""

    max_tokens: NotRequired[int]
    """Maximum tokens per response."""

    timeout: NotRequired[int]
    """Request timeout in seconds."""

    max_retries: NotRequired[int]
    """Maximum retry attempts for failed requests."""

    rate_limit: NotRequired[int]
    """Requests per minute limit."""
