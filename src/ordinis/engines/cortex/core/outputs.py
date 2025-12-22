"""
Cortex output types and data structures.

All Cortex outputs are advisory and must be validated before use.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class OutputType(Enum):
    """Type of Cortex output."""

    RESEARCH = "research"  # Research findings and synthesis
    HYPOTHESIS = "hypothesis"  # Trading hypothesis
    STRATEGY_SPEC = "strategy_spec"  # Strategy specification
    PARAM_PROPOSAL = "param_proposal"  # Parameter recommendations
    REVIEW = "review"  # Review of engine outputs
    CODE_ANALYSIS = "code_analysis"  # Code analysis and suggestions
    MARKET_INSIGHT = "market_insight"  # Market condition analysis


# --- Pydantic Models for Structured LLM Output ---


class StructuredHypothesis(BaseModel):
    """
    Pydantic model for structured LLM hypothesis output.
    
    Used with LangChain's with_structured_output for JSON mode.
    """
    
    name: str = Field(
        ..., 
        description="Short name for the strategy hypothesis",
        min_length=3,
        max_length=100,
    )
    description: str = Field(
        ..., 
        description="Detailed description of the trading strategy",
        min_length=10,
        max_length=1000,
    )
    rationale: str = Field(
        ..., 
        description="Reasoning behind the strategy based on market conditions",
        min_length=10,
        max_length=2000,
    )
    
    # Strategy classification
    instrument_class: Literal["equity", "options", "futures", "forex", "crypto"] = Field(
        default="equity",
        description="Asset class for the strategy",
    )
    time_horizon: Literal["intraday", "swing", "position", "long_term"] = Field(
        default="swing",
        description="Trading time horizon",
    )
    strategy_type: Literal["mean_reversion", "trend_following", "momentum", "arbitrage", "adaptive"] = Field(
        default="adaptive",
        description="Core strategy type",
    )
    
    # Entry/exit conditions
    entry_conditions: list[str] = Field(
        default_factory=list,
        description="List of conditions that must be met to enter a trade",
        min_length=1,
        max_length=10,
    )
    exit_conditions: list[str] = Field(
        default_factory=list,
        description="List of conditions that trigger trade exit",
        min_length=1,
        max_length=10,
    )
    
    # Risk parameters
    max_position_size_pct: float = Field(
        default=5.0,
        description="Maximum position size as percentage of portfolio",
        ge=0.1,
        le=25.0,
    )
    stop_loss_pct: float = Field(
        default=2.0,
        description="Stop loss percentage from entry price",
        ge=0.1,
        le=50.0,
    )
    take_profit_pct: float | None = Field(
        default=None,
        description="Take profit percentage from entry price",
        ge=0.1,
        le=100.0,
    )
    
    # Performance expectations
    expected_sharpe: float | None = Field(
        default=None,
        description="Expected Sharpe ratio",
        ge=-5.0,
        le=10.0,
    )
    expected_win_rate: float | None = Field(
        default=None,
        description="Expected win rate (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence level in the hypothesis (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    
    # Additional parameters
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific parameters like lookback periods, thresholds",
    )


@dataclass
class CortexOutput:
    """
    Standard output format from Cortex engine.

    All outputs are advisory - they propose actions but don't execute them.
    Must pass through validation (ProofBench) before implementation.
    """

    output_type: OutputType
    content: dict[str, Any]
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Audit trail of AI reasoning
    requires_validation: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    # AI model attribution
    model_used: str | None = None
    model_version: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self):
        """Validate output."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Convert output to dictionary."""
        return {
            "output_type": self.output_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "requires_validation": self.requires_validation,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "model_used": self.model_used,
            "model_version": self.model_version,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class StrategyHypothesis:
    """
    Trading strategy hypothesis generated by Cortex.

    Represents a testable trading idea that must be validated via ProofBench.
    """

    hypothesis_id: str
    name: str
    description: str
    rationale: str

    # Strategy parameters
    instrument_class: str  # "equity", "options", "futures", etc.
    time_horizon: str  # "intraday", "swing", "position"
    strategy_type: str  # "mean_reversion", "trend_following", etc.

    # Proposed parameters
    parameters: dict[str, Any]

    # Entry/exit conditions
    entry_conditions: list[str]
    exit_conditions: list[str]

    # Risk parameters
    max_position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float | None = None

    # Expected performance (from hypothesis)
    expected_sharpe: float | None = None
    expected_win_rate: float | None = None

    # Metadata
    confidence: float = 0.5
    requires_validation: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert hypothesis to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "name": self.name,
            "description": self.description,
            "rationale": self.rationale,
            "instrument_class": self.instrument_class,
            "time_horizon": self.time_horizon,
            "strategy_type": self.strategy_type,
            "parameters": self.parameters,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "max_position_size_pct": self.max_position_size_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "expected_sharpe": self.expected_sharpe,
            "expected_win_rate": self.expected_win_rate,
            "confidence": self.confidence,
            "requires_validation": self.requires_validation,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
