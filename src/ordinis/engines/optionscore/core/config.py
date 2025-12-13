"""
OptionsCore Engine Configuration

Type-safe configuration classes for the OptionsCore pricing and Greeks engine.
Follows existing patterns from ModelConfig, PluginConfig across the platform.

Author: Ordinis Project
License: MIT
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptionsEngineConfig:
    """
    Configuration for OptionsCore engine.

    Attributes:
        engine_id: Unique identifier for this engine instance
        enabled: Whether engine is active
        cache_ttl_seconds: Time-to-live for cached results (default 5 minutes)
        default_risk_free_rate: Risk-free rate for pricing (annual, decimal)
        default_dividend_yield: Dividend yield for pricing (annual, decimal)
        calculation_mode: Pricing model mode ('european' for MVP)
        enable_iv_calculation: Enable implied volatility calculations (future)
        metadata: Additional configuration parameters

    Example:
        >>> config = OptionsEngineConfig(
        ...     engine_id="optionscore_main",
        ...     default_risk_free_rate=0.05,
        ...     cache_ttl_seconds=300
        ... )
    """

    engine_id: str
    enabled: bool = True
    cache_ttl_seconds: int = 300
    default_risk_free_rate: float = 0.05
    default_dividend_yield: float = 0.0
    calculation_mode: str = "european"
    enable_iv_calculation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.engine_id:
            raise ValueError("engine_id cannot be empty")

        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative")

        if self.default_risk_free_rate < 0:
            raise ValueError("default_risk_free_rate must be non-negative")

        if self.default_dividend_yield < 0:
            raise ValueError("default_dividend_yield must be non-negative")

        if self.calculation_mode not in ["european", "american"]:
            raise ValueError(
                f"calculation_mode must be 'european' or 'american', got '{self.calculation_mode}'"
            )

        if self.calculation_mode == "american":
            raise NotImplementedError("American options pricing not yet implemented")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "engine_id": self.engine_id,
            "enabled": self.enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "default_risk_free_rate": self.default_risk_free_rate,
            "default_dividend_yield": self.default_dividend_yield,
            "calculation_mode": self.calculation_mode,
            "enable_iv_calculation": self.enable_iv_calculation,
            "metadata": self.metadata,
        }
