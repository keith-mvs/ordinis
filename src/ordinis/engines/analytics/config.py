from dataclasses import dataclass, field


@dataclass
class AnalyticsConfig:
    """Configuration for the Analytics Engine."""

    enabled: bool = True
    storage_path: str = "data/analytics"
    metrics_retention_days: int = 30
    real_time_calculation: bool = True
    performance_metrics: list[str] = field(default_factory=lambda: ["pnl", "win_rate", "drawdown"])
