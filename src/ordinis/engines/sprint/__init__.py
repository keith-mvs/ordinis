"""
Sprint Engine - GPU-accelerated strategy development and testing.

This module provides:
- GPU-accelerated numerical computations for backtesting
- AI-powered parameter optimization using LLMs
- Walk-forward validation framework
- Strategy visualization and reporting
"""

from ordinis.engines.sprint.core.accelerator import GPUBacktestEngine, GPUConfig
from ordinis.engines.sprint.core.optimizer import (
    AIOptimizerConfig,
    AIStrategyOptimizer,
    StrategyProfile,
)
from ordinis.engines.sprint.core.runner import (
    AcceleratedSprintRunner,
    SprintConfig,
    StrategyResult,
    run_sprint,
)
from ordinis.engines.sprint.strategies import (
    EVT_TAIL_PROFILE,
    GARCH_BREAKOUT_PROFILE,
    HMM_REGIME_PROFILE,
    KALMAN_TREND_PROFILE,
    MI_LEAD_LAG_PROFILE,
    MTF_MOMENTUM_PROFILE,
    NETWORK_REGIME_PROFILE,
    OU_PAIRS_PROFILE,
    STRATEGY_PROFILES,
)
from ordinis.engines.sprint.viz import (
    StrategyVisualizer,
    generate_all_visualizations,
)

__all__ = [
    # Core components
    "GPUBacktestEngine",
    "GPUConfig",
    "AIStrategyOptimizer",
    "AIOptimizerConfig",
    "StrategyProfile",
    "AcceleratedSprintRunner",
    "SprintConfig",
    "StrategyResult",
    "run_sprint",
    # Strategy profiles
    "GARCH_BREAKOUT_PROFILE",
    "KALMAN_TREND_PROFILE",
    "HMM_REGIME_PROFILE",
    "OU_PAIRS_PROFILE",
    "EVT_TAIL_PROFILE",
    "MTF_MOMENTUM_PROFILE",
    "MI_LEAD_LAG_PROFILE",
    "NETWORK_REGIME_PROFILE",
    "STRATEGY_PROFILES",
    # Visualization
    "StrategyVisualizer",
    "generate_all_visualizations",
]
