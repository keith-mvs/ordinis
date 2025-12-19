"""Strategy profiles for AI optimization."""

from ordinis.engines.sprint.core.optimizer import StrategyProfile

# GARCH Breakout Strategy
GARCH_BREAKOUT_PROFILE = StrategyProfile(
    name="GARCH Breakout",
    description="Volatility breakout strategy using GARCH to detect unusual price moves. "
    "Enters when volatility ratio exceeds threshold, exits on ATR-based stops.",
    param_definitions={
        "breakout_threshold": {
            "type": "float",
            "min": 1.0,
            "max": 3.0,
            "default": 1.5,
            "description": "Volatility ratio threshold for entry signals",
        },
        "garch_lookback": {
            "type": "int",
            "min": 20,
            "max": 120,
            "default": 60,
            "description": "GARCH estimation window in days",
        },
        "atr_stop_mult": {
            "type": "float",
            "min": 1.0,
            "max": 4.0,
            "default": 2.0,
            "description": "ATR multiplier for stop-loss",
        },
        "atr_tp_mult": {
            "type": "float",
            "min": 1.5,
            "max": 6.0,
            "default": 3.0,
            "description": "ATR multiplier for take-profit",
        },
    },
    objective="sharpe",
    constraints={"max_drawdown": 0.25},
)

# Kalman Trend Filter Strategy
KALMAN_TREND_PROFILE = StrategyProfile(
    name="Kalman Trend Filter",
    description="Trend-following strategy using Kalman filter to estimate hidden price trends. "
    "Smooth filtering reduces noise, trend_threshold controls signal sensitivity.",
    param_definitions={
        "process_variance": {
            "type": "float",
            "min": 0.001,
            "max": 0.1,
            "default": 0.01,
            "description": "Process noise for Kalman state evolution",
        },
        "measurement_variance": {
            "type": "float",
            "min": 0.01,
            "max": 1.0,
            "default": 0.1,
            "description": "Measurement noise for observations",
        },
        "trend_threshold": {
            "type": "float",
            "min": 0.005,
            "max": 0.05,
            "default": 0.01,
            "description": "Minimum trend magnitude for entry",
        },
        "lookback": {
            "type": "int",
            "min": 10,
            "max": 60,
            "default": 30,
            "description": "Trend calculation lookback",
        },
    },
    objective="sharpe",
    constraints={"max_drawdown": 0.20},
)

# HMM Regime Strategy
HMM_REGIME_PROFILE = StrategyProfile(
    name="HMM Regime",
    description="Hidden Markov Model regime detection strategy. "
    "Identifies market regimes (trending/ranging) and trades regime transitions.",
    param_definitions={
        "n_regimes": {
            "type": "int",
            "min": 2,
            "max": 5,
            "default": 3,
            "description": "Number of hidden states/regimes",
        },
        "lookback": {
            "type": "int",
            "min": 60,
            "max": 504,
            "default": 252,
            "description": "Training window for regime detection",
        },
        "transition_threshold": {
            "type": "float",
            "min": 0.5,
            "max": 0.9,
            "default": 0.7,
            "description": "Probability threshold for regime change",
        },
    },
    objective="sharpe",
    constraints={"max_drawdown": 0.30},
)

# OU Pairs Trading Strategy
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

# EVT Tail Risk Strategy
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

# Mutual Information Lead-Lag Strategy
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

# Network Correlation Regime Strategy
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

# All strategy profiles
STRATEGY_PROFILES = {
    "garch_breakout": GARCH_BREAKOUT_PROFILE,
    "kalman_trend": KALMAN_TREND_PROFILE,
    "hmm_regime": HMM_REGIME_PROFILE,
    "ou_pairs": OU_PAIRS_PROFILE,
    "evt_tail": EVT_TAIL_PROFILE,
    "mtf_momentum": MTF_MOMENTUM_PROFILE,
    "mi_lead_lag": MI_LEAD_LAG_PROFILE,
    "network_regime": NETWORK_REGIME_PROFILE,
}
