#!/usr/bin/env python3
"""
Network Parity Optimization Configuration

Defines all configuration dataclasses and constants for the Network Parity
portfolio optimization system.

Author: Ordinis Quantitative Research
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
import json

# =============================================================================
# PATHS
# =============================================================================

# Scripts are in data/backtests_new/080202a_NETWORK_PARITY/scripts/
SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPTS_DIR.parent  # 080202a_NETWORK_PARITY
PROJECT_ROOT = OUTPUT_DIR.parent.parent.parent  # ordinis/
DATA_DIR = PROJECT_ROOT / "data"
MASSIVE_DATA_DIR = DATA_DIR / "massive"
HISTORICAL_DATA_DIR = OUTPUT_DIR / "historical_data"  # Downloaded historical periods (v1: 2004, 2008, 2010, 2017, 2024)
HISTORICAL_DATA_DIR_V2 = OUTPUT_DIR / "historical_data_v2"  # Alternative periods (v2: 2006, 2012, 2015, 2019, 2022, 2023)

# Ensure output directories exist
(OUTPUT_DIR / "configs").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "iterations").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "baseline").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "summary").mkdir(parents=True, exist_ok=True)


# =============================================================================
# NVIDIA API CONFIGURATION
# =============================================================================

NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
NEMOTRON_MODEL = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
NEMOTRON_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"


# =============================================================================
# SECTOR DEFINITIONS
# =============================================================================

SMALL_CAP_SECTORS: dict[str, list[str]] = {
    "technology": ["RIOT", "MARA", "AI", "IONQ", "SOUN", "KULR"],
    "healthcare": ["BNGO", "SNDL", "TLRY", "CGC", "ACB", "XXII"],
    "energy_materials": ["PLUG", "FCEL", "BE", "CHPT", "BLNK", "EVGO"],
    "financials": ["SOFI", "HOOD", "AFRM", "UPST", "OPEN"],
    "consumer": ["GME", "AMC", "WKHS", "WISH", "CLOV"],
    "industrials": ["GOEV", "BITF", "CLSK", "CIFR", "WULF"],
}

ALL_SMALL_CAPS = [sym for sector in SMALL_CAP_SECTORS.values() for sym in sector]


# =============================================================================
# PARAMETER BOUNDS
# =============================================================================

PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "corr_lookback": (5, 20),  # Reduced for limited data (21 trading days)
    "corr_threshold": (0.2, 0.7),
    "recalc_frequency": (1, 10),
    "weight_decay": (0.2, 1.0),
    "min_weight": (0.01, 0.05),
    "max_weight": (0.15, 0.50),
    "momentum_lookback": (3, 15),  # Reduced for limited data
    "vol_target": (0.08, 0.25),
    "z_score_entry": (1.0, 3.0),
    "z_score_exit": (0.0, 1.5),
    "stop_loss_pct": (0.02, 0.10),
    "take_profit_pct": (0.05, 0.20),
    "trailing_stop_pct": (0.01, 0.05),
}


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class NetworkParityParams:
    """Network Parity strategy parameters."""

    # Correlation network (reduced for available data - 21 trading days)
    corr_lookback: int = 10  # Reduced from 60 for limited data
    corr_threshold: float = 0.30
    recalc_frequency: int = 3  # Reduced from 5
    centrality_method: str = "eigenvector"  # eigenvector, degree, betweenness

    # Weight calculation
    weight_decay: float = 0.5
    min_weight: float = 0.02
    max_weight: float = 0.30
    vol_target: float = 0.15

    # Signal generation (reduced for available data)
    momentum_lookback: int = 5  # Reduced from 20
    z_score_entry: float = 2.0
    z_score_exit: float = 0.5
    momentum_threshold: float = 0.0

    # Risk management
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    trailing_stop_pct: float = 0.03
    max_portfolio_drawdown: float = 0.15
    max_positions: int = 10

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "corr_lookback": self.corr_lookback,
            "corr_threshold": self.corr_threshold,
            "recalc_frequency": self.recalc_frequency,
            "centrality_method": self.centrality_method,
            "weight_decay": self.weight_decay,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "vol_target": self.vol_target,
            "momentum_lookback": self.momentum_lookback,
            "z_score_entry": self.z_score_entry,
            "z_score_exit": self.z_score_exit,
            "momentum_threshold": self.momentum_threshold,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "max_portfolio_drawdown": self.max_portfolio_drawdown,
            "max_positions": self.max_positions,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NetworkParityParams":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> list[str]:
        """Validate parameters against bounds. Returns list of violations."""
        violations = []
        for param, (min_val, max_val) in PARAMETER_BOUNDS.items():
            if hasattr(self, param):
                val = getattr(self, param)
                if val < min_val or val > max_val:
                    violations.append(f"{param}={val} outside [{min_val}, {max_val}]")
        return violations


@dataclass
class TransactionCostConfig:
    """Transaction cost model configuration."""

    commission_bps: float = 10.0  # Commission in basis points
    slippage_bps: float = 15.0  # Slippage/market impact
    spread_bps: float = 5.0  # Half bid-ask spread

    @property
    def total_cost_bps(self) -> float:
        """Total round-trip cost in basis points."""
        return self.commission_bps + self.slippage_bps + self.spread_bps

    @property
    def total_cost_pct(self) -> float:
        """Total round-trip cost as decimal percentage."""
        return self.total_cost_bps / 10000.0


@dataclass
class BacktestingConfig:
    """Backtesting configuration."""

    time_aggregations: list[str] = field(default_factory=lambda: ["1min", "1D"])
    sample_years: list[int] = field(default_factory=lambda: [2004, 2008, 2010, 2017, 2024])
    period_length_days: int = 21
    min_stocks_required: int = 30
    initial_capital: float = 100_000.0
    transaction_costs: TransactionCostConfig = field(default_factory=TransactionCostConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time_aggregations": self.time_aggregations,
            "sample_years": self.sample_years,
            "period_length_days": self.period_length_days,
            "min_stocks_required": self.min_stocks_required,
            "initial_capital": self.initial_capital,
            "transaction_cost_bps": self.transaction_costs.total_cost_bps,
        }


@dataclass
class NemoConfig:
    """NVIDIA Nemo integration configuration."""

    model: str = NEMOTRON_MODEL
    temperature: float = 0.7
    max_tokens: int = 2048
    max_suggestions: int = 3
    confidence_threshold: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_suggestions": self.max_suggestions,
            "confidence_threshold": self.confidence_threshold,
        }


@dataclass
class OptimizationHyperparams:
    """Optimization hyperparameters."""

    max_iterations: int = 50
    convergence_threshold: float = 0.001
    early_stopping_patience: int = 5

    # Objective weights
    return_weight: float = 0.40
    sortino_weight: float = 0.35
    win_rate_weight: float = 0.15
    drawdown_penalty_weight: float = 0.10

    # Hard constraints
    min_sharpe: float = 1.0
    max_drawdown: float = 0.20
    min_trades: int = 20

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "early_stopping_patience": self.early_stopping_patience,
            "objective_weights": {
                "return": self.return_weight,
                "sortino": self.sortino_weight,
                "win_rate": self.win_rate_weight,
                "drawdown_penalty": self.drawdown_penalty_weight,
            },
            "constraints": {
                "min_sharpe": self.min_sharpe,
                "max_drawdown": self.max_drawdown,
                "min_trades": self.min_trades,
            },
        }


@dataclass
class EquityUniverseConfig:
    """Equity universe configuration."""

    market_cap_category: str = "small_cap"
    sector_count: int = 6
    target_stocks_per_sector: int = 5
    min_total_stocks: int = 30
    sectors: dict[str, list[str]] = field(default_factory=lambda: SMALL_CAP_SECTORS.copy())

    @property
    def symbols(self) -> list[str]:
        """All symbols in universe."""
        return [sym for sector in self.sectors.values() for sym in sector]

    @property
    def total_stocks(self) -> int:
        """Total number of stocks."""
        return len(self.symbols)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market_cap_category": self.market_cap_category,
            "sector_count": len(self.sectors),
            "target_stocks_per_sector": self.target_stocks_per_sector,
            "total_stocks": self.total_stocks,
            "symbols": self.symbols,
            "sector_allocation": self.sectors,
        }


@dataclass
class OptimizationConfig:
    """Complete optimization configuration."""

    random_seed: int = 42
    universe: EquityUniverseConfig = field(default_factory=EquityUniverseConfig)
    baseline_params: NetworkParityParams = field(default_factory=NetworkParityParams)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)
    nemo: NemoConfig = field(default_factory=NemoConfig)
    hyperparams: OptimizationHyperparams = field(default_factory=OptimizationHyperparams)

    _config_id: str = field(default="", init=False)
    _timestamp: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Generate config ID and timestamp."""
        self._timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Hash based on key configuration values
        hash_input = json.dumps({
            "seed": self.random_seed,
            "symbols": self.universe.symbols,
            "params": self.baseline_params.to_dict(),
        }, sort_keys=True)
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        self._config_id = f"cfg_{self._timestamp}_{hash_suffix}"

    @property
    def config_id(self) -> str:
        """Unique configuration identifier."""
        return self._config_id

    @property
    def timestamp(self) -> str:
        """Configuration timestamp."""
        return self._timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to full JSON-serializable dictionary."""
        return {
            "schema_version": "1.0",
            "configuration_id": self.config_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "random_seed": self.random_seed,
            "equity_universe": self.universe.to_dict(),
            "baseline_strategy": {
                "name": "Network Parity",
                "version": "1.0.0",
                "initial_params": self.baseline_params.to_dict(),
            },
            "backtesting_config": self.backtesting.to_dict(),
            "nemo_parameters": self.nemo.to_dict(),
            "optimization_hyperparameters": self.hyperparams.to_dict(),
        }

    def save(self, path: Path | None = None) -> Path:
        """Save configuration to JSON file."""
        if path is None:
            path = OUTPUT_DIR / "configs" / f"{self.config_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: Path) -> "OptimizationConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        config = cls(
            random_seed=data.get("random_seed", 42),
        )
        config.universe = EquityUniverseConfig(
            sectors=data.get("equity_universe", {}).get("sector_allocation", SMALL_CAP_SECTORS),
        )
        if "baseline_strategy" in data:
            config.baseline_params = NetworkParityParams.from_dict(
                data["baseline_strategy"].get("initial_params", {})
            )
        return config


if __name__ == "__main__":
    # Test configuration
    config = OptimizationConfig()
    print(f"Config ID: {config.config_id}")
    print(f"Universe: {config.universe.total_stocks} stocks across {len(config.universe.sectors)} sectors")
    print(f"Symbols: {config.universe.symbols[:5]}... ({len(config.universe.symbols)} total)")

    # Validate baseline params
    violations = config.baseline_params.validate()
    if violations:
        print(f"Parameter violations: {violations}")
    else:
        print("All parameters within bounds")

    # Save config
    path = config.save()
    print(f"Saved to: {path}")
