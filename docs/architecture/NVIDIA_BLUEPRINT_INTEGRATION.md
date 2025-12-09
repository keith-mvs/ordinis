# NVIDIA AI Blueprint Integration Plan

Integration of NVIDIA Quantitative Portfolio Optimization and AI Model Distillation blueprints into the Ordinis trading engine.

**Version**: 1.0.0
**Last Updated**: 2025-12-08
**Status**: Implementation Plan

---

## Executive Summary

This document defines the integration strategy for two NVIDIA AI Blueprints:

1. **Quantitative Portfolio Optimization** - GPU-accelerated Mean-CVaR optimization with 100-160x speedups
2. **AI Model Distillation for Financial Data** - Teacher-student LLM distillation achieving 98% cost reduction

The integration creates a unified workflow from data ingestion through portfolio optimization with distilled model inference.

---

## 1. Repository Analysis

### 1.1 Quantitative Portfolio Optimization

**Source**: https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization

| Component | Technology | Purpose |
|-----------|------------|---------|
| Scenario Generation | cuDF, cuML | GPU-accelerated return distribution sampling |
| Optimization | cuOpt | Mean-CVaR portfolio optimization solver |
| Backtesting | CUDA-X HPC | Strategy backtesting and refinement |

**Hardware Requirements** (Full Blueprint):
- NVIDIA H100 SXM (compute capability >= 9.0)
- CUDA 13.0, drivers 580.65.06+
- 32+ CPU cores, 64+ GB RAM

**Key Artifacts**:
- `notebooks/cvar_basic.ipynb` - CVaR optimization fundamentals
- `notebooks/efficient_frontier.ipynb` - Frontier construction
- `notebooks/rebalancing_strategies.ipynb` - Dynamic rebalancing

### 1.2 AI Model Distillation for Financial Data

**Source**: https://github.com/NVIDIA-AI-Blueprints/ai-model-distillation-for-financial-data

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Flywheel | FastAPI, Celery | Automated training loop orchestration |
| Fine-tuning | NeMo Customizer (LoRA) | Parameter-efficient model adaptation |
| Evaluation | NeMo Evaluator | F1-score assessment pipeline |
| Inference | NeMo NIMs | Optimized model serving |

**Hardware Requirements** (Full Blueprint):
- 2-6x H100 or A100 GPUs
- 200GB disk, Docker, K8s cluster-admin

**Performance Benchmarks** (from repository):

| Dataset | Model | Base F1 | Fine-tuned F1 |
|---------|-------|---------|---------------|
| 25K samples | Llama 3.2 1B | 0.32 | 0.95 |
| 25K samples | Llama 3.2 3B | 0.72 | 0.95 |

---

## 1.3 Consumer GPU Implementation Path

The full blueprints require datacenter GPUs (H100/A100). For development systems with consumer hardware (e.g., RTX 2080 Ti, RTX 3090, RTX 4090), a hybrid approach is recommended.

### Hardware Comparison

| Requirement | Full Blueprint | RTX 2080 Ti | RTX 4090 |
|-------------|----------------|-------------|----------|
| Compute Capability | 9.0+ (H100) | 7.5 | 8.9 |
| VRAM | 40-80GB | 11GB | 24GB |
| CUDA | 13.0+ | 12.x | 12.x |
| Multi-GPU | 2-6x | 1x | 1x |

### Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CONSUMER GPU HYBRID ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LOCAL (RTX 2080 Ti / 11GB VRAM)            NVIDIA API (Cloud)          │
│  ═══════════════════════════════            ════════════════════        │
│                                                                          │
│  ✓ Text embeddings (300M model)             ✓ Code embeddings (7B)      │
│  ✓ Data preprocessing (pandas/numpy)        ✓ LLM inference (70B)       │
│  ✓ Mean-Variance optimization (scipy)       ✓ Reranking (500M)          │
│  ✓ Vector search (ChromaDB)                 ✓ Fine-tuned model serving  │
│  ✓ Technical indicators                                                  │
│  ✓ Risk rule evaluation                     ON-DEMAND CLOUD (Optional)  │
│                                             ════════════════════════     │
│                                             ✓ Model fine-tuning          │
│                                             ✓ Large-scale optimization   │
│                                             ✓ Batch scenario generation  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Mapping

| Blueprint Component | Full Implementation | Consumer GPU Fallback |
|--------------------|--------------------|----------------------|
| **Portfolio Optimization** | cuOpt (100-160x speedup) | SciPy SLSQP/CVXPY (~1-5s for 50 assets) |
| **Scenario Generation** | cuML multivariate (GPU) | NumPy multivariate_normal (CPU) |
| **Text Embeddings** | NeMo 300M (local GPU) | NeMo 300M (local GPU) - fits in 11GB |
| **Code Embeddings** | nv-embedcode-7b (local) | NVIDIA API fallback (~$0.0005/query) |
| **LLM Inference** | NeMo NIM (local) | NVIDIA API (~$0.0002-0.005/query) |
| **Model Distillation** | Local LoRA training | Cloud GPU rental (on-demand) |
| **Reranking** | NeMo 500M (local) | NVIDIA API (~$0.0003/query) |

### Estimated Monthly Costs (Consumer GPU + API)

| Usage Level | API Queries/Day | Estimated Monthly Cost |
|-------------|-----------------|----------------------|
| Light (dev/testing) | 50 | $3-8 |
| Moderate (daily trading) | 200 | $12-30 |
| Heavy (active strategies) | 500 | $30-75 |
| Batch retraining (quarterly) | N/A | $20-50 (cloud GPU) |

All scenarios fit well within a $500/month budget.

### Code Changes for Consumer GPU

The existing RAG config already supports this pattern:

```python
# src/rag/config.py - Already implemented
from rag.config import RAGConfig, set_config

# Consumer GPU configuration
config = RAGConfig(
    use_local_embeddings=True,     # 300M text fits in 11GB
    use_local_code_embeddings=False,  # 7B too large, use API
    api_fallback=True,             # Automatic fallback on VRAM errors
    vram_threshold_gb=9.0,         # Reserve 2GB for system
)
set_config(config)
```

For PortOpt, add similar fallback:

```python
# src/engines/portopt/core/engine.py (to be implemented)
class PortOptEngine:
    def __init__(self, use_gpu: bool = True, gpu_fallback: bool = True):
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.gpu_fallback = gpu_fallback

    def _check_gpu_available(self) -> bool:
        """Check if cuOpt-compatible GPU is available."""
        try:
            import cupy as cp
            device = cp.cuda.Device()
            # cuOpt requires compute capability 7.0+
            # Full performance requires 9.0+ (H100)
            cc = device.compute_capability
            if cc[0] < 7:
                return False
            if cc[0] < 9:
                logger.warning(
                    f"GPU compute capability {cc[0]}.{cc[1]} < 9.0. "
                    "Using CPU fallback for optimal stability."
                )
                return False
            return True
        except ImportError:
            return False

    def optimize(self, ...) -> OptimizationResult:
        if self.use_gpu:
            return self._optimize_gpu(...)
        return self._optimize_cpu(...)  # scipy.optimize fallback
```

### Recommended Development Workflow

1. **Local Development**: Use CPU fallbacks for optimization, local 300M embeddings
2. **Testing**: Same as local, with occasional API calls for code search
3. **Production**: Mix of local (fast ops) and API (heavy inference)
4. **Quarterly Retraining**: Rent cloud GPU for 2-4 hours (~$10-20)

---

## 2. Target Engine Architecture

### 2.1 Current Ordinis Engine Structure

```
src/engines/
├── cortex/          # Strategy generation, RAG
├── signalcore/      # Signal generation models
├── riskguard/       # Risk management rules
├── proofbench/      # Backtesting engine
├── flowroute/       # Order execution
└── governance/      # Compliance, ethics
```

### 2.2 Proposed New Components

```
src/engines/
├── portopt/         # NEW: Portfolio Optimization Engine
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py           # Main optimization orchestrator
│   │   ├── scenarios.py        # Scenario generation (cuDF/cuML)
│   │   ├── cvar.py             # Mean-CVaR solver wrapper (cuOpt)
│   │   └── constraints.py      # Portfolio constraint definitions
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── rebalancing.py      # Rebalancing strategies
│   │   └── frontier.py         # Efficient frontier computation
│   └── adapters/
│       ├── __init__.py
│       └── cuda.py             # CUDA device management
│
├── distillery/      # NEW: Model Distillation Engine
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py           # Flywheel orchestrator
│   │   ├── flywheel.py         # Data flywheel implementation
│   │   └── config.py           # Distillation configuration
│   ├── training/
│   │   ├── __init__.py
│   │   ├── lora.py             # LoRA fine-tuning wrapper
│   │   ├── dataset.py          # Dataset curation
│   │   └── logging.py          # Production log capture
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # F1, accuracy metrics
│   │   └── benchmark.py        # Model comparison
│   └── serving/
│       ├── __init__.py
│       └── nim.py              # NeMo NIM deployment
```

---

## 3. Integration Points

### 3.1 Data Flow Integration

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED TRADING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  INGESTION  │───>│  SIGNALCORE │───>│   PORTOPT   │───>│  RISKGUARD  │      │
│  │  (Market    │    │  (Distilled │    │  (cuOpt     │    │  (Rule      │      │
│  │   Data)     │    │   Models)   │    │   CVaR)     │    │   Engine)   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│        │                  │                  │                  │              │
│        │                  │                  │                  │              │
│        ▼                  ▼                  ▼                  ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ Alpha       │    │ Llama 1B    │    │ Optimal     │    │ Validated   │      │
│  │ Vantage     │    │ (Distilled) │    │ Weights     │    │ Orders      │      │
│  │ Finnhub     │    │             │    │             │    │             │      │
│  │ Polygon     │    │             │    │             │    │             │      │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                                                 │
│                              │                                                  │
│                              ▼                                                  │
│                    ┌─────────────────────┐                                      │
│                    │     FLOWROUTE       │                                      │
│                    │  (Order Execution)  │                                      │
│                    └─────────────────────┘                                      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          DISTILLERY (Background)                         │   │
│  │  Production Logs → Dataset Curation → LoRA Training → Model Evaluation  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Engine Integration Matrix

| Source Engine | Target Engine | Integration Type | Data Flow |
|---------------|---------------|------------------|-----------|
| Market Data Plugins | PortOpt | Direct | OHLCV → Scenarios |
| SignalCore | PortOpt | Signal Batch | Signals → Expected Returns |
| PortOpt | RiskGuard | Optimization Output | Weights → ProposedTrades |
| RiskGuard | FlowRoute | Validated Orders | Orders → Execution |
| Cortex | Distillery | Training Data | LLM Logs → Flywheel |
| Distillery | SignalCore | Model Serving | Distilled Model → Inference |

### 3.3 Dependency Constraints

**Hard Dependencies** (must satisfy):

```python
# pyproject.toml additions
[project.optional-dependencies]
portopt = [
    "cudf-cu12>=24.10",      # GPU DataFrames
    "cuml-cu12>=24.10",      # GPU ML
    "cuopt>=24.10",          # Optimization solver
]
distillery = [
    "nemo-toolkit>=2.0",     # NeMo framework
    "peft>=0.7.0",           # LoRA implementation
    "mlflow>=2.10",          # Experiment tracking
    "celery>=5.3",           # Task orchestration
    "elasticsearch>=8.12",   # Log storage
]
nvidia-full = [
    "ordinis[portopt,distillery]",
]
```

**Soft Dependencies** (graceful degradation):

| Component | Without GPU | Fallback Behavior |
|-----------|-------------|-------------------|
| cuDF | pandas | 10-100x slower data processing |
| cuOpt | scipy.optimize | No CVaR, basic MVO only |
| NeMo NIM | NVIDIA API | Remote inference (paid) |
| Distillery | None | Use teacher model directly |

---

## 4. Component Specifications

### 4.1 PortOpt Engine

#### 4.1.1 Core Interface

```python
# src/engines/portopt/core/engine.py

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    min_weight: float = 0.0           # Minimum asset weight
    max_weight: float = 0.20          # Maximum asset weight (20%)
    max_sector_weight: float = 0.30   # Maximum sector exposure
    target_volatility: Optional[float] = None
    max_leverage: float = 1.0         # No leverage by default
    long_only: bool = True

@dataclass
class CVaRConfig:
    """Mean-CVaR optimization configuration."""
    confidence_level: float = 0.95    # CVaR at 95% confidence
    num_scenarios: int = 10000        # Monte Carlo scenarios
    risk_aversion: float = 0.5        # Lambda: 0=max return, 1=min risk
    use_gpu: bool = True

@dataclass
class OptimizationResult:
    """Portfolio optimization output."""
    weights: dict[str, float]         # Symbol -> weight
    expected_return: float
    expected_volatility: float
    cvar: float                       # Conditional Value at Risk
    sharpe_ratio: float
    efficient_frontier: Optional[np.ndarray] = None
    solver_time_ms: float = 0.0
    gpu_accelerated: bool = False

class PortOptEngine:
    """
    GPU-accelerated portfolio optimization engine.

    Integrates NVIDIA cuOpt for Mean-CVaR optimization.
    """

    def __init__(
        self,
        constraints: PortfolioConstraints | None = None,
        cvar_config: CVaRConfig | None = None,
        nvidia_api_key: str | None = None,
    ):
        self.constraints = constraints or PortfolioConstraints()
        self.cvar_config = cvar_config or CVaRConfig()
        self._nvidia_api_key = nvidia_api_key
        self._scenario_generator = None
        self._solver = None

    def optimize(
        self,
        expected_returns: dict[str, float],
        covariance_matrix: np.ndarray,
        symbols: list[str],
        current_weights: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """
        Run Mean-CVaR portfolio optimization.

        Args:
            expected_returns: Symbol -> expected return
            covariance_matrix: NxN covariance matrix
            symbols: List of symbols (order matches covariance)
            current_weights: Current portfolio weights (for rebalancing)

        Returns:
            Optimal portfolio weights and metrics
        """
        ...

    def compute_efficient_frontier(
        self,
        expected_returns: dict[str, float],
        covariance_matrix: np.ndarray,
        symbols: list[str],
        num_points: int = 50,
    ) -> list[OptimizationResult]:
        """Compute efficient frontier points."""
        ...

    def generate_scenarios(
        self,
        historical_returns: np.ndarray,
        num_scenarios: int | None = None,
    ) -> np.ndarray:
        """
        Generate Monte Carlo scenarios using GPU acceleration.

        Uses cuML for distribution fitting and sampling.
        """
        ...
```

#### 4.1.2 SignalCore Integration

```python
# src/engines/portopt/adapters/signalcore.py

from ..core.engine import PortOptEngine, OptimizationResult
from ...signalcore.core.signal import SignalBatch, Signal
from ...riskguard.core.engine import ProposedTrade

class SignalToPortfolioAdapter:
    """
    Converts SignalCore outputs to PortOpt inputs.
    """

    def __init__(self, portopt_engine: PortOptEngine):
        self.engine = portopt_engine

    def signals_to_expected_returns(
        self,
        signal_batch: SignalBatch,
    ) -> dict[str, float]:
        """
        Convert signal expected returns to optimizer input.

        Args:
            signal_batch: Batch of signals from SignalCore

        Returns:
            Symbol -> expected return mapping
        """
        return {
            signal.symbol: signal.expected_return
            for signal in signal_batch.signals
            if signal.expected_return is not None
        }

    def optimization_to_trades(
        self,
        result: OptimizationResult,
        portfolio_value: float,
        current_positions: dict[str, float],
        price_data: dict[str, float],
    ) -> list[ProposedTrade]:
        """
        Convert optimization result to proposed trades.

        Args:
            result: Optimization output with target weights
            portfolio_value: Total portfolio value
            current_positions: Current position values by symbol
            price_data: Current prices by symbol

        Returns:
            List of proposed trades for RiskGuard evaluation
        """
        trades = []

        for symbol, target_weight in result.weights.items():
            target_value = portfolio_value * target_weight
            current_value = current_positions.get(symbol, 0.0)
            delta_value = target_value - current_value

            if abs(delta_value) < 100:  # Skip small rebalances
                continue

            price = price_data.get(symbol)
            if not price:
                continue

            quantity = int(abs(delta_value) / price)
            if quantity == 0:
                continue

            trades.append(ProposedTrade(
                symbol=symbol,
                direction="long" if delta_value > 0 else "short",
                quantity=quantity,
                entry_price=price,
            ))

        return trades
```

### 4.2 Distillery Engine

#### 4.2.1 Flywheel Implementation

```python
# src/engines/distillery/core/flywheel.py

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from datetime import datetime

class FlywheelStage(Enum):
    """Data flywheel stages."""
    INGEST = "ingest"
    CURATE = "curate"
    STORE = "store"
    TRAIN = "train"
    EVALUATE = "evaluate"
    DEPLOY = "deploy"

@dataclass
class FlywheelConfig:
    """Flywheel configuration."""
    # Teacher model (source of labels)
    teacher_model: str = "meta/llama-3.3-70b-instruct"

    # Student model (to be fine-tuned)
    student_model: str = "meta/llama-3.2-1b-instruct"

    # Training parameters
    min_samples: int = 5000
    validation_split: float = 0.1
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    num_epochs: int = 3

    # Evaluation thresholds
    min_f1_improvement: float = 0.10  # 10% improvement required

    # Infrastructure
    elasticsearch_url: str = "http://localhost:9200"
    mlflow_tracking_uri: str = "http://localhost:5000"

@dataclass
class FlywheelRun:
    """Flywheel execution record."""
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    stage: FlywheelStage = FlywheelStage.INGEST
    samples_collected: int = 0
    samples_curated: int = 0
    base_f1: Optional[float] = None
    trained_f1: Optional[float] = None
    model_artifact: Optional[str] = None
    deployed: bool = False
    error: Optional[str] = None

class DataFlywheel:
    """
    Automated model distillation flywheel.

    Implements the NVIDIA Blueprint pattern:
    Ingest → Curate → Store → Train → Evaluate → Deploy
    """

    def __init__(self, config: FlywheelConfig):
        self.config = config
        self._elasticsearch = None
        self._nemo_customizer = None
        self._nemo_evaluator = None
        self._mlflow = None

    async def ingest_logs(
        self,
        source: str = "cortex",
        since: datetime | None = None,
    ) -> int:
        """
        Ingest production LLM logs from Elasticsearch.

        Args:
            source: Engine source (cortex, signalcore, riskguard)
            since: Only ingest logs after this timestamp

        Returns:
            Number of samples ingested
        """
        ...

    async def curate_dataset(
        self,
        run_id: str,
        stratify_by: str = "task_type",
    ) -> tuple[int, int]:
        """
        Curate balanced train/eval datasets.

        Args:
            run_id: Flywheel run identifier
            stratify_by: Field to stratify samples

        Returns:
            (train_samples, eval_samples)
        """
        ...

    async def train_student(
        self,
        run_id: str,
        resume_from: str | None = None,
    ) -> str:
        """
        Execute LoRA fine-tuning via NeMo Customizer.

        Args:
            run_id: Flywheel run identifier
            resume_from: Checkpoint to resume from

        Returns:
            Model artifact path
        """
        ...

    async def evaluate_model(
        self,
        run_id: str,
        model_path: str,
    ) -> dict[str, float]:
        """
        Evaluate fine-tuned model via NeMo Evaluator.

        Returns:
            Metrics dict (f1, precision, recall, etc.)
        """
        ...

    async def deploy_model(
        self,
        run_id: str,
        model_path: str,
        target_engine: str = "signalcore",
    ) -> bool:
        """
        Deploy fine-tuned model to target engine.

        Uses NeMo NIM for optimized serving.
        """
        ...

    async def run_full_cycle(self) -> FlywheelRun:
        """
        Execute complete flywheel cycle.

        Orchestrates all stages with proper error handling
        and checkpointing.
        """
        ...
```

#### 4.2.2 SignalCore Model Integration

```python
# src/engines/signalcore/models/distilled.py

from typing import Optional
from ..core.model import BaseModel, ModelConfig
from ..core.signal import Signal
from ...distillery.serving.nim import DistilledModelClient

class DistilledSignalModel(BaseModel):
    """
    Signal model using distilled LLM for classification.

    Uses a fine-tuned Llama 1B/3B model instead of 70B,
    achieving 98% cost reduction with maintained accuracy.
    """

    def __init__(
        self,
        config: ModelConfig,
        model_endpoint: str | None = None,
        fallback_to_teacher: bool = True,
    ):
        super().__init__(config)
        self._model_endpoint = model_endpoint
        self._fallback_to_teacher = fallback_to_teacher
        self._client: Optional[DistilledModelClient] = None

    def generate(self, data: Any, timestamp: Any) -> Signal:
        """
        Generate signal using distilled model inference.

        Falls back to teacher model if distilled unavailable.
        """
        ...

    @property
    def model_size(self) -> str:
        """Return model size (1B, 3B, or teacher)."""
        ...

    @property
    def cost_per_inference(self) -> float:
        """Estimated cost per inference call."""
        # Distilled 1B: ~$0.0001 per call
        # Teacher 70B: ~$0.005 per call
        ...
```

---

## 5. Unified Workflow

### 5.1 Complete Pipeline Sequence

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        UNIFIED TRADING WORKFLOW                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  PHASE 1: DATA INGESTION                                                       │
│  ═══════════════════════                                                       │
│  1.1 Market data plugins fetch OHLCV (Alpha Vantage, Finnhub, Polygon)        │
│  1.2 Data validation and normalization                                         │
│  1.3 Store in time-series cache                                                │
│                                                                                │
│  PHASE 2: FEATURE ENGINEERING                                                  │
│  ═══════════════════════════                                                   │
│  2.1 Technical indicators (RSI, MACD, Bollinger) via SignalCore               │
│  2.2 GPU-accelerated feature computation (cuDF)                               │
│  2.3 Return distribution estimation                                            │
│                                                                                │
│  PHASE 3: SIGNAL GENERATION                                                    │
│  ═════════════════════════                                                     │
│  3.1 Distilled model inference (Llama 1B/3B via NIM)                          │
│  3.2 Signal batch generation with probabilities                               │
│  3.3 Expected return estimates per symbol                                      │
│                                                                                │
│  PHASE 4: PORTFOLIO OPTIMIZATION                                               │
│  ═══════════════════════════════                                               │
│  4.1 Scenario generation (10K+ Monte Carlo via cuML)                          │
│  4.2 Mean-CVaR optimization (cuOpt solver)                                    │
│  4.3 Constraint satisfaction (sector, leverage, position limits)              │
│  4.4 Optimal weight computation                                                │
│                                                                                │
│  PHASE 5: RISK VALIDATION                                                      │
│  ════════════════════════                                                      │
│  5.1 Convert weights to ProposedTrades                                         │
│  5.2 RiskGuard rule evaluation                                                 │
│  5.3 Position sizing adjustments                                               │
│  5.4 Kill switch checks                                                        │
│                                                                                │
│  PHASE 6: ORDER EXECUTION                                                      │
│  ════════════════════════                                                      │
│  6.1 Validated orders to FlowRoute                                             │
│  6.2 Broker adapter execution (Alpaca, Paper)                                  │
│  6.3 Fill confirmation and position updates                                    │
│                                                                                │
│  PHASE 7: MONITORING & FEEDBACK                                                │
│  ═══════════════════════════                                                   │
│  7.1 ProofBench performance tracking                                           │
│  7.2 LLM interaction logging (for Distillery)                                  │
│  7.3 Governance audit trail                                                    │
│                                                                                │
│  BACKGROUND: MODEL DISTILLATION                                                │
│  ══════════════════════════════                                                │
│  B.1 Collect production logs (Cortex, SignalCore)                             │
│  B.2 Curate training datasets                                                  │
│  B.3 LoRA fine-tuning cycle                                                    │
│  B.4 Evaluation and deployment                                                 │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Workflow Code Example

```python
# scripts/unified_workflow_example.py

import asyncio
from datetime import datetime, timedelta
import numpy as np

from src.plugins.market_data import AlphaVantagePlugin
from src.engines.signalcore import SignalCoreEngine
from src.engines.signalcore.models.distilled import DistilledSignalModel
from src.engines.portopt import PortOptEngine, PortfolioConstraints, CVaRConfig
from src.engines.portopt.adapters.signalcore import SignalToPortfolioAdapter
from src.engines.riskguard import RiskGuardEngine, PortfolioState
from src.engines.flowroute import FlowRouteEngine
from src.engines.proofbench import ProofBenchEngine

async def run_unified_workflow():
    """Execute complete trading workflow with NVIDIA acceleration."""

    # Initialize engines
    market_data = AlphaVantagePlugin(api_key="...")

    signal_engine = SignalCoreEngine(
        models=[DistilledSignalModel(
            config=ModelConfig(model_id="distilled-classifier"),
            model_endpoint="http://localhost:8000/v1",  # NIM endpoint
        )]
    )

    portopt = PortOptEngine(
        constraints=PortfolioConstraints(
            max_weight=0.15,
            max_sector_weight=0.30,
            long_only=True,
        ),
        cvar_config=CVaRConfig(
            confidence_level=0.95,
            num_scenarios=10000,
            risk_aversion=0.5,
            use_gpu=True,
        ),
    )

    adapter = SignalToPortfolioAdapter(portopt)

    riskguard = RiskGuardEngine()  # Load standard rules
    flowroute = FlowRouteEngine(adapter="paper")

    # Universe definition
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

    # PHASE 1: Data Ingestion
    print("[1/6] Fetching market data...")
    price_data = {}
    historical_returns = []

    for symbol in symbols:
        quote = await market_data.get_quote(symbol)
        price_data[symbol] = quote["price"]

        history = await market_data.get_historical(
            symbol,
            period="1y",
            interval="1d"
        )
        returns = history["close"].pct_change().dropna()
        historical_returns.append(returns.values)

    returns_matrix = np.column_stack(historical_returns)
    covariance = np.cov(returns_matrix.T)

    # PHASE 2-3: Feature Engineering + Signal Generation
    print("[2/6] Generating signals with distilled model...")
    signal_batch = await signal_engine.generate_batch(
        symbols=symbols,
        data=price_data,
        timestamp=datetime.utcnow(),
    )

    expected_returns = adapter.signals_to_expected_returns(signal_batch)

    # PHASE 4: Portfolio Optimization (GPU-accelerated)
    print("[3/6] Running Mean-CVaR optimization...")
    optimization_result = portopt.optimize(
        expected_returns=expected_returns,
        covariance_matrix=covariance,
        symbols=symbols,
    )

    print(f"    Solver time: {optimization_result.solver_time_ms:.1f}ms")
    print(f"    GPU accelerated: {optimization_result.gpu_accelerated}")
    print(f"    Expected return: {optimization_result.expected_return:.2%}")
    print(f"    CVaR (95%): {optimization_result.cvar:.2%}")

    # PHASE 5: Risk Validation
    print("[4/6] Validating through RiskGuard...")
    portfolio_value = 100_000.0
    current_positions = {}  # Fresh portfolio

    proposed_trades = adapter.optimization_to_trades(
        result=optimization_result,
        portfolio_value=portfolio_value,
        current_positions=current_positions,
        price_data=price_data,
    )

    portfolio_state = PortfolioState(
        equity=portfolio_value,
        cash=portfolio_value,
        peak_equity=portfolio_value,
        daily_pnl=0.0,
        daily_trades=0,
        open_positions={},
        total_positions=0,
        total_exposure=0.0,
    )

    validated_trades = []
    for trade in proposed_trades:
        signal = signal_batch.get_by_symbol(trade.symbol)
        if signal:
            passed, results, adjusted = riskguard.evaluate_signal(
                signal, trade, portfolio_state
            )
            if passed:
                validated_trades.append(trade)

    print(f"    Proposed: {len(proposed_trades)}, Validated: {len(validated_trades)}")

    # PHASE 6: Order Execution
    print("[5/6] Executing validated orders...")
    for trade in validated_trades:
        order_result = await flowroute.submit_order(trade)
        print(f"    {trade.symbol}: {trade.direction} {trade.quantity} @ ${trade.entry_price:.2f}")

    # PHASE 7: Monitoring
    print("[6/6] Recording performance metrics...")
    # ProofBench tracking would happen here

    print("\nWorkflow complete.")
    return optimization_result

if __name__ == "__main__":
    asyncio.run(run_unified_workflow())
```

---

## 6. Implementation Plan

### 6.1 Phase Breakdown

| Phase | Duration | Deliverables | Dependencies |
|-------|----------|--------------|--------------|
| **Phase 1**: Foundation | 2 weeks | PortOpt engine skeleton, CUDA adapters | H100 access |
| **Phase 2**: Optimization Core | 3 weeks | cuOpt integration, CVaR solver | Phase 1 |
| **Phase 3**: Distillery Foundation | 2 weeks | Flywheel skeleton, Elasticsearch setup | Docker, K8s |
| **Phase 4**: Training Pipeline | 3 weeks | LoRA training, NeMo Customizer | Phase 3, 2x H100 |
| **Phase 5**: Integration | 2 weeks | Adapter layers, unified workflow | Phases 2, 4 |
| **Phase 6**: Testing & Refinement | 2 weeks | Integration tests, benchmarks | Phase 5 |

### 6.2 Sequential Task List

#### Phase 1: Foundation (Tasks 1-8)

```
[ ] 1.1  Create src/engines/portopt/ directory structure
[ ] 1.2  Define PortfolioConstraints, CVaRConfig dataclasses
[ ] 1.3  Implement PortOptEngine skeleton with interface methods
[ ] 1.4  Create CUDA device detection and fallback logic
[ ] 1.5  Add cuDF/cuML conditional imports with graceful degradation
[ ] 1.6  Create src/engines/distillery/ directory structure
[ ] 1.7  Define FlywheelConfig, FlywheelRun dataclasses
[ ] 1.8  Implement DataFlywheel skeleton
```

#### Phase 2: Optimization Core (Tasks 9-18)

```
[ ] 2.1  Implement scenario generation with cuML multivariate sampling
[ ] 2.2  Implement CPU fallback scenario generation (numpy)
[ ] 2.3  Integrate cuOpt solver for Mean-CVaR optimization
[ ] 2.4  Implement constraint handling (leverage, sector, position)
[ ] 2.5  Implement efficient frontier computation
[ ] 2.6  Add rebalancing strategy support
[ ] 2.7  Create OptimizationResult serialization
[ ] 2.8  Benchmark GPU vs CPU performance
[ ] 2.9  Write unit tests for optimization routines
[ ] 2.10 Document optimization API
```

#### Phase 3: Distillery Foundation (Tasks 19-26)

```
[ ] 3.1  Set up Elasticsearch for log ingestion
[ ] 3.2  Implement production log capture from Cortex
[ ] 3.3  Implement dataset curation with stratified sampling
[ ] 3.4  Set up MLflow for experiment tracking
[ ] 3.5  Create Celery task queue for flywheel orchestration
[ ] 3.6  Implement GPU allocation serialization (prevent oversubscription)
[ ] 3.7  Add Docker Compose for local development
[ ] 3.8  Write flywheel stage tests
```

#### Phase 4: Training Pipeline (Tasks 27-34)

```
[ ] 4.1  Integrate NeMo Customizer for LoRA fine-tuning
[ ] 4.2  Implement dataset formatting for NeMo
[ ] 4.3  Configure LoRA hyperparameters (rank, alpha, lr)
[ ] 4.4  Implement training checkpointing
[ ] 4.5  Integrate NeMo Evaluator for F1 scoring
[ ] 4.6  Implement model comparison benchmarking
[ ] 4.7  Set up NeMo NIM for model serving
[ ] 4.8  Create DistilledModelClient for inference
```

#### Phase 5: Integration (Tasks 35-42)

```
[ ] 5.1  Implement SignalToPortfolioAdapter
[ ] 5.2  Integrate PortOpt with SignalCore signal batches
[ ] 5.3  Integrate optimization output with RiskGuard
[ ] 5.4  Implement DistilledSignalModel in SignalCore
[ ] 5.5  Add distilled model fallback to teacher
[ ] 5.6  Create unified workflow orchestrator
[ ] 5.7  Implement background flywheel scheduling
[ ] 5.8  Add monitoring hooks for Distillery data collection
```

#### Phase 6: Testing & Refinement (Tasks 43-50)

```
[ ] 6.1  Integration tests for PortOpt → RiskGuard flow
[ ] 6.2  Integration tests for Distillery → SignalCore flow
[ ] 6.3  End-to-end workflow test
[ ] 6.4  Performance benchmarking (latency, throughput)
[ ] 6.5  GPU memory optimization
[ ] 6.6  Documentation: user guides
[ ] 6.7  Documentation: API reference
[ ] 6.8  Example notebooks
```

---

## 7. Required Refactoring

### 7.1 SignalCore Changes

| File | Change | Reason |
|------|--------|--------|
| `signalcore/core/signal.py` | Add `covariance_contribution` field | For portfolio optimization |
| `signalcore/core/model.py` | Add `get_expected_returns()` method | PortOpt input |
| `signalcore/__init__.py` | Export `DistilledSignalModel` | New model type |

### 7.2 RiskGuard Changes

| File | Change | Reason |
|------|--------|--------|
| `riskguard/core/engine.py` | Add `evaluate_portfolio_weights()` | Batch weight validation |
| `riskguard/rules/standard.py` | Add CVaR-based rules | Align with optimization |

### 7.3 Cortex Changes

| File | Change | Reason |
|------|--------|--------|
| `cortex/core/engine.py` | Add `log_interaction()` hook | Flywheel data collection |

### 7.4 pyproject.toml Changes

```toml
[project.optional-dependencies]
portopt = [
    "cudf-cu12>=24.10",
    "cuml-cu12>=24.10",
    "cuopt>=24.10",
]
distillery = [
    "nemo-toolkit>=2.0",
    "peft>=0.7.0",
    "mlflow>=2.10",
    "celery>=5.3",
    "elasticsearch>=8.12",
    "redis>=5.0",
]
nvidia-full = [
    "ordinis[portopt,distillery,ai]",
]
```

---

## 8. Future Extensions

### 8.1 Risk-Scoring Modules

The architecture supports adding CVaR-based risk scoring:

```python
# Future: src/engines/riskguard/rules/cvar_rules.py

class CVaRPositionRule(RiskRule):
    """Limit position size based on CVaR contribution."""

    def __init__(self, max_cvar_contribution: float = 0.02):
        super().__init__(
            rule_id="CVAR_POSITION",
            name="CVaR Position Limit",
            threshold=max_cvar_contribution,
            ...
        )
```

### 8.2 Rule-Based Overlays

Optimization results can be overlaid with discretionary rules:

```python
# Future: src/engines/portopt/overlays/discretionary.py

class DiscretionaryOverlay:
    """Apply discretionary rules to optimization output."""

    def apply(
        self,
        result: OptimizationResult,
        overrides: dict[str, float],
    ) -> OptimizationResult:
        """
        Apply manual weight overrides to optimization result.

        Useful for:
        - Earnings blackout periods
        - Sector rotation views
        - Liquidity constraints
        """
        ...
```

### 8.3 Multi-Period Optimization

```python
# Future: src/engines/portopt/strategies/multiperiod.py

class MultiPeriodOptimizer:
    """
    Multi-period portfolio optimization with transaction costs.

    Considers rebalancing costs and tax efficiency.
    """
    ...
```

---

## 9. Monitoring & Evaluation

### 9.1 Key Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Optimization latency | < 500ms for 50 assets | ProofBench timing |
| Distillation F1 | > 0.90 | NeMo Evaluator |
| Inference cost reduction | > 90% | Token counting |
| GPU utilization | > 70% during optimization | nvidia-smi |

### 9.2 Alerting Rules

```yaml
# Prometheus alerting rules for NVIDIA components
groups:
  - name: nvidia_blueprints
    rules:
      - alert: PortOptSlowOptimization
        expr: portopt_optimization_duration_seconds > 2.0
        for: 5m
        labels:
          severity: warning

      - alert: DistilleryFlywheelFailed
        expr: distillery_flywheel_failures_total > 0
        for: 1m
        labels:
          severity: critical

      - alert: DistilledModelF1Drop
        expr: distillery_model_f1 < 0.85
        for: 1h
        labels:
          severity: warning
```

---

## 10. References

### 10.1 NVIDIA Documentation

- [cuOpt Documentation](https://docs.nvidia.com/cuopt/)
- [RAPIDS cuDF](https://docs.rapids.ai/api/cudf/stable/)
- [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/)
- [NeMo Framework](https://docs.nvidia.com/nemo-framework/)
- [NeMo NIMs](https://developer.nvidia.com/nemo-microservices)

### 10.2 Repository Sources

- [Quantitative Portfolio Optimization Blueprint](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization)
- [AI Model Distillation Blueprint](https://github.com/NVIDIA-AI-Blueprints/ai-model-distillation-for-financial-data)

### 10.3 Academic References

- Rockafellar & Uryasev (2000). Optimization of Conditional Value-at-Risk.
- Hinton et al. (2015). Distilling the Knowledge in a Neural Network.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.

---

**Document Status**: Implementation Plan Complete
**Next Action**: Begin Phase 1 implementation pending hardware availability
