# ML Systems Enhancement Review

**Version:** 1.0  
**Date:** 2025-01-21  
**Author:** AI ML Architecture Review  
**Status:** Comprehensive Analysis Complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current ML Architecture Map](#2-current-ml-architecture-map)
3. [Critical Gaps & Risks](#3-critical-gaps--risks)
4. [High-Impact Enhancement Proposals](#4-high-impact-enhancement-proposals)
5. [ONNX Runtime Integration Plan](#5-onnx-runtime-integration-plan)
6. [MLOS-Inspired Optimization Framework](#6-mlos-inspired-optimization-framework)
7. [Testing & Evaluation Plan](#7-testing--evaluation-plan)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Appendix: Repository Evidence](#appendix-repository-evidence)

---

## 1. Executive Summary

### Top 5 Gaps (Prioritized)

| Rank | Gap | Severity | Impact | Evidence |
|------|-----|----------|--------|----------|
| **G1** | **No model serialization/persistence** | 🔴 Critical | Models lost on restart; no reproducibility | [model.py#L100-200](../src/ordinis/engines/signalcore/core/model.py) - `ModelRegistry._models` is in-memory dict only |
| **G2** | **Training/serving skew in LSTM** | 🔴 Critical | Incorrect inference predictions | [lstm_model.py#L160](../src/ordinis/engines/signalcore/models/lstm_model.py) - "wrong for inference but ok for this demo" |
| **G3** | **No feature store implementation** | 🟡 High | Feature drift, inconsistent preprocessing | Only doc references exist; no `FeatureStore` class in `src/` |
| **G4** | **No inference latency benchmarking** | 🟡 High | Cannot meet p95 ≤ 200ms target | No `@timer` decorators or benchmark harness for model inference |
| **G5** | **No mixed-precision/quantization** | 🟡 High | Suboptimal GPU utilization | Zero matches for `cuda.amp`, `autocast`, `GradScaler` |

### Top 5 Improvements (ROI-Ordered)

| Rank | Improvement | Effort | Impact | Business Value |
|------|-------------|--------|--------|----------------|
| **I1** | **ONNX Runtime for production inference** | Medium | High | 2-5x latency reduction, CPU/GPU portable |
| **I2** | **Model checkpointing with scaler persistence** | Low | Critical | Fixes training/serving skew, enables reproducibility |
| **I3** | **Feature Store with versioning** | Medium | High | Eliminates preprocessing drift, enables feature reuse |
| **I4** | **MLOS-inspired tuning framework** | Medium | High | Automated hyperparameter optimization with telemetry |
| **I5** | **Mixed-precision training (AMP)** | Low | Medium | 1.5-2x training speedup on RTX GPUs |

### Current vs Target State

```
                    CURRENT STATE                          TARGET STATE
                    ─────────────                          ────────────
┌─────────────────────────────────────┐    ┌─────────────────────────────────────┐
│  Training                           │    │  Training                           │
│  ├─ Manual torch.save (only LSTM)   │    │  ├─ MLflow artifact store           │
│  ├─ In-memory model registry        │───▶│  ├─ Versioned model registry        │
│  ├─ Inline normalization (skew!)    │    │  ├─ FeatureStore with scalers       │
│  └─ No AMP/quantization             │    │  └─ AMP + INT8 quantization         │
├─────────────────────────────────────┤    ├─────────────────────────────────────┤
│  Inference                          │    │  Inference                          │
│  ├─ PyTorch eager mode              │    │  ├─ ONNX Runtime sessions           │
│  ├─ No batching strategy            │───▶│  ├─ Dynamic batching                │
│  └─ No latency monitoring           │    │  └─ Prometheus p50/p95/p99 metrics  │
├─────────────────────────────────────┤    ├─────────────────────────────────────┤
│  Evaluation                         │    │  Evaluation                         │
│  ├─ WalkForward optimizer exists    │    │  ├─ PurgedKFold for financial data  │
│  ├─ GridSearch in scripts           │───▶│  ├─ Optuna/MLOS integration         │
│  └─ No systematic drift detection   │    │  └─ Automated drift → retrain loop  │
└─────────────────────────────────────┘    └─────────────────────────────────────┘
```

---

## 2. Current ML Architecture Map

### 2.1 Model Taxonomy

```
src/ordinis/engines/signalcore/models/
├── lstm_model.py              # PyTorch LSTM (trainable)
├── hmm_regime.py              # Hidden Markov Model (trainable, custom impl)
├── garch_breakout.py          # GARCH volatility model
├── kalman_hybrid.py           # Kalman filter
├── mi_ensemble.py             # Mutual information ensemble
├── bollinger_bands.py         # Technical indicator (rule-based)
├── macd.py                    # Technical indicator (rule-based)
├── rsi_mean_reversion.py      # Technical indicator (rule-based)
├── momentum_breakout.py       # Technical indicator (rule-based)
├── sma_crossover.py           # Technical indicator (rule-based)
├── forecasting/
│   ├── statsforecast_model.py # statsforecast wrapper
│   └── volatility_model.py    # Volatility forecasting
├── sentiment/                 # Sentiment-based models
└── fundamental/               # Fundamental analysis models
```

**Model Base Class** ([model.py#L1-100](../src/ordinis/engines/signalcore/core/model.py)):

```python
@dataclass
class ModelConfig:
    model_id: str
    model_type: str  # "ml" | "technical" | "ensemble"
    version: str = "1.0.0"
    parameters: dict[str, Any] = field(default_factory=dict)
    min_data_points: int = 100

class Model(ABC):
    @abstractmethod
    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal: ...
    # ⚠️ MISSING: save(), load(), export_onnx()
```

### 2.2 Training Infrastructure

**Entry Point**: [train.py](../src/ordinis/engines/signalcore/train.py)

| Component | Location | Status | Gap |
|-----------|----------|--------|-----|
| CLI Training | `train.py:run_training()` | ✅ Works | No experiment tracking |
| Data Loading | `train.py:load_data()` | ✅ Works | No validation split |
| Model Save | `train.py:142` | ⚠️ Partial | Only saves `state_dict`, no scaler |
| Hyperparameters | `train.py:args` | ✅ Works | Manual tuning only |

**Training Flow**:

```
train.py --model lstm --data data/historical
    │
    ▼
┌─────────────────────────────────────┐
│ 1. load_data() → pd.DataFrame       │
│ 2. ModelConfig(parameters={...})    │
│ 3. LSTMModel(config)                │
│ 4. model.train(data)                │ ◀─ Inline normalization (BUG!)
│ 5. torch.save(state_dict, path)     │ ◀─ No scaler saved
└─────────────────────────────────────┘
```

### 2.3 Inference Pipeline

**SignalCore Engine** ([engine.py](../src/ordinis/engines/signalcore/core/engine.py)):

```python
class SignalCoreEngine(BaseEngine[SignalCoreEngineConfig]):
    async def generate_signals(self, data: Any) -> list[Any]:
        # Calls model.generate() for each registered model
        # ⚠️ No batching, no latency tracking, no ONNX
```

**LSTM Inference** ([lstm_model.py#L140-186](../src/ordinis/engines/signalcore/models/lstm_model.py)):

```python
async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
    # Line 160: CRITICAL BUG
    # "Use stored mean/std if available, else compute 
    #  (which is wrong for inference but ok for this demo)"
    if hasattr(self, "mean"):
        features = (features - self.mean) / (self.std + 1e-8)
    # ⚠️ If model loaded from state_dict, mean/std are LOST
```

### 2.4 Evaluation Infrastructure

| Component | Location | Status | Gap |
|-----------|----------|--------|-----|
| Walk-Forward | [walk_forward.py](../src/ordinis/engines/proofbench/analysis/walk_forward.py) | ✅ Production | Good foundation |
| Monte Carlo | `walk_forward.py:MonteCarloMethod` | ✅ Production | Bootstrap, block bootstrap |
| Cross-Validation | LearningEngineConfig | ⚠️ Config only | No `PurgedKFold` implementation |
| Hyperparameter | [backtest_gs_quant_strategies.py](../scripts/backtest_gs_quant_strategies.py) | ⚠️ Scripts only | GridSearch, no Bayesian |

### 2.5 Learning Engine Integration

**LearningEngine** ([engine.py](../src/ordinis/engines/learning/core/engine.py)):

```python
class LearningEngine(BaseEngine[LearningEngineConfig]):
    _model_versions: dict[str, list[ModelVersion]] = {}  # ⚠️ In-memory only
    _training_jobs: dict[str, TrainingJob] = {}
    
    # Drift detection exists but not wired to auto-retrain
    # See: LEARNING_ENGINE_REVIEW.md for full analysis
```

---

## 3. Critical Gaps & Risks

### GAP-1: No Model Serialization/Persistence (🔴 Critical)

**Evidence**:

- [model.py#L100-200](../src/ordinis/engines/signalcore/core/model.py): `ModelRegistry._models` is `dict[str, Model]`
- No `save()`, `load()`, or `to_dict()` methods on `Model` ABC
- Only LSTM has partial save via `train.py:142`

**Risk**:

- Models lost on process restart
- No reproducibility for regulatory audits
- Cannot A/B test model versions

**Fix**:

```python
# Proposed: model.py additions
class Model(ABC):
    @abstractmethod
    def save(self, path: Path) -> None: ...
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Model": ...
    
    def export_onnx(self, path: Path, example_input: torch.Tensor) -> None:
        torch.onnx.export(self.model, example_input, path, ...)
```

### GAP-2: Training/Serving Skew (🔴 Critical)

**Evidence** ([lstm_model.py#L160](../src/ordinis/engines/signalcore/models/lstm_model.py)):

```python
# Line 65-67: Normalization during training
self.mean = features.mean(axis=0)
self.std = features.std(axis=0)
features = (features - self.mean) / (self.std + 1e-8)

# Line 160: Normalization during inference
# "wrong for inference but ok for this demo"
if hasattr(self, "mean"):
    features = (features - self.mean) / (self.std + 1e-8)
```

**Risk**:

- If model loaded from `state_dict`, `self.mean`/`self.std` are undefined
- Inference uses wrong normalization → garbage predictions
- Silent failure (no error raised)

**Fix**:

```python
# Proposed: Persist scaler alongside model
def save(self, path: Path) -> None:
    torch.save({
        "model_state_dict": self.model.state_dict(),
        "scaler": {"mean": self.mean, "std": self.std},
        "config": asdict(self.config),
    }, path)

@classmethod
def load(cls, path: Path) -> "LSTMModel":
    checkpoint = torch.load(path)
    config = ModelConfig(**checkpoint["config"])
    instance = cls(config)
    instance.model.load_state_dict(checkpoint["model_state_dict"])
    instance.mean = checkpoint["scaler"]["mean"]
    instance.std = checkpoint["scaler"]["std"]
    return instance
```

### GAP-3: No Feature Store (🟡 High)

**Evidence**:

- `grep "FeatureStore"` → Only in [LEARNING_ENGINE_REVIEW.md](LEARNING_ENGINE_REVIEW.md) proposals
- `scripts/docs/document_modules.py:72` references `ordinis.data.feature_store` → module doesn't exist
- Each model does inline feature engineering with no caching

**Risk**:

- Feature drift between training and inference
- Duplicated preprocessing code across models
- No feature versioning for reproducibility

**Fix**: See Section 4.2

### GAP-4: No Inference Latency Monitoring (🟡 High)

**Evidence**:

- `grep "latency|throughput|benchmark"` → Only doc references
- No `@timer` decorators on `model.generate()`
- No Prometheus histograms for inference time

**Risk**:

- Cannot verify p95 ≤ 200ms SLA from README
- No visibility into model performance degradation

**Fix**: See Section 6.2

### GAP-5: No Mixed Precision / Quantization (🟡 High)

**Evidence**:

- `grep "cuda.amp|autocast|GradScaler"` → 0 matches in src/
- `grep "quantize|int8"` → Only in docs/knowledge-base references
- LSTM training uses FP32 throughout

**Risk**:

- 50% slower training than possible on RTX GPUs
- Larger memory footprint limits batch sizes
- No INT8 inference optimization path

**Fix**:

```python
# Proposed: Add AMP to LSTM training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for epoch in range(epochs):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        with autocast():  # FP16 forward pass
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

---

## 4. High-Impact Enhancement Proposals

### 4.1 Model Checkpointing with Scaler Persistence

**Priority**: P0 (Blocks all other improvements)

**Implementation**:

```python
# src/ordinis/engines/signalcore/core/checkpoint.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch

@dataclass
class ModelCheckpoint:
    """Complete model checkpoint with preprocessing state."""
    
    model_id: str
    version: str
    model_state_dict: dict[str, torch.Tensor]
    scaler_state: dict[str, Any]  # {"mean": np.ndarray, "std": np.ndarray}
    config: dict[str, Any]
    training_metrics: dict[str, float]
    created_at: str
    
    def save(self, path: Path) -> None:
        torch.save(self.__dict__, path)
    
    @classmethod
    def load(cls, path: Path) -> "ModelCheckpoint":
        data = torch.load(path, weights_only=False)
        return cls(**data)

class CheckpointManager:
    """Manages model checkpoint lifecycle."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: "Model",
        version: str,
        metrics: dict[str, float],
    ) -> Path:
        checkpoint = ModelCheckpoint(
            model_id=model.config.model_id,
            version=version,
            model_state_dict=model.model.state_dict(),
            scaler_state=model.get_scaler_state(),
            config=asdict(model.config),
            training_metrics=metrics,
            created_at=datetime.now(UTC).isoformat(),
        )
        path = self.base_dir / f"{model.config.model_id}_v{version}.pt"
        checkpoint.save(path)
        return path
    
    def load_checkpoint(self, model_id: str, version: str) -> ModelCheckpoint:
        path = self.base_dir / f"{model_id}_v{version}.pt"
        return ModelCheckpoint.load(path)
```

**Test**:

```python
# tests/test_engines/test_signalcore/test_checkpoint.py
def test_checkpoint_preserves_scaler():
    model = LSTMModel(config)
    model.train(training_data)
    
    manager = CheckpointManager(tmp_path)
    path = manager.save_checkpoint(model, "1.0.0", {"loss": 0.5})
    
    # Load and verify scaler preserved
    checkpoint = manager.load_checkpoint(model.config.model_id, "1.0.0")
    loaded_model = LSTMModel.from_checkpoint(checkpoint)
    
    np.testing.assert_array_equal(loaded_model.mean, model.mean)
    np.testing.assert_array_equal(loaded_model.std, model.std)
```

### 4.2 Feature Store with Versioning

**Priority**: P1

**Design**:

```python
# src/ordinis/data/feature_store.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import hashlib
import pandas as pd
import numpy as np

@dataclass
class FeatureDefinition:
    """Immutable feature definition."""
    name: str
    transform: Callable[[pd.DataFrame], pd.Series]
    dependencies: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    
    @property
    def signature(self) -> str:
        """Unique hash for this feature version."""
        import inspect
        code = inspect.getsource(self.transform)
        return hashlib.md5(f"{self.name}:{self.version}:{code}".encode()).hexdigest()[:12]

class FeatureStore:
    """Centralized feature computation and caching."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._registry: dict[str, FeatureDefinition] = {}
        self._scalers: dict[str, dict[str, np.ndarray]] = {}
    
    def register(self, feature: FeatureDefinition) -> None:
        self._registry[feature.name] = feature
    
    def compute(
        self,
        df: pd.DataFrame,
        features: list[str],
        fit_scalers: bool = False,
    ) -> pd.DataFrame:
        """Compute features with optional scaler fitting."""
        result = pd.DataFrame(index=df.index)
        
        for name in features:
            feat = self._registry[name]
            result[name] = feat.transform(df)
            
            if fit_scalers:
                self._scalers[name] = {
                    "mean": result[name].mean(),
                    "std": result[name].std(),
                }
            
            if name in self._scalers:
                result[name] = (
                    result[name] - self._scalers[name]["mean"]
                ) / (self._scalers[name]["std"] + 1e-8)
        
        return result
    
    def save_scalers(self, path: Path) -> None:
        import json
        serializable = {
            k: {k2: v2.tolist() if hasattr(v2, 'tolist') else v2 
                for k2, v2 in v.items()}
            for k, v in self._scalers.items()
        }
        path.write_text(json.dumps(serializable))
    
    def load_scalers(self, path: Path) -> None:
        import json
        data = json.loads(path.read_text())
        self._scalers = {
            k: {k2: np.array(v2) for k2, v2 in v.items()}
            for k, v in data.items()
        }
```

**Standard Features Registry**:

```python
# src/ordinis/data/standard_features.py
STANDARD_FEATURES = [
    FeatureDefinition(
        name="returns_5d",
        transform=lambda df: df["close"].pct_change(5),
        version="1.0.0",
    ),
    FeatureDefinition(
        name="volatility_20d",
        transform=lambda df: df["close"].pct_change().rolling(20).std(),
        version="1.0.0",
    ),
    FeatureDefinition(
        name="rsi_14",
        transform=compute_rsi,  # from analysis module
        dependencies=["close"],
        version="1.0.0",
    ),
]
```

### 4.3 Mixed-Precision Training

**Priority**: P2

**Implementation** ([lstm_model.py](../src/ordinis/engines/signalcore/models/lstm_model.py) patch):

```python
# Add to LSTMModel.train()
from torch.cuda.amp import autocast, GradScaler

def train(self, data: pd.DataFrame) -> dict[str, float]:
    # ... existing setup ...
    
    use_amp = self.device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    
    self.model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            
            with autocast(enabled=use_amp):
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")
    
    return {"final_loss": total_loss / len(loader)}
```

---

## 5. ONNX Runtime Integration Plan

### 5.1 Overview

ONNX Runtime provides production-grade inference with:

- **2-5x latency reduction** over PyTorch eager mode
- **CPU/GPU portability** with execution providers
- **INT8/FP16 quantization** for further speedup
- **Dynamic batching** for throughput optimization

### 5.2 Export Pipeline

```python
# src/ordinis/engines/signalcore/onnx/exporter.py
from pathlib import Path
import torch
import onnx
import onnxruntime as ort

class ONNXExporter:
    """Export PyTorch models to optimized ONNX format."""
    
    @staticmethod
    def export_lstm(
        model: "LSTMModel",
        output_path: Path,
        sequence_length: int = 60,
        optimize: bool = True,
    ) -> None:
        """Export LSTM model with dynamic batch size."""
        # Create example input
        example_input = torch.randn(
            1, sequence_length, 5,  # [batch, seq, features]
            device=model.device,
        )
        
        # Export
        torch.onnx.export(
            model.model,
            example_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=17,
        )
        
        if optimize:
            # Optimize with onnxruntime
            import onnxruntime.transformers.optimizer as optimizer
            optimized = optimizer.optimize_model(
                str(output_path),
                model_type="bert",  # LSTM uses similar optimizations
                use_gpu=True,
            )
            optimized.save_model_to_file(str(output_path))
        
        # Validate
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
```

### 5.3 Inference Session Manager

```python
# src/ordinis/engines/signalcore/onnx/session_manager.py
from pathlib import Path
from typing import Optional
import numpy as np
import onnxruntime as ort

class ONNXSessionManager:
    """Manage ONNX Runtime inference sessions with caching."""
    
    def __init__(
        self,
        model_dir: Path,
        use_gpu: bool = True,
        enable_profiling: bool = False,
    ):
        self.model_dir = model_dir
        self._sessions: dict[str, ort.InferenceSession] = {}
        
        # Configure providers
        if use_gpu:
            self.providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                }),
                "CPUExecutionProvider",
            ]
        else:
            self.providers = ["CPUExecutionProvider"]
        
        # Session options
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session_options.intra_op_num_threads = 4
        if enable_profiling:
            self.session_options.enable_profiling = True
    
    def get_session(self, model_id: str) -> ort.InferenceSession:
        """Get or create inference session."""
        if model_id not in self._sessions:
            model_path = self.model_dir / f"{model_id}.onnx"
            self._sessions[model_id] = ort.InferenceSession(
                str(model_path),
                self.session_options,
                providers=self.providers,
            )
        return self._sessions[model_id]
    
    def predict(
        self,
        model_id: str,
        input_data: np.ndarray,
    ) -> np.ndarray:
        """Run inference with latency tracking."""
        session = self.get_session(model_id)
        
        # Ensure correct dtype
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        result = session.run(
            [output_name],
            {input_name: input_data},
        )
        
        return result[0]
    
    def warmup(self, model_id: str, num_iterations: int = 10) -> None:
        """Warmup session for accurate benchmarking."""
        session = self.get_session(model_id)
        input_shape = session.get_inputs()[0].shape
        # Replace dynamic dims with reasonable values
        shape = [s if isinstance(s, int) else 1 for s in input_shape]
        dummy_input = np.random.randn(*shape).astype(np.float32)
        
        for _ in range(num_iterations):
            self.predict(model_id, dummy_input)
```

### 5.4 Quantization Pipeline

```python
# src/ordinis/engines/signalcore/onnx/quantizer.py
from pathlib import Path
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
    QuantType,
)

class SignalModelQuantizer:
    """Quantize ONNX models for faster inference."""
    
    @staticmethod
    def dynamic_quantize(
        input_path: Path,
        output_path: Path,
        quant_type: QuantType = QuantType.QInt8,
    ) -> None:
        """Dynamic INT8 quantization (no calibration data needed)."""
        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=quant_type,
        )
    
    @staticmethod
    def static_quantize(
        input_path: Path,
        output_path: Path,
        calibration_data: np.ndarray,
    ) -> None:
        """Static INT8 quantization with calibration."""
        class CalibrationReader(CalibrationDataReader):
            def __init__(self, data: np.ndarray):
                self.data = data
                self.index = 0
            
            def get_next(self):
                if self.index >= len(self.data):
                    return None
                batch = {"input": self.data[self.index:self.index+1]}
                self.index += 1
                return batch
        
        quantize_static(
            str(input_path),
            str(output_path),
            CalibrationReader(calibration_data),
        )
```

### 5.5 Integration with SignalCore

```python
# Proposed modifications to SignalCoreEngine
class SignalCoreEngine(BaseEngine[SignalCoreEngineConfig]):
    def __init__(self, ...):
        # ... existing init ...
        
        # Add ONNX session manager
        if self.config.use_onnx_runtime:
            self._onnx_manager = ONNXSessionManager(
                model_dir=self.config.onnx_model_dir,
                use_gpu=self.config.onnx_use_gpu,
                enable_profiling=self.config.onnx_profiling,
            )
    
    async def generate_signal_onnx(
        self,
        model_id: str,
        data: pd.DataFrame,
        timestamp: datetime,
    ) -> Signal:
        """Generate signal using ONNX Runtime."""
        # Prepare features (use FeatureStore)
        features = self._feature_store.compute(
            data, 
            self._registry.get(model_id).feature_names,
            fit_scalers=False,  # Use saved scalers
        )
        
        # Convert to numpy
        input_data = features.values.astype(np.float32)
        
        # Run ONNX inference
        with self._metrics.inference_latency.time():
            output = self._onnx_manager.predict(model_id, input_data)
        
        # Post-process to Signal
        return self._output_to_signal(model_id, output, timestamp)
```

### 5.6 Benchmark Targets

| Metric | Current (PyTorch) | Target (ONNX) | Improvement |
|--------|-------------------|---------------|-------------|
| p50 latency | ~50ms (estimated) | ≤20ms | 2.5x |
| p95 latency | ~150ms (estimated) | ≤50ms | 3x |
| p99 latency | ~250ms (estimated) | ≤100ms | 2.5x |
| Throughput | ~20 signals/sec | ≥100 signals/sec | 5x |
| Memory | ~500MB (CUDA) | ~200MB (INT8) | 2.5x |

---

## 6. MLOS-Inspired Optimization Framework

### 6.1 Overview

[MLOS (Machine Learning for Operating Systems)](https://github.com/microsoft/MLOS) provides a framework for automated system optimization. Key concepts to adopt:

1. **Tunable Parameters** - Declarative parameter spaces with constraints
2. **Telemetry Loop** - Observe → Suggest → Apply → Measure
3. **Optimizer Backends** - Bayesian optimization, random search, grid search
4. **Configuration Spaces** - Hierarchical, conditional parameters

### 6.2 Ordinis Optimization Framework Design

```python
# src/ordinis/optimization/framework.py
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import json

class ParameterType(Enum):
    """Tunable parameter types."""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"

@dataclass
class TunableParameter:
    """Single tunable parameter definition."""
    name: str
    param_type: ParameterType
    default: Any
    range: tuple[Any, Any] | list[Any]  # (min, max) or [options]
    log_scale: bool = False
    conditional_on: dict[str, Any] | None = None  # e.g., {"optimizer": "adam"}

@dataclass
class TunableComponent:
    """Component with tunable parameters."""
    name: str
    parameters: list[TunableParameter]
    apply_fn: Callable[[dict[str, Any]], None]  # Apply config to component
    telemetry_fn: Callable[[], dict[str, float]]  # Collect metrics
    
    def get_config_space(self) -> dict:
        """Export as configuration space."""
        return {
            p.name: {
                "type": p.param_type.value,
                "default": p.default,
                "range": p.range,
                "log_scale": p.log_scale,
            }
            for p in self.parameters
        }
```

### 6.3 Signal Model Tunables

```python
# src/ordinis/engines/signalcore/tunables.py
from ordinis.optimization.framework import (
    TunableComponent,
    TunableParameter,
    ParameterType,
)

LSTM_TUNABLES = TunableComponent(
    name="lstm_model",
    parameters=[
        TunableParameter(
            name="hidden_dim",
            param_type=ParameterType.INTEGER,
            default=64,
            range=(16, 256),
        ),
        TunableParameter(
            name="num_layers",
            param_type=ParameterType.INTEGER,
            default=2,
            range=(1, 4),
        ),
        TunableParameter(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            default=0.001,
            range=(1e-5, 1e-1),
            log_scale=True,
        ),
        TunableParameter(
            name="batch_size",
            param_type=ParameterType.CATEGORICAL,
            default=32,
            range=[16, 32, 64, 128],
        ),
        TunableParameter(
            name="sequence_length",
            param_type=ParameterType.INTEGER,
            default=60,
            range=(20, 120),
        ),
        TunableParameter(
            name="dropout",
            param_type=ParameterType.CONTINUOUS,
            default=0.2,
            range=(0.0, 0.5),
        ),
    ],
    apply_fn=apply_lstm_config,
    telemetry_fn=collect_lstm_metrics,
)

INFERENCE_TUNABLES = TunableComponent(
    name="inference_engine",
    parameters=[
        TunableParameter(
            name="onnx_intra_op_threads",
            param_type=ParameterType.INTEGER,
            default=4,
            range=(1, 16),
        ),
        TunableParameter(
            name="onnx_inter_op_threads",
            param_type=ParameterType.INTEGER,
            default=1,
            range=(1, 4),
        ),
        TunableParameter(
            name="batch_size",
            param_type=ParameterType.CATEGORICAL,
            default=1,
            range=[1, 8, 16, 32],
        ),
        TunableParameter(
            name="use_gpu",
            param_type=ParameterType.CATEGORICAL,
            default=True,
            range=[True, False],
        ),
    ],
    apply_fn=apply_inference_config,
    telemetry_fn=collect_inference_metrics,
)
```

### 6.4 Optimization Loop

```python
# src/ordinis/optimization/optimizer.py
from dataclasses import dataclass
from typing import Protocol
import optuna

class OptimizerBackend(Protocol):
    """Optimizer backend interface."""
    def suggest(self, config_space: dict) -> dict[str, Any]: ...
    def observe(self, config: dict, metrics: dict[str, float]) -> None: ...

class OptunaBackend:
    """Optuna-based Bayesian optimization."""
    
    def __init__(self, study_name: str, direction: str = "maximize"):
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage="sqlite:///artifacts/optuna_studies.db",
            load_if_exists=True,
        )
        self._current_trial: optuna.Trial | None = None
    
    def suggest(self, config_space: dict) -> dict[str, Any]:
        trial = self.study.ask()
        self._current_trial = trial
        
        config = {}
        for name, spec in config_space.items():
            if spec["type"] == "continuous":
                if spec.get("log_scale"):
                    config[name] = trial.suggest_float(
                        name, spec["range"][0], spec["range"][1], log=True
                    )
                else:
                    config[name] = trial.suggest_float(
                        name, spec["range"][0], spec["range"][1]
                    )
            elif spec["type"] == "integer":
                config[name] = trial.suggest_int(
                    name, spec["range"][0], spec["range"][1]
                )
            elif spec["type"] == "categorical":
                config[name] = trial.suggest_categorical(name, spec["range"])
        
        return config
    
    def observe(self, config: dict, metrics: dict[str, float]) -> None:
        if self._current_trial:
            # Report primary metric
            primary_metric = metrics.get("sharpe_ratio", metrics.get("accuracy", 0))
            self.study.tell(self._current_trial, primary_metric)
            self._current_trial = None

@dataclass
class OptimizationLoop:
    """Main optimization loop."""
    
    component: TunableComponent
    backend: OptimizerBackend
    max_iterations: int = 100
    
    def run(self) -> dict[str, Any]:
        """Run optimization loop."""
        config_space = self.component.get_config_space()
        best_config = None
        best_metric = float("-inf")
        
        for i in range(self.max_iterations):
            # Suggest configuration
            config = self.backend.suggest(config_space)
            
            # Apply to component
            self.component.apply_fn(config)
            
            # Collect telemetry
            metrics = self.component.telemetry_fn()
            
            # Report to optimizer
            self.backend.observe(config, metrics)
            
            # Track best
            primary = metrics.get("sharpe_ratio", metrics.get("accuracy", 0))
            if primary > best_metric:
                best_metric = primary
                best_config = config
            
            print(f"Iteration {i+1}: {primary:.4f} (best: {best_metric:.4f})")
        
        return best_config
```

### 6.5 Telemetry Collection

```python
# src/ordinis/optimization/telemetry.py
from dataclasses import dataclass, field
from typing import Callable
from contextlib import contextmanager
import time

@dataclass
class TelemetryCollector:
    """Collect metrics for optimization."""
    
    _latencies: list[float] = field(default_factory=list)
    _throughputs: list[float] = field(default_factory=list)
    
    @contextmanager
    def measure_latency(self):
        """Context manager for latency measurement."""
        start = time.perf_counter()
        yield
        self._latencies.append(time.perf_counter() - start)
    
    def record_throughput(self, items_per_second: float) -> None:
        self._throughputs.append(items_per_second)
    
    def get_metrics(self) -> dict[str, float]:
        import numpy as np
        return {
            "latency_p50": np.percentile(self._latencies, 50) if self._latencies else 0,
            "latency_p95": np.percentile(self._latencies, 95) if self._latencies else 0,
            "latency_p99": np.percentile(self._latencies, 99) if self._latencies else 0,
            "throughput_mean": np.mean(self._throughputs) if self._throughputs else 0,
        }
    
    def reset(self) -> None:
        self._latencies.clear()
        self._throughputs.clear()
```

---

## 7. Testing & Evaluation Plan

### 7.1 ML-Specific Test Categories

| Category | Purpose | Coverage Target |
|----------|---------|-----------------|
| Unit Tests | Isolated model behavior | 90% code coverage |
| Integration Tests | End-to-end signal generation | All models |
| Parity Tests | Training vs inference consistency | All trainable models |
| Performance Tests | Latency/throughput benchmarks | p95 ≤ 200ms |
| Drift Tests | Feature/concept drift detection | Weekly automated |

### 7.2 Parity Test Framework

```python
# tests/test_ml/test_parity.py
import pytest
import numpy as np
from ordinis.engines.signalcore.models.lstm_model import LSTMModel

class TestTrainingServingParity:
    """Ensure training and inference produce consistent results."""
    
    @pytest.fixture
    def trained_model(self, sample_training_data):
        model = LSTMModel()
        model.train(sample_training_data)
        return model
    
    def test_scaler_preserved_after_save_load(self, trained_model, tmp_path):
        """Scaler state must survive serialization."""
        # Save
        checkpoint_path = tmp_path / "model.pt"
        trained_model.save(checkpoint_path)
        
        # Load
        loaded_model = LSTMModel.load(checkpoint_path)
        
        # Verify scaler state
        np.testing.assert_array_almost_equal(
            trained_model.mean, loaded_model.mean
        )
        np.testing.assert_array_almost_equal(
            trained_model.std, loaded_model.std
        )
    
    def test_inference_matches_after_save_load(
        self, trained_model, sample_inference_data, tmp_path
    ):
        """Predictions must match before and after serialization."""
        # Predict before save
        signal_before = trained_model.generate(
            sample_inference_data, datetime.now()
        )
        
        # Save and load
        checkpoint_path = tmp_path / "model.pt"
        trained_model.save(checkpoint_path)
        loaded_model = LSTMModel.load(checkpoint_path)
        
        # Predict after load
        signal_after = loaded_model.generate(
            sample_inference_data, datetime.now()
        )
        
        # Verify match
        assert signal_before.direction == signal_after.direction
        assert abs(signal_before.strength - signal_after.strength) < 1e-5
    
    def test_onnx_matches_pytorch(self, trained_model, sample_inference_data, tmp_path):
        """ONNX output must match PyTorch output."""
        # PyTorch prediction
        pytorch_signal = trained_model.generate(
            sample_inference_data, datetime.now()
        )
        
        # Export to ONNX
        onnx_path = tmp_path / "model.onnx"
        trained_model.export_onnx(onnx_path)
        
        # ONNX prediction
        session = ort.InferenceSession(str(onnx_path))
        # ... prepare input ...
        onnx_output = session.run(None, {"input": input_data})[0]
        
        # Compare (allow small floating point differences)
        np.testing.assert_array_almost_equal(
            pytorch_output, onnx_output, decimal=4
        )
```

### 7.3 Performance Benchmark Suite

```python
# tests/test_ml/test_performance.py
import pytest
import time
import numpy as np
from ordinis.engines.signalcore.onnx.session_manager import ONNXSessionManager

class TestInferencePerformance:
    """Benchmark inference latency and throughput."""
    
    @pytest.fixture
    def onnx_session(self, onnx_model_path):
        manager = ONNXSessionManager(onnx_model_path.parent)
        manager.warmup("test_model", num_iterations=100)
        return manager
    
    def test_p95_latency_under_target(self, onnx_session, benchmark_data):
        """p95 latency must be under 50ms for ONNX."""
        latencies = []
        
        for sample in benchmark_data:
            start = time.perf_counter()
            onnx_session.predict("test_model", sample)
            latencies.append(time.perf_counter() - start)
        
        p95 = np.percentile(latencies, 95) * 1000  # ms
        assert p95 < 50, f"p95 latency {p95:.1f}ms exceeds 50ms target"
    
    def test_throughput_meets_target(self, onnx_session, benchmark_data):
        """Throughput must exceed 100 signals/sec."""
        batch = np.stack(benchmark_data[:100])
        
        start = time.perf_counter()
        onnx_session.predict("test_model", batch)
        elapsed = time.perf_counter() - start
        
        throughput = 100 / elapsed
        assert throughput > 100, f"Throughput {throughput:.1f}/sec below 100/sec target"
```

### 7.4 Walk-Forward Validation Integration

Leverage existing [walk_forward.py](../src/ordinis/engines/proofbench/analysis/walk_forward.py):

```python
# tests/test_ml/test_walkforward.py
from ordinis.engines.proofbench.analysis.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardConfig,
    OptimizationObjective,
)

def test_lstm_walkforward_no_overfitting(sample_data):
    """Walk-forward should not show >50% degradation."""
    config = WalkForwardConfig(
        in_sample_size=252,
        out_sample_size=63,
        step_size=21,
        min_windows=4,
        objective=OptimizationObjective.SHARPE_RATIO,
    )
    
    optimizer = WalkForwardOptimizer(config)
    result = optimizer.optimize(
        data=sample_data,
        param_grid={"hidden_dim": [32, 64], "learning_rate": [0.001, 0.01]},
        model_factory=lambda params: LSTMModel(ModelConfig(parameters=params)),
    )
    
    # Check for overfitting
    assert result.average_degradation < 0.5, \
        f"Average degradation {result.average_degradation:.1%} indicates overfitting"
    assert result.robustness_score > 0.6, \
        f"Robustness score {result.robustness_score:.2f} too low"
```

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

| Task | Owner | Deliverable | Dependencies |
|------|-------|-------------|--------------|
| Model checkpoint with scaler | ML Eng | `checkpoint.py`, `LSTMModel.save/load` | None |
| Fix training/serving skew | ML Eng | Updated `lstm_model.py` | Task 1 |
| Basic ONNX export | ML Eng | `exporter.py` | Task 1 |
| Parity tests | QA | `test_parity.py` | Tasks 1-2 |

**Acceptance Criteria**:

- [ ] `LSTMModel.save()` persists scaler state
- [ ] `LSTMModel.load()` restores scaler state
- [ ] Parity tests pass for save/load cycle
- [ ] ONNX export produces valid model

### Phase 2: ONNX Integration (Weeks 3-4)

| Task | Owner | Deliverable | Dependencies |
|------|-------|-------------|--------------|
| ONNX session manager | ML Eng | `session_manager.py` | Phase 1 |
| INT8 quantization | ML Eng | `quantizer.py` | Task 1 |
| SignalCore ONNX integration | Backend | Updated `engine.py` | Tasks 1-2 |
| Performance benchmarks | QA | `test_performance.py` | Task 3 |

**Acceptance Criteria**:

- [ ] p95 latency ≤ 50ms for ONNX inference
- [ ] Throughput ≥ 100 signals/sec
- [ ] ONNX output matches PyTorch within 1e-4

### Phase 3: Feature Store (Weeks 5-6)

| Task | Owner | Deliverable | Dependencies |
|------|-------|-------------|--------------|
| Feature store implementation | ML Eng | `feature_store.py` | None |
| Standard features registry | ML Eng | `standard_features.py` | Task 1 |
| Migrate LSTM to feature store | ML Eng | Updated `lstm_model.py` | Tasks 1-2 |
| Feature versioning tests | QA | `test_feature_store.py` | Task 3 |

**Acceptance Criteria**:

- [ ] Feature computation is idempotent
- [ ] Scalers persist with features
- [ ] Feature version changes are tracked

### Phase 4: MLOS Framework (Weeks 7-8)

| Task | Owner | Deliverable | Dependencies |
|------|-------|-------------|--------------|
| Tunable parameter framework | ML Eng | `framework.py` | None |
| Optuna backend | ML Eng | `optimizer.py` | Task 1 |
| LSTM tunables definition | ML Eng | `tunables.py` | Task 2 |
| Inference tunables | ML Eng | `tunables.py` | Phase 2 |
| Optimization CLI | ML Eng | `scripts/optimize_model.py` | Tasks 1-4 |

**Acceptance Criteria**:

- [ ] Optimization loop runs end-to-end
- [ ] Best config improves Sharpe by ≥10%
- [ ] Results persisted to Optuna storage

### Phase 5: Mixed Precision & Observability (Weeks 9-10)

| Task | Owner | Deliverable | Dependencies |
|------|-------|-------------|--------------|
| AMP training | ML Eng | Updated `lstm_model.py` | None |
| Prometheus metrics | Infra | `metrics.py` | None |
| Grafana dashboard | Infra | `ml_dashboard.json` | Task 2 |
| E2E integration tests | QA | `test_e2e.py` | All phases |

**Acceptance Criteria**:

- [ ] AMP training 1.5x faster on RTX 4090
- [ ] Metrics exported: inference_latency, throughput, drift_score
- [ ] Dashboard shows real-time ML health

---

## Appendix: Repository Evidence

### Files Analyzed

| File | Lines | Key Findings |
|------|-------|--------------|
| [train.py](../src/ordinis/engines/signalcore/train.py) | 278 | CLI training, partial model save |
| [lstm_model.py](../src/ordinis/engines/signalcore/models/lstm_model.py) | 186 | Training/serving skew at L160 |
| [model.py](../src/ordinis/engines/signalcore/core/model.py) | 277 | In-memory registry, no persistence |
| [engine.py](../src/ordinis/engines/signalcore/core/engine.py) | 463 | No ONNX integration |
| [walk_forward.py](../src/ordinis/engines/proofbench/analysis/walk_forward.py) | 771 | Good WFO foundation |
| [feedback.py](../src/ordinis/engines/learning/collectors/feedback.py) | 1933 | Circuit breakers, drift types |
| [engine.py](../src/ordinis/engines/learning/core/engine.py) | 623 | In-memory model versions |
| [confidence_calibrator.py](../src/ordinis/optimizations/confidence_calibrator.py) | 258 | sklearn/numpy calibration |

### Grep Search Results

| Pattern | Matches | Key Finding |
|---------|---------|-------------|
| `torch.save\|torch.load` | 1 (in train.py) | Only LSTM has save |
| `onnx\|ONNX` | 0 in src/ | No ONNX integration |
| `cuda.amp\|autocast` | 0 | No mixed precision |
| `FeatureStore` | 0 in src/ | No implementation |
| `model.eval\(\)` | 2 | Only LSTM, docs |
| `quantize\|int8` | Docs only | Not implemented |

---

**Document End**

*This review follows the structure established in [LEARNING_ENGINE_REVIEW.md](LEARNING_ENGINE_REVIEW.md) and provides complementary coverage of SignalCore ML systems.*
