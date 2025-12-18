---
title: SignalEngine (SignalCore) API Reference
date: 2025-12-18
version: 1.0.0
type: api reference
description: >
  Public Python API surface for SignalEngine (SignalCore): models, signals, configuration, and governance hooks.
source_of_truth: ../inbox/documents/system-specification.md
---

# `ordinis.engines.signalcore`

SignalEngine (SignalCore) generates quantitative trading signals. It consumes market data (typically
OHLCV as a `pandas.DataFrame`) and produces `Signal` outputs for downstream gating by
`RiskEngine (RiskGuard)` and routing by `ExecutionEngine (FlowRoute)`.

## Key exports

- `SignalCoreEngine`: orchestrates models and produces signals (`generate_signal`, `generate_batch`)
- `SignalCoreEngineConfig`: engine configuration
- `Model`, `ModelConfig`, `ModelRegistry`: model abstraction and registration
- `Signal`, `SignalBatch`: output types

## Included model implementations

- `SMACrossoverModel`
- `RSIMeanReversionModel`

## Governance hooks

- `SignalCoreGovernanceHook`
- `DataQualityRule`
- `SignalThresholdRule`
- `ModelValidationRule`

## Quick start

```python
from ordinis.engines.signalcore import (
    ModelConfig,
    SignalCoreEngine,
    SignalCoreEngineConfig,
    SMACrossoverModel,
)

config = SignalCoreEngineConfig(
    min_probability=0.6,
    min_score=0.3,
    enable_governance=True,
)

engine = SignalCoreEngine(config)
await engine.initialize()

model = SMACrossoverModel(
    ModelConfig(
        model_id="sma_crossover_v1",
        model_type="technical",
        parameters={"short_period": 10, "long_period": 50},
    )
)

engine.register_model(model)

signal = await engine.generate_signal(symbol="AAPL", data=df)
print(signal)
```

## Related documentation

- Architecture: `docs/knowledge-base/engines/signalcore.md`
- Engine definitions: `docs/knowledge-base/inbox/documents/system-specification.md`
