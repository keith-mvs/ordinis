---
title: NVIDIA AI Integration
date: 2025-11-29
version: 1.0.0
type: integration guide
description: >
  NVIDIA model integration across Cortex, SignalEngine (SignalCore), RiskEngine (RiskGuard), and AnalyticsEngine (ProofBench).
source_of_truth: ../inbox/documents/system-specification.md
---

# NVIDIA AI Integration

Complete NVIDIA AI model integration across all Ordinis engines, providing natural language insights, explanations, and optimizations throughout the trading workflow.

## Overview

The system integrates NVIDIA's state-of-the-art language models to enhance every stage of the trading pipeline:

1. **Strategy Generation** (Cortex) - AI-powered hypothesis generation and code analysis
2. **Signal Interpretation** (SignalEngine (SignalCore)) - Natural language signal explanations
3. **Risk Evaluation** (RiskEngine (RiskGuard)) - AI-explained trade evaluations
4. **Performance Analysis** (AnalyticsEngine (ProofBench)) - Backtest narration and optimization

## NVIDIA Models Deployed

### Meta Llama 3.1 405B Instruct
- **Engine:** Cortex
- **Purpose:** Deep code analysis and strategy reasoning
- **Temperature:** 0.2 (deterministic)
- **Use Cases:**
  - Strategy code review and optimization
  - Complex reasoning about market conditions
  - Strategy hypothesis generation

### Meta Llama 3.1 70B Instruct
- **Engines:** SignalEngine (SignalCore), RiskEngine (RiskGuard), AnalyticsEngine (ProofBench)
- **Purpose:** Real-time interpretation and explanation
- **Temperature:** 0.3-0.4 (balanced)
- **Use Cases:**
  - Signal interpretation and feature suggestions
  - Trade risk explanations
  - Performance narration and optimization

### NVIDIA NV-Embed-QA E5 V5
- **Engine:** Cortex
- **Purpose:** Semantic understanding and research synthesis
- **Use Cases:**
  - Document similarity and search
  - Research synthesis from multiple sources
  - Knowledge base queries

### NVIDIA Llama 3.3 Nemotron Super 49B v1.5
- **Engine:** Cortex (default NVIDIA path)
- **Purpose:** Primary reasoning/code-review LLM
- **Defaults:** `model=nvidia/llama-3.3-nemotron-super-49b-v1.5`, `base_url=https://integrate.api.nvidia.com/v1`
- **Notes:** Requires `pip install langchain-nvidia-ai-endpoints`; falls back to rule-based when not configured.

## Engine-Specific Integration

### Cortex (LLM Reasoning Engine)

**File:** `src/ordinis/engines/cortex/core/engine.py`

**AI Capabilities:**
- Strategy hypothesis generation based on market context
- Code analysis and review with AI insights
- Research synthesis from multiple sources
- Output review and recommendations

**Models:**
- Llama 3.3 Nemotron Super 49B v1.5 for code analysis (default NVIDIA path)
- NV-Embed-QA for embeddings

**Example:**
```python
from ordinis.engines.cortex import CortexEngine

cortex = CortexEngine(
    nvidia_api_key="nvapi-...",
    usd_code_enabled=True,
    embeddings_enabled=True
)

# Generate strategy hypothesis
hypothesis = cortex.generate_hypothesis(
    market_context={"regime": "trending", "volatility": "low"},
    constraints={"max_position_pct": 0.10}
)

# Analyze strategy code
analysis = cortex.analyze_code(code_string, "review")
print(analysis.content['llm_analysis'])
```

The NVIDIA provider is the default and uses `nvidia/llama-3.3-nemotron-super-49b-v1.5`.

### SignalEngine (SignalCore)

**File:** `src/ordinis/engines/signalcore/models/llm_enhanced.py`

**AI Capabilities:**
- Signal interpretation with natural language explanations
- Feature engineering suggestions
- Feature importance explanations

**Models:**
- Llama 3.1 70B for signal interpretation

**Example:**
```python
from ordinis.engines.signalcore.models import LLMEnhancedModel, RSIMeanReversionModel
from ordinis.engines.signalcore.core.model import ModelConfig

# Create base model
config = ModelConfig(
    model_id="rsi-model",
    model_type="mean_reversion",
    parameters={"rsi_period": 14}
)
base_model = RSIMeanReversionModel(config)

# Enhance with LLM
enhanced = LLMEnhancedModel(
    base_model=base_model,
    nvidia_api_key="nvapi-...",
    llm_enabled=True
)

# Generate signal with AI interpretation
signal = enhanced.generate(data, timestamp)
print(signal.metadata['llm_interpretation'])
```

### RiskEngine (RiskGuard)

**File:** `src/ordinis/engines/riskguard/core/llm_enhanced.py`

**AI Capabilities:**
- Trade evaluation explanations
- Risk scenario analysis
- Rule optimization suggestions
- Plain language rule explanations

**Models:**
- Llama 3.1 70B for risk analysis

**Example:**
```python
from ordinis.engines.riskguard import LLMEnhancedRiskGuard, RiskGuardEngine, STANDARD_RISK_RULES

# Create AI-enhanced risk guard
base_engine = RiskGuardEngine(rules=STANDARD_RISK_RULES.copy())
riskguard = LLMEnhancedRiskGuard(
    base_engine=base_engine,
    nvidia_api_key="nvapi-...",
    llm_enabled=True
)

# Evaluate trade with AI explanation
passed, results, adjusted_signal = riskguard.evaluate_signal(
    signal, proposed_trade, portfolio
)

if adjusted_signal:
    print(adjusted_signal.metadata['risk_explanation'])
```

### AnalyticsEngine (ProofBench)

**File:** `src/ordinis/engines/proofbench/analytics/llm_enhanced.py`

**AI Capabilities:**
- Performance narration with insights
- Multi-strategy comparison
- Optimization suggestions by focus area
- Trade pattern analysis
- Metric explanations

**Models:**
- Llama 3.1 70B for performance analysis

**Example:**
```python
from ordinis.engines.proofbench import LLMPerformanceNarrator

narrator = LLMPerformanceNarrator(nvidia_api_key="nvapi-...")

# Narrate backtest results
narration = narrator.narrate_results(simulation_results)
print(narration['narration'])

# Compare multiple strategies
comparison = narrator.compare_results([
    ("Strategy A", results_a),
    ("Strategy B", results_b),
])
print(comparison['comparison'])

# Get optimization suggestions
suggestions = narrator.suggest_optimizations(
    results, focus="returns"
)
```

## Setup and Configuration

### 1. Get NVIDIA API Key

Visit https://build.nvidia.com/ and create a free account to get your API key.

### 2. Install Dependencies

```bash
pip install langchain-nvidia-ai-endpoints langchain openai
```

Or install the AI extras:

```bash
pip install -e ".[ai]"
```

### 3. Configure API Key

**Option 1: Environment Variable (Recommended)**
```bash
export NVIDIA_API_KEY='nvapi-...'
```

**Option 2: Direct Pass to Engine**
```python
engine = CortexEngine(nvidia_api_key="nvapi-...")
```

### 4. Enable LLM Features

Each engine requires explicit enabling:

```python
# Cortex
cortex = CortexEngine(
    nvidia_api_key=api_key,
    usd_code_enabled=True,      # Enable code analysis
    embeddings_enabled=True      # Enable embeddings
)

# SignalEngine (SignalCore)
enhanced = LLMEnhancedModel(
    base_model=model,
    nvidia_api_key=api_key,
    llm_enabled=True             # Enable signal interpretation
)

# RiskEngine (RiskGuard)
riskguard = LLMEnhancedRiskGuard(
    base_engine=engine,
    nvidia_api_key=api_key,
    llm_enabled=True             # Enable risk explanations
)

# AnalyticsEngine (ProofBench)
narrator = LLMPerformanceNarrator(
    nvidia_api_key=api_key       # Auto-enables when key provided
)
```

## Features and Capabilities

### Graceful Fallback

All engines work without NVIDIA API keys by falling back to rule-based logic:

- **No API Key:** System uses deterministic rule-based fallbacks
- **With API Key:** Full AI-powered insights and explanations
- **API Errors:** Automatic fallback to ensure system reliability

### Temperature Settings

Different tasks use different temperature settings for optimal results:

| Engine | Temperature | Purpose |
|--------|-------------|---------|
| Cortex (405B) | 0.2 | Deterministic code analysis |
| SignalEngine (SignalCore) | 0.3 | Balanced signal interpretation |
| RiskEngine (RiskGuard) | 0.3-0.4 | Risk scenario creativity |
| AnalyticsEngine (ProofBench) | 0.4 | Creative performance insights |

### Token Limits

Models are configured with appropriate token limits:

- **Cortex 405B:** 2048 tokens (deep analysis)
- **SignalEngine (SignalCore) 70B:** 512 tokens (concise interpretation)
- **RiskEngine (RiskGuard) 70B:** 512-1024 tokens (risk explanations)
- **AnalyticsEngine (ProofBench) 70B:** 1024 tokens (performance narration)

## Testing

### Unit Tests

All AI integrations have comprehensive test coverage:

```bash
# Test specific engine
pytest tests/test_engines/test_cortex/test_engine.py -v
pytest tests/test_engines/test_signalcore/test_llm_enhanced.py -v
pytest tests/test_engines/test_riskguard/test_llm_enhanced.py -v
pytest tests/test_engines/test_proofbench/test_llm_enhanced.py -v

# Test all
pytest -v
```

### Test Coverage

- **Total Tests:** 238 (all passing)
- **Overall Coverage:** 71.78%
- **LLM-Specific Coverage:**
  - Cortex: 67.39%
  - SignalEngine (SignalCore): 52.03%
  - RiskEngine (RiskGuard): 57.86%
  - AnalyticsEngine (ProofBench): 74.07%

### Integration Tests

Run the complete workflow example:

```bash
export NVIDIA_API_KEY='nvapi-...'
python docs/examples/integrated_nvidia_example.py
```

## Examples

### Individual Engine Examples

Each engine has a dedicated example file:

- **Cortex:** `docs/examples/cortex_nvidia_example.py`
- **RiskGuard:** `docs/examples/riskguard_nvidia_example.py`
- **ProofBench:** `docs/examples/proofbench_nvidia_example.py`

### Complete Workflow

**Integrated Example:** `docs/examples/integrated_nvidia_example.py`

Demonstrates all engines working together in a complete trading workflow.

## Performance Considerations

### API Rate Limits

NVIDIA API has rate limits. For production:

1. Implement request caching
2. Use batch processing where possible
3. Monitor API usage
4. Implement exponential backoff

### Cost Optimization

To minimize API costs:

1. **Cache Results:** Store AI explanations for repeated queries
2. **Selective Enhancement:** Only use AI where most valuable
3. **Batch Requests:** Group similar requests together
4. **Fallback First:** Use rule-based when sufficient

### Response Times

Typical response times with NVIDIA API:

- **Code Analysis (405B):** 3-5 seconds
- **Signal Interpretation (70B):** 1-2 seconds
- **Risk Explanation (70B):** 1-2 seconds
- **Performance Narration (70B):** 2-3 seconds

## Best Practices

### 1. API Key Security

```python
#  Don't hardcode API keys
api_key = "nvapi-abc123..."

#  Use environment variables
import os
api_key = os.getenv("NVIDIA_API_KEY")
```

### 2. Error Handling

All engines handle API errors gracefully:

```python
# No need for try/catch - engines handle errors internally
signal = llm_model.generate(data, timestamp)

# Check if AI was used
if 'llm_interpretation' in signal.metadata:
    print("AI interpretation available")
else:
    print("Using rule-based fallback")
```

### 3. Monitoring AI Usage

Track when AI is used vs. fallback:

```python
# Cortex tracks model usage
outputs = cortex.get_outputs()
for output in outputs:
    print(f"Model used: {output.model_used}")

# Check signal metadata
if signal.metadata.get('llm_model'):
    print(f"AI model: {signal.metadata['llm_model']}")
```

### 4. Progressive Enhancement

Start without AI, add progressively:

```python
# Phase 1: Rule-based (no API key)
engine = CortexEngine()

# Phase 2: Add AI when ready
engine = CortexEngine(
    nvidia_api_key=api_key,
    usd_code_enabled=True
)
```

## Troubleshooting

### Common Issues

**1. "Install: pip install langchain-nvidia-ai-endpoints"**
```bash
pip install langchain-nvidia-ai-endpoints langchain
```

**2. "NVIDIA API key required"**
```bash
export NVIDIA_API_KEY='nvapi-...'
```

**3. API timeout errors**
- Check internet connection
- Verify API key is valid
- System falls back to rule-based automatically

**4. Unexpected fallback to rule-based**
- Verify `llm_enabled=True` was set
- Check API key is set correctly
- Review logs for API errors

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Architecture

### Wrapper Pattern

All AI integrations use a wrapper pattern:

```
Base Engine → AI-Enhanced Wrapper → Same Interface
```

This ensures:
- **Backward Compatibility:** Drop-in replacement
- **Graceful Degradation:** Works without AI
- **Easy Testing:** Can test with/without AI

### Lazy Initialization

LLM clients are initialized only when needed:

```python
class LLMEnhancedModel:
    def __init__(self, ...):
        self._llm_client = None  # Not initialized

    def _init_llm(self):
        # Only called when needed
        if self._llm_client is None:
            self._llm_client = ChatNVIDIA(...)
```

This reduces:
- Startup time
- Memory usage
- Unnecessary API calls

## Future Enhancements

Potential improvements:

1. **Fine-tuning:** Custom models for specific strategies
2. **Streaming:** Real-time token streaming for long responses
3. **Caching:** Redis cache for repeated queries
4. **A/B Testing:** Compare AI vs. rule-based performance
5. **Multi-model:** Support for other LLM providers

## Support

For issues or questions:

1. Check examples in `docs/examples/`
2. Review test files for usage patterns
3. Open GitHub issue with details
4. Visit NVIDIA documentation: https://build.nvidia.com/

## License

AI integration uses the same license as the main project.

---

**Last Updated:** 2025-11-29
**Version:** 1.0.0
**Status:** Production Ready
