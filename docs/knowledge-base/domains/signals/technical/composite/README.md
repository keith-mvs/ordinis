# Composite Indicators

## Overview

Composite indicators combine multiple calculations or indicator types into a single signal generator. They typically provide more robust signals by filtering noise through multiple layers of smoothing or confirmation.

---

## Indicators in This Category

| File | Indicator | Components | Use Case |
|------|-----------|------------|----------|
| [macd.md](macd.md) | MACD | 2 EMAs + Signal line | Trend + momentum |
| [momentum.md](momentum.md) | ROC, Momentum, PPO | Price change calculations | Pure momentum |
| — | CompositeIndicator | Weighted/voting mix of any indicators | Consensus signal scoring |

### Quick Usage

```python
from ordinis.analysis.technical import CompositeIndicator

def rsi_signal(df):  # returns -1/0/1
    return 1 if df["rsi"].iloc[-1] < 30 else -1 if df["rsi"].iloc[-1] > 70 else 0

def ma_alignment(df):  # returns normalized score -1..1
    fast = df["close"].ewm(span=20, adjust=False).mean()
    slow = df["close"].ewm(span=50, adjust=False).mean()
    return 1.0 if fast.iloc[-1] > slow.iloc[-1] else -1.0

composite = CompositeIndicator(method="weighted_sum", weights={"rsi": 0.4, "ma": 0.6})
score = composite.combine({"rsi": rsi_signal(data), "ma": ma_alignment(data)})
print(score.signal, score.value)  # e.g., 'buy' with a weighted score
```

---

## Why Composite Indicators?

1. **Noise Reduction**: Multiple smoothing layers filter out false signals
2. **Multi-Dimensional**: Capture both trend and momentum
3. **Signal Clarity**: Clear crossover or divergence signals
4. **Versatility**: Work across multiple market conditions

---

## Common Composite Indicator Types

### Moving Average Based
- MACD (EMA difference)
- PPO (Percentage Price Oscillator)
- TRIX (Triple smoothed EMA)

### Momentum Based
- ROC (Rate of Change)
- Momentum (Price difference)
- RSI variants (Stoch RSI, Connors RSI)

### Hybrid
- KST (Know Sure Thing - weighted ROC)
- Ultimate Oscillator (weighted timeframes)

---

## Best Practices

1. **Understand components**: Know what each piece measures
2. **Use for confirmation**: Not primary entry signals
3. **Watch for divergence**: Often most reliable signal
4. **Respect the lag**: Composite indicators are slower
