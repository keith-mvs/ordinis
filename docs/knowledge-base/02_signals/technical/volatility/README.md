# Volatility Indicators

## Overview

Volatility indicators measure the degree of price dispersion over time. They're essential for position sizing, stop placement, and identifying potential breakouts.

---

## Indicators in This Category

| File | Indicator | Measures | Use |
|------|-----------|----------|-----|
| [atr.md](atr.md) | ATR | Average True Range | Stops, position sizing |
| [implied_realized.md](implied_realized.md) | IV vs HV | Option volatility | Volatility trades |

---

## Key Concepts

### Volatility Types
- **Historical (Realized)**: Past price movements
- **Implied**: Market's expectation (from options)
- **Intraday**: Within-session volatility
- **Overnight**: Gap risk

### Volatility Regimes
```python
# ATR percentile for regime
VOL_PERCENTILE = percentile_rank(ATR, 252)

LOW_VOL = VOL_PERCENTILE < 25
NORMAL_VOL = 25 <= VOL_PERCENTILE <= 75
HIGH_VOL = VOL_PERCENTILE > 75
```

---

## Applications

1. **Position Sizing**: Inverse volatility scaling
2. **Stop Placement**: ATR-based dynamic stops
3. **Breakout Detection**: Volatility squeeze/expansion
4. **Risk Management**: Reduce size in high vol

---

## Best Practices

1. **Normalize stops by volatility**: Use ATR multiples
2. **Scale position size inversely**: Smaller in high vol
3. **Watch for expansion after contraction**: Breakout signals
4. **Compare IV to RV**: Options trading edge
