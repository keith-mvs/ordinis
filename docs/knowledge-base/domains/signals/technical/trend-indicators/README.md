# Trend Indicators

## Overview

Trend indicators measure the presence, direction, and strength of price trends. They help determine whether to use trend-following or mean-reversion strategies.

---

## Indicators in This Category

| File | Indicator | Measures | Key Level |
|------|-----------|----------|-----------|
| [adx_dmi.md](adx_dmi.md) | ADX/DMI | Trend strength + direction | ADX > 25 |
| [parabolic_sar.md](parabolic_sar.md) | Parabolic SAR | Trend + trailing stops | Price vs SAR |
| [aroon.md](aroon.md) | Aroon | Time since high/low | >70 trending |

---

## Key Concept: Trending vs Ranging

```python
# ADX-based regime detection
if ADX(14) > 25:
    regime = "TRENDING"
    use_trend_following_strategies()
else:
    regime = "RANGING"
    use_mean_reversion_strategies()
```

---

## Common Applications

1. **Strategy Selection**: Trend/range determines indicator choice
2. **Filter Signals**: Only trade momentum signals in trends
3. **Exit Signals**: Trend weakening suggests profit-taking
4. **Position Sizing**: Larger in strong trends

---

## Best Practices

1. **Confirm with price action**: Indicators lag price
2. **Use multiple timeframes**: Align trend direction
3. **Watch for exhaustion**: High ADX can precede reversals
4. **Combine indicators**: ADX + DI for complete picture
