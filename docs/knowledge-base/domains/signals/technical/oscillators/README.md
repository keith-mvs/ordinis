# Oscillator Indicators

## Overview

Oscillators are momentum indicators that fluctuate between fixed boundaries, typically used to identify overbought and oversold conditions, as well as momentum divergences.

---

## Indicators in This Category

| File | Indicator | Range | Primary Use |
|------|-----------|-------|-------------|
| [rsi.md](rsi.md) | RSI | 0-100 | Overbought/oversold, divergence |
| [stochastic.md](stochastic.md) | Stochastic | 0-100 | Mean reversion, crossovers |
| [cci.md](cci.md) | CCI | Unbounded | Trend extremes |
| [williams_r.md](williams_r.md) | Williams %R | -100 to 0 | Quick reversals |

---

## Common Characteristics

1. **Bounded Range**: Most oscillate between 0-100 or similar
2. **Mean-Reverting**: Tend to return to neutral zone
3. **Divergence Detection**: Can signal trend exhaustion
4. **Overbought/Oversold**: Extreme readings suggest reversal potential

---

## Key Concepts

### Overbought vs Oversold
- **Overbought** (RSI > 70): Price has risen "too far, too fast"
- **Oversold** (RSI < 30): Price has fallen "too far, too fast"
- **Warning**: Prices can remain extreme in strong trends

### Divergence Types
```python
# Bullish divergence (potential reversal up)
BULLISH_DIV = price.lower_low() AND oscillator.higher_low()

# Bearish divergence (potential reversal down)
BEARISH_DIV = price.higher_high() AND oscillator.lower_high()

# Hidden divergence (trend continuation)
HIDDEN_BULL_DIV = price.higher_low() AND oscillator.lower_low()
HIDDEN_BEAR_DIV = price.lower_high() AND oscillator.higher_high()
```

---

## Best Practices

1. **Use trend filter**: Only take signals in direction of higher TF trend
2. **Wait for confirmation**: Don't enter on extreme reading alone
3. **Combine oscillators**: RSI + Stochastic for confluence
4. **Adjust thresholds**: Use 80/20 in strong trends instead of 70/30
