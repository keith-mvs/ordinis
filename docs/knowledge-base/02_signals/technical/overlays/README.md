# Overlay Indicators

## Overview

Overlay indicators are plotted directly on the price chart, providing visual representation of trend direction, dynamic support/resistance levels, and volatility bands.

---

## Indicators in This Category

| File | Indicator | Description |
|------|-----------|-------------|
| [moving_averages.md](moving_averages.md) | MA Family | SMA, EMA, WMA, Hull MA, KAMA, VWAP |
| [bollinger_bands.md](bollinger_bands.md) | Bollinger Bands | Volatility-based bands around MA |
| [keltner_channels.md](keltner_channels.md) | Keltner Channels | ATR-based price channels |
| [envelopes.md](envelopes.md) | Price Envelopes | Fixed percentage bands |

---

## Common Characteristics

1. **Price-Relative**: Values move with price, providing dynamic levels
2. **Trend Identification**: Help determine trend direction and strength
3. **Support/Resistance**: Act as dynamic S/R levels
4. **Visual Clarity**: Easy to interpret on price chart

---

## Typical Usage Patterns

### Trend Following
```python
UPTREND = price > EMA(50) AND EMA(20) > EMA(50)
ENTRY_PULLBACK = UPTREND AND price touches EMA(20)
```

### Mean Reversion
```python
OVERSOLD = price < lower_bollinger_band
ENTRY = OVERSOLD AND price > price[1]  # Turning up
```

### Breakout Confirmation
```python
BREAKOUT = price > upper_band AND volume_surge
```

---

## Best Practices

1. **Use multiple MAs** for trend confirmation
2. **Combine with oscillators** for timing
3. **Adjust periods** based on trading timeframe
4. **Respect the bands** but don't fade strong trends
