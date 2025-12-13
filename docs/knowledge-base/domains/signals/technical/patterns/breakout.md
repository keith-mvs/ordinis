# Breakout Detection

## Overview

Breakouts occur when price escapes a well-defined range, consolidation, or pattern (flags, triangles) and is confirmed by expanding volume. They can fail frequently without confirmation, so combine with volume and higher-timeframe trend filters.

---

## Signal Heuristics

- Range high/low based on lookback window (default 20 bars)
- Close above range high ⇒ bullish breakout; close below range low ⇒ bearish breakout
- Volume confirmation: current volume must exceed recent average (default 1.5×)
- Optional stop level: opposite side of the range

---

## Implementation

```python
import pandas as pd
from ordinis.analysis.technical.patterns import BreakoutDetector

detector = BreakoutDetector()

# DataFrame must include high, low, close, volume
signal = detector.detect_breakout(
    high=data["high"],
    low=data["low"],
    close=data["close"],
    volume=data["volume"],
    lookback=20,
    volume_multiplier=1.5,
)

if signal.type != "none":
    print(f"{signal.type} breakout at {signal.price:.2f} (vol_confirm={signal.volume_confirmed})")
```

**Best Practices**
- Wait for close above/below the level, not intrabar spikes
- Require volume confirmation; skip low-liquidity names
- Align with higher timeframe trend; fade breakouts against strong trends with caution
- Set stops just inside the broken range; take partial profits on quick extensions
