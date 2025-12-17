# Married Put Strategy

## Purpose

Hedge downside risk on a long stock position by buying a 1:1 put, capping losses while retaining upside.

## Setup

- Long 100 shares of the underlying.
- Long 1 put option (same expiry); strike typically ATM/near ATM.

## Payoff Profile

- Max loss ≈ put premium + (entry price - strike) if strike < entry.
- Upside: stock gains minus put premium; protected below strike.

## Greeks (approx)

- Delta: stock (+1) minus put delta (negative) → net < 1.
- Vega: positive (hedge benefits from vol up).
- Theta: negative from put decay.

## Python Example (payoff and breakeven)

```python
def married_put_pnl(stock_price, stock_entry, put_strike, put_premium):
    stock_leg = stock_price - stock_entry
    put_leg = max(0, put_strike - stock_price) - put_premium
    return stock_leg + put_leg

def married_put_breakeven(stock_entry, put_premium):
    return stock_entry + put_premium

entry = 100
put_k = 100
put_px = 2.5
prices = [80, 90, 100, 110, 120]
pnl = [married_put_pnl(p, entry, put_k, put_px) for p in prices]
print("P&L:", list(zip(prices, pnl)))
print("Breakeven:", married_put_breakeven(entry, put_px))
```

## Management Notes

- If price rallies: consider selling/rolling put to lock gains or reduce decay drag.
- If price drops: keep/roll hedge or exit stock+put; reassess strike/tenor.
- Watch IV: high IV raises hedge cost; low IV reduces protection quality if vol spikes.
