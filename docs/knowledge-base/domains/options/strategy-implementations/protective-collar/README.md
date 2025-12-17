# Protective Collar Strategy

## Purpose

Cap downside on a long stock position by pairing a protective put with a covered call to offset hedge cost.

## Setup

- Long 100 shares.
- Long 1 put (protection) at/below ATM.
- Short 1 call (income) at/above ATM.
- Align expiries; pick strikes per risk/return and cost targets.

## Payoff Profile

- Downside limited below put strike (minus net premium).
- Upside capped at call strike (plus net premium).
- Net cost can be low/zero if call premium offsets put.

## Greeks (approx)

- Delta: stock (+), put (-), call (+); net < 1 and capped.
- Vega: put +, call -; near neutral depending on strikes.
- Theta: call income offsets put decay; near neutral.

## Python Example (collar payoff)

```python
def collar_pnl(stock_price, stock_entry, put_k, put_px, call_k, call_px):
    stock_leg = stock_price - stock_entry
    put_leg = max(0, put_k - stock_price) - put_px
    call_leg = -max(0, stock_price - call_k) + call_px
    return stock_leg + put_leg + call_leg

entry = 100
put_k, put_px = 95, 2.0
call_k, call_px = 110, 2.2
prices = [80, 90, 100, 110, 120]
pnl = [collar_pnl(p, entry, put_k, put_px, call_k, call_px) for p in prices]
print("P&L:", list(zip(prices, pnl)))
```

## Management Notes

- If price nears call strike: let assign, or roll up/out to keep upside.
- If price drops: keep/roll put; roll call lower to harvest premium (raises cap risk).
- Avoid short calls into ex-dividend if early assignment risk matters.
