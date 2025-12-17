# Vertical Spreads

## Purpose

Directional structures using two strikes and one expiry with defined risk/reward. Covers debit (bull call, bear put) and credit (bull put, bear call) variants.

## Components

- Two options on same underlying and expiry.
- One long, one short; width = |long strike - short strike|.
- Net debit (pay) or net credit (receive).

## Payoff Formulas

- Debit spreads: max loss = net debit; max profit = width - net debit; breakeven = long strike ± net debit (call: +, put: -).
- Credit spreads: max profit = net credit; max loss = width - net credit; breakeven = short strike ± net credit (put: -, call: +).

## Greek Profile (typical)

- Delta: modest directional; credit spreads often closer to 0 at open.
- Theta: credit spreads generally positive; debit spreads negative.
- Vega: debit spreads benefit from IV rise; credit spreads from IV fall.

## Construction Guidelines

- Pick width based on risk budget/margin; prefer stable spacing (e.g., $5 or $10).
- Debit (directional, lower IV): buy near ATM, sell OTM to offset cost.
- Credit (range/neutral, higher IV): sell nearer money; buy further OTM for protection; avoid tiny widths that are all fees.
- Align expiry to thesis; avoid unintended event risk unless desired.

## Risk Controls

- Set exits: profit target (e.g., 50–75% of max) and max loss (e.g., 1–1.5x debit).
- Monitor assignment on short leg (dividends for calls, deep ITM puts).
- Roll or close when most credit captured or thesis invalidates.

## Example (Bull Call Debit)

- Buy 1 K1 call @ $3.00, sell 1 K2 call @ $1.50 (K1 < K2).
- Net debit = $1.50; width = K2 - K1.
- Max profit = width - debit; max loss = debit.

## Implementation Notes

- Enforce strike spacing/width limits and margin checks in engine.
- Backtest with realistic fills, early exercise, and slippage.
- Standardize sizing across spread types; cap exposure per underlying/sector.

## Python Example (debit vs credit spread metrics)

```python
def debit_spread_metrics(long_k, short_k, long_px, short_px, call=True):
    width = abs(short_k - long_k)
    debit = long_px - short_px
    max_loss = debit
    max_profit = width - debit
    breakeven = long_k + debit if call else long_k - debit
    return {"width": width, "debit": debit, "max_profit": max_profit, "max_loss": max_loss, "breakeven": breakeven}

def credit_spread_metrics(short_k, long_k, short_px, long_px, call=True):
    width = abs(long_k - short_k)
    credit = short_px - long_px
    max_profit = credit
    max_loss = width - credit
    breakeven = short_k + credit if call else short_k - credit
    return {"width": width, "credit": credit, "max_profit": max_profit, "max_loss": max_loss, "breakeven": breakeven}

# Examples
bull_call = debit_spread_metrics(long_k=100, short_k=105, long_px=3.5, short_px=1.5, call=True)
bear_call = credit_spread_metrics(short_k=100, long_k=105, short_px=3.0, long_px=1.0, call=True)
print("Bull call metrics:", bull_call)
print("Bear call metrics:", bear_call)
```
