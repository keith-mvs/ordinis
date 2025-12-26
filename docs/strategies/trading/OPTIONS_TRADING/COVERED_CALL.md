# Covered Call Strategy

Income-oriented with long delta, tighter triggers, and quantified exits.

---

## Setup
- Underlying: Liquid large-cap/ETF; IV rank 40–75; ATR(14)/Price < 3% (avoid high vol).
- Position: Long 100 shares per contract; sell OTM call delta 0.25–0.30; 30–45 DTE.
- Max allocation: 8% of portfolio per underlying; max 2 underlyings; net short call vega < 20% of total vega.

## Entry (must pass all)
1. Trend: 20d SMA > 50d SMA AND close > 20d SMA.
2. Momentum guard: RSI(14) 45–65; no up-day gap >2% on entry.
3. IV filter: IV rank 40–75; skip if VIX > 2σ above 1y mean.
4. Liquidity: Bid/ask ≤ $0.15 (ETF) or ≤ $0.25 (stock) and OI ≥ 1,000 at strike.

## Management / Playbook
- Profit: BTC at 50% credit or at 21 DTE, whichever first.
- Early harvest: If premium <40% of credit within 5 trading days and delta <0.15, close.
- Defense ladder:
  - Short call delta 0.35: roll up/out to re-center ~0.25 delta; keep same expiration +7–14 DTE.
  - Price > short strike and delta >0.45: roll out 21–30 DTE, strike 1–2 steps higher; only if trend intact.
  - Trend break (20d < 50d) or price gap down ≥5% intraday: close stock + call.
- Assignment: Accept; if called away and trend intact, re-enter via cash-secured put delta ~0.30, 30–45 DTE.

## Risk Controls
- Max loss per position: 0.8% of portfolio (stock stop 8–10% below entry).
- Portfolio caps: Options risk budget ≤10% of portfolio; per-name exposure ≤8%.
- Event filter: No entries inside 5 trading days pre-earnings or macro (CPI/FOMC) for index ETFs.

---

## Deployment Notes
- Monitoring: Alert on short call delta >0.40, premium at 50% target, or stock stop hit.
- Sizing: Enforce share count = floor(0.08 * portfolio value / (100 * stock price)).
- Record: Log entry IV rank, credit, delta, and exits for expectancy tracking.

---

## Document Metadata

```yaml
version: "1.1.0"
created: "2025-01-01"
last_updated: "2025-01-01"
status: "draft"
```

---

**END OF DOCUMENT**
