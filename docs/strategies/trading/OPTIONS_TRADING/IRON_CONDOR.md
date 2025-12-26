# Iron Condor Strategy

Market-neutral, defined-risk with explicit regime gates and adjustment ladder.

---

## Setup
- Structure: Short OTM call (delta 0.12–0.18) + long further OTM call; short OTM put (delta 0.12–0.18) + long further OTM put.
- Width: 3–5 strikes per side; 30–45 DTE; target credit 30–40% of max risk.
- Underlying: High liquidity, low event risk; IV rank 45–75.
- Allocation: Position risk ≤1% of portfolio; max 3 concurrent condors; net short vega from condors ≤20% of portfolio vega.

## Entry (all required)
1. Regime: 10d realized vol < 20d IV; price inside 20d Bollinger Bands; no earnings (single name) or major macro (index) within 7 trading days.
2. Skew/placement: Favor richer skew (often put side). Keep short strikes ≥1.2× ATR(14) from spot.
3. Vol filter: IV rank 45–75; skip if VIX > 2σ above 1y mean.
4. Liquidity: Bid/ask per wing ≤ $0.10 (index ETFs) or ≤ $0.20 (liquid stocks); OI ≥ 1,000 at shorts.

## Management / Playbook
- Profit: Close at 50–60% of max profit or 21 DTE.
- Quick win exit: If premium decays to 40% of entry within 5 days and both short deltas <0.12, close.
- Defense ladder:
  - Short delta hits 0.25: roll tested side out-in-time (keep width), re-center delta ~0.15; maintain defined risk.
  - Price tags short strike OR wing value >170% of entry: roll whole condor out 21–30 DTE, re-center strikes; if IV expanded materially, consider widening safe side to keep credit/risk >25%.
  - Gap/event risk emerges: close entire structure; do not hold through surprise events.

## Risk Controls
- Position risk = max loss of condor; cap 1% per trade and aggregate options risk ≤10% of portfolio.
- Event blackout: No entries inside 7 days of earnings (single names) or within 3 days of CPI/FOMC (indexes). Skip week of OPEX if liquidity/volatility degraded.

---

## Deployment Notes
- Monitoring: Alert when any short delta >0.25, value >170% of entry, or 21 DTE.
- Sizing: contracts = floor((0.01 * portfolio_value) / max_loss_per_condor).
- Skew tuning: If downside skew rich, move put side further OTM and take slightly closer calls to balance credit.

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
