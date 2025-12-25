# Bull Put Spread Strategy

Defined-risk bullish-to-neutral with quantified gates and defense ladder.

---

## Setup
- Structure: Short put delta 0.22–0.28; long put delta ~0.08–0.12; 30–45 DTE.
- Width: 3–5 strikes; target credit 25–35% of width.
- Underlying: Up-trending, liquid options; IV rank 35–65.
- Allocation: Max risk 1% of portfolio per spread; max 4 concurrent; net short vega from spreads ≤20% of portfolio vega.

## Entry (all required)
1. Trend: 20d SMA > 50d SMA; close > 20d SMA; no lower-low in past 5 bars.
2. Support: Short strike below recent swing low or anchored VWAP; distance ≥1.0× ATR(14) from current price.
3. Vol: IV rank 35–65; skip if VIX > 2σ above 1y mean.
4. Liquidity: Bid/ask ≤ $0.10 (ETFs) or ≤ $0.20 (stocks); OI ≥ 1,000 at short strike.
5. Credit: ≥25% of width; expected pop (per platform) ≥60%.

## Management / Playbook
- Profit: Close at 50–60% max profit or 21 DTE.
- Early exit: If credit decays to <40% of entry within 5 days and short delta <0.12, close.
- Defense ladder:
  - Short put delta 0.30–0.35: roll down/out 14–21 DTE, maintain width, re-center delta ~0.22.
  - Price tags short strike OR spread value >150% of entry: roll entire spread out-in-time (same width) or close to defined loss ≤1% of portfolio.
  - Trend break: 20d < 50d or close < swing low: close position.

## Risk Controls
- Max risk per spread = defined spread width minus credit; size so per-spread risk ≤1% and aggregate options risk ≤10%.
- Event filter: No entries inside 5 trading days pre-earnings (single names) or major macro (index ETFs: CPI/FOMC within 3 days).

---

## Deployment Notes
- Monitoring: Alert on short delta >0.35, spread value >150% of entry, or 21 DTE.
- Sizing formula: contracts = floor((0.01 * portfolio_value) / max_loss_per_contract).
- Record IV rank, credit %, distance to support, and exit reason for expectancy tracking.

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
