# Options Trading Strategies

Practical option strategies for directional, neutral, and income objectives with risk controls.

---

## Overview

This guide outlines three production-ready option strategies with clear entry/exit rules, sizing,
and risk management. Each uses liquid underlyings (large-cap equities/ETFs), 30–60 DTE options,
and position sizing capped by portfolio risk budgets.

---

## 1. Covered Call (Income With Long Delta)

**Intent:** Generate premium while holding or accumulating stock; harvest theta in flat to mildly
bullish regimes.

**Setup**
- Underlying: Liquid large-cap/ETF, IV rank 30–70.
- Position: Long 100 shares per contract; sell 1 OTM call (delta 0.25–0.35), 30–45 DTE.
- Max allocation: 10% of portfolio per underlying; max 3 concurrent underlyings.

**Entry**
1. Trend filter: 20d SMA > 50d SMA; daily RSI(14) between 45 and 70.
2. IV filter: IV rank 30–70 to balance premium and assignment risk.
3. Execute: Buy/hold shares; sell OTM call at target delta.

**Management**
- Profit target: Buy-to-close at 50% of collected credit or 21 DTE (whichever first).
- Roll: If price approaches strike and trend still up, roll out-in-time to maintain delta ~0.25.
- Exit: Close stock and call if 20d SMA crosses below 50d SMA or price gaps down >5% intraday.

**Risk Controls**
- Max loss per position: 1% of portfolio (stock stop 8–10% below entry).
- Assignment: Acceptable; if assigned, reset by selling cash-secured puts to re-enter stock.

---

## 2. Bull Put Spread (Defined-Risk Long Delta)

**Intent:** Express bullish-to-neutral view with capped downside and favorable theta.

**Setup**
- Structure: Short put (delta ~0.25) and buy further OTM put (delta ~0.10), 30–45 DTE.
- Width: 3–5 strikes; target 25–35% max profit on risk.
- Underlying: Up-trending, liquid options; IV rank 30–60.
- Max allocation: 1% of portfolio risk per spread; max 4 concurrent spreads.

**Entry**
1. Trend filter: 20d SMA > 50d SMA; close above 20d SMA on entry day.
2. Support check: Short strike below recent swing low or below anchored VWAP.
3. Credit filter: Minimum credit = 25% of spread width.

**Management**
- Profit target: Close at 50–60% of max profit or 21 DTE.
- Defense: If price breaches short strike or delta of short put >0.35, roll down/out to
  re-center delta and maintain defined risk.

**Risk Controls**
- Position risk = max loss of spread; size so total risk ≤1% of portfolio.
- No new entries if VIX > 2σ above 1y mean (vol shock filter).

---

## 3. Iron Condor (Market Neutral Income)

**Intent:** Collect premium in range-bound regimes with defined risk on both tails.

**Setup**
- Structure: Short OTM call (delta 0.15–0.20) + long further OTM call; short OTM put
  (delta 0.15–0.20) + long further OTM put.
- Width: 3–5 strikes each side; 30–45 DTE; target 30–40% max profit on risk.
- Underlying: High liquidity, low event risk; IV rank 40–70.
- Max allocation: 1% of portfolio risk per condor; max 3 concurrent.

**Entry**
1. Regime: Realized vol < implied vol; price inside 20d Bollinger Bands; no major earnings/FOMC
   within 10 days.
2. Skew check: Favor put side if downside skew rich; shift strikes to balance credit.

**Management**
- Profit target: Close at 50–60% of max profit or 21 DTE.
- Wing adjustment: If price tags either short strike or side delta >0.35, roll tested side
  out-in-time and re-center; keep total risk defined.
- Early exit: Close entire condor if IV crush achieved and profit target hit quickly.

**Risk Controls**
- Position risk = max loss of condor; cap at 1% of portfolio.
- Halt entries during earnings week for single-name underlyings; for index ETFs, avoid major
  macro events (CPI/FOMC) in next 3 trading days.

---

## Deployment Notes

- Liquidity: Use underlyings with tight spreads (<$0.10 on liquid ETFs, <$0.20 on large caps) and
  ≥500 contracts open interest at chosen strikes.
- Data/monitoring: Track position greeks daily; alert if short strike delta >0.35 or spread value
  <40% of initial credit (profit target) or >150% (defense trigger).
- Sizing: Enforce per-position risk caps; aggregate options risk budget ≤10% of portfolio.
- Automation: Encode filters (trend, IV rank, event calendar), strike selection by target delta,
  and profit/defense rules in the execution engine.

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-01-01"
last_updated: "2025-01-01"
status: "draft"
```

---

**END OF DOCUMENT**
