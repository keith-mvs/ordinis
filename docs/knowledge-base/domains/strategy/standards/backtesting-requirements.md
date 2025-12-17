# Backtesting Requirements

## Purpose

Set minimum standards so results are reproducible, realistic, and decision-grade before any live exposure.

## Data Requirements

- Point-in-time prices/factors; include delisted names; consistent calendars/timezones.
- Corporate actions applied (splits/dividends); adjust volume with splits.
- Record data manifest (source, range, checksum); pin data snapshots per run.

## Methodology

- Splits: train/validate/test or walk-forward; never tune on test.
- Use realistic latencies and bar alignment; no lookahead/leakage.
- Execution model: slippage (spread + impact), commissions, partial fills, rejects.
- Position sizing follows risk rules used in live (caps, concentration, cash/margin).

## Costs & Slippage (defaults to override)

- Spread: half-spread per side (e.g., 10 bps); impact: sqrt(order/ADV) with cap.
- Commissions: per-share or bps; min ticket cost; borrow/financing if shorting.
- Reject/latency modeling for thin names or venue outages.

## Validation Criteria (fail if breached)

- Sample: ≥3 years and ≥100 trades (per book) unless justified.
- Risk/return: Sharpe ≥ target, max DD ≤ limit, profit factor ≥ 1.5 (set per strategy class).
- Robustness: OOS performance degradation ≤ 30% vs IS; passes walk-forward or k-fold time splits.
- Profitable after modeled costs; sensitivity stable across small parameter changes.

## Robustness Tests

- Parameter sweeps: small perturbations should not flip sign.
- Regime tests: bull/bear/sideways, high/low vol.
- Monte Carlo trade reordering to assess path risk and drawdown tails.
- Stress scenarios: gaps, data holes, widened spreads.

## Reporting (per run)

- Version info: code tag, config, data manifest, execution model settings.
- Metrics: return, Sharpe/Sortino, DD (depth/duration), PF, hit rate, turnover, costs.
- Plots: equity and drawdown, rolling Sharpe, exposure, slippage vs budget.
- Notes: anomalies, data issues, parameter choices, and next actions.
