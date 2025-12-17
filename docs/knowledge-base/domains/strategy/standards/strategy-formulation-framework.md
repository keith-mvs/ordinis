# Strategy Formulation Framework

## Purpose

Provide a repeatable path from idea to validated strategy spec.

## Process

1) Hypothesis
- Define edge and economic intuition; target instruments/universe/timeframes; expected regime fit.

2) Signals & Features
- List inputs (prices, fundamentals, alt data); feature engineering plan; label/target definition.
- Latency/freshness requirements; handling of missing/stale data.

3) Risk & Sizing
- Position sizing rules; caps (symbol/sector/book); DD and limit settings; stop/target logic.
- Correlation/overlap with existing books; hedging plan if needed.

4) Execution Assumptions
- Order types, venues, routing; expected spreads/impact; slippage/commission budgets.
- Latency constraints; failure/retry behavior.

5) Backtest Design
- Data sets and manifes; point-in-time requirement; universe rules (include delistings).
- Split plan (train/val/test or walk-forward); regimes to test; costs model to use.
- Acceptance criteria (align with backtesting requirements).

6) Robustness & Validation
- Parameter sensitivity, regime splits, Monte Carlo/shuffle tests.
- Stress scenarios (gaps, vol spikes, data holes).

7) Monitoring & Ops
- Live metrics to watch: PnL, DD, hit rate, turnover, slippage vs budget, factor drift.
- Alerts and kill-switch thresholds; who owns and responds.
- Runbooks referenced (live-trading-workflows, version control, retirement criteria).

8) Deliverables

- Strategy spec (this doc), risk limits, backtest report, data manifest, code/config versions.
- Plan for rollout (shadow/AB), review dates, and decay/health thresholds.
