# Performance Attribution

## Purpose

Explain what is driving returns, detect alpha decay, and compare against appropriate benchmarks so you can act (tune, pause, retire).

## Data Requirements

- Clean, point-in-time prices and factors; include delisted names.
- Same calendar/clock across portfolio, benchmark, and factors.
- Gross vs net PnL separated; costs/slippage modeled explicitly.

## Factor Attribution (practical)

1) Choose factors relevant to the universe (e.g., mkt, size, value, momentum, quality).
2) Regress excess returns on factor returns; use rolling windows to see drift.
3) Inspect betas, alpha, t-stats, R²; watch for beta drift or alpha collapse.

Minimal example (Python):
```python
import pandas as pd
import statsmodels.api as sm

df = pd.merge(portfolio, factors, left_index=True, right_index=True).dropna()
Y = df['portfolio_return'] - df['risk_free']
X = sm.add_constant(df[['mkt','smb','hml','mom','quality']])
model = sm.OLS(Y, X).fit()
print(model.params, model.tvalues, model.rsquared_adj)
```

Actions:
- Rising factor beta + falling alpha: strategy drift; revisit signals/sizing.
- Low R² with noisy alpha: likely unstable; increase sample or simplify model.

## Alpha Decay

- Track rolling alpha and half-life; alarm if half-life < threshold (e.g., 60 trading days).
- Test by regime: bull/bear/sideways; expect alpha stability across regimes or document why not.
- Watch decay after slippage/cost updates; adjust execution or reduce size.

## Benchmarking

- Pick benchmark that matches universe/exposure (index or custom factor basket).
- Monitor tracking error, info ratio, active drawdown vs benchmark.
- Revisit benchmark if universe or style mix changes.

## Visuals to keep

- Rolling betas and alpha with confidence bands.
- Cumulative active return vs benchmark; active drawdown.
- Factor contribution bars (positive/negative).

## Pitfalls

- Look-ahead or survivorship bias in factors/universe.
- Mixing gross/net returns; inconsistent cost assumptions.
- Overfitting factor sets; constantly swapping factors to chase R².
- Using the wrong benchmark (e.g., large-cap index for small-cap book).

## Cadence and Ownership

- Weekly: update attribution, alpha decay, and benchmark metrics.
- Monthly: regime-split analysis; review factor drift.
- Owner: strategy PM/quant; Risk reviews alerts on decay/TE breaches.

## Exit/Adjust Triggers

- Alpha half-life below threshold for N weeks.
- Info ratio below target while costs/slippage are rising.
- Persistent factor drift into unintended styles.
