# Kalman Trend Signal

## Purpose

Estimate underlying trend versus noise using a Kalman filter; emit a trend strength signal for trend-follow or mean-revert logic.

## Inputs

- Price series (mid/close), aligned calendar/tz.
- Optional volatility proxy to adapt noise settings.

## Method (outline)

1) Model: state_t = state_{t-1} + w_t ; price_t = state_t + v_t, with process noise Q, observation noise R.
2) Tune Q/R: smaller Q = smoother trend; larger Q = more responsive.
3) Run forward-only Kalman filter; extract state (trend level) and residual.
4) Derive:
   - trend_slope (first difference of state)
   - residual_z (residual normalized by residual std)

## Outputs

- trend_level, trend_slope
- residual, residual_z
- confidence = 1 / state_variance (for weighting)

## Usage Notes

- Avoid lookahead: no smoothing with future bars; fit online.
- Re-tune Q/R across regimes; validate after volatility shifts.
- Combine with execution budgets (slippage/impact) before trading.

## Python Example

```python
import numpy as np
import pandas as pd

def kalman_trend(prices: pd.Series, q: float = 1e-5, r: float = 1e-3) -> pd.DataFrame:
    """
    1D Kalman filter for price trend.
    q: process noise (higher = more responsive)
    r: observation noise (higher = smoother)
    """
    x = 0.0   # state estimate (trend level)
    p = 1.0   # state covariance
    trend = []
    resid = []
    var = []
    for z in prices:
        # Predict
        x_pred = x
        p_pred = p + q
        # Update
        k = p_pred / (p_pred + r)
        x = x_pred + k * (z - x_pred)
        p = (1 - k) * p_pred
        trend.append(x)
        resid.append(z - x)
        var.append(p)
    df = pd.DataFrame(
        {"trend_level": trend, "residual": resid, "state_var": var},
        index=prices.index,
    )
    df["trend_slope"] = df["trend_level"].diff()
    rolling = df["residual"].rolling(100, min_periods=25)
    df["residual_z"] = (df["residual"] - rolling.mean()) / rolling.std(ddof=0)
    df["confidence"] = 1 / (df["state_var"] + 1e-9)
    return df

def load_prices(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
    prices = df["close"].ffill().dropna()
    return prices

prices = load_prices("prices.csv")
signals = kalman_trend(prices, q=1e-6, r=1e-3)

# Example: trend-follow with confidence and residual filter
trade_signal = (
    (signals["trend_slope"] > 0) &
    (signals["residual_z"].abs() < 1.0) &
    (signals["confidence"] > signals["confidence"].quantile(0.2))
)

# Parameter sweep to pick q/r (proxy objective)
def sweep_qr(prices, q_vals, r_vals):
    rows = []
    for q in q_vals:
        for r in r_vals:
            s = kalman_trend(prices, q=q, r=r)
            sharpe = np.sqrt(252) * s["trend_slope"].mean() / s["trend_slope"].std(ddof=0)
            rows.append({"q": q, "r": r, "sharpe_proxy": sharpe})
    return pd.DataFrame(rows)

grid = sweep_qr(prices, q_vals=[1e-7, 1e-6, 1e-5], r_vals=[1e-4, 1e-3, 1e-2])
print(grid.sort_values("sharpe_proxy", ascending=False).head())
```

## Ensemble Hook

Provide trend_slope and confidence to the ensemble; down-weight when state_variance is high.
