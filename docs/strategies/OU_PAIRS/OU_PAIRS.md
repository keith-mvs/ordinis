# Ornstein-Uhlenbeck Pairs Trading Strategy

---

**Title:** OU Pairs Trading
**Description:** Pairs trading with OU process parameter estimation for dynamic thresholds
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** pairs-trading, cointegration, ornstein-uhlenbeck, mean-reversion, statistical-arbitrage
**References:** Ornstein & Uhlenbeck (1930), Vidyamurthy (2004)

---

## Overview

The OU Pairs strategy identifies cointegrated asset pairs and models their spread as an Ornstein-Uhlenbeck process. This allows estimation of mean-reversion speed (half-life), enabling dynamic entry thresholds and position sizing.

## Mathematical Basis

### Cointegration

Two price series $P_A$ and $P_B$ are cointegrated if:

$$
S_t = P_A - \beta P_B \sim I(0)
$$

Where:
- $\beta$ = hedge ratio (from OLS regression)
- $S_t$ = spread (stationary)
- $I(0)$ = integrated of order zero (stationary)

### Ornstein-Uhlenbeck Process

The spread follows:

$$
dS_t = \theta(\mu - S_t)dt + \sigma dW_t
$$

Where:
- $\theta$ = mean-reversion speed
- $\mu$ = long-run mean
- $\sigma$ = process volatility
- $W_t$ = Wiener process

### Half-Life

Time for spread to revert halfway to mean:

$$
\tau_{1/2} = \frac{\ln(2)}{\theta}
$$

**Interpretation:**
- $\tau_{1/2} < 5$ days → Fast reversion, tight stops
- $\tau_{1/2} \in [5, 20]$ days → Optimal trading range
- $\tau_{1/2} > 20$ days → Slow reversion, may not be tradeable

### Parameter Estimation (OLS)

Discrete approximation:

$$
\Delta S_t = \alpha + \beta S_{t-1} + \epsilon_t
$$

Then:
- $\theta = -\beta$
- $\mu = \alpha / \theta$
- $\sigma = \text{std}(\epsilon)$

## Signal Logic

| Spread Z-Score | Action | Position |
|----------------|--------|----------|
| `z < -2.0` | **LONG SPREAD** | Buy A, Sell $\beta$ × B |
| `z > 2.0` | **SHORT SPREAD** | Sell A, Buy $\beta$ × B |
| `|z| < 0.5` | **EXIT** | Close position |
| `|z| > 4.0` | **STOP LOSS** | Close position |

### Dynamic Thresholds

Entry threshold scales with half-life:

$$
z_{\text{entry}} = 1.5 + 0.1 \times \min(\tau_{1/2}, 20)
$$

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coint_lookback` | 252 | Days for cointegration test |
| `coint_pvalue` | 0.05 | Required p-value |
| `hedge_lookback` | 60 | Days for hedge ratio estimation |
| `ou_lookback` | 60 | Days for OU parameter estimation |
| `min_halflife` | 2 | Minimum half-life filter |
| `max_halflife` | 60 | Maximum half-life filter |
| `entry_z` | 2.0 | Z-score for entry |
| `exit_z` | 0.5 | Z-score for exit |
| `stop_z` | 4.0 | Z-score for stop loss |

## Pair Selection

### Cointegration Test (Engle-Granger)

```python
from statsmodels.tsa.stattools import coint

score, pvalue, _ = coint(prices_a, prices_b)
if pvalue < 0.05:
    print("Cointegrated pair found!")
```

### Pair Discovery Function

```python
from ordinis.engines.signalcore.models.ou_pairs import find_cointegrated_pairs

pairs = find_cointegrated_pairs(
    prices_df,           # Columns = symbols
    pvalue_threshold=0.05
)

# Returns: [(sym_a, sym_b, pvalue, hedge_ratio), ...]
```

## Edge Source

1. **Statistical Arbitrage:** Exploits temporary mispricings in related assets
2. **Market Neutral:** Long/short cancels market beta
3. **Dynamic Calibration:** Half-life adapts thresholds to spread dynamics
4. **Avoids Slow Pairs:** Filters out pairs that won't revert in trading horizon

## Implementation Notes

```python
from ordinis.engines.signalcore.models import OUPairsModel
from ordinis.engines.signalcore.core.model import ModelConfig

config = ModelConfig(
    model_id="ou_pairs",
    model_type="pairs",
    parameters={
        "entry_z": 2.0,
        "min_halflife": 2,
        "max_halflife": 30,
    }
)

model = OUPairsModel(config)

# Analyze pair
stats = model.analyze_pair("AAPL", "MSFT", prices_a, prices_b)
print(f"Hedge ratio: {stats.hedge_ratio:.3f}")
print(f"Half-life: {stats.ou_params.halflife:.1f} days")
print(f"Current z-score: {stats.spread_z:.2f}")
print(f"Valid pair: {stats.is_valid}")

# Generate signal
signal = await model.generate_pair_signal(
    "AAPL", "MSFT", prices_a, prices_b, timestamp
)
```

## Example Pairs

| Pair | Sector | Typical Half-Life |
|------|--------|-------------------|
| XOM/CVX | Energy | 5-15 days |
| KO/PEP | Consumer | 8-20 days |
| JPM/BAC | Financials | 7-18 days |
| GOOGL/META | Tech | 10-25 days |

## Risk Considerations

- **Regime Breaks:** Cointegration can break permanently
- **Execution Risk:** Requires simultaneous execution of both legs
- **Margin Requirements:** Short positions require margin
- **Convergence Risk:** Spread may widen before reverting

## Performance Expectations

- **Win Rate:** 60-70%
- **Profit Factor:** 1.8-2.5
- **Sharpe Ratio:** 1.5-2.5 (market-neutral)
- **Best Conditions:** Stable correlation regimes
- **Worst Conditions:** Structural breaks, merger events

---

**File:** `src/ordinis/engines/signalcore/models/ou_pairs.py`
**Status:** ✅ Complete
