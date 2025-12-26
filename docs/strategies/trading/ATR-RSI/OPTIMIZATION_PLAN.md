# ATR-RSI Systematic Optimization & Backtesting Plan

**Strategy:** ATR-Optimized RSI Mean Reversion
**Target:** CAGR > S&P 500 + 15% (Risk-Adjusted)
**Date:** 2025-12-23

---

## 1. Executive Summary

This document outlines the comprehensive optimization and backtesting strategy for the ATR-RSI algorithm. The objective is to scientifically derive a parameter set that consistently outperforms the S&P 500 by a significant margin (15%+) over a 5-10 year horizon, using high-granularity data (30s bars) across a universe of 30 Mid/Small-Cap stocks.

## 2. Assumptions & Constraints

### 2.1 Strategy Logic (from Strategy Card)
- **Type:** Mean Reversion.
- **Entry:** RSI < `rsi_oversold` (filtered by Regime & Volume).
- **Exit:** RSI > `rsi_exit` OR Price hits ATR-based Stop/Target.
- **Risk:** Position sizing capped (default 3%), max concurrent positions (10).

### 2.2 Data Requirements
- **Source:** Massive (Polygon.io).
- **Universe:** 30 Equities (15 Mid-Cap, 15 Small-Cap).
- **Granularity:** 30-second aggregated bars (captures microstructure noise vs. signal).
- **Horizon:** 2015-2025 (covering Volmageddon 2018, Covid 2020, 2022 Bear).
- **Adjustments:** Splits and dividends must be adjusted.

## 3. Tunable Parameter Space

We will optimize the following hyperparameters using Bayesian Optimization (TPE):

| Parameter | Range | Type | Description |
|-----------|-------|------|-------------|
| `rsi_period` | 5 - 40 | Integer | Lookback for RSI calculation. |
| `rsi_oversold` | 15 - 45 | Integer | Threshold for entry signal. |
| `rsi_exit` | 45 - 80 | Integer | Threshold for mean-reversion exit. |
| `atr_period` | 5 - 60 | Integer | Lookback for volatility normalization. |
| `atr_stop_mult`| 0.5 - 5.0 | Float | Width of stop loss in ATR units. |
| `atr_tp_mult` | 1.0 - 10.0| Float | Width of take profit in ATR units. |
| `atr_scale` | 1.0 - 50.0| Float | Scaling factor for price tiers (Mid vs Small cap). |
| `regime_sma` | 50 - 300 | Integer | Lookback for regime filter (Trend definition). |
| `volume_filter`| True/False| Categorical | Require volume spike on entry? |

## 4. Optimization Methodology

### 4.1 Framework: Walk-Forward Optimization (WFO)
To prevent overfitting, we will NOT use a simple train/test split. We will use **Anchored Walk-Forward Analysis**:

1.  **Train Window:** 2 Years.
2.  **Test Window:** 6 Months.
3.  **Step:** Move forward 6 months.

**Cycle:**
- Optimize on 2015-2016 -> Test on H1 2017.
- Optimize on 2015-2017 -> Test on H2 2017.
- ...and so on.

### 4.2 Search Algorithm: Tree-structured Parzen Estimator (TPE)
We use **Optuna** for Bayesian optimization. TPE is superior to Grid Search (too slow) and Random Search (less efficient) as it models the probability of a parameter set improving the objective function (`P(y|x)`).

### 4.3 Objective Function
We maximize a custom **Stability Score** rather than raw CAGR to ensure robustness:

$$ Score = \text{CAGR} \times \frac{\text{Sortino Ratio}}{|\text{Max Drawdown}|} $$

This penalizes volatility and deep drawdowns heavily.

## 5. Machine Learning Integration

### 5.1 Regime Detection (Pre-Filtering)
Instead of hardcoded filters, we treat Regime Detection as a supervised classification problem (if labels exist) or unsupervised clustering (K-Means on Volatility/Trend).
- **Input:** VIX, Sector Momentum, ADX.
- **Action:** The optimizer selects whether to trade Long-Only, Long/Short, or Cash based on the regime.

### 5.2 Overfitting Safeguards
- **Parameter Stability check:** If the top 10% of parameter sets are clustered together, the solution is robust. If they are scattered, the model is overfitting noise.
- **Deflated Sharpe Ratio (DSR):** We will calculate DSR to adjust for the number of trials (multiple testing bias).

## 6. Execution Plan (Code Implementation)

1.  **Data Ingestion:** Script to fetch/cache 30 stocks from Massive (Polygon).
2.  **Vectorized Backtest:** Use `pandas`/`numpy` for high-speed simulation of the 30s bars (looping is too slow).
3.  **Optimizer Loop:** Implement the WFO loop wrapping the vector backtest.
4.  **Reporting:** Generate a "Tearsheet" comparing Strategy vs. SPY.

## 7. Universe Selection (Proposed)

**Mid-Cap (Growth/Vol):**
DKNG, ROKU, PLTR, NET, DDOG, ZS, TWLO, SQ, U, RBLX, SOFI, AFRM, HOOD, COIN, SNOW

**Small-Cap (High Beta):**
RDFN, OPEN, LC, UPST, FUBO, SPCE, NKLA, QS, LAZR, MVIS, CLOV, WISH, SENS, GME, AMC

---

*Prepared by Ordinis Quantitative Research*
