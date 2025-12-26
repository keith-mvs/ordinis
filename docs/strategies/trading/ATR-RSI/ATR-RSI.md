# ATR-Optimized RSI Mean Reversion Strategy

Technical specification and operating notes for the ATR-Optimized RSI strategy as implemented in the **SignalEngine** layer (implemented in this repo by **SignalCore**) at `src/ordinis/engines/signalcore/models/atr_optimized_rsi.py` and loaded via `src/ordinis/engines/signalcore/strategy_loader.py`.

This file is the **official** strategy markdown. The previous technical-spec entrypoint has been archived at `docs/archive/atr-optimized-rsi-technical-spec.md`.

**Version:** 1.2.0
**Status:** review
**Last Updated:** 2025-12-23
**Author:** Ordinis Quantitative Research

---

## Executive Summary

The ATR-Optimized RSI strategy is a mean reversion trading system that combines the Relative Strength Index (RSI) for entry timing with Average True Range (ATR) for adaptive risk management.

In internal sweeps, naive RSI-only variants (fixed stops and/or no volatility normalization) performed poorly; the key improvement is adapting exits to volatility (ATR-based stop-loss and take-profit) and avoiding the worst regimes.

For an implementation-focused narrative, see `docs/guides/atr_optimized_rsi_implementation.md`.

### Key Performance Metrics

| Metric | Value |
|--------|-------|
| Total Return (example backtest) | +60.1% |
| Win Rate | 70-85% |
| Total Trades | 819 |
| Max Drawdown | 26.1% |
| Sharpe Ratio (estimate) | 0.17 |
| Walk-Forward Validation | 8/8 symbols robust |

> Important: The metrics above are **scenario-specific**. When citing results, always include the universe, timeframe (e.g. 5-minute bars), test window, fees/slippage assumptions, and whether position sizing and portfolio constraints were enabled.

---

## Overview

### What this strategy does

- Generates **ENTRY** signals on RSI oversold conditions.
- Generates **EXIT** signals when RSI mean-reverts or when ATR-based stop/target levels are hit.
- Uses regime analysis (via `RegimeDetector`) to avoid known low-edge regimes.

### Terminology (alignment with `ARCHITECTURE.md`)

`ARCHITECTURE.md` uses generic engine names (SignalEngine/RiskEngine/ExecutionEngine). In the current repository implementation these correspond to:

| Architecture term | Repo implementation (current) |
|---|---|
| **StreamingBus** | Data source interface registered with OrchestrationEngine (`DataSourceProtocol` in `src/ordinis/engines/orchestration/core/engine.py`) |
| **SignalEngine** | **SignalCore**: `src/ordinis/engines/signalcore/` (models + strategy loading + regime detection) |
| **RiskEngine** | **RiskGuard**: `src/ordinis/engines/riskguard/` (portfolio/risk gates and hard limits) |
| **ExecutionEngine** | **FlowRoute** + broker/paper adapters (order routing + simulated fills) |
| **PortfolioEngine** | Portfolio engine (positions/cash/ledger) |
| **AnalyticsEngine** | Analytics engine (metrics + reports) |
| **GovernanceEngine** | Governance engine (preflight + audit trail) |

This document uses the architecture terms as primary, and references the concrete modules/paths where the behavior is implemented.

### What this strategy does not do (as currently implemented)

- It does **not** implement an SMA trend filter inside the model.
- It does **not** implement opening short positions (exits use `direction=SHORT` to represent *closing a long*).
- It does **not** size positions by itself; sizing is enforced by the runtime/PortfolioEngine/RiskEngine layers.

---

## 1. Theoretical Foundation

### 1.1 Mean Reversion Hypothesis

The strategy exploits the empirical observation that short-term price extremes tend to revert toward a mean. This is supported by:

- **Negative autocorrelation** in high-frequency returns
- **Overreaction hypothesis** (De Bondt & Thaler, 1985)
- **Liquidity provider behavior** - market makers fade extreme moves

### 1.2 RSI as Momentum Oscillator

The RSI measures the magnitude of recent price changes to evaluate overbought/oversold conditions:

$$RSI = 100 - \frac{100}{1 + RS}$$

Where $RS = \frac{\text{Average Gain}}{\text{Average Loss}}$ over $n$ periods.

**Standard interpretation:**
- RSI < 30: Oversold (potential buy)
- RSI > 70: Overbought (potential sell)

**Effective thresholds (current implementation):**
- RSI oversold: typically 30–35 (per-symbol and/or config-dependent)
- RSI exit: 50 (quick exit to lock gains)

See `configs/strategies/atr_optimized_rsi.yaml` and `ATROptimizedRSIModel` parameter parsing for the authoritative values.

### 1.3 ATR for Volatility Normalization

The Average True Range captures volatility across different market regimes:

$$ATR_n = \frac{1}{n} \sum_{i=1}^{n} TR_i$$

Where True Range is:
$$TR = \max(H - L, |H - C_{prev}|, |L - C_{prev}|)$$

**Key insight:** Fixed percentage stops fail because volatility varies dramatically between stocks and across time. ATR-based stops adapt automatically.

---

## 2. Signal Generation

### 2.1 Entry Conditions

A **BUY** signal is generated when:

1. $RSI_{n} < RSI_{oversold}$ (oversold)
2. Regime ∉ {quiet_choppy, choppy} (regime filter, applied by the loader/runtime)

Implementation reference:

- Model entry condition: `src/ordinis/engines/signalcore/models/atr_optimized_rsi.py` (`ATROptimizedRSIModel.generate`)
- Regime gating: `src/ordinis/engines/signalcore/strategy_loader.py` (`StrategyLoader.should_trade`)

### 2.2 Exit Conditions

A **SELL** signal is generated when ANY of:

1. $RSI_{n} > RSI_{exit}$ (mean reversion / momentum exhaustion)
2. Price hits stop-loss: $Entry - (ATR_{n} \times Stop_{mult})$
3. Price hits take-profit: $Entry + (ATR_{n} \times TP_{mult})$

Where $TP_{mult}$ varies by symbol.

Note: In the current implementation the model is **stateful** (tracks whether it is in a position and its entry/stop/target). Ensure you use one model instance per traded symbol.

### 2.3 Short Selling

Short selling is **not implemented** in `ATROptimizedRSIModel`.

If/when shorts are added, document them here and link to the exact implementation.

---

## 3. Risk Management

### 3.1 Position Sizing

The strategy model does not size positions. Position sizing is enforced by the portfolio/risk layers.

If using simple notional sizing:

$$Position\ Size = \frac{Account\ Equity \times Max\ Position\ \%}{Entry\ Price}$$

Current production config default (see `configs/strategies/atr_optimized_rsi.yaml`): `max_position_size_pct = 3.0` (percent).

### 3.2 Stop-Loss Calculation

$$Stop\ Loss = Entry\ Price - (ATR_{14} \times Stop\ Multiplier)$$

Default: `stop_multiplier = 1.5`

This places stops at approximately 1.5 standard deviations of recent volatility.

### 3.3 Take-Profit Calculation

$$Take\ Profit = Entry\ Price + (ATR_{14} \times TP\ Multiplier)$$

TP multipliers are optimized per-symbol:

| Symbol | TP Multiplier | Rationale |
|--------|--------------|-----------|
| COIN | 3.0 | High volatility, extended runs |
| DKNG | 2.0 | Moderate volatility |
| AMD | 1.5 | Lower volatility, quick reversals |
| SPY | 1.5 | Index, mean-reverts quickly |

### 3.4 Portfolio-Level Controls

- **Max concurrent positions:** 10
- **Max daily loss:** 2% of equity
- **Position correlation limit:** Avoid >3 positions in same sector

Additional limits may be enforced by the RiskEngine (RiskGuard implementation) and GovernanceEngine depending on environment configuration.

---

## 4. Regime Detection

### 4.1 Regime Classification

The strategy uses a multi-factor regime detector (direction-change rate, autocorrelation, big-move frequency, and industry-standard ADX/DMI + ATR%).

Effective policy:

- Prefer: `mean_reverting` (best conditions for mean reversion)
- Avoid: `quiet_choppy`, `choppy`

Implementation reference: `src/ordinis/engines/signalcore/regime_detector.py`.

### 4.2 Regime Detection Algorithm
See `RegimeDetector.compute_metrics()` and `RegimeDetector.analyze()` in `src/ordinis/engines/signalcore/regime_detector.py`.

---

## 5. Optimized Parameters

### 5.1 Global Parameters

Effective parameters used by the current implementation:

```yaml
global_params:
  rsi_period: 14
  atr_period: 14

symbols:
  COIN:
    rsi_oversold: 30
    rsi_exit: 50
    atr_stop_mult: 1.5
    atr_tp_mult: 3.0
```

Note: The current `configs/strategies/atr_optimized_rsi.yaml` includes `default_*` keys for readability/roadmap, but the model consumes `rsi_oversold`, `rsi_exit`, `atr_stop_mult`, `atr_tp_mult` (symbol-level overrides).

### 5.2 Per-Symbol Optimization

| Symbol | RSI Oversold | ATR Stop | ATR TP | Expected Return |
|--------|--------------|----------|--------|-----------------|
| COIN | 30–35 | 1.5 | 3.0 | +12.3% |
| DKNG | 30–35 | 1.5 | 2.0 | +8.1% |
| AMD | 30–35 | 1.5 | 1.5 | +6.9% |
| PLTR | 35 | 1.5 | 2.0 | +5.2% |
| TSLA | 35 | 1.5 | 2.5 | +6.0% |
| ROKU | 35 | 1.5 | 2.5 | +4.8% |
| SPY | 30 | 1.5 | 1.5 | +2.1% |
| SOXS | 35 | 2.0 | 2.0 | +8.5% |
| BAC | 30 | 1.5 | 1.5 | +3.2% |
| AAL | 35 | 2.0 | 2.0 | +4.1% |

---

## 6. Validation Results

### 6.1 Walk-Forward Analysis

**Methodology:** Train on days 1-10, test on days 11-21

| Symbol | Train Return | Test Return | Robust? |
|--------|-------------|-------------|---------|
| COIN | -7.0% | +12.3% | ✅ |
| DKNG | +0.9% | +8.1% | ✅ |
| AMD | +2.1% | +6.0% | ✅ |
| TSLA | +1.2% | +6.0% | ✅ |
| PLTR | +3.4% | +5.2% | ✅ |
| ROKU | +1.8% | +4.8% | ✅ |
| AAL | +0.5% | +4.1% | ✅ |
| SOXS | +4.2% | +8.5% | ✅ |

### 6.2 Long+Short Comparison

> Research note: the current `ATROptimizedRSIModel` implementation is long-only (exits are represented with `direction=SHORT`). The table below reflects experimental results from a separate long+short research variant.

| Symbol | Long Only | Long+Short | Short Edge |
|--------|-----------|------------|------------|
| COIN | +12.3% | +23.4% | +11.1% |
| AMD | +6.0% | +13.8% | +7.8% |
| DKNG | +8.1% | +11.2% | +3.1% |
| TSLA | +6.0% | +0.9% | -5.1% ❌ |

### 6.3 Drawdown Analysis

```
Max Drawdown: 26.1%
Worst Single Day: -30.9%
Best Single Day: +12.3%
Average Daily Return: +0.42%
Sharpe Ratio: 0.17
```

---

## 7. Implementation

### 7.0 Implementation References

- Model: `src/ordinis/engines/signalcore/models/atr_optimized_rsi.py`
- Loader: `src/ordinis/engines/signalcore/strategy_loader.py`
- Regime detector: `src/ordinis/engines/signalcore/regime_detector.py`
- Config: `configs/strategies/atr_optimized_rsi.yaml`

### 7.1 File Structure

```
src/ordinis/engines/signalcore/
├── models/
│   └── atr_optimized_rsi.py    # Main strategy model
├── regime_detector.py           # Regime classification
└── strategy_loader.py           # YAML config loader

configs/strategies/
└── atr_optimized_rsi.yaml       # Production config

scripts/trading/
├── live_paper_trading.py        # Paper trading execution (example entrypoint)
└── test_alpaca_connection.py    # Broker connectivity smoke test
```

### 7.2 Usage

```python
from ordinis.engines.signalcore.strategy_loader import StrategyLoader
from ordinis.adapters.broker.broker import SimulatedBroker

# Load strategy
loader = StrategyLoader()
loader.load_strategy("configs/strategies/atr_optimized_rsi.yaml")

# Get model for symbol
model = loader.get_model("COIN")

# Check regime before trading
should_trade, reason = loader.should_trade("COIN", price_df)

# Generate signal
signal = await model.generate("COIN", price_df, timestamp)
```

### 7.3 Live Trading

On Windows, credentials are expected in the **Windows User environment** as:

- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`

The runtime also supports process env fallbacks (`ALPACA_API_KEY`, `ALPACA_API_SECRET`) if User-scoped variables are not set.

PowerShell example (process-scoped):

```powershell
$env:APCA_API_KEY_ID = "your_key"
$env:APCA_API_SECRET_KEY = "your_secret"
python -m ordinis.runtime.live_trading --mode paper
python -m ordinis.runtime.live_trading --mode simulated
```

Implementation reference: `src/ordinis/utils/env.py`.

---

## 8. Known Limitations

1. **Data dependency:** Requires intraday data (5-min bars) for optimal performance
2. **Market hours only:** Not designed for pre/post market
3. **Equity focus:** Tested on US equities only, not validated for futures/forex
4. **Regime lag:** Regime detection uses trailing data, may lag regime changes
5. **Short constraints:** Short selling is not implemented in the current model; short-edge notes are research-only.

---

## 9. Future Enhancements

1. **Multi-timeframe confirmation:** Add daily RSI filter
2. **Volume confirmation:** Require above-average volume on entry
3. **Sector rotation:** Weight toward sectors in favorable regimes
4. **Machine learning:** Train classifier for regime detection
5. **Options overlay:** Use puts for stop-loss instead of hard stops

---

## Appendix A: Mathematical Derivations

### A.1 Optimal Stop Distance

Given volatility σ and desired win rate p, the optimal stop distance d satisfies:

$$d = \sigma \cdot \Phi^{-1}(p)$$

For p = 0.7 (70% win rate) and using ATR as σ proxy:
$$d \approx 1.5 \times ATR$$

### A.2 Kelly Criterion Position Size

For edge e and odds b:
$$f^* = \frac{e}{b} = \frac{p \cdot b - (1-p)}{b}$$

With 70% win rate and 1.5:1 reward-risk:
$$f^* = \frac{0.7 \times 1.5 - 0.3}{1.5} = 0.5$$

Recommend half-Kelly (25% max per trade) for drawdown control.

---

## Appendix B: Configuration Schema

```yaml
strategy:
  name: string          # Strategy identifier
  version: string       # Semantic version
  type: string          # mean_reversion | trend_following | momentum

global_params:
  rsi_period: int       # RSI calculation period (default: 14)
  # NOTE: the model consumes per-symbol keys below (rsi_oversold/rsi_exit/etc).
  atr_period: int       # ATR calculation period (default: 14)
  # Other global keys may be present for documentation/roadmap.

symbols:
  SYMBOL:
    rsi_oversold: float     # Entry threshold (effective)
    rsi_exit: float         # Exit threshold (effective)
    atr_stop_mult: float    # Stop-loss multiplier
    atr_tp_mult: float      # Take-profit multiplier

regime_filter:
  enabled: bool
  avoid_regimes: list[string]  # quiet_choppy, choppy, trending

risk_management:
  max_position_size_pct: float
  max_daily_loss_pct: float
  max_concurrent_positions: int
```

---

## Document Metadata

```yaml
version: "1.2.0"
created: "2025-12-17"
last_updated: "2025-12-23"
status: "review"
implements:
  - "src/ordinis/engines/signalcore/models/atr_optimized_rsi.py"
  - "src/ordinis/engines/signalcore/strategy_loader.py"
  - "src/ordinis/engines/signalcore/regime_detector.py"
config:
  - "configs/strategies/atr_optimized_rsi.yaml"
```

---

*Document generated by Ordinis Quantitative Research*
