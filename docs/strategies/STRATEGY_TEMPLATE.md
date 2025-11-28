# Strategy Specification Template

## Overview

This template defines the standard format for documenting trading strategies in a machine-implementable manner. All strategies must be specified using this format to ensure they can be coded, backtested, and deployed.

---

## Strategy Specification Schema

```yaml
strategy:
  # Identification
  id: "STRAT-XXX"
  name: "Strategy Name"
  version: "1.0.0"
  author: "Author Name"
  created: "YYYY-MM-DD"
  last_updated: "YYYY-MM-DD"
  status: "development|testing|paper|live|retired"

  # Classification
  category: "momentum|mean_reversion|breakout|trend|arbitrage|hybrid"
  style: "intraday|swing|position|mixed"
  instruments: ["equities", "options", "futures", "forex", "crypto"]

  # Description
  description: |
    Brief description of the strategy logic and what edge it exploits.

  hypothesis: |
    The market inefficiency or behavioral pattern this strategy exploits.

  # Universe & Filters
  universe:
    market: "US"
    exchanges: ["NYSE", "NASDAQ"]
    filters:
      - min_price: 10
      - max_price: 500
      - min_avg_volume: 500000
      - min_market_cap: 1000000000
      - exclude_sectors: []
      - exclude_tickers: []

  # Entry Rules
  entry:
    conditions:
      - condition_1: "description"
      - condition_2: "description"
    logic: "AND|OR combinations"
    confirmation:
      - volume_required: true
      - price_confirmation: true
    timing:
      - allowed_hours: "09:30-15:30"
      - avoid_days: ["FOMC", "NFP", "Earnings"]

  # Exit Rules
  exit:
    profit_target:
      method: "fixed_pct|atr_multiple|resistance"
      value: 0.05
    stop_loss:
      method: "fixed_pct|atr_multiple|support"
      value: 0.02
    trailing_stop:
      enabled: true
      method: "pct|atr"
      value: 0.03
    time_stop:
      enabled: true
      max_hold_days: 10

  # Position Sizing
  position_sizing:
    method: "risk_based|fixed_dollar|volatility_adjusted"
    risk_per_trade: 0.01
    max_position_pct: 0.10

  # Risk Parameters
  risk:
    max_positions: 5
    max_sector_exposure: 0.25
    max_correlation: 0.70
    max_daily_loss: 0.03
    max_drawdown: 0.15

  # Data Requirements
  data:
    price_data: "1min|5min|15min|daily"
    lookback_period: 50
    indicators:
      - name: "SMA"
        params: {period: 20}
      - name: "RSI"
        params: {period: 14}
    fundamental_data: false
    news_data: false
    options_data: false

  # Evaluation Metrics
  evaluation:
    primary_metric: "sharpe_ratio"
    minimum_thresholds:
      sharpe: 1.0
      win_rate: 0.40
      profit_factor: 1.5
      max_drawdown: 0.20
    required_sample_size: 100
```

---

## Example Strategy 1: Momentum Breakout

```yaml
strategy:
  id: "STRAT-001"
  name: "Momentum Breakout"
  version: "1.0.0"
  status: "development"

  category: "breakout"
  style: "swing"
  instruments: ["equities"]

  description: |
    Enter long positions when price breaks above a consolidation range
    with above-average volume, in stocks showing positive momentum.

  hypothesis: |
    Stocks consolidating in a range often accumulate buying pressure.
    When price breaks out with conviction (volume), the move tends
    to continue in the breakout direction.

  universe:
    market: "US"
    exchanges: ["NYSE", "NASDAQ"]
    filters:
      - min_price: 15
      - max_price: 200
      - min_avg_volume: 1000000
      - min_market_cap: 2000000000
      - exclude_sectors: ["Utilities"]

  entry:
    conditions:
      - consolidation: |
          RANGE = highest(high, 20) - lowest(low, 20)
          RANGE_PCT = RANGE / close
          RANGE_PCT < 0.10  # Tight range (<10%)
      - breakout: |
          close > highest(high, 20)
      - volume_confirmation: |
          volume > SMA(volume, 20) * 1.5
      - trend_filter: |
          close > SMA(close, 50)
          SMA(close, 50) > SMA(close, 200)
      - momentum: |
          RSI(14) > 50
          RSI(14) < 80  # Not overbought
    logic: "consolidation AND breakout AND volume_confirmation AND trend_filter AND momentum"
    confirmation:
      - must_close_above: "breakout_level"
    timing:
      - allowed_hours: "09:45-15:45"
      - avoid_days: ["FOMC_day", "NFP_day"]
      - avoid_earnings: 3  # Days before earnings

  exit:
    profit_target:
      method: "atr_multiple"
      value: 3.0  # 3x ATR from entry
    stop_loss:
      method: "atr_multiple"
      value: 1.5  # 1.5x ATR below entry
      placement: "below_breakout_level"
    trailing_stop:
      enabled: true
      activation: "after_1atr_profit"
      method: "atr"
      value: 2.0  # Trail by 2 ATR
    time_stop:
      enabled: true
      max_hold_days: 15
      condition: "exit if no profit after 5 days"

  position_sizing:
    method: "risk_based"
    risk_per_trade: 0.01
    max_position_pct: 0.08
    sizing_formula: |
      risk_dollars = equity * 0.01
      stop_distance = ATR(14) * 1.5
      shares = risk_dollars / stop_distance

  risk:
    max_positions: 6
    max_sector_exposure: 0.25
    max_correlation: 0.70
    max_daily_loss: 0.03

  data:
    price_data: "daily"
    lookback_period: 200
    indicators:
      - name: "SMA"
        params: [{period: 20}, {period: 50}, {period: 200}]
      - name: "RSI"
        params: {period: 14}
      - name: "ATR"
        params: {period: 14}
      - name: "Volume_SMA"
        params: {period: 20}

  evaluation:
    minimum_thresholds:
      sharpe: 1.0
      win_rate: 0.45
      profit_factor: 1.5
      max_drawdown: 0.15
    backtest_period: "5 years"
    out_of_sample: "1 year"
```

---

## Example Strategy 2: Mean Reversion RSI

```yaml
strategy:
  id: "STRAT-002"
  name: "Oversold Bounce"
  version: "1.0.0"
  status: "development"

  category: "mean_reversion"
  style: "swing"
  instruments: ["equities"]

  description: |
    Buy quality stocks when they become oversold in an uptrend,
    expecting a reversion to the mean.

  hypothesis: |
    Strong stocks in uptrends experience temporary pullbacks.
    When oversold, they tend to revert to their mean, providing
    a high-probability entry with defined risk.

  universe:
    market: "US"
    filters:
      - min_price: 20
      - min_avg_volume: 500000
      - min_market_cap: 5000000000
      - require: "profitable company (positive EPS TTM)"

  entry:
    conditions:
      - uptrend: |
          close > SMA(close, 200)
          SMA(close, 50) > SMA(close, 200)
      - oversold: |
          RSI(14) < 30
      - not_in_freefall: |
          close > lowest(low, 10) * 1.02  # At least 2% off lows
      - volume_present: |
          volume > SMA(volume, 20) * 0.5  # At least half avg volume
    logic: "uptrend AND oversold AND not_in_freefall AND volume_present"
    confirmation:
      - wait_for_green: "close > open on entry day"
    timing:
      - avoid_earnings: 5

  exit:
    profit_target:
      method: "indicator"
      condition: "close > SMA(close, 20) OR RSI(14) > 60"
    stop_loss:
      method: "fixed_pct"
      value: 0.05  # 5% stop
    trailing_stop:
      enabled: false
    time_stop:
      enabled: true
      max_hold_days: 10

  position_sizing:
    method: "risk_based"
    risk_per_trade: 0.01
    max_position_pct: 0.10

  risk:
    max_positions: 8
    max_sector_exposure: 0.30

  data:
    price_data: "daily"
    lookback_period: 200
    indicators:
      - name: "SMA"
        params: [{period: 20}, {period: 50}, {period: 200}]
      - name: "RSI"
        params: {period: 14}
    fundamental_data: true
    fundamental_fields: ["eps_ttm"]
```

---

## Example Strategy 3: Iron Condor Premium Selling

```yaml
strategy:
  id: "STRAT-003"
  name: "High IV Iron Condor"
  version: "1.0.0"
  status: "development"

  category: "options_premium"
  style: "swing"
  instruments: ["options"]

  description: |
    Sell iron condors on liquid ETFs when implied volatility is elevated,
    collecting premium with defined risk.

  hypothesis: |
    Implied volatility is often overpriced relative to realized volatility.
    Selling options during high IV periods captures this premium as IV
    reverts to the mean.

  universe:
    underlyings: ["SPY", "QQQ", "IWM"]
    filters:
      - option_volume: "> 10000 daily"
      - option_spread: "< 0.05 bid-ask as % of mid"

  entry:
    conditions:
      - high_iv: |
          IV_RANK > 0.50
      - iv_premium: |
          IV > HV(20) * 1.10
      - range_bound: |
          ADX(14) < 25
      - no_earnings: |
          earnings_date > expiration_date OR earnings_date < today
    logic: "high_iv AND iv_premium AND range_bound AND no_earnings"
    timing:
      - dte_range: [30, 45]
      - avoid_fomc: 2  # Days before FOMC

  structure:
    type: "iron_condor"
    put_spread:
      short_strike_delta: 0.16
      width: 5  # $5 wide
    call_spread:
      short_strike_delta: 0.16
      width: 5  # $5 wide
    target_credit: "> 0.30 * width"  # At least 30% of width

  exit:
    profit_target:
      method: "percent_of_max"
      value: 0.50  # Close at 50% of max profit
    stop_loss:
      method: "percent_of_credit"
      value: 2.00  # Close if loss = 200% of credit
    management:
      - tested_short: "Roll out and away OR close if loss > 100%"
      - time_decay: "Close if < 7 DTE and profit < 25%"
    time_stop:
      enabled: true
      close_at_dte: 7

  position_sizing:
    method: "max_loss_based"
    max_loss_per_trade: 0.02  # 2% of equity
    formula: |
      max_loss_per_condor = (width - credit) * 100
      contracts = (equity * 0.02) / max_loss_per_condor

  risk:
    max_positions: 3  # Max 3 iron condors at once
    max_delta_exposure: 0.05  # Low delta exposure
    avoid_same_underlying: true  # One condor per underlying

  data:
    price_data: "daily"
    options_data: true
    options_fields:
      - iv_rank
      - iv_percentile
      - delta
      - credit_available
    lookback_period: 252  # For IV rank calculation
```

---

## Strategy Implementation Checklist

Before moving a strategy to backtesting:

### Specification Complete
- [ ] All entry conditions are explicit and codeable
- [ ] All exit conditions are explicit and codeable
- [ ] Position sizing rules are defined
- [ ] Risk parameters are set
- [ ] Data requirements are listed
- [ ] Minimum thresholds are set

### Logic Validation
- [ ] Entry conditions are mutually consistent
- [ ] Exit conditions cover all scenarios
- [ ] Risk rules are enforceable
- [ ] No circular dependencies

### Data Feasibility
- [ ] Required data is available
- [ ] Historical data exists for backtest period
- [ ] Real-time data source identified for live trading

### Risk Review
- [ ] Maximum loss per trade is acceptable
- [ ] Maximum drawdown is acceptable
- [ ] Concentration limits are appropriate

---

## Strategy Documentation Requirements

Each strategy folder should contain:

```
strategies/
└── STRAT-001-momentum-breakout/
    ├── specification.yaml      # Full spec (this template)
    ├── rationale.md           # Detailed hypothesis and logic
    ├── backtest_results/      # Backtest outputs
    │   ├── summary.json
    │   ├── trades.csv
    │   └── equity_curve.png
    ├── parameter_sensitivity/  # Parameter analysis
    ├── implementation/        # Code files
    │   ├── signals.py
    │   ├── entry.py
    │   ├── exit.py
    │   └── sizing.py
    └── changelog.md           # Version history
```

---

## Notes

1. **Strategies are hypotheses**: They must be tested, not trusted
2. **Simpler is better**: Fewer parameters = more robust
3. **Define everything**: No ambiguity in automated systems
4. **Version control**: Track all changes
5. **Out-of-sample required**: Never trust in-sample only
6. **Risk first**: Define risk before potential reward
