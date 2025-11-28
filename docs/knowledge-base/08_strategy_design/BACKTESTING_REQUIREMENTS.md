# Data, Backtesting & Evaluation Requirements

## Overview

This document specifies the data requirements, backtesting methodology, and evaluation criteria for validating trading strategies before deployment.

---

## 1. Data Requirements

### 1.1 Price Data

| Data Type | Frequency | Lookback | Use Case |
|-----------|-----------|----------|----------|
| OHLCV (Daily) | End of day | 10+ years | Swing/position strategies |
| OHLCV (Intraday) | 1min, 5min, 15min | 2+ years | Intraday strategies |
| Tick Data | Every trade | 1+ year | High-frequency analysis |
| Adjusted Prices | Daily | Full history | Accurate returns calculation |

**Required Fields**:
```python
PRICE_DATA_SCHEMA = {
    'timestamp': datetime,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': int,
    'adjusted_close': float,  # Split/dividend adjusted
    'split_factor': float,
    'dividend': float
}
```

---

### 1.2 Fundamental Data

| Data Type | Frequency | Source |
|-----------|-----------|--------|
| Financial statements | Quarterly | SEC EDGAR, data providers |
| Earnings estimates | Weekly updates | I/B/E/S, FactSet |
| Economic indicators | Varies | FRED |
| Sector classification | Static/annual | GICS |

**Point-in-Time Requirement**:
```python
# CRITICAL: Use data available at signal time
# Avoid lookahead bias by using announcement dates, not period end dates

def get_fundamental(symbol: str, field: str, as_of: datetime) -> float:
    """
    Return the value of fundamental field that was KNOWN as of the date.
    Not the value for that period, but what was publicly available.
    """
    return query_point_in_time(symbol, field, as_of)
```

---

### 1.3 Options Data

| Data Type | Frequency | Fields |
|-----------|-----------|--------|
| Options chain | Daily EOD | Strike, expiry, bid, ask, IV, greeks |
| Implied volatility | Daily | IV by strike, term structure |
| Open interest | Daily | OI by strike and expiry |

**Options Chain Schema**:
```python
OPTIONS_CHAIN_SCHEMA = {
    'underlying': str,
    'underlying_price': float,
    'expiration': date,
    'strike': float,
    'option_type': str,  # 'call' or 'put'
    'bid': float,
    'ask': float,
    'mid': float,
    'last': float,
    'volume': int,
    'open_interest': int,
    'implied_volatility': float,
    'delta': float,
    'gamma': float,
    'theta': float,
    'vega': float,
    'rho': float
}
```

---

### 1.4 News & Sentiment Data

| Data Type | Frequency | Source |
|-----------|-----------|--------|
| News headlines | Real-time | News APIs |
| Earnings dates | Updated daily | Earnings calendars |
| Economic calendar | Updated daily | FRED, Bloomberg |
| Sentiment scores | Real-time | NLP providers |

---

### 1.5 Data Quality Checks

```python
def validate_data(data: pd.DataFrame) -> ValidationReport:
    """
    Comprehensive data quality validation.
    """
    checks = {
        'missing_values': check_missing(data),
        'price_gaps': check_gaps(data),
        'volume_zeros': check_zero_volume(data),
        'price_outliers': check_outliers(data),
        'split_adjustments': verify_splits(data),
        'survivorship_bias': check_survivorship(data),
        'lookahead_bias': check_lookahead(data),
        'corporate_actions': verify_corporate_actions(data)
    }
    return ValidationReport(checks)

# Specific checks
def check_survivorship(data: pd.DataFrame) -> bool:
    """
    Ensure delisted stocks are included in historical data.
    Survivorship bias inflates backtested returns.
    """
    pass

def check_lookahead(data: pd.DataFrame) -> bool:
    """
    Verify data timestamps represent when data was available,
    not when events occurred.
    """
    pass
```

---

## 2. Backtesting Methodology

### 2.1 Train/Test Split

```
Historical Data
├── Training Set (60-70%)     → Parameter optimization
├── Validation Set (15-20%)   → Model selection
└── Test Set (15-20%)         → Final evaluation (ONE TIME ONLY)

Example (10 years):
├── Train: Years 1-6
├── Validate: Years 7-8
└── Test: Years 9-10
```

**Rules**:
1. Never optimize on test data
2. Test set is used ONCE, at the end
3. Record all tests to prevent data snooping

---

### 2.2 Walk-Forward Testing

```python
class WalkForwardTest:
    """
    Rolling window optimization to detect overfitting.
    """
    def __init__(
        self,
        in_sample_window: int = 252,  # 1 year
        out_sample_window: int = 63,   # 3 months
        step_size: int = 21            # 1 month
    ):
        self.is_window = in_sample_window
        self.oos_window = out_sample_window
        self.step = step_size

    def run(self, data: pd.DataFrame, strategy: Strategy) -> List[WFResult]:
        results = []
        total_bars = len(data)

        for start in range(0, total_bars - self.is_window - self.oos_window, self.step):
            # In-sample period
            is_start = start
            is_end = start + self.is_window

            # Out-of-sample period
            oos_start = is_end
            oos_end = oos_start + self.oos_window

            # Optimize on in-sample
            best_params = self.optimize(data.iloc[is_start:is_end], strategy)

            # Test on out-of-sample
            oos_metrics = self.test(
                data.iloc[oos_start:oos_end],
                strategy,
                best_params
            )

            results.append(WFResult(
                period=(oos_start, oos_end),
                params=best_params,
                metrics=oos_metrics
            ))

        return results

    def analyze(self, results: List[WFResult]) -> WFAnalysis:
        """
        Compare in-sample vs out-of-sample performance.
        """
        is_sharpe = mean([r.is_sharpe for r in results])
        oos_sharpe = mean([r.oos_sharpe for r in results])

        degradation = (is_sharpe - oos_sharpe) / is_sharpe

        return WFAnalysis(
            is_avg_sharpe=is_sharpe,
            oos_avg_sharpe=oos_sharpe,
            degradation_pct=degradation,
            stable=degradation < 0.30  # Less than 30% degradation
        )
```

---

### 2.3 Out-of-Sample Validation

```python
# Strict out-of-sample protocol
class OutOfSampleProtocol:
    def __init__(self, test_data: pd.DataFrame):
        self.test_data = test_data
        self.tested = False

    def run_final_test(self, strategy: Strategy, params: dict) -> FinalResult:
        """
        Run once and only once on test data.
        """
        if self.tested:
            raise Exception("Test data already used! Cannot rerun.")

        result = backtest(self.test_data, strategy, params)
        self.tested = True

        # Log for audit
        log_test_run(strategy, params, result)

        return result
```

---

### 2.4 Transaction Cost Modeling

```python
class TransactionCostModel:
    """
    Realistic cost modeling for backtests.
    """
    def __init__(self, config: CostConfig):
        self.config = config

    def calculate_costs(
        self,
        order: Order,
        market_data: MarketData
    ) -> TradeCosts:
        """
        Calculate all trading costs.
        """
        # Commission
        commission = self._commission(order)

        # Spread cost (half the spread)
        spread_cost = self._spread_cost(order, market_data)

        # Market impact
        impact_cost = self._market_impact(order, market_data)

        # Slippage
        slippage_cost = self._slippage(order, market_data)

        return TradeCosts(
            commission=commission,
            spread=spread_cost,
            impact=impact_cost,
            slippage=slippage_cost,
            total=commission + spread_cost + impact_cost + slippage_cost
        )

    def _market_impact(self, order: Order, market_data: MarketData) -> float:
        """
        Estimate price impact of order.
        Square-root model: impact ~ sqrt(order_size / ADV)
        """
        order_value = order.quantity * order.price
        adv = market_data.average_daily_volume * market_data.price

        impact_pct = self.config.impact_coefficient * np.sqrt(order_value / adv)
        return order_value * impact_pct


# Default cost assumptions
DEFAULT_COSTS = CostConfig(
    commission_per_share=0.005,
    min_commission=1.00,
    spread_pct=0.0010,  # 10 bps
    impact_coefficient=0.10,
    slippage_pct=0.0005  # 5 bps
)
```

---

## 3. Evaluation Metrics

### 3.1 Return Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Total Return | (Final - Initial) / Initial | > 0 |
| CAGR | (Final/Initial)^(1/years) - 1 | > Risk-free + premium |
| Monthly Returns | Grouped by month | Consistent |

```python
def cagr(equity_curve: pd.Series) -> float:
    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    years = len(equity_curve) / 252
    return (end / start) ** (1 / years) - 1
```

---

### 3.2 Risk-Adjusted Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Sharpe Ratio | (Return - Rf) / StdDev | > 1.0 |
| Sortino Ratio | (Return - Rf) / Downside Dev | > 1.5 |
| Calmar Ratio | CAGR / Max Drawdown | > 1.0 |

```python
def sharpe_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
    excess = returns - risk_free / 252
    if returns.std() == 0:
        return 0
    return excess.mean() / returns.std() * np.sqrt(252)

def sortino_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
    excess = returns - risk_free / 252
    downside = returns[returns < 0].std()
    if downside == 0:
        return 0
    return excess.mean() / downside * np.sqrt(252)

def calmar_ratio(equity_curve: pd.Series) -> float:
    annual_return = cagr(equity_curve)
    max_dd = max_drawdown(equity_curve)
    if max_dd == 0:
        return 0
    return annual_return / abs(max_dd)
```

---

### 3.3 Drawdown Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Max Drawdown | Largest peak-to-trough decline | < 20% |
| Max DD Duration | Longest time underwater | < 1 year |
| Avg Drawdown | Average of all drawdowns | < 10% |

```python
def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

def drawdown_duration(equity_curve: pd.Series) -> int:
    rolling_max = equity_curve.cummax()
    underwater = equity_curve < rolling_max

    # Find longest consecutive underwater period
    groups = (underwater != underwater.shift()).cumsum()
    underwater_periods = underwater.groupby(groups).sum()
    return underwater_periods.max()
```

---

### 3.4 Trade Statistics

| Metric | Formula | Target |
|--------|---------|--------|
| Win Rate | Wins / Total Trades | > 40% |
| Profit Factor | Gross Profit / Gross Loss | > 1.5 |
| Expectancy | (Win% × Avg Win) - (Loss% × Avg Loss) | > 0 |
| Avg Trade | Total P&L / Total Trades | > Costs |

```python
def trade_statistics(trades: List[Trade]) -> TradeStats:
    if not trades:
        return TradeStats()

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]

    win_rate = len(wins) / len(trades)
    avg_win = mean([t.pnl for t in wins]) if wins else 0
    avg_loss = mean([t.pnl for t in losses]) if losses else 0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

    return TradeStats(
        total_trades=len(trades),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        largest_win=max(t.pnl for t in trades),
        largest_loss=min(t.pnl for t in trades)
    )
```

---

### 3.5 Risk Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Volatility | Annualized std dev of returns | Context-dependent |
| VaR 95% | 5th percentile daily loss | < 2% |
| CVaR 95% | Avg loss beyond VaR | < 3% |
| Risk of Ruin | Probability of losing X% | < 1% |

```python
def var_95(returns: pd.Series) -> float:
    """Value at Risk at 95% confidence."""
    return returns.quantile(0.05)

def cvar_95(returns: pd.Series) -> float:
    """Conditional VaR (Expected Shortfall)."""
    var = var_95(returns)
    return returns[returns <= var].mean()

def risk_of_ruin(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    risk_per_trade: float,
    ruin_threshold: float = 0.50
) -> float:
    """
    Estimate probability of losing ruin_threshold of capital.
    Simplified formula for illustration.
    """
    edge = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)
    if edge <= 0:
        return 1.0  # Certain ruin with negative expectancy

    # Simplified approximation
    a = (1 - win_rate) / win_rate
    ruin_prob = a ** (ruin_threshold / risk_per_trade)
    return min(ruin_prob, 1.0)
```

---

## 4. Minimum Validation Criteria

### Required for Production

```python
MINIMUM_CRITERIA = {
    # Sample size
    'min_trades': 100,
    'min_years': 3,

    # Performance
    'min_sharpe': 1.0,
    'min_sortino': 1.5,
    'min_profit_factor': 1.5,
    'min_win_rate': 0.35,

    # Risk
    'max_drawdown': 0.20,
    'max_drawdown_duration_days': 252,

    # Robustness
    'out_of_sample_required': True,
    'walk_forward_required': True,
    'oos_degradation_max': 0.30,  # Max 30% degradation vs in-sample

    # Costs
    'profitable_after_costs': True
}

def validate_strategy(results: BacktestResults) -> ValidationResult:
    """
    Check if strategy meets minimum criteria.
    """
    failures = []

    if results.total_trades < MINIMUM_CRITERIA['min_trades']:
        failures.append(f"Insufficient trades: {results.total_trades}")

    if results.sharpe < MINIMUM_CRITERIA['min_sharpe']:
        failures.append(f"Sharpe too low: {results.sharpe:.2f}")

    if results.max_drawdown < -MINIMUM_CRITERIA['max_drawdown']:
        failures.append(f"Drawdown too large: {results.max_drawdown:.1%}")

    # ... more checks

    return ValidationResult(
        passed=len(failures) == 0,
        failures=failures
    )
```

---

## 5. Robustness Testing

### 5.1 Parameter Sensitivity

```python
def parameter_sensitivity(
    strategy: Strategy,
    base_params: dict,
    param_ranges: dict,
    data: pd.DataFrame
) -> SensitivityReport:
    """
    Test how results change with parameter variations.
    """
    results = []

    for param, values in param_ranges.items():
        for value in values:
            test_params = base_params.copy()
            test_params[param] = value

            metrics = backtest(data, strategy, test_params)
            results.append({
                'param': param,
                'value': value,
                'sharpe': metrics.sharpe,
                'return': metrics.total_return
            })

    # Strategy should be robust to small parameter changes
    return analyze_sensitivity(results)
```

### 5.2 Monte Carlo Simulation

```python
def monte_carlo(
    trades: List[Trade],
    n_simulations: int = 10000
) -> MonteCarloResults:
    """
    Shuffle trade order to understand distribution of outcomes.
    """
    results = []

    for _ in range(n_simulations):
        shuffled = random.sample(trades, len(trades))
        equity = simulate_equity(shuffled)
        results.append({
            'final_equity': equity[-1],
            'max_drawdown': calc_max_drawdown(equity),
            'sharpe': calc_sharpe(equity)
        })

    return MonteCarloResults(
        median_equity=np.median([r['final_equity'] for r in results]),
        p5_equity=np.percentile([r['final_equity'] for r in results], 5),
        p95_equity=np.percentile([r['final_equity'] for r in results], 95),
        p95_drawdown=np.percentile([r['max_drawdown'] for r in results], 95)
    )
```

### 5.3 Regime Testing

```python
def test_across_regimes(
    strategy: Strategy,
    data: pd.DataFrame,
    regimes: Dict[str, Tuple[datetime, datetime]]
) -> RegimeReport:
    """
    Test strategy across different market regimes.
    """
    results = {}

    for regime_name, (start, end) in regimes.items():
        regime_data = data[start:end]
        metrics = backtest(regime_data, strategy)
        results[regime_name] = metrics

    return RegimeReport(results)

# Example regimes
REGIMES = {
    'bull_2013_2015': ('2013-01-01', '2015-12-31'),
    'volatility_2018': ('2018-01-01', '2018-12-31'),
    'covid_crash_2020': ('2020-02-01', '2020-04-30'),
    'recovery_2020': ('2020-04-01', '2020-12-31'),
    'bear_2022': ('2022-01-01', '2022-12-31'),
}
```

---

## 6. Data Sources

### 6.1 Free/Low-Cost Sources

| Source | Data Type | Quality | Cost |
|--------|-----------|---------|------|
| Yahoo Finance | OHLCV | Medium | Free |
| Alpha Vantage | OHLCV, Fundamentals | Medium | Free tier |
| FRED | Economic | High | Free |
| SEC EDGAR | Filings | High | Free |
| Polygon.io | OHLCV | High | $$ |

### 6.2 Premium Sources

| Source | Data Type | Quality | Cost |
|--------|-----------|---------|------|
| Bloomberg | Everything | Highest | $$$$ |
| Refinitiv | Everything | Highest | $$$$ |
| FactSet | Fundamentals | High | $$$ |
| Quandl/Nasdaq | Various | High | $-$$ |
| IEX Cloud | OHLCV | High | $ |

---

## Academic References

1. **Bailey, D.H. & López de Prado, M. (2014)**: "The Deflated Sharpe Ratio" - Correcting for multiple testing
2. **Harvey, C.R. et al. (2016)**: "...and the Cross-Section of Expected Returns" - Data mining pitfalls
3. **de Prado, M.L. (2018)**: "Advances in Financial Machine Learning" - ML backtesting
4. **Aronson, D.R. (2006)**: "Evidence-Based Technical Analysis" - Statistical validation
5. **Bailey, D.H. & López de Prado, M. (2012)**: "The Sharpe Ratio Efficient Frontier"
