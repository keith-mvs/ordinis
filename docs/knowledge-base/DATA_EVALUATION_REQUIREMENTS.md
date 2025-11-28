# Data, Backtesting & Evaluation Requirements

## Overview

This document specifies the complete data requirements, backtesting infrastructure, and evaluation metrics needed for the Intelligent Investor automated trading system.

---

## 1. Data Requirements

### 1.1 Historical OHLCV Data

| Asset Class | Frequency | Minimum History | Fields | Priority |
|-------------|-----------|-----------------|--------|----------|
| US Equities | Daily | 10+ years | OHLCV + Adj Close | Critical |
| US Equities | 1-min/5-min | 2+ years | OHLCV | High |
| ETFs | Daily | 10+ years | OHLCV + Adj Close | Critical |
| Options | Daily EOD | 5+ years | Bid/Ask/Last/Volume/OI | High |
| Futures | Daily | 5+ years | OHLCV + OI | Medium |
| Forex | Daily/Hourly | 5+ years | OHLCV | Medium |
| Crypto | Daily/Hourly | 3+ years | OHLCV | Medium |

**Data Schema - Equities**:
```python
@dataclass
class OHLCVBar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: float  # Split/dividend adjusted
    vwap: Optional[float]  # Volume-weighted avg price

# Quality requirements
QUALITY_REQUIREMENTS = {
    'max_gap_days': 5,           # Max consecutive missing days
    'min_completeness': 0.99,    # 99% of expected bars present
    'price_precision': 4,        # Decimal places
    'volume_precision': 0,       # Whole numbers
    'timezone': 'America/New_York'
}
```

---

### 1.2 Corporate Actions

| Action Type | Fields Required | Use Case |
|-------------|-----------------|----------|
| Stock Splits | Date, ratio, symbol | Price adjustment |
| Dividends | Ex-date, pay date, amount, type | Yield calc, adjustment |
| Mergers/Acquisitions | Date, terms, status | Universe filtering |
| Spinoffs | Date, ratio, new symbol | Position tracking |
| Name/Ticker Changes | Old/new symbol, date | Symbol mapping |
| Delistings | Date, reason | Survivorship bias |

**Corporate Actions Schema**:
```python
@dataclass
class CorporateAction:
    symbol: str
    action_type: str  # 'split', 'dividend', 'merger', etc.
    ex_date: date
    effective_date: date
    details: Dict[str, Any]
    # Split: {'ratio': 4.0}
    # Dividend: {'amount': 0.22, 'type': 'regular'}
    # Merger: {'acquirer': 'XYZ', 'terms': '0.5 shares'}
```

**Adjustment Requirements**:
```python
def adjust_for_split(price: float, split_ratio: float) -> float:
    """Adjust historical prices for splits."""
    return price / split_ratio

def adjust_for_dividend(price: float, dividend: float) -> float:
    """Adjust historical prices for dividends."""
    return price - dividend

# CRITICAL: Use point-in-time adjustments
# Store both raw and adjusted prices
# Re-adjustment capability required
```

---

### 1.3 Fundamental Data

| Category | Metrics | Frequency | Source |
|----------|---------|-----------|--------|
| Income Statement | Revenue, Net Income, EPS, Margins | Quarterly | SEC, Data Providers |
| Balance Sheet | Assets, Liabilities, Equity, Cash, Debt | Quarterly | SEC, Data Providers |
| Cash Flow | Operating CF, Free CF, CapEx | Quarterly | SEC, Data Providers |
| Valuation | P/E, P/S, P/B, EV/EBITDA | Daily (derived) | Calculated |
| Quality | ROE, ROA, ROIC, Debt/Equity | Quarterly | Calculated |
| Growth | Revenue Growth, EPS Growth | Quarterly | Calculated |
| Estimates | EPS Est, Rev Est, Guidance | Weekly updates | I/B/E/S, Refinitiv |

**Fundamental Data Schema**:
```python
@dataclass
class FundamentalSnapshot:
    symbol: str
    period_end: date         # Fiscal period end
    report_date: date        # When publicly available (CRITICAL)
    period_type: str         # 'Q1', 'Q2', 'Q3', 'Q4', 'FY'

    # Income Statement
    revenue: float
    gross_profit: float
    operating_income: float
    net_income: float
    eps_basic: float
    eps_diluted: float

    # Balance Sheet
    total_assets: float
    total_liabilities: float
    total_equity: float
    cash_and_equivalents: float
    total_debt: float

    # Cash Flow
    operating_cash_flow: float
    capital_expenditures: float
    free_cash_flow: float

    # Shares
    shares_outstanding: float
    shares_diluted: float

# POINT-IN-TIME CRITICAL
# Use report_date (when data became public), NOT period_end
# This prevents lookahead bias
```

---

### 1.4 News & Sentiment Data

| Data Type | Fields | Latency | Source |
|-----------|--------|---------|--------|
| News Headlines | Timestamp, ticker, headline, source, sentiment | Real-time | News APIs |
| News Full Text | Body text for NLP analysis | Minutes | News APIs |
| Earnings Announcements | Date, time, actual vs estimate | Real-time | Earnings calendars |
| SEC Filings | 10-K, 10-Q, 8-K alerts | Minutes | SEC EDGAR |
| Analyst Actions | Upgrades, downgrades, price targets | Same-day | Data providers |
| Social Sentiment | Aggregated sentiment scores | Hourly | Sentiment providers |

**News Schema**:
```python
@dataclass
class NewsItem:
    id: str
    timestamp: datetime
    source: str
    source_tier: int  # 1=highest credibility, 4=lowest

    # Content
    headline: str
    summary: Optional[str]
    body: Optional[str]
    url: str

    # Classification
    tickers: List[str]
    categories: List[str]  # 'earnings', 'ma', 'regulatory', etc.

    # Sentiment (if pre-scored)
    sentiment_score: Optional[float]  # -1 to 1
    sentiment_confidence: Optional[float]

@dataclass
class EarningsEvent:
    symbol: str
    fiscal_period: str
    report_date: date
    report_time: str  # 'BMO', 'AMC', 'DMH'

    # Estimates (before)
    eps_estimate: float
    revenue_estimate: float

    # Actuals (after)
    eps_actual: Optional[float]
    revenue_actual: Optional[float]

    # Guidance
    guidance_eps: Optional[Tuple[float, float]]
    guidance_revenue: Optional[Tuple[float, float]]
```

---

### 1.5 Options Chain Data

| Field | Description | Update Frequency |
|-------|-------------|------------------|
| Underlying | Current stock price | Real-time |
| Strike | Strike prices available | Daily |
| Expiration | Expiration dates | Daily |
| Bid/Ask | Current bid and ask | Real-time |
| Last | Last trade price | Real-time |
| Volume | Daily volume | Real-time |
| Open Interest | Open contracts | Daily |
| Implied Volatility | Option-implied vol | Real-time |
| Greeks | Delta, Gamma, Theta, Vega, Rho | Real-time |

**Options Chain Schema**:
```python
@dataclass
class OptionContract:
    underlying: str
    expiration: date
    strike: float
    option_type: str  # 'call' or 'put'

    # Pricing
    bid: float
    ask: float
    mid: float
    last: float

    # Activity
    volume: int
    open_interest: int

    # Volatility
    implied_volatility: float

    # Greeks
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    # Metadata
    timestamp: datetime
    underlying_price: float

@dataclass
class VolatilitySurface:
    underlying: str
    timestamp: datetime

    # Term structure
    atm_iv_by_expiry: Dict[date, float]

    # Skew
    iv_by_strike_by_expiry: Dict[date, Dict[float, float]]

    # Summary metrics
    iv_rank: float      # Percentile over 1 year
    iv_percentile: float
    hv_20: float        # 20-day realized vol
    hv_60: float        # 60-day realized vol
```

---

### 1.6 Implied Volatility Data

| Metric | Description | Use Case |
|--------|-------------|----------|
| IV by Strike | IV for each strike | Skew analysis |
| ATM IV | At-the-money IV | Baseline volatility |
| IV Term Structure | IV by expiration | Calendar trades |
| IV Rank | Current vs 52-week range | Strategy selection |
| IV Percentile | % of days with lower IV | Strategy selection |
| VIX/VXST | Market-wide volatility | Regime detection |
| Realized Vol | Historical volatility | IV premium analysis |

**Volatility Data Schema**:
```python
@dataclass
class VolatilityMetrics:
    symbol: str
    timestamp: datetime

    # Implied Volatility
    iv_atm_30d: float       # 30-day ATM IV
    iv_atm_60d: float       # 60-day ATM IV
    iv_rank: float          # 0-100
    iv_percentile: float    # 0-100

    # Skew
    put_call_skew: float    # 25 delta put IV - 25 delta call IV

    # Historical/Realized
    hv_10: float            # 10-day realized vol
    hv_20: float            # 20-day realized vol
    hv_30: float            # 30-day realized vol
    hv_60: float            # 60-day realized vol

    # Premium
    iv_hv_spread: float     # IV - HV (vol premium)
```

---

### 1.7 Economic & Macro Data

| Category | Indicators | Source | Frequency |
|----------|------------|--------|-----------|
| Interest Rates | Fed Funds, 2Y, 10Y, 30Y Treasury | FRED | Daily |
| Inflation | CPI, PPI, PCE | FRED | Monthly |
| Employment | NFP, Unemployment, Claims | FRED | Weekly/Monthly |
| Growth | GDP, Industrial Production | FRED | Quarterly/Monthly |
| Sentiment | Consumer Confidence, PMI | Various | Monthly |
| Market | VIX, Credit Spreads | Exchanges | Daily |

**Macro Data Schema**:
```python
@dataclass
class MacroIndicator:
    indicator_id: str
    name: str
    value: float
    timestamp: datetime
    release_date: datetime  # When publicly released
    period: str             # 'daily', 'weekly', 'monthly', etc.

    # Expectations
    consensus_estimate: Optional[float]
    prior_value: Optional[float]
    revision: Optional[float]

# Key indicators to track
MACRO_INDICATORS = [
    'FED_FUNDS_RATE',
    'TREASURY_10Y',
    'TREASURY_2Y',
    'YIELD_CURVE_10Y_2Y',
    'VIX',
    'CPI_YOY',
    'UNEMPLOYMENT_RATE',
    'NFP_CHANGE',
    'GDP_GROWTH',
    'ISM_PMI',
    'CONSUMER_CONFIDENCE',
    'HY_SPREAD',  # High yield credit spread
]
```

---

## 2. Backtesting Requirements

### 2.1 Train/Test/Validation Splits

```
┌────────────────────────────────────────────────────────────────┐
│                    DATA SPLITTING STRATEGY                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [====== TRAIN (60%) ======][== VALIDATE (20%) ==][= TEST (20%) =]
│                                                                 │
│  Years 1-6                   Years 7-8            Years 9-10    │
│  Parameter optimization      Model selection      Final eval    │
│                             No optimization       ONE TIME ONLY │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Split Configuration**:
```python
@dataclass
class DataSplitConfig:
    train_pct: float = 0.60
    validate_pct: float = 0.20
    test_pct: float = 0.20

    # Options
    gap_between_splits: int = 0  # Days gap to prevent leakage
    time_based: bool = True       # Must be True for financial data
    shuffle: bool = False         # Must be False for time series

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """Split data chronologically."""
        n = len(data)
        train_end = int(n * self.train_pct)
        val_end = int(n * (self.train_pct + self.validate_pct))

        train = data.iloc[:train_end - self.gap_between_splits]
        validate = data.iloc[train_end:val_end - self.gap_between_splits]
        test = data.iloc[val_end:]

        return train, validate, test
```

---

### 2.2 Walk-Forward Testing

```
Walk-Forward Analysis (Rolling Windows)
────────────────────────────────────────

Window 1: [===IS===][OOS]........................
Window 2: .....[===IS===][OOS]...................
Window 3: ..........[===IS===][OOS]..............
Window 4: ...............[===IS===][OOS].........
Window 5: ....................[===IS===][OOS]....
Window 6: .........................[===IS===][OOS]

IS = In-Sample (Optimize)
OOS = Out-of-Sample (Test)
```

**Walk-Forward Configuration**:
```python
@dataclass
class WalkForwardConfig:
    in_sample_days: int = 252      # 1 year training
    out_sample_days: int = 63      # 3 months testing
    step_days: int = 21            # 1 month step
    min_trades_per_window: int = 20

    # Optimization settings
    param_grid: Dict[str, List]
    optimization_metric: str = 'sharpe_ratio'

    # Results tracking
    track_param_stability: bool = True
    track_degradation: bool = True

class WalkForwardAnalyzer:
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.results = []

    def run(self, data: pd.DataFrame, strategy_class) -> WalkForwardResults:
        """Execute walk-forward analysis."""
        windows = self._generate_windows(data)

        for window in windows:
            # Optimize on in-sample
            best_params = self._optimize(
                data[window.is_start:window.is_end],
                strategy_class
            )

            # Test on out-of-sample
            oos_metrics = self._test(
                data[window.oos_start:window.oos_end],
                strategy_class,
                best_params
            )

            self.results.append({
                'window': window,
                'params': best_params,
                'is_metrics': is_metrics,
                'oos_metrics': oos_metrics,
                'degradation': self._calc_degradation(is_metrics, oos_metrics)
            })

        return self._aggregate_results()

    def _calc_degradation(self, is_metrics, oos_metrics) -> float:
        """Calculate performance degradation IS vs OOS."""
        return (is_metrics.sharpe - oos_metrics.sharpe) / is_metrics.sharpe
```

---

### 2.3 Out-of-Sample Testing Protocol

```python
class OutOfSampleProtocol:
    """
    Strict protocol for out-of-sample testing.
    Test data should be used ONCE and ONLY ONCE.
    """
    def __init__(self, test_data: pd.DataFrame):
        self.test_data = test_data
        self.used = False
        self.usage_log = []

    def run_final_test(
        self,
        strategy: Strategy,
        params: Dict,
        reason: str
    ) -> TestResult:
        """
        Execute final out-of-sample test.
        WARNING: Can only be run once per strategy version.
        """
        if self.used:
            raise RuntimeError(
                "Test data already used! "
                "Create new test data or use different strategy version."
            )

        # Log the usage
        self.usage_log.append({
            'timestamp': datetime.now(),
            'strategy': strategy.__class__.__name__,
            'params': params,
            'reason': reason
        })

        # Run backtest
        result = self._backtest(self.test_data, strategy, params)

        self.used = True

        return result

    def get_usage_log(self) -> List[Dict]:
        """Return audit trail of test data usage."""
        return self.usage_log
```

---

### 2.4 Transaction Costs & Slippage

**Cost Model Components**:
```python
@dataclass
class TransactionCostModel:
    # Commission structure
    commission_type: str = 'per_share'  # 'per_share', 'per_trade', 'percentage'
    commission_rate: float = 0.005      # $0.005 per share
    commission_minimum: float = 1.00    # $1 minimum
    commission_maximum: float = None    # No maximum

    # Spread costs
    spread_model: str = 'fixed'  # 'fixed', 'dynamic', 'historical'
    fixed_spread_bps: float = 5  # 5 basis points half-spread

    # Market impact
    impact_model: str = 'sqrt'  # 'linear', 'sqrt', 'almgren_chriss'
    impact_coefficient: float = 0.1

    # Slippage
    slippage_model: str = 'volatility'  # 'fixed', 'volatility', 'volume'
    slippage_base_bps: float = 2

    def calculate_total_cost(
        self,
        order: Order,
        market_data: MarketData
    ) -> TradeCosts:
        """Calculate all transaction costs."""
        commission = self._calc_commission(order)
        spread = self._calc_spread_cost(order, market_data)
        impact = self._calc_market_impact(order, market_data)
        slippage = self._calc_slippage(order, market_data)

        return TradeCosts(
            commission=commission,
            spread=spread,
            impact=impact,
            slippage=slippage,
            total=commission + spread + impact + slippage,
            total_bps=self._to_bps(
                commission + spread + impact + slippage,
                order
            )
        )

    def _calc_market_impact(
        self,
        order: Order,
        market_data: MarketData
    ) -> float:
        """
        Square-root market impact model.
        Impact = coefficient * sigma * sqrt(Q/V)
        """
        if self.impact_model == 'sqrt':
            sigma = market_data.volatility
            q_over_v = order.quantity / market_data.adv
            impact_pct = self.impact_coefficient * sigma * np.sqrt(q_over_v)
            return order.notional * impact_pct
        # ... other models
```

**Realistic Cost Assumptions**:
```python
# Default cost assumptions by asset class
COST_ASSUMPTIONS = {
    'us_equities_liquid': {
        'commission_per_share': 0.005,
        'spread_bps': 2,
        'impact_coefficient': 0.05,
        'slippage_bps': 2,
        'total_estimate_bps': 8  # ~8 bps round-trip
    },
    'us_equities_illiquid': {
        'commission_per_share': 0.005,
        'spread_bps': 15,
        'impact_coefficient': 0.15,
        'slippage_bps': 10,
        'total_estimate_bps': 40  # ~40 bps round-trip
    },
    'options': {
        'commission_per_contract': 0.65,
        'spread_pct_of_mid': 0.05,  # 5% of mid price
        'slippage_pct': 0.02
    },
    'futures': {
        'commission_per_contract': 2.25,
        'spread_ticks': 1,
        'slippage_ticks': 0.5
    }
}
```

---

## 3. Evaluation Metrics

### 3.1 Return Metrics

| Metric | Formula | Target | Description |
|--------|---------|--------|-------------|
| Total Return | (End - Start) / Start | > 0 | Absolute return |
| CAGR | (End/Start)^(1/years) - 1 | > Rf + premium | Annualized return |
| Monthly Return | Monthly compounded | Consistent | Distribution analysis |
| Rolling Returns | 12M rolling | Stable | Trend analysis |

```python
def calculate_cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    years = len(equity_curve) / 252  # Trading days
    return (end_value / start_value) ** (1 / years) - 1

def calculate_rolling_returns(
    equity_curve: pd.Series,
    window: int = 252
) -> pd.Series:
    """Rolling annualized returns."""
    return equity_curve.pct_change(window).dropna()
```

---

### 3.2 Risk-Adjusted Metrics

| Metric | Formula | Target | Description |
|--------|---------|--------|-------------|
| Sharpe Ratio | (R - Rf) / σ | > 1.0 | Return per unit risk |
| Sortino Ratio | (R - Rf) / σ_down | > 1.5 | Return per downside risk |
| Calmar Ratio | CAGR / Max DD | > 1.0 | Return per drawdown |
| Information Ratio | (R - Rb) / TE | > 0.5 | Active return per tracking error |
| Omega Ratio | Σ gains / Σ losses | > 1.5 | Gain/loss probability weighted |

```python
@dataclass
class RiskAdjustedMetrics:
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    omega_ratio: float

    @classmethod
    def calculate(
        cls,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ) -> 'RiskAdjustedMetrics':
        rf_daily = risk_free_rate / 252
        excess_returns = returns - rf_daily

        # Sharpe
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)

        # Sortino
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # Calmar
        max_dd = cls._max_drawdown(returns)
        cagr = cls._cagr_from_returns(returns)
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Information Ratio
        if benchmark_returns is not None:
            active_returns = returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            ir = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        else:
            ir = None

        # Omega
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        omega = gains / losses if losses > 0 else float('inf')

        return cls(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=ir,
            omega_ratio=omega
        )
```

---

### 3.3 Drawdown Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Max Drawdown | Largest peak-to-trough | < 20% |
| Max DD Duration | Longest underwater period | < 252 days |
| Avg Drawdown | Mean of all drawdowns | < 10% |
| Recovery Time | Time to recover from max DD | < 2x DD duration |
| Ulcer Index | RMS of drawdowns | Lower is better |

```python
@dataclass
class DrawdownMetrics:
    max_drawdown: float
    max_drawdown_duration: int  # Days
    avg_drawdown: float
    drawdown_count: int
    time_underwater_pct: float
    ulcer_index: float
    recovery_factor: float  # Total return / Max DD

    @classmethod
    def calculate(cls, equity_curve: pd.Series) -> 'DrawdownMetrics':
        # Calculate drawdown series
        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve - rolling_max) / rolling_max

        # Max drawdown
        max_dd = drawdowns.min()

        # Drawdown duration
        underwater = drawdowns < 0
        dd_groups = (~underwater).cumsum()
        dd_durations = underwater.groupby(dd_groups).sum()
        max_dd_duration = dd_durations.max()

        # Average drawdown (excluding zero periods)
        dd_values = drawdowns[drawdowns < 0]
        avg_dd = dd_values.mean() if len(dd_values) > 0 else 0

        # Ulcer Index (RMS of drawdowns)
        ulcer = np.sqrt((drawdowns ** 2).mean())

        # Recovery factor
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        recovery_factor = total_return / abs(max_dd) if max_dd != 0 else 0

        return cls(
            max_drawdown=max_dd,
            max_drawdown_duration=int(max_dd_duration),
            avg_drawdown=avg_dd,
            drawdown_count=len(dd_durations[dd_durations > 0]),
            time_underwater_pct=underwater.mean(),
            ulcer_index=ulcer,
            recovery_factor=recovery_factor
        )
```

---

### 3.4 Win Rate & Payoff Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Win Rate | Wins / Total Trades | > 40% |
| Loss Rate | Losses / Total Trades | < 60% |
| Avg Win | Mean of winning trades | > 1.5 × Avg Loss |
| Avg Loss | Mean of losing trades | < Avg Win / 1.5 |
| Payoff Ratio | Avg Win / Avg Loss | > 1.5 |
| Profit Factor | Gross Profit / Gross Loss | > 1.5 |
| Expectancy | (Win% × Avg Win) - (Loss% × Avg Loss) | > 0 |

```python
@dataclass
class TradeMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int

    win_rate: float
    loss_rate: float

    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    payoff_ratio: float
    profit_factor: float
    expectancy: float
    expectancy_ratio: float  # Expectancy / Avg Loss

    avg_trade: float
    avg_holding_period: float  # Days

    @classmethod
    def calculate(cls, trades: List[Trade]) -> 'TradeMetrics':
        if not trades:
            return cls.empty()

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        breakeven = [t for t in trades if t.pnl == 0]

        n = len(trades)
        win_rate = len(wins) / n
        loss_rate = len(losses) / n

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        payoff = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))

        return cls(
            total_trades=n,
            winning_trades=len(wins),
            losing_trades=len(losses),
            breakeven_trades=len(breakeven),
            win_rate=win_rate,
            loss_rate=loss_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=max(t.pnl for t in trades),
            largest_loss=min(t.pnl for t in trades),
            payoff_ratio=payoff,
            profit_factor=pf,
            expectancy=expectancy,
            expectancy_ratio=expectancy / abs(avg_loss) if avg_loss != 0 else 0,
            avg_trade=np.mean([t.pnl for t in trades]),
            avg_holding_period=np.mean([t.holding_days for t in trades])
        )
```

---

### 3.5 Risk-of-Ruin Analysis

```python
def risk_of_ruin(
    win_rate: float,
    payoff_ratio: float,
    risk_per_trade: float,
    ruin_threshold: float = 0.50  # 50% drawdown = ruin
) -> float:
    """
    Calculate probability of reaching ruin threshold.

    Uses simplified formula for illustration.
    More accurate methods: Monte Carlo simulation.
    """
    # Edge calculation
    edge = win_rate * payoff_ratio - (1 - win_rate)

    if edge <= 0:
        return 1.0  # Certain ruin with negative expectancy

    # Number of losing trades to hit ruin
    trades_to_ruin = ruin_threshold / risk_per_trade

    # Simplified risk of ruin (binomial approximation)
    loss_rate = 1 - win_rate
    ror = (loss_rate / win_rate) ** trades_to_ruin

    return min(ror, 1.0)

def monte_carlo_risk_of_ruin(
    trades: List[Trade],
    initial_capital: float,
    ruin_threshold: float = 0.50,
    n_simulations: int = 10000
) -> Dict[str, float]:
    """
    Monte Carlo simulation for risk of ruin.
    """
    ruin_count = 0
    min_equity_values = []

    for _ in range(n_simulations):
        # Shuffle trades
        shuffled = random.sample(trades, len(trades))

        # Simulate equity curve
        equity = initial_capital
        min_equity = equity
        ruined = False

        for trade in shuffled:
            equity += trade.pnl
            min_equity = min(min_equity, equity)

            if equity < initial_capital * (1 - ruin_threshold):
                ruined = True
                break

        if ruined:
            ruin_count += 1
        min_equity_values.append(min_equity)

    return {
        'risk_of_ruin': ruin_count / n_simulations,
        'median_min_equity': np.median(min_equity_values),
        'percentile_5_min_equity': np.percentile(min_equity_values, 5),
        'worst_min_equity': min(min_equity_values)
    }
```

---

### 3.6 Stability Across Regimes

```python
@dataclass
class RegimePerformance:
    regime_name: str
    start_date: date
    end_date: date

    # Core metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

    # Comparison to baseline
    vs_baseline_return: float
    vs_baseline_sharpe: float

class RegimeAnalyzer:
    """Analyze strategy performance across market regimes."""

    # Predefined regimes
    REGIMES = {
        'bull_2013_2015': ('2013-01-01', '2015-12-31'),
        'china_crash_2015': ('2015-08-01', '2015-10-31'),
        'volatility_2018': ('2018-01-01', '2018-12-31'),
        'covid_crash_2020': ('2020-02-01', '2020-03-31'),
        'recovery_2020': ('2020-04-01', '2020-12-31'),
        'bull_2021': ('2021-01-01', '2021-12-31'),
        'bear_2022': ('2022-01-01', '2022-10-31'),
        'recovery_2023': ('2023-01-01', '2023-12-31'),
    }

    def analyze(
        self,
        equity_curve: pd.Series,
        trades: List[Trade]
    ) -> List[RegimePerformance]:
        """Analyze performance across all regimes."""
        results = []

        for regime_name, (start, end) in self.REGIMES.items():
            regime_equity = equity_curve[start:end]
            regime_trades = [t for t in trades if start <= t.exit_date <= end]

            if len(regime_equity) > 20:  # Minimum data
                metrics = self._calculate_regime_metrics(
                    regime_name, start, end,
                    regime_equity, regime_trades
                )
                results.append(metrics)

        return results

    def check_stability(
        self,
        regime_results: List[RegimePerformance]
    ) -> StabilityReport:
        """Check if strategy is stable across regimes."""
        sharpes = [r.sharpe_ratio for r in regime_results]
        returns = [r.total_return for r in regime_results]

        return StabilityReport(
            sharpe_std=np.std(sharpes),
            sharpe_min=min(sharpes),
            return_std=np.std(returns),
            worst_regime=min(regime_results, key=lambda x: x.sharpe_ratio),
            best_regime=max(regime_results, key=lambda x: x.sharpe_ratio),
            stable=all(r.sharpe_ratio > 0 for r in regime_results)
        )
```

---

## 4. Minimum Validation Thresholds

```python
# Strategy must meet ALL of these to proceed to paper trading
MINIMUM_VALIDATION_THRESHOLDS = {
    # Sample size
    'min_trades': 100,
    'min_years': 3,
    'min_out_of_sample_trades': 30,

    # Performance
    'min_sharpe': 1.0,
    'min_sortino': 1.5,
    'min_calmar': 0.75,
    'min_profit_factor': 1.5,
    'min_win_rate': 0.35,
    'min_expectancy': 0,  # Must be positive

    # Risk
    'max_drawdown': -0.20,  # -20%
    'max_drawdown_duration': 252,  # 1 year
    'max_risk_of_ruin': 0.01,  # 1%

    # Robustness
    'max_oos_degradation': 0.30,  # 30% max degradation vs in-sample
    'min_regimes_profitable': 0.75,  # 75% of regimes must be profitable
    'max_parameter_sensitivity': 0.25,  # 25% max sharpe change with params

    # Costs
    'profitable_after_costs': True,
    'min_avg_trade_after_costs': 0  # Must be positive
}

def validate_strategy(metrics: StrategyMetrics) -> ValidationResult:
    """Check if strategy meets all thresholds."""
    failures = []

    for metric, threshold in MINIMUM_VALIDATION_THRESHOLDS.items():
        value = getattr(metrics, metric, None)
        if value is None:
            failures.append(f"Missing metric: {metric}")
        elif metric.startswith('min_') and value < threshold:
            failures.append(f"{metric}: {value:.3f} < {threshold}")
        elif metric.startswith('max_') and value > threshold:
            failures.append(f"{metric}: {value:.3f} > {threshold}")

    return ValidationResult(
        passed=len(failures) == 0,
        failures=failures,
        metrics=metrics
    )
```

---

## 5. Data Sources Summary

### Recommended Providers

| Data Type | Free Options | Paid Options (Recommended) |
|-----------|--------------|---------------------------|
| OHLCV | Yahoo Finance, Alpha Vantage | Polygon.io, IEX Cloud |
| Fundamentals | SEC EDGAR, Yahoo | FactSet, S&P Capital IQ |
| Options | Limited | CBOE, Polygon, OptionMetrics |
| News | NewsAPI (limited) | Bloomberg, Refinitiv |
| Macro | FRED | Bloomberg, Refinitiv |
| Sentiment | Limited | RavenPack, Sentifi |

### Data Quality Checklist

- [ ] Point-in-time data available (no lookahead bias)
- [ ] Corporate actions handled correctly
- [ ] Survivorship bias addressed (includes delisted)
- [ ] Sufficient history (10+ years for daily)
- [ ] Reasonable latency for live trading
- [ ] API rate limits sufficient for needs
- [ ] Data validation/cleaning pipeline exists
