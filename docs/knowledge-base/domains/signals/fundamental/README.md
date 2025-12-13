# Fundamental & Macro Analysis - Knowledge Base

## Purpose

Fundamental and macroeconomic analysis provides **universe filters**, **directional bias**, and **rule-based overrides** for the automated trading system. This section documents how to integrate fundamentals systematically.

---

## 1. Financial Statement Analysis

### 1.1 Income Statement Metrics

**Key Metrics**:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Revenue | Top line sales | Growth indicator |
| Gross Margin | (Revenue - COGS) / Revenue | Pricing power |
| Operating Margin | Operating Income / Revenue | Operational efficiency |
| Net Margin | Net Income / Revenue | Bottom line profitability |
| EPS | Net Income / Shares Outstanding | Per-share earnings |

**Rule Templates**:
```python
# Profitability filters
PROFITABLE = net_income > 0
POSITIVE_OPERATING_INCOME = operating_income > 0

# Margin thresholds
HIGH_GROSS_MARGIN = gross_margin > 0.40  # >40%
STRONG_OPERATING_MARGIN = operating_margin > 0.15  # >15%
HEALTHY_NET_MARGIN = net_margin > 0.05  # >5%

# Growth filters
REVENUE_GROWTH = revenue_yoy_growth > 0.10  # >10% YoY
EARNINGS_GROWTH = eps_yoy_growth > 0.15  # >15% YoY
ACCELERATING_GROWTH = revenue_growth_qoq > revenue_growth_qoq[4]  # Accelerating

# Quality filters
CONSISTENT_PROFITABILITY = all(quarterly_eps > 0 for last 8 quarters)
```

---

### 1.2 Balance Sheet Metrics

**Key Metrics**:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Current Ratio | Current Assets / Current Liabilities | Short-term liquidity |
| Quick Ratio | (Current Assets - Inventory) / Current Liabilities | Immediate liquidity |
| Debt/Equity | Total Debt / Total Equity | Leverage |
| Debt/EBITDA | Total Debt / EBITDA | Debt servicing ability |

**Rule Templates**:
```python
# Liquidity filters
ADEQUATE_LIQUIDITY = current_ratio > 1.5
STRONG_LIQUIDITY = quick_ratio > 1.0

# Leverage filters
LOW_DEBT = debt_to_equity < 0.5
MODERATE_DEBT = debt_to_equity < 1.0
HIGH_DEBT = debt_to_equity > 2.0  # Warning
OVERLEVERAGED = debt_to_ebitda > 4.0  # Danger zone

# Balance sheet quality
NET_CASH = cash > total_debt  # Net cash position
IMPROVING_BALANCE_SHEET = debt_to_equity < debt_to_equity[4]  # Deleveraging
```

---

### 1.3 Cash Flow Metrics

**Key Metrics**:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Operating Cash Flow | Cash from operations | Core cash generation |
| Free Cash Flow | OCF - CapEx | Discretionary cash |
| FCF Margin | FCF / Revenue | Cash efficiency |
| Cash Conversion | FCF / Net Income | Earnings quality |

**Rule Templates**:
```python
# Cash flow quality
POSITIVE_OCF = operating_cash_flow > 0
POSITIVE_FCF = free_cash_flow > 0
FCF_POSITIVE_STREAK = all(quarterly_fcf > 0 for last 4 quarters)

# Cash conversion
HIGH_CASH_CONVERSION = fcf / net_income > 0.80  # 80%+ conversion
EARNINGS_QUALITY_CHECK = operating_cash_flow > net_income  # Cash backs earnings

# FCF yield
ATTRACTIVE_FCF_YIELD = fcf / market_cap > 0.05  # >5% FCF yield

# Self-funding
SELF_SUSTAINING = fcf > dividends + buybacks  # Funds shareholder returns
```

---

## 2. Valuation Metrics

### 2.1 Price-Based Ratios

| Metric | Formula | Typical Range | Notes |
|--------|---------|---------------|-------|
| P/E | Price / EPS | 10-25 | Earnings-based |
| Forward P/E | Price / Forward EPS | 10-25 | Forward-looking |
| P/S | Price / Revenue per Share | 1-5 | Revenue-based |
| P/B | Price / Book Value | 1-4 | Asset-based |
| P/FCF | Price / FCF per Share | 10-25 | Cash-based |

**Rule Templates**:
```python
# Value thresholds (sector-dependent)
VALUE_PE = pe_ratio < 15
GROWTH_PE = pe_ratio > 25
REASONABLE_PE = 10 < pe_ratio < 25

# Relative value
UNDERVALUED_VS_SECTOR = pe_ratio < sector_median_pe * 0.80
OVERVALUED_VS_SECTOR = pe_ratio > sector_median_pe * 1.20

# PEG ratio (P/E to Growth)
PEG_ATTRACTIVE = peg_ratio < 1.0  # P/E < growth rate
PEG_FAIR = 1.0 < peg_ratio < 2.0
PEG_EXPENSIVE = peg_ratio > 2.0

# Multiple expansion/contraction
MULTIPLE_EXPANSION = pe_ratio > pe_ratio_1y_ago
MULTIPLE_CONTRACTION = pe_ratio < pe_ratio_1y_ago
```

---

### 2.2 Enterprise Value Ratios

| Metric | Formula | Notes |
|--------|---------|-------|
| EV | Market Cap + Debt - Cash | Total firm value |
| EV/EBITDA | EV / EBITDA | Comparable across capital structures |
| EV/Sales | EV / Revenue | Revenue-based enterprise value |
| EV/FCF | EV / Free Cash Flow | Cash-based |

**Rule Templates**:
```python
# EV/EBITDA thresholds
LOW_EV_EBITDA = ev_ebitda < 8  # Potentially cheap
MODERATE_EV_EBITDA = 8 < ev_ebitda < 12
HIGH_EV_EBITDA = ev_ebitda > 15  # Potentially expensive

# M&A screen (acquisition targets)
POTENTIAL_TARGET = ev_ebitda < 10 AND fcf_positive AND revenue_growing
```

---

## 3. Quality Metrics

### 3.1 Profitability Ratios

| Metric | Formula | Purpose |
|--------|---------|---------|
| ROE | Net Income / Equity | Return on shareholder capital |
| ROA | Net Income / Total Assets | Asset efficiency |
| ROIC | NOPAT / Invested Capital | Capital efficiency |

**Rule Templates**:
```python
# Return thresholds
HIGH_ROE = roe > 0.15  # >15%
STRONG_ROA = roa > 0.08  # >8%
EXCEPTIONAL_ROIC = roic > 0.15  # >15%

# Quality screen
QUALITY_COMPANY = (
    roe > 0.12 AND
    roic > 0.10 AND
    debt_to_equity < 1.0 AND
    fcf_positive
)

# DuPont decomposition
ROE_FROM_MARGINS = high_net_margin contributing to roe
ROE_FROM_LEVERAGE = high_leverage inflating roe  # Warning
```

---

### 3.2 Earnings Quality Indicators

**Warning Signs**:
```python
# Accruals check (high accruals = lower quality)
ACCRUALS = (net_income - operating_cash_flow) / total_assets
HIGH_ACCRUALS = ACCRUALS > 0.10  # Warning sign

# Revenue quality
RECEIVABLES_GROWING_FASTER = receivables_growth > revenue_growth  # Warning

# One-time items
RECURRING_EARNINGS = exclude_one_time_items(net_income)
ADJUSTED_EPS_INFLATED = adjusted_eps > gaap_eps * 1.20  # Suspicious adjustments

# Audit quality
AUDIT_OPINION_CLEAN = no_qualified_opinion
NO_RESTATEMENTS = no_recent_restatements
```

---

## 4. Growth Metrics

### 4.1 Growth Rates

**Key Metrics**:
```python
# Revenue growth
REVENUE_YOY = (revenue - revenue_1y_ago) / revenue_1y_ago
REVENUE_QOQ = (revenue - revenue_1q_ago) / revenue_1q_ago
REVENUE_CAGR_3Y = (revenue / revenue_3y_ago) ** (1/3) - 1

# Earnings growth
EPS_YOY = (eps - eps_1y_ago) / eps_1y_ago
EPS_CAGR_3Y = (eps / eps_3y_ago) ** (1/3) - 1

# Growth thresholds
HIGH_GROWTH = revenue_yoy > 0.20  # >20%
MODERATE_GROWTH = 0.05 < revenue_yoy < 0.20
SLOW_GROWTH = 0 < revenue_yoy < 0.05
DECLINING = revenue_yoy < 0
```

---

### 4.2 Growth Quality

```python
# Sustainable growth
ORGANIC_GROWTH = growth from operations, not acquisitions
EFFICIENT_GROWTH = revenue_growth > capex_growth  # Scaling efficiently

# Growth consistency
CONSISTENT_GROWER = std(revenue_growth_quarterly) < 0.10
VOLATILE_GROWTH = std(revenue_growth_quarterly) > 0.20

# Growth vs valuation
GROWTH_AT_REASONABLE_PRICE = revenue_growth > 0.15 AND pe_ratio < 25
```

---

## 5. Sector Analysis

### 5.1 Sector Classification

**GICS Sectors** (Standard):
1. Information Technology
2. Health Care
3. Financials
4. Consumer Discretionary
5. Communication Services
6. Industrials
7. Consumer Staples
8. Energy
9. Utilities
10. Real Estate
11. Materials

**Rule Templates**:
```python
# Sector exposure limits
MAX_SECTOR_EXPOSURE = 0.25  # Max 25% in any sector
SECTOR_CONCENTRATION = sum(position in sector) / total_portfolio

# Sector momentum
SECTOR_RELATIVE_STRENGTH = sector_return / market_return
SECTOR_LEADING = sector_relative_strength > 1.0
SECTOR_LAGGING = sector_relative_strength < 1.0

# Sector rotation
OVERWEIGHT_SECTOR = IF sector_momentum_rank in top_3
UNDERWEIGHT_SECTOR = IF sector_momentum_rank in bottom_3
```

---

### 5.2 Sector-Specific Metrics

**Technology**:
```python
TECH_METRICS = {
    'revenue_growth': required > 0.10,
    'gross_margin': required > 0.50,
    'r_and_d_ratio': informational,
    'saas_metrics': arr_growth, net_retention
}
```

**Financials**:
```python
FINANCIAL_METRICS = {
    'book_value': primary_valuation,
    'roa': efficiency,
    'net_interest_margin': bank_profitability,
    'capital_ratios': regulatory_compliance
}
```

**Energy**:
```python
ENERGY_METRICS = {
    'ev_ebitda': primary_valuation,
    'production_growth': growth,
    'reserve_replacement': sustainability,
    'breakeven_cost': profitability_sensitivity
}
```

---

## 6. Macro Indicators

### 6.1 Interest Rates

**Key Rates**:
- Federal Funds Rate
- 10-Year Treasury Yield
- 2-Year Treasury Yield
- Yield Curve (10Y - 2Y spread)

**Rule Templates**:
```python
# Rate environment
RISING_RATES = fed_funds_rate > fed_funds_rate_3m_ago
FALLING_RATES = fed_funds_rate < fed_funds_rate_3m_ago

# Yield curve
YIELD_CURVE = treasury_10y - treasury_2y
NORMAL_CURVE = YIELD_CURVE > 0.50  # Positive slope
FLAT_CURVE = -0.25 < YIELD_CURVE < 0.25
INVERTED_CURVE = YIELD_CURVE < -0.25  # Recession warning

# Rate sensitivity
AVOID_RATE_SENSITIVE = IF RISING_RATES: underweight utilities, reits
FAVOR_FINANCIALS = IF RISING_RATES AND normal_curve: overweight banks
```

---

### 6.2 Inflation

**Key Indicators**:
- CPI (Consumer Price Index)
- PPI (Producer Price Index)
- PCE (Personal Consumption Expenditures)
- Inflation Expectations (TIPS breakevens)

**Rule Templates**:
```python
# Inflation regime
HIGH_INFLATION = cpi_yoy > 0.04  # >4%
MODERATE_INFLATION = 0.02 < cpi_yoy < 0.04
LOW_INFLATION = cpi_yoy < 0.02
DEFLATION_RISK = cpi_yoy < 0

# Inflation-adjusted returns
REAL_RETURN = nominal_return - inflation_rate

# Inflation hedges
IF HIGH_INFLATION:
    overweight: commodities, tips, real_assets
    underweight: long_duration_bonds, growth_stocks
```

---

### 6.3 Economic Growth

**Key Indicators**:
- GDP Growth
- ISM Manufacturing/Services PMI
- Unemployment Rate
- Retail Sales
- Industrial Production

**Rule Templates**:
```python
# Growth regime
EXPANSION = gdp_growth > 0 AND pmi > 50
CONTRACTION = gdp_growth < 0 OR pmi < 50
SLOWING = gdp_growth > 0 AND gdp_growth < gdp_growth_prior

# PMI signals
PMI_EXPANSION = pmi > 50
PMI_CONTRACTION = pmi < 50
PMI_ACCELERATING = pmi > pmi_3m_ago
PMI_DECELERATING = pmi < pmi_3m_ago

# Economic cycle positioning
IF EXPANSION AND ACCELERATING:
    favor: cyclicals, growth, small_caps
IF EXPANSION AND DECELERATING:
    favor: quality, defensives
IF CONTRACTION:
    favor: defensives, cash, treasuries
```

---

### 6.4 Risk Regimes

**Risk-On / Risk-Off Indicators**:
```python
# Volatility regime
VIX_LOW = vix < 15  # Complacency
VIX_MODERATE = 15 < vix < 25  # Normal
VIX_ELEVATED = 25 < vix < 35  # Caution
VIX_HIGH = vix > 35  # Fear

# Credit spreads
CREDIT_SPREAD = high_yield_spread - treasury_yield
CREDIT_STRESS = credit_spread > credit_spread_avg_1y * 1.5

# Risk regime classification
RISK_ON = vix < 20 AND credit_spread normal AND pmi > 50
RISK_OFF = vix > 25 OR credit_spread elevated OR pmi < 50

# Dollar strength
DOLLAR_STRONG = dxy > dxy_200ma  # Risk-off typically
DOLLAR_WEAK = dxy < dxy_200ma   # Risk-on typically
```

---

## 7. Fundamental Filters & Overrides

### 7.1 Universe Filters

```python
# Quality universe filter
QUALITY_UNIVERSE = (
    profitable == True AND
    positive_fcf == True AND
    debt_to_equity < 2.0 AND
    roe > 0.08 AND
    no_accounting_issues == True
)

# Growth universe filter
GROWTH_UNIVERSE = (
    revenue_growth_yoy > 0.15 AND
    eps_growth_yoy > 0.20 AND
    gross_margin > 0.40
)

# Value universe filter
VALUE_UNIVERSE = (
    pe_ratio < sector_median_pe AND
    ev_ebitda < 12 AND
    fcf_yield > 0.04 AND
    dividend_yield > 0.02
)

# Combined filter (quality at reasonable price)
GARP_UNIVERSE = (
    QUALITY_UNIVERSE AND
    peg_ratio < 2.0 AND
    pe_ratio < 25
)
```

---

### 7.2 Directional Bias Rules

```python
# Macro-driven bias
IF EXPANSION AND RISK_ON:
    bias = "bullish"
    favor = ["cyclicals", "small_caps", "high_beta"]

IF CONTRACTION OR RISK_OFF:
    bias = "cautious"
    favor = ["defensives", "quality", "low_volatility"]

IF HIGH_INFLATION:
    bias = "selective"
    favor = ["pricing_power", "real_assets", "commodities"]

# Sector rotation
IF EARLY_CYCLE:
    overweight = ["financials", "industrials", "materials"]
IF LATE_CYCLE:
    overweight = ["energy", "materials", "defensives"]
IF RECESSION:
    overweight = ["utilities", "staples", "healthcare"]
```

---

### 7.3 Event Overrides

```python
# Earnings-related
NO_NEW_POSITIONS = days_to_earnings < 3  # Avoid earnings risk
REDUCE_POSITION = earnings_announced AND significant_miss

# Macro events
NO_NEW_POSITIONS = major_fed_meeting within 24 hours
REDUCE_EXPOSURE = major_economic_data_release within 1 hour

# Company-specific
EXIT_POSITION = (
    earnings_miss > 20% OR
    guidance_cut > 10% OR
    key_executive_departure OR
    sec_investigation OR
    downgrade_to_sell by multiple analysts
)
```

---

## 8. Data Sources

### 8.1 Fundamental Data Providers

**Primary (Regulatory/Official)**:
- SEC EDGAR (Company filings)
- Federal Reserve FRED (Macro data)
- Bureau of Labor Statistics (Employment, inflation)
- Bureau of Economic Analysis (GDP)

**Commercial Providers**:
- Bloomberg, Refinitiv (Comprehensive)
- S&P Capital IQ, FactSet (Fundamentals)
- Quandl/Nasdaq Data Link (Alternative data)

### 8.2 Data Quality Considerations

```python
# Data freshness
STALE_DATA_WARNING = last_update > 30 days ago
REQUIRE_TTM = use trailing twelve months for consistency

# Point-in-time (avoid lookahead bias)
USE_POINT_IN_TIME = use data available at signal time
DONT_USE_RESTATED = use original reported values

# Standardization
NORMALIZE_ACCOUNTING = adjust for different accounting standards
SECTOR_ADJUST = compare within sector, not across all stocks
```

---

## Academic References

1. **Graham, B. & Dodd, D.**: "Security Analysis" - Foundational value investing text
2. **Fama, E. & French, K.**: Three-Factor Model, Five-Factor Model papers
3. **Piotroski, J. (2000)**: "Value Investing: The Use of Historical Financial Statement Information" - F-Score
4. **Greenblatt, J.**: "The Little Book That Beats the Market" - Magic Formula
5. **Asness, C. et al. (AQR)**: Quality factor research papers
6. **Sloan, R. (1996)**: "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows?"

---

## Key Takeaways

1. **Use fundamentals as filters**: Screen universe, not timing signals
2. **Sector context matters**: Compare within sectors, not across all stocks
3. **Quality over quantity**: Focus on few reliable metrics
4. **Macro sets the backdrop**: Adjust strategy based on economic regime
5. **Avoid binary decisions**: Use fundamentals for bias, not absolutes
6. **Point-in-time data**: Prevent lookahead bias in backtesting
7. **Combine with technicals**: Fundamentals say what, technicals say when
