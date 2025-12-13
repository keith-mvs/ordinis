# Crisis Period Coverage - Complete Market Cycle Testing

## Objective

Ensure trading strategies are robust across **all major market crises, crashes, and regime changes** from 1995-2025 to validate consistent equity signals in extreme conditions.

---

## Historical Crisis Periods Covered

### 1. **Dot-Com Bubble & Burst** (1995-2002)
**Peak**: March 2000
**Trough**: October 2002
**Decline**: -78% (NASDAQ)
**Duration**: 30 months

**Characteristics**:
- Tech stocks massively overvalued (P/E ratios >100)
- Rapid collapse in 2000-2002
- Flight to quality (bonds, utilities)
- Best for testing: Mean reversion strategies, value signals

**Key Events**:
- 1995-1999: Irrational exuberance, tech IPO mania
- March 2000: Bubble peak
- 2000-2002: 78% NASDAQ decline
- 2003: Recovery begins

### 2. **9/11 Terrorist Attacks** (2001)
**Event**: September 11, 2001
**Market Closure**: 4 trading days
**Decline**: -11.6% (S&P 500) in first week
**Recovery**: 6 months

**Characteristics**:
- Unprecedented market closure
- Extreme volatility spike
- Defensive sectors outperformed
- Airlines, insurance sectors crashed

### 3. **Financial Crisis / Housing Crash** (2007-2009)
**Peak**: October 2007
**Trough**: March 2009
**Decline**: -57% (S&P 500)
**Duration**: 17 months

**Characteristics**:
- Subprime mortgage collapse
- Bank failures (Lehman Brothers)
- Credit freeze, liquidity crisis
- Massive government intervention (TARP, QE1)
- Defensive stocks (healthcare, staples) outperformed

**Key Events**:
- Aug 2007: Bear Stearns funds collapse
- Sep 2008: Lehman Brothers bankruptcy
- Oct 2008: Worst monthly decline (-16.8%)
- Mar 2009: Bottom (SPX 666)
- 2009-2020: Longest bull market in history

### 4. **Flash Crash** (May 6, 2010)
**Duration**: 36 minutes
**Decline**: -9% (intraday)
**Recovery**: Same day

**Characteristics**:
- Algorithmic trading malfunction
- Extreme intraday volatility
- Liquidity evaporation
- Circuit breakers activated

### 5. **European Debt Crisis** (2010-2012)
**Peak Volatility**: Aug 2011
**Decline**: -19.4% (S&P 500) Apr-Oct 2011
**Duration**: Intermittent 2010-2012

**Characteristics**:
- Sovereign debt fears (PIIGS countries)
- Euro currency crisis
- U.S. debt downgrade (Aug 2011)
- Safe haven flows to USD, gold

### 6. **Taper Tantrum** (May-June 2013)
**Trigger**: Fed hints at QE tapering
**Bond Yield Spike**: 10Y from 1.6% to 3%
**Equity Decline**: -5.8% (S&P 500)

**Characteristics**:
- Interest rate sensitivity
- Growth vs value rotation
- Emerging market selloff

### 7. **China Devaluation & Oil Crash** (Aug 2015 - Feb 2016)
**Decline**: -14% (S&P 500)
**Oil Crash**: $107 → $26 (WTI)
**Duration**: 6 months

**Characteristics**:
- China Yuan devaluation fears
- Commodity collapse
- Energy sector decimated
- Flight to quality

### 8. **Volmageddon** (Feb 5, 2018)
**VIX Spike**: 11 → 50 in one day
**Decline**: -12% (S&P 500) in 2 weeks

**Characteristics**:
- Short volatility strategy implosion
- XIV ETN liquidation
- Algorithm-driven selloff

### 9. **Q4 2018 Correction** (Oct-Dec 2018)
**Decline**: -19.8% (S&P 500)
**Trigger**: Fed rate hikes, trade war fears
**Recovery**: Q1 2019

**Characteristics**:
- Interest rate sensitivity
- Trade war escalation
- Tech sector correction

### 10. **COVID-19 Pandemic Crash** (Feb-Mar 2020)
**Decline**: -34% (S&P 500) in 33 days
**Fastest bear market in history**
**Recovery**: 5 months to new highs

**Characteristics**:
- Global pandemic shock
- Circuit breakers triggered multiple times
- Unprecedented fiscal/monetary stimulus
- V-shaped recovery
- Work-from-home winners (ZOOM, PTON)
- Travel/hospitality collapse

**Key Events**:
- Feb 19, 2020: All-time high
- Mar 23, 2020: Bottom (-34%)
- Aug 2020: New all-time highs
- Stay-at-home vs reopening rotation

### 11. **Inflation / Rate Hike Cycle** (2022)
**Decline**: -25% (S&P 500) Jan-Oct 2022
**Trigger**: 40-year high inflation (9.1%)
**Fed Response**: Fastest rate hikes since 1980s

**Characteristics**:
- Tech crash (growth stocks hammered)
- Value outperformance
- Crypto collapse
- Bond selloff (worst year for 60/40 portfolio)
- Energy sector leadership

### 12. **Banking Crisis** (March 2023)
**Events**: SVB, Signature Bank, Credit Suisse failures
**Decline**: -8% regional banks
**Duration**: 2 weeks

**Characteristics**:
- Rate hike aftershocks
- Bank deposit flight
- Contagion fears
- Flight to mega-cap tech

### 13. **2023-2024 AI Boom**
**Leaders**: NVDA, MSFT, GOOGL
**Concentration**: Magnificent 7 drive 80% of S&P gains

**Characteristics**:
- Narrow market leadership
- AI hype cycle
- Tech mega-caps vs everything else
- Rotation risks

---

## Dataset Time Periods (Optimized for Crisis Coverage)

### Option 1: **30 Years** (1995-2025) - RECOMMENDED
**Covers**:
- Dot-com bubble (1995-2002) ✓
- 9/11 (2001) ✓
- Financial crisis (2007-2009) ✓
- Flash crash (2010) ✓
- European debt crisis (2010-2012) ✓
- Taper tantrum (2013) ✓
- China/oil crash (2015-2016) ✓
- Volmageddon (2018) ✓
- COVID crash (2020) ✓
- Inflation crisis (2022) ✓
- Banking crisis (2023) ✓
- AI boom (2023-2024) ✓

**Pros**:
- Complete market cycle coverage
- Tests strategies in all crisis types
- 7,500+ bars per symbol

**Cons**:
- Some stocks don't have 30-year history
- Survivorship bias for failed companies

### Option 2: **25 Years** (2000-2025)
**Covers**: All major crises except late 1990s tech bubble buildup

### Option 3: **20 Years** (2005-2025) - CURRENT
**Covers**: Financial crisis onwards, misses dot-com era

---

## Crisis-Specific Test Windows

For targeted validation, create windowed datasets for each crisis:

```python
CRISIS_WINDOWS = {
    "dotcom_bubble": ("1999-01-01", "2002-12-31"),      # 4 years
    "financial_crisis": ("2007-01-01", "2009-12-31"),   # 3 years
    "covid_crash": ("2019-01-01", "2021-12-31"),        # 3 years
    "inflation_2022": ("2021-01-01", "2023-12-31"),     # 3 years
    "flash_crash": ("2010-01-01", "2010-12-31"),        # 1 year
    "volmageddon": ("2017-01-01", "2018-12-31"),        # 2 years
}
```

---

## Backtesting Strategy by Crisis Type

### Bull Market Testing (Trending)
- 2009-2020: Longest bull market
- 2023-2024: AI boom
- **Strategies to favor**: Trend-following (MACD, PSAR), momentum

### Bear Market Testing (Declining)
- 2000-2002: Dot-com crash
- 2007-2009: Financial crisis
- 2020: COVID crash
- 2022: Inflation bear
- **Strategies to favor**: Mean reversion (RSI, Bollinger), defensive sectors

### Sideways Market Testing (Choppy)
- 2015-2016: Range-bound post-oil crash
- 2011-2012: European crisis uncertainty
- **Strategies to favor**: Range trading (Fibonacci levels, Bollinger)

### Crisis Volatility Testing (Extreme)
- 2008 Q4: Financial panic
- 2020 Mar: COVID crash
- 2010 May: Flash crash
- 2018 Feb: Volmageddon
- **Strategies to favor**: Adaptive stops (PSAR with ADX filter)

---

## Recommended Implementation

### Phase 1: Extend Historical Data to 30 Years
```bash
# Fetch 30-year datasets for crisis coverage
python scripts/fetch_parallel.py --years 30 --output data --format csv
```

**Result**: 256 symbols × 30 years = 1.9 million bars total

### Phase 2: Create Crisis Windows
```bash
# Generate windowed datasets for each crisis
python scripts/create_crisis_windows.py \
  --input data/historical \
  --output data/crisis_windows \
  --windows dotcom,financial_crisis,covid,inflation_2022
```

### Phase 3: Run Comprehensive Backtests

**Test Matrix**:
```
6 strategies × 256 symbols × 6 crisis windows × 2 timeframes = 18,432 backtests
```

**This will validate**:
- Which strategies survive ALL crises
- Which sectors outperform in each crisis type
- Market cap rotation during crises
- Optimal defensive positioning

### Phase 4: Generate Crisis Performance Report

**Metrics by Crisis**:
- Max drawdown during crisis
- Recovery time to breakeven
- Sharpe ratio during crisis vs normal
- Win rate in volatile regimes
- Defensive vs cyclical performance

---

## Expected Insights

### By Crisis Type

**Tech Crashes** (2000, 2022):
- Value outperforms growth
- Defensive sectors (utilities, staples) lead
- Small cap underperforms
- Mean reversion strategies excel

**Financial Crises** (2008, 2023):
- Flight to quality (bonds, gold)
- Financials collapse
- Government intervention creates opportunities
- Adaptive strategies (ADX filter) crucial

**Pandemic Shocks** (2020):
- Sector rotation extreme (travel down, tech up)
- V-shaped recovery favors trend-following
- Stay-at-home vs reopening trades

**Inflation Crises** (2022):
- Energy and commodities outperform
- Bonds and tech underperform
- Value over growth
- Real assets (REITs) hedge

### Universal Crisis Patterns

1. **Volatility spike** → PSAR and ATR-based stops essential
2. **Correlation breakdown** → Diversification works better
3. **Liquidity crunch** → Execution slippage increases
4. **Mean reversion opportunities** → Oversold bounces
5. **Sector rotation** → Defensive sectors outperform early

---

## Next Steps

1. **Extend dataset fetch to 30 years** (running: 256 symbols)
2. Create crisis window extraction script
3. Run 18,432-test comprehensive backtest
4. Generate crisis performance analysis
5. Identify crisis-resistant strategies

---

**Status**: 256-symbol fetch in progress (process 7e8cb4)
**Target**: 30-year coverage for complete crisis testing
**ETA**: 5-10 minutes for fetch completion
