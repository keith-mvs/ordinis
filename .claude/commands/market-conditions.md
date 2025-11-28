# Market Conditions

Assess current market regime and trading environment.

## Usage
/market-conditions

## Instructions

Perform a comprehensive market conditions assessment to determine the current trading environment.

### 1. Market Indices
Search for current levels and daily performance of:
- S&P 500 (SPY)
- Nasdaq 100 (QQQ)
- Russell 2000 (IWM)
- Dow Jones (DIA)

### 2. Volatility Assessment
- VIX current level and recent trend
- VIX term structure (if available)
- Classify volatility regime:
  - Low: VIX < 15
  - Normal: 15-20
  - Elevated: 20-25
  - High: 25-35
  - Extreme: > 35

### 3. Breadth Indicators (if available)
- Advance/decline ratio
- New highs vs new lows
- Percentage of stocks above key MAs

### 4. Sector Performance
Rank sectors by recent performance:
- XLK (Technology)
- XLF (Financials)
- XLE (Energy)
- XLV (Healthcare)
- XLY (Consumer Discretionary)
- XLP (Consumer Staples)
- XLI (Industrials)
- XLB (Materials)
- XLU (Utilities)
- XLRE (Real Estate)

### 5. Economic Calendar
Check for upcoming high-impact events:
- FOMC meetings/announcements
- Employment reports (NFP)
- CPI/PPI releases
- GDP releases
- Other Fed speeches

### 6. Regime Classification
Based on analysis, classify current regime:
- **Risk-On**: VIX low, breadth positive, cyclicals leading
- **Risk-Off**: VIX elevated, defensive sectors leading
- **Choppy/Uncertain**: Mixed signals, range-bound
- **Trending**: Clear directional bias

## Output Format

```
## Market Conditions Report
Generated: [date/time]

### Market Overview
| Index | Price | Daily | Weekly | Trend |
|-------|-------|-------|--------|-------|
| SPY | $XXX | ±X.X% | ±X.X% | [Up/Down/Flat] |
| QQQ | $XXX | ±X.X% | ±X.X% | [Up/Down/Flat] |
| IWM | $XXX | ±X.X% | ±X.X% | [Up/Down/Flat] |

### Volatility
- VIX: XX.XX (±X.X%)
- Regime: [Low/Normal/Elevated/High/Extreme]
- Trend: [Rising/Falling/Stable]

### Sector Rotation
**Leaders (Top 3)**:
1. [Sector] +X.X%
2. [Sector] +X.X%
3. [Sector] +X.X%

**Laggards (Bottom 3)**:
1. [Sector] -X.X%
2. [Sector] -X.X%
3. [Sector] -X.X%

### Economic Calendar (Next 5 Days)
| Date | Event | Expected Impact |
|------|-------|-----------------|
| [date] | [event] | [High/Med/Low] |

### Regime Assessment
**Current Regime**: [Risk-On / Risk-Off / Choppy / Trending]

**Rationale**: [Brief explanation of regime classification]

### Trading Implications
Based on the Knowledge Base risk rules:

- **Position Sizing**: [Normal / Reduced / Minimal]
- **Strategy Bias**: [Trend / Mean Reversion / Both / Caution]
- **Sectors to Favor**: [list]
- **Sectors to Avoid**: [list]
- **Special Considerations**: [any warnings or notes]

### Sources
- [list sources used]
```

## Notes
- Run this at the start of each trading session
- Update assessment if significant intraday changes occur
- Reference `docs/knowledge-base/04_fundamental_analysis/README.md` for macro indicator interpretation
- Reference `docs/knowledge-base/07_risk_management/README.md` for regime-based risk adjustments
