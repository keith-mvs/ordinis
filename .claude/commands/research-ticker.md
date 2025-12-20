# Research Ticker

Perform comprehensive research on a stock or ETF symbol for trading analysis.

## Usage
/research-ticker AAPL

## Instructions

When this command is invoked with a ticker symbol, perform the following research:

### 1. Price & Technical Analysis
- Use web search to find current price, daily change, and 52-week range
- Determine trend status (relationship to 50/200 day moving averages if available)
- Note recent volume compared to average
- Identify key support/resistance levels if apparent

### 2. Fundamental Snapshot
- Market cap and sector
- P/E ratio, P/S ratio
- Revenue and earnings growth (YoY)
- Recent analyst ratings summary

### 3. News & Events
- Search for recent significant news (last 7 days)
- Check for upcoming earnings date
- Note any pending ex-dividend dates
- Flag any unusual events (FDA decisions, lawsuits, M&A)

### 4. Sentiment Assessment
- Overall news sentiment (positive/negative/neutral)
- Note any analyst upgrades/downgrades

### 5. Trading Considerations
Based on the Knowledge Base rules in `docs/knowledge-base/`:
- Would this pass liquidity filters? (volume > 500K ADV, etc.)
- Any earnings blackout concerns?
- Current IV environment (if options data available)

## Output Format

```
## [SYMBOL] Research Summary
Generated: [date/time]

### Price Action
- Current Price: $XXX.XX (±X.X%)
- 52-Week Range: $XXX - $XXX
- Volume: X.XM (vs X.XM avg)
- Trend: [Uptrend/Downtrend/Sideways] [brief explanation]

### Key Levels
- Resistance: $XXX
- Support: $XXX

### Fundamentals
| Metric | Value | Notes |
|--------|-------|-------|
| Market Cap | $XXB | |
| P/E | XX.X | [vs sector] |
| P/S | X.X | |
| Rev Growth | XX% | YoY |

### Upcoming Events
- Earnings: [date] [AMC/BMO]
- Ex-Dividend: [date]
- Other: [if any]

### Recent News
1. [headline] - [source] - [sentiment]
2. [headline] - [source] - [sentiment]

### Trading Assessment
- Liquidity: [Pass/Fail] - [reason]
- Volatility: [IV Rank if available]
- Earnings Risk: [Yes/No] - [days until]
- Recommendation: [Tradeable/Avoid/Caution] - [reason]

### Sources
- [list sources used]
```

## Notes
- Always cite sources for factual data
- Separate facts from interpretation
- Flag if any data is stale or unavailable
- Cross-reference key claims with multiple sources when possible
