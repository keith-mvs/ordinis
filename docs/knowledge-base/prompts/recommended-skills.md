# Recommended Claude Code Skills for Ordinis

## Overview

This document recommends custom Claude Code skill files (`.md` files in `.claude/commands/`) that would accelerate development of the Ordinis automated trading system.

Skills are reusable prompt templates that Claude can invoke to perform specialized tasks consistently.

---

## Recommended Skills

### 1. Market Research & Analysis

#### `/research-ticker <SYMBOL>`
**Purpose**: Deep dive research on a specific stock or ETF

**Functionality**:
- Fetch current price, volume, and key statistics
- Pull recent news and sentiment
- Analyze technical setup (trend, support/resistance)
- Summarize fundamental metrics
- Identify upcoming events (earnings, ex-div)
- Cross-reference with KB entry/exit rules

**Example Output**:
```
## AAPL Research Summary

### Price Action
- Current: $185.23 (+1.2%)
- Trend: Uptrend (above 50/200 MA)
- RSI(14): 58 (neutral)

### Fundamentals
- P/E: 28.5, P/S: 7.2
- Revenue Growth: +8% YoY
- FCF Yield: 3.2%

### Events
- Earnings: Jan 25 AMC
- Ex-Div: Feb 9

### Signal Assessment
[Based on KB rules...]
```

---

#### `/sector-scan <SECTOR>`
**Purpose**: Analyze sector health and top movers

**Functionality**:
- Sector performance vs SPY
- Top 5 gainers/losers
- Sector-wide volume analysis
- Macro factors affecting sector
- Relative strength ranking

---

#### `/market-conditions`
**Purpose**: Daily market regime assessment

**Functionality**:
- VIX level and trend
- Breadth indicators
- Sector rotation analysis
- Risk-on/risk-off assessment
- Fed/macro calendar check
- Recommended positioning bias

---

### 2. Strategy Development

#### `/design-strategy <description>`
**Purpose**: Generate strategy specification from natural language description

**Functionality**:
- Parse strategy description
- Generate YAML specification
- Identify required data sources
- Suggest entry/exit rules
- Propose risk parameters
- Flag potential issues

---

#### `/validate-strategy <strategy-file>`
**Purpose**: Validate strategy specification for completeness

**Functionality**:
- Check all required fields present
- Validate rule syntax
- Check for logical inconsistencies
- Verify data availability
- Assess overfitting risk
- Suggest improvements

---

#### `/optimize-params <strategy> <param-ranges>`
**Purpose**: Guide parameter optimization process

**Functionality**:
- Generate parameter grid
- Suggest optimization methodology
- Warn about overfitting
- Recommend validation approach
- Structure walk-forward test

---

### 3. Backtesting & Analysis

#### `/analyze-backtest <results-file>`
**Purpose**: Deep analysis of backtest results

**Functionality**:
- Calculate all performance metrics
- Identify best/worst periods
- Analyze trade distribution
- Detect potential issues
- Compare to benchmarks
- Generate improvement suggestions

---

#### `/compare-strategies <strat1> <strat2>`
**Purpose**: Side-by-side strategy comparison

**Functionality**:
- Performance metrics comparison
- Correlation analysis
- Complementary potential
- Combined portfolio simulation
- Recommend allocation

---

#### `/regime-analysis <strategy> <period>`
**Purpose**: Analyze strategy performance across market regimes

**Functionality**:
- Identify regime periods
- Calculate regime-specific metrics
- Highlight vulnerabilities
- Suggest adaptations

---

### 4. Risk Management

#### `/risk-report`
**Purpose**: Generate current portfolio risk assessment

**Functionality**:
- Position-level risk metrics
- Portfolio-level exposure
- Concentration analysis
- Greeks summary (for options)
- VaR/CVaR estimates
- Recommendations

---

#### `/position-size <entry> <stop> <risk-pct>`
**Purpose**: Calculate position size based on risk parameters

**Functionality**:
- Calculate shares/contracts
- Verify against limits
- Show dollar risk
- Consider existing positions

---

#### `/stress-test <scenario>`
**Purpose**: Analyze portfolio under stress scenarios

**Functionality**:
- Apply historical scenario
- Calculate hypothetical losses
- Identify vulnerable positions
- Suggest hedges

---

### 5. Options-Specific

#### `/options-chain <SYMBOL>`
**Purpose**: Analyze options chain for a symbol

**Functionality**:
- Display IV rank/percentile
- Show key strikes (delta-based)
- Calculate expected move
- Suggest strategies based on IV
- Show put/call skew

---

#### `/options-strategy <type> <SYMBOL>`
**Purpose**: Generate options strategy recommendation

**Functionality**:
- Select strikes based on rules
- Calculate max profit/loss
- Show breakevens
- Display Greeks
- Estimate probability of profit

---

#### `/roll-analysis <position>`
**Purpose**: Analyze rolling options for an existing position

**Functionality**:
- Show current position status
- Calculate roll credit/debit
- Compare expiration choices
- Recommend action

---

### 6. Code Generation

#### `/generate-indicator <name> <params>`
**Purpose**: Generate indicator calculation code

**Functionality**:
- Create Python function
- Include docstring
- Add parameter validation
- Include unit tests
- Match KB specification

---

#### `/generate-strategy-code <spec-file>`
**Purpose**: Generate strategy implementation from YAML spec

**Functionality**:
- Create strategy class
- Implement entry/exit logic
- Add position sizing
- Include risk checks
- Generate tests

---

#### `/generate-backtest <strategy>`
**Purpose**: Generate backtest script for a strategy

**Functionality**:
- Load data
- Initialize strategy
- Run simulation
- Generate report
- Create visualizations

---

### 7. Documentation & Learning

#### `/explain-concept <topic>`
**Purpose**: Explain a trading/finance concept from KB

**Functionality**:
- Retrieve from knowledge base
- Explain in plain language
- Provide examples
- Show rule templates
- Cite academic sources

---

#### `/kb-search <query>`
**Purpose**: Search knowledge base for relevant rules/concepts

**Functionality**:
- Search across all KB sections
- Rank by relevance
- Return rule templates
- Show related concepts

---

#### `/academic-sources <topic>`
**Purpose**: Find academic references for a topic

**Functionality**:
- Search for relevant papers
- Summarize key findings
- Provide citation
- Note practical implications

---

### 8. Operations

#### `/daily-prep`
**Purpose**: Pre-market preparation routine

**Functionality**:
- Check overnight moves
- Review economic calendar
- Check earnings calendar
- Assess market conditions
- Review open positions
- Generate watchlist

---

#### `/eod-review`
**Purpose**: End-of-day review routine

**Functionality**:
- Summarize day's trades
- Calculate daily PnL
- Review position status
- Log lessons learned
- Prepare for next day

---

#### `/trade-journal <trade-id>`
**Purpose**: Generate trade journal entry

**Functionality**:
- Record trade details
- Document rationale
- Note market conditions
- Identify lessons
- Track for pattern analysis

---

## Implementation Priority

### High Priority (Implement First)
1. `/research-ticker` - Core research capability
2. `/market-conditions` - Daily regime assessment
3. `/position-size` - Essential risk management
4. `/analyze-backtest` - Strategy validation
5. `/generate-strategy-code` - Accelerate development

### Medium Priority
6. `/design-strategy` - Streamline strategy creation
7. `/options-chain` - Options analysis
8. `/risk-report` - Portfolio monitoring
9. `/daily-prep` - Operations efficiency
10. `/explain-concept` - Learning aid

### Lower Priority (Nice to Have)
11. `/sector-scan` - Market analysis
12. `/roll-analysis` - Options management
13. `/academic-sources` - Research depth
14. `/trade-journal` - Record keeping

---

## Skill File Structure

Each skill should be placed in `.claude/commands/` with the following format:

```markdown
# /skill-name

## Description
Brief description of what this skill does.

## Parameters
- `param1`: Description of first parameter
- `param2`: Description of second parameter

## Instructions

[Detailed instructions for Claude on how to execute this skill,
including what tools to use, what data to fetch, and how to
format the output.]

## Output Format

[Template for expected output structure]

## Examples

[Example invocations and outputs]
```

---

## Integration with Knowledge Base

Skills should reference the knowledge base for:
- Rule templates (entry/exit conditions)
- Risk parameters (limits, thresholds)
- Indicator specifications (formulas, parameters)
- Evaluation criteria (minimum thresholds)

Example reference in skill:
```
Refer to docs/knowledge-base/02_technical_analysis/README.md for
indicator definitions and rule templates when analyzing price action.
```

---

## Next Steps

1. Create `.claude/commands/` directory
2. Implement high-priority skills first
3. Test each skill with real scenarios
4. Iterate based on usage patterns
5. Add more skills as needs emerge
