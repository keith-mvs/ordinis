# Financial Ratios and Metrics Reference

## Purpose
Comprehensive reference for calculating and interpreting financial ratios used in investment analysis, credit assessment, and operational performance evaluation.

## Profitability Ratios

### Gross Profit Margin
**Formula:**
```
Gross Profit Margin = (Revenue - COGS) / Revenue
```

**Interpretation:**
- Measures production efficiency
- Higher is better (more profit per dollar of sales)
- Industry-specific benchmarks
- Track trends over time

**Typical Ranges:**
- Software/SaaS: 70-90%
- Retail: 20-40%
- Manufacturing: 25-35%
- Services: 40-60%

### Operating Margin (EBIT Margin)
**Formula:**
```
Operating Margin = EBIT / Revenue
```

**Interpretation:**
- Measures operational efficiency before financing costs
- Excludes non-operating items
- Key profitability metric for comparisons

**Excel Implementation:**
```excel
=EBIT/Revenue
// Or
=(Revenue - COGS - Operating_Expenses) / Revenue
```

### Net Profit Margin
**Formula:**
```
Net Profit Margin = Net Income / Revenue
```

**Interpretation:**
- Bottom-line profitability
- After all expenses, interest, and taxes
- Most comprehensive profitability measure

**Typical Ranges:**
- High margin businesses: 15-25%+
- Average: 5-15%
- Low margin: 1-5%

### Return on Assets (ROA)
**Formula:**
```
ROA = Net Income / Average Total Assets
```

**Interpretation:**
- Efficiency of asset utilization
- How much profit generated per dollar of assets
- Compare within same industry

**Excel Implementation:**
```excel
=Net_Income / ((Beginning_Assets + Ending_Assets) / 2)
```

### Return on Equity (ROE)
**Formula:**
```
ROE = Net Income / Average Shareholders' Equity
```

**Interpretation:**
- Return generated for shareholders
- Key metric for investors
- Should exceed cost of capital

**DuPont Analysis (Decomposed ROE):**
```
ROE = (Net Income / Revenue) × (Revenue / Assets) × (Assets / Equity)
    = Profit Margin × Asset Turnover × Financial Leverage
```

**Typical Ranges:**
- Excellent: 15-20%+
- Good: 10-15%
- Average: 5-10%
- Poor: <5%

### Return on Invested Capital (ROIC)
**Formula:**
```
ROIC = NOPAT / Invested Capital

Where:
NOPAT = Net Operating Profit After Tax = EBIT × (1 - Tax Rate)
Invested Capital = Total Assets - Non-interest Bearing Current Liabilities
```

**Interpretation:**
- Most important profitability metric for valuation
- Measures returns on capital employed in business
- Should exceed WACC for value creation

**Excel Implementation:**
```excel
NOPAT = EBIT * (1 - Tax_Rate)
Invested_Capital = Total_Assets - (Current_Liabilities - Short_Term_Debt)
ROIC = NOPAT / Invested_Capital
```

## Liquidity Ratios

### Current Ratio
**Formula:**
```
Current Ratio = Current Assets / Current Liabilities
```

**Interpretation:**
- Ability to pay short-term obligations
- Ratio > 1 indicates positive working capital
- Too high may indicate inefficient asset use

**Benchmarks:**
- Healthy: 1.5 - 3.0
- Minimum acceptable: 1.0
- Concern: < 1.0

### Quick Ratio (Acid Test)
**Formula:**
```
Quick Ratio = (Current Assets - Inventory) / Current Liabilities
// Or
Quick Ratio = (Cash + Marketable Securities + AR) / Current Liabilities
```

**Interpretation:**
- More conservative than current ratio
- Excludes inventory (less liquid)
- Better indicator of immediate liquidity

**Benchmarks:**
- Strong: > 1.0
- Acceptable: 0.7 - 1.0
- Concern: < 0.5

### Cash Ratio
**Formula:**
```
Cash Ratio = (Cash + Cash Equivalents) / Current Liabilities
```

**Interpretation:**
- Most conservative liquidity measure
- Immediate payment capacity
- Useful for distressed situations

### Working Capital
**Formula:**
```
Working Capital = Current Assets - Current Liabilities
```

**Interpretation:**
- Absolute liquidity buffer
- Should be positive
- Needs vary by business model

**Working Capital Ratio:**
```
WC Ratio = Working Capital / Revenue
```

## Leverage Ratios

### Debt to Equity
**Formula:**
```
Debt to Equity = Total Debt / Total Equity
```

**Interpretation:**
- Financial leverage degree
- Risk indicator
- Industry-specific norms

**Typical Ranges:**
- Conservative: < 0.5
- Moderate: 0.5 - 1.5
- Aggressive: > 2.0

### Debt to Assets
**Formula:**
```
Debt to Assets = Total Debt / Total Assets
```

**Interpretation:**
- Percentage of assets financed by debt
- 0.5 = 50% debt-financed
- Lower is more conservative

### Debt to EBITDA
**Formula:**
```
Debt to EBITDA = Total Debt / EBITDA
```

**Interpretation:**
- Debt payback period in years
- Key metric in lending covenants
- Industry-specific benchmarks

**Typical Ranges:**
- Investment grade: < 3x
- Acceptable: 3-5x
- High risk: > 5x

### Interest Coverage
**Formula:**
```
Interest Coverage = EBIT / Interest Expense
```

**Interpretation:**
- Ability to service debt
- Higher is safer
- Key covenant metric

**Benchmarks:**
- Strong: > 5x
- Adequate: 2.5 - 5x
- Concern: < 2x
- Danger: < 1x (not covering interest)

### Debt Service Coverage Ratio (DSCR)
**Formula:**
```
DSCR = Net Operating Income / Total Debt Service

Where:
Total Debt Service = Principal Payments + Interest Payments
```

**Interpretation:**
- Ability to cover all debt obligations
- Critical for lending decisions
- Should be > 1.25 minimum

**Excel Implementation:**
```excel
=Net_Operating_Income / (Principal_Payments + Interest_Payments)
```

## Efficiency Ratios

### Asset Turnover
**Formula:**
```
Asset Turnover = Revenue / Average Total Assets
```

**Interpretation:**
- Revenue generated per dollar of assets
- Higher indicates better efficiency
- Varies significantly by industry

**Typical Ranges:**
- Capital intensive: 0.5 - 1.5x
- Retail: 2 - 3x
- Asset-light: 3 - 5x+

### Inventory Turnover
**Formula:**
```
Inventory Turnover = COGS / Average Inventory
```

**Interpretation:**
- How many times inventory sold per year
- Higher generally better (less capital tied up)
- Too high may indicate stockouts

**Excel Implementation:**
```excel
=COGS / ((Beginning_Inventory + Ending_Inventory) / 2)
```

### Days Inventory Outstanding (DIO)
**Formula:**
```
DIO = (Average Inventory / COGS) × 365
// Or
DIO = 365 / Inventory Turnover
```

**Interpretation:**
- Average days inventory held
- Lower is generally better
- Industry and business model dependent

### Accounts Receivable Turnover
**Formula:**
```
AR Turnover = Revenue / Average Accounts Receivable
```

### Days Sales Outstanding (DSO)
**Formula:**
```
DSO = (Average AR / Revenue) × 365
// Or
DSO = 365 / AR Turnover
```

**Interpretation:**
- Average collection period
- Lower is better (faster collections)
- Compare to payment terms

**Typical Ranges:**
- Fast collection: < 30 days
- Standard: 30-45 days
- Concern: > 60 days

### Accounts Payable Turnover
**Formula:**
```
AP Turnover = COGS / Average Accounts Payable
```

### Days Payable Outstanding (DPO)
**Formula:**
```
DPO = (Average AP / COGS) × 365
// Or
DPO = 365 / AP Turnover
```

**Interpretation:**
- Average payment period to suppliers
- Higher preserves cash (but may strain relationships)
- Should align with terms received

### Cash Conversion Cycle (CCC)
**Formula:**
```
CCC = DIO + DSO - DPO
```

**Interpretation:**
- Days from cash payment to suppliers to cash collection from customers
- Lower is better (faster cash generation)
- Can be negative (collect before paying)

**Example:**
```
DIO: 60 days
DSO: 45 days
DPO: 30 days
CCC = 60 + 45 - 30 = 75 days
```

**Excel Implementation:**
```excel
DIO = (Average_Inventory / COGS) * 365
DSO = (Average_AR / Revenue) * 365
DPO = (Average_AP / COGS) * 365
CCC = DIO + DSO - DPO
```

### Fixed Asset Turnover
**Formula:**
```
Fixed Asset Turnover = Revenue / Average Net PP&E
```

**Interpretation:**
- Efficiency of fixed asset utilization
- Industry-specific benchmarks
- Higher indicates better utilization

## Valuation Ratios

### Price to Earnings (P/E)
**Formula:**
```
P/E Ratio = Stock Price / Earnings Per Share
// Or
P/E Ratio = Market Capitalization / Net Income
```

**Interpretation:**
- How much investors pay per dollar of earnings
- Compare to:
  - Historical company P/E
  - Industry average
  - Market average (S&P 500 ≈ 15-25x)

**Forward P/E:**
```
Forward P/E = Current Price / Expected Next Year EPS
```

### Price to Earnings Growth (PEG)
**Formula:**
```
PEG Ratio = P/E Ratio / Annual EPS Growth Rate (%)
```

**Interpretation:**
- P/E adjusted for growth
- < 1.0: Potentially undervalued
- = 1.0: Fairly valued
- > 1.0: Potentially overvalued or high-quality premium

### Price to Book (P/B)
**Formula:**
```
P/B Ratio = Market Price per Share / Book Value per Share
// Or
P/B Ratio = Market Capitalization / Book Value of Equity
```

**Interpretation:**
- Market value relative to accounting book value
- < 1.0: Trading below book value
- > 1.0: Market expects value creation

**Typical Ranges:**
- Asset-heavy: 0.5 - 2.0x
- Services: 2 - 5x
- High-growth tech: 5 - 15x+

### Price to Sales (P/S)
**Formula:**
```
P/S Ratio = Market Capitalization / Revenue
```

**Interpretation:**
- Useful for unprofitable but growing companies
- Less manipulable than earnings
- Industry-specific benchmarks

### Enterprise Value to EBITDA (EV/EBITDA)
**Formula:**
```
EV/EBITDA = Enterprise Value / EBITDA

Where:
EV = Market Cap + Total Debt - Cash
```

**Interpretation:**
- Most common valuation multiple
- Comparable across capital structures
- Industry-specific benchmarks

**Typical Ranges:**
- Mature, low growth: 4-8x
- Average: 8-12x
- High growth: 12-20x+
- Very high growth: 20x+

**Excel Implementation:**
```excel
EV = Market_Cap + Total_Debt + Preferred_Stock - Cash
EV_EBITDA = EV / EBITDA
```

### Enterprise Value to Revenue (EV/Revenue)
**Formula:**
```
EV/Revenue = Enterprise Value / Revenue
```

**Interpretation:**
- For pre-EBITDA companies
- High-growth SaaS companies
- Lower multiples for mature businesses

### Dividend Yield
**Formula:**
```
Dividend Yield = Annual Dividends per Share / Stock Price
```

**Interpretation:**
- Cash return to shareholders
- Compare to bond yields
- High yield may indicate distress or value

### Dividend Payout Ratio
**Formula:**
```
Payout Ratio = Dividends / Net Income
```

**Interpretation:**
- Percentage of earnings paid as dividends
- Remainder retained for growth
- Varies by company maturity

**Typical Ranges:**
- Growth companies: 0-30%
- Mature companies: 30-60%
- Income stocks: 60-90%

## Growth Metrics

### Revenue Growth Rate
**Formula:**
```
Revenue Growth = (Current Period Revenue / Prior Period Revenue) - 1
```

**Compound Annual Growth Rate (CAGR):**
```
CAGR = ((Ending Value / Beginning Value)^(1/Number of Years)) - 1
```

**Excel Implementation:**
```excel
// Year-over-year growth
=(Current_Year_Revenue / Prior_Year_Revenue) - 1

// CAGR
=((Ending_Revenue / Beginning_Revenue)^(1/Years)) - 1
```

### Earnings Growth Rate
**Formula:**
```
EPS Growth = (Current EPS / Prior EPS) - 1
```

### Sustainable Growth Rate
**Formula:**
```
Sustainable Growth Rate = ROE × (1 - Dividend Payout Ratio)
```

**Interpretation:**
- Maximum growth without external financing
- Based on retained earnings
- Assumes constant leverage and margins

## Credit Analysis Metrics

### Altman Z-Score (Public Companies)
**Formula:**
```
Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5

Where:
X1 = Working Capital / Total Assets
X2 = Retained Earnings / Total Assets
X3 = EBIT / Total Assets
X4 = Market Value of Equity / Book Value of Total Liabilities
X5 = Sales / Total Assets
```

**Interpretation:**
- Bankruptcy prediction model
- Z > 2.99: Safe zone
- 1.81 < Z < 2.99: Gray zone
- Z < 1.81: Distress zone

**Excel Implementation:**
```excel
X1 = Working_Capital / Total_Assets
X2 = Retained_Earnings / Total_Assets
X3 = EBIT / Total_Assets
X4 = Market_Value_Equity / Book_Value_Liabilities
X5 = Revenue / Total_Assets
Z_Score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
```

### Fixed Charge Coverage
**Formula:**
```
Fixed Charge Coverage = (EBIT + Lease Payments) / (Interest + Lease Payments)
```

**Interpretation:**
- Ability to cover all fixed obligations
- More comprehensive than interest coverage
- Should be > 1.5x minimum

## Market Value Metrics

### Market Capitalization
**Formula:**
```
Market Cap = Stock Price × Shares Outstanding
```

**Classifications:**
- Mega Cap: > $200B
- Large Cap: $10B - $200B
- Mid Cap: $2B - $10B
- Small Cap: $300M - $2B
- Micro Cap: $50M - $300M
- Nano Cap: < $50M

### Enterprise Value (EV)
**Formula:**
```
EV = Market Cap + Total Debt + Preferred Stock + Minority Interest - Cash
```

**Interpretation:**
- Theoretical takeover price
- Used for valuation multiples
- More comprehensive than market cap

## Cash Flow Metrics

### Free Cash Flow (FCF)
**Formula:**
```
FCF = Operating Cash Flow - Capital Expenditures
```

**Alternative:**
```
FCF = EBIT × (1 - Tax Rate) + Depreciation - CapEx - Change in NWC
```

### Free Cash Flow Yield
**Formula:**
```
FCF Yield = Free Cash Flow / Enterprise Value
```

**Interpretation:**
- Similar to earnings yield
- Higher is better
- Compare to WACC and bond yields

### Operating Cash Flow Ratio
**Formula:**
```
OCF Ratio = Operating Cash Flow / Current Liabilities
```

**Interpretation:**
- Liquidity measure using cash flow
- > 1.0 is healthy
- More reliable than current ratio

## Industry-Specific Metrics

### SaaS / Software Metrics

**Annual Recurring Revenue (ARR):**
```
ARR = Monthly Recurring Revenue × 12
```

**Customer Acquisition Cost (CAC):**
```
CAC = Sales & Marketing Expense / New Customers Acquired
```

**Lifetime Value (LTV):**
```
LTV = (Average Revenue per Customer / Churn Rate) - CAC
```

**LTV/CAC Ratio:**
```
LTV/CAC = Lifetime Value / Customer Acquisition Cost
```
- Healthy: > 3.0x
- Good: 2.0 - 3.0x
- Concern: < 2.0x

**Rule of 40:**
```
Rule of 40 = Revenue Growth Rate (%) + FCF Margin (%)
```
- Should be ≥ 40% for healthy SaaS business

### Retail Metrics

**Same-Store Sales Growth:**
```
Comp Sales Growth = (Current Period Sales - Prior Period Sales) / Prior Period Sales
// For stores open > 12 months
```

**Sales per Square Foot:**
```
Sales per Sq Ft = Annual Sales / Retail Square Footage
```

### Banking Metrics

**Net Interest Margin (NIM):**
```
NIM = (Interest Income - Interest Expense) / Average Earning Assets
```

**Efficiency Ratio:**
```
Efficiency Ratio = Non-Interest Expense / (Net Interest Income + Non-Interest Income)
```
- Lower is better
- < 50% is excellent

**Loan-to-Deposit Ratio:**
```
Loan-to-Deposit Ratio = Total Loans / Total Deposits
```

## Financial Health Dashboard

### Comprehensive Health Check
Create a dashboard evaluating:

**Profitability Health:**
- Gross Margin: Trending up/down?
- Operating Margin: Above/below industry?
- ROE: Exceeds cost of capital?
- ROIC: Above WACC?

**Liquidity Health:**
- Current Ratio: > 1.5?
- Quick Ratio: > 1.0?
- Days Cash on Hand: > 30 days?

**Leverage Health:**
- Debt/Equity: Within industry norms?
- Interest Coverage: > 3x?
- Debt/EBITDA: < 4x?

**Efficiency Health:**
- Cash Conversion Cycle: Improving?
- Asset Turnover: Above average?
- Inventory Days: Optimized?

**Growth Health:**
- Revenue Growth: Accelerating?
- Market Share: Gaining?
- ROIC > Growth Rate? (quality check)

## Excel Template for Ratio Dashboard

```excel
// Profitability Section
Gross_Margin = (Revenue - COGS) / Revenue
Operating_Margin = EBIT / Revenue
Net_Margin = Net_Income / Revenue
ROA = Net_Income / ((Begin_Assets + End_Assets) / 2)
ROE = Net_Income / ((Begin_Equity + End_Equity) / 2)
ROIC = (EBIT * (1 - Tax_Rate)) / Invested_Capital

// Liquidity Section
Current_Ratio = Current_Assets / Current_Liabilities
Quick_Ratio = (Current_Assets - Inventory) / Current_Liabilities
Cash_Ratio = Cash / Current_Liabilities

// Leverage Section
Debt_to_Equity = Total_Debt / Total_Equity
Debt_to_Assets = Total_Debt / Total_Assets
Interest_Coverage = EBIT / Interest_Expense

// Efficiency Section
Asset_Turnover = Revenue / Avg_Total_Assets
Inventory_Turnover = COGS / Avg_Inventory
DSO = (Avg_AR / Revenue) * 365
DPO = (Avg_AP / COGS) * 365
CCC = DIO + DSO - DPO

// Growth Section
Revenue_Growth = (Current_Revenue / Prior_Revenue) - 1
EPS_Growth = (Current_EPS / Prior_EPS) - 1
```

## Interpretation Best Practices

1. **Context Matters**
   - Industry norms vary significantly
   - Business model affects ratios
   - Size and maturity impact benchmarks

2. **Trends Over Time**
   - Single period ratios can mislead
   - Look for improvement/deterioration
   - Compare to 3-5 year history

3. **Peer Comparison**
   - Compare to direct competitors
   - Use industry averages
   - Adjust for company size

4. **Ratio Interrelationships**
   - Ratios don't exist in isolation
   - High ROE may come from high leverage
   - Strong margins may indicate monopoly power

5. **Quality of Inputs**
   - GIGO: Garbage In, Garbage Out
   - Verify data accuracy
   - Adjust for one-time items
   - Normalize for comparability

This reference provides the foundation for comprehensive financial analysis and performance evaluation in any financial modeling context.
