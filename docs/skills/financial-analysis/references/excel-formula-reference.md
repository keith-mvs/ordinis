# Excel Formula Reference for Financial Modeling

## Purpose
Comprehensive reference for Excel formulas commonly used in financial modeling, with emphasis on audit-friendly patterns, error handling, and best practices.

## Financial Functions

### Net Present Value (NPV)
```excel
=NPV(discount_rate, value1, [value2], ...)
```
**Critical Notes:**
- NPV assumes cash flows occur at END of period
- First cash flow (time 0) should be added separately
- Correct pattern: `=InitialInvestment + NPV(rate, Year1:Year5)`

**Audit-Friendly Alternative:**
```excel
=SUMPRODUCT(CashFlows/(1+DiscountRate)^Periods)
```
This makes the calculation transparent and traceable.

### Internal Rate of Return (IRR)
```excel
=IRR(values, [guess])
```
**Best Practices:**
- Include initial investment (negative value) in cash flow array
- Provide guess if convergence issues occur (typically 0.1)
- Use XIRR for irregular time periods

**With Error Handling:**
```excel
=IFERROR(IRR(B5:B15, 0.1), "No valid IRR")
```

### Extended IRR (XIRR)
```excel
=XIRR(values, dates, [guess])
```
**Usage:**
- Use when cash flows occur at irregular intervals
- Dates must be actual Excel date values
- More accurate than IRR for real-world scenarios

### Present Value (PV)
```excel
=PV(rate, nper, pmt, [fv], [type])
```
**Parameters:**
- rate: Discount rate per period
- nper: Total number of periods
- pmt: Payment per period (negative for outflows)
- fv: Future value (optional)
- type: 0 for end of period, 1 for beginning

### Future Value (FV)
```excel
=FV(rate, nper, pmt, [pv], [type])
```

### Payment (PMT)
```excel
=PMT(rate, nper, pv, [fv], [type])
```
**Example - Loan Payment:**
```excel
=PMT(5%/12, 30*12, -500000)  // $2,684.11 monthly payment
```

## Date and Time Functions

### Today's Date
```excel
=TODAY()
```
**Warning:** Volatile function - recalculates on every change. Avoid in large models.

**Better Approach:**
- Hard-code date in Assumptions sheet
- Update manually or with VBA on open

### End of Month
```excel
=EOMONTH(start_date, months)
```
**Example:**
```excel
=EOMONTH(A1, 0)  // End of current month
=EOMONTH(A1, 1)  // End of next month
```

### Year Fraction
```excel
=YEARFRAC(start_date, end_date, [basis])
```
**Basis Codes:**
- 0 or omitted: US (NASD) 30/360
- 1: Actual/actual
- 2: Actual/360
- 3: Actual/365
- 4: European 30/360

### Network Days (Business Days)
```excel
=NETWORKDAYS(start_date, end_date, [holidays])
```

## Lookup and Reference Functions

### VLOOKUP
```excel
=VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])
```
**Best Practice Pattern:**
```excel
=IFERROR(VLOOKUP(A2, $D$2:$F$100, 2, FALSE), "Not Found")
```
- Always use FALSE for exact match in financial models
- Use absolute references ($) for table_array
- Wrap in IFERROR for robustness

### XLOOKUP (Excel 365)
```excel
=XLOOKUP(lookup_value, lookup_array, return_array, [if_not_found], [match_mode], [search_mode])
```
**Advantages over VLOOKUP:**
- Can look left (return columns before lookup column)
- Built-in error handling with if_not_found
- More flexible and faster

**Example:**
```excel
=XLOOKUP(A2, CompanyNames, Revenues, "Company not found", 0)
```

### INDEX-MATCH (Most Robust)
```excel
=INDEX(return_range, MATCH(lookup_value, lookup_range, 0))
```
**Why INDEX-MATCH:**
- More flexible than VLOOKUP
- Can look left or right
- Faster with large datasets
- Column insertions don't break formula

**With Error Handling:**
```excel
=IFERROR(INDEX($F$2:$F$100, MATCH(A2, $D$2:$D$100, 0)), 0)
```

### SUMIF and SUMIFS
```excel
=SUMIF(range, criteria, [sum_range])
=SUMIFS(sum_range, criteria_range1, criteria1, [criteria_range2, criteria2], ...)
```
**Example - Sum Revenue by Region:**
```excel
=SUMIFS(Revenue, Region, "West", Year, 2024)
```

### COUNTIF and COUNTIFS
```excel
=COUNTIF(range, criteria)
=COUNTIFS(criteria_range1, criteria1, [criteria_range2, criteria2], ...)
```

## Conditional Logic

### IF Statement
```excel
=IF(logical_test, value_if_true, value_if_false)
```
**Nested IF Example:**
```excel
=IF(Revenue>1000000, "Large", IF(Revenue>100000, "Medium", "Small"))
```

**Better Alternative - IFS (Excel 365):**
```excel
=IFS(Revenue>1000000, "Large", Revenue>100000, "Medium", TRUE, "Small")
```

### AND, OR, NOT
```excel
=AND(logical1, [logical2], ...)
=OR(logical1, [logical2], ...)
=NOT(logical)
```
**Example - Multiple Conditions:**
```excel
=IF(AND(Revenue>1000000, Margin>0.2), "Target Met", "Below Target")
```

## Error Handling

### IFERROR
```excel
=IFERROR(value, value_if_error)
```
**Standard Pattern:**
```excel
=IFERROR(A1/B1, 0)  // Return 0 if division by zero
```

### IFNA
```excel
=IFNA(value, value_if_na)
```
**Usage:**
```excel
=IFNA(VLOOKUP(A2, Table, 2, FALSE), "Not Found")
```
More specific than IFERROR - only catches #N/A errors.

### ISERROR, ISNA, ISBLANK
```excel
=ISERROR(value)  // Returns TRUE if any error
=ISNA(value)     // Returns TRUE if #N/A error
=ISBLANK(value)  // Returns TRUE if cell is empty
```

## Array Formulas and Advanced Functions

### SUMPRODUCT
```excel
=SUMPRODUCT(array1, [array2], [array3], ...)
```
**Powerful Uses:**
1. **Weighted Average:**
```excel
=SUMPRODUCT(Values, Weights) / SUM(Weights)
```

2. **Multiple Criteria Sum:**
```excel
=SUMPRODUCT((Region="West")*(Year=2024)*Revenue)
```

3. **NPV Calculation (Audit-Friendly):**
```excel
=SUMPRODUCT(CashFlows/(1+DiscountRate)^Periods)
```

### Array Constants
```excel
{1,2,3,4,5}  // Horizontal array
{1;2;3;4;5}  // Vertical array
```

**Example - Create Period Numbers:**
```excel
=COLUMN(A1:E1)  // Returns {1,2,3,4,5}
```

## Text Functions

### CONCATENATE / CONCAT / TEXTJOIN
```excel
=CONCATENATE(text1, [text2], ...)
=CONCAT(text1, [text2], ...)  // Excel 2016+
=TEXTJOIN(delimiter, ignore_empty, text1, [text2], ...)  // Excel 2016+
```

**Example - Build Scenario Name:**
```excel
=TEXTJOIN(" - ", TRUE, ScenarioType, "Case", Year)
// Output: "Base - Case - 2024"
```

### TEXT (Number Formatting)
```excel
=TEXT(value, format_text)
```
**Common Formats:**
```excel
=TEXT(A1, "$#,##0")          // Currency without decimals
=TEXT(A1, "$#,##0.00")       // Currency with decimals
=TEXT(A1, "0.0%")            // Percentage
=TEXT(A1, "yyyy-mm-dd")      // Date format
=TEXT(A1, "mmmm yyyy")       // Month and year
```

### LEFT, RIGHT, MID
```excel
=LEFT(text, [num_chars])
=RIGHT(text, [num_chars])
=MID(text, start_num, num_chars)
```

### TRIM, CLEAN
```excel
=TRIM(text)   // Removes extra spaces
=CLEAN(text)  // Removes non-printable characters
```

## Statistical Functions

### AVERAGE, MEDIAN, MODE
```excel
=AVERAGE(number1, [number2], ...)
=MEDIAN(number1, [number2], ...)
=MODE.SNGL(number1, [number2], ...)  // Most frequent value
```

### Standard Deviation
```excel
=STDEV.S(number1, [number2], ...)  // Sample standard deviation
=STDEV.P(number1, [number2], ...)  // Population standard deviation
```

### Percentile
```excel
=PERCENTILE.INC(array, k)  // Inclusive (k between 0 and 1)
=PERCENTILE.EXC(array, k)  // Exclusive (k between 0 and 1)
```

**Example - Calculate VaR:**
```excel
=PERCENTILE.INC(Returns, 0.05)  // 5th percentile for 95% VaR
```

### Correlation and Covariance
```excel
=CORREL(array1, array2)
=COVARIANCE.S(array1, array2)  // Sample covariance
```

## Growth and Trend Functions

### Compound Annual Growth Rate (CAGR)
```excel
=((Ending_Value/Beginning_Value)^(1/Number_of_Years))-1
```

**Example:**
```excel
=((F5/B5)^(1/5))-1  // 5-year CAGR
```

### GROWTH (Exponential Growth)
```excel
=GROWTH(known_y's, [known_x's], [new_x's], [const])
```

### TREND (Linear Trend)
```excel
=TREND(known_y's, [known_x's], [new_x's], [const])
```

## Financial Ratios - Formula Patterns

### Profitability Ratios
```excel
// Gross Margin
=(Revenue - COGS) / Revenue

// Operating Margin
=Operating_Income / Revenue

// Net Margin
=Net_Income / Revenue

// Return on Assets (ROA)
=Net_Income / Total_Assets

// Return on Equity (ROE)
=Net_Income / Shareholders_Equity

// Return on Invested Capital (ROIC)
=NOPAT / Invested_Capital
```

### Liquidity Ratios
```excel
// Current Ratio
=Current_Assets / Current_Liabilities

// Quick Ratio (Acid Test)
=(Current_Assets - Inventory) / Current_Liabilities

// Cash Ratio
=(Cash + Marketable_Securities) / Current_Liabilities
```

### Leverage Ratios
```excel
// Debt to Equity
=Total_Debt / Total_Equity

// Debt to Assets
=Total_Debt / Total_Assets

// Interest Coverage
=EBIT / Interest_Expense

// Debt Service Coverage
=(Net_Operating_Income) / (Debt_Service)
```

### Efficiency Ratios
```excel
// Asset Turnover
=Revenue / Average_Total_Assets

// Inventory Turnover
=COGS / Average_Inventory

// Days Sales Outstanding (DSO)
=(Accounts_Receivable / Revenue) * 365

// Days Inventory Outstanding (DIO)
=(Inventory / COGS) * 365

// Days Payable Outstanding (DPO)
=(Accounts_Payable / COGS) * 365

// Cash Conversion Cycle
=DSO + DIO - DPO
```

### Valuation Ratios
```excel
// Price to Earnings (P/E)
=Stock_Price / Earnings_Per_Share

// Price to Book (P/B)
=Stock_Price / Book_Value_Per_Share

// Enterprise Value to EBITDA (EV/EBITDA)
=Enterprise_Value / EBITDA

// PEG Ratio
=PE_Ratio / Earnings_Growth_Rate
```

## Best Practice Patterns

### Named Ranges Best Practice
Instead of:
```excel
=B5*(1+B10)^C5
```

Use named ranges:
```excel
=BaseRevenue*(1+GrowthRate)^Period
```

Define names: Insert > Name > Define Name
- BaseRevenue = Assumptions!$B$5
- GrowthRate = Assumptions!$B$10
- Period = Calculations!C5

### Absolute vs Relative References
```excel
A1      // Relative (both row and column adjust when copied)
$A$1    // Absolute (neither row nor column adjust)
$A1     // Mixed (column absolute, row relative)
A$1     // Mixed (row absolute, column relative)
```

**Common Pattern:**
```excel
=B5*$B$10  // B5 changes when copied down, $B$10 stays fixed
```

### Avoiding Circular References
If you need circular logic (e.g., interest calculated on debt, but debt includes accrued interest):

1. **Enable Iterative Calculation:**
   File > Options > Formulas > Enable iterative calculation

2. **Use Goal Seek or Solver** for one-time calculations

3. **Break the circle** with helper columns:
```excel
// Instead of circular: Debt = PriorDebt + Interest (where Interest = Debt * Rate)
// Use:
DebtBefore = PriorDebt
Interest = DebtBefore * Rate
DebtAfter = DebtBefore + Interest
```

### Dynamic Ranges with OFFSET
```excel
=OFFSET(reference, rows, cols, [height], [width])
```

**Example - Dynamic Named Range:**
```excel
=OFFSET(Assumptions!$A$1, 0, 0, COUNTA(Assumptions!$A:$A), 1)
```
This creates a range that automatically expands as you add data.

### Data Validation for Inputs
Use data validation on all input cells:
1. Select input cells
2. Data > Data Validation
3. Set criteria (List, Number, Date, etc.)
4. Add Input Message and Error Alert

**Example - Dropdown List:**
```excel
Source: ="Base Case,Upside Case,Downside Case"
```

Or reference a range:
```excel
Source: =ScenarioList
```

## Performance Optimization Formulas

### Avoid Volatile Functions
These recalculate on any worksheet change:
- TODAY()
- NOW()
- RAND()
- RANDBETWEEN()
- OFFSET() (in some uses)
- INDIRECT()

**Better Alternatives:**
- Use static dates updated via VBA on workbook open
- Minimize use of OFFSET; use INDEX instead
- Avoid INDIRECT when direct references work

### Replace Complex Nested IFs
Instead of:
```excel
=IF(A1>90, "A", IF(A1>80, "B", IF(A1>70, "C", IF(A1>60, "D", "F"))))
```

Use VLOOKUP with a table:
```excel
=VLOOKUP(A1, GradeTable, 2, TRUE)
```

Or IFS (Excel 365):
```excel
=IFS(A1>90, "A", A1>80, "B", A1>70, "C", A1>60, "D", TRUE, "F")
```

### Minimize Entire Column References
Avoid:
```excel
=SUM(A:A)  // Checks 1,048,576 rows
```

Use:
```excel
=SUM(A1:A1000)  // Checks only needed range
```

Or create a dynamic range:
```excel
=SUM(OFFSET(A1, 0, 0, COUNTA(A:A), 1))
```

## Common Model Calculations

### Working Capital Change
```excel
=((Current_Assets_End - Current_Liabilities_End) -
  (Current_Assets_Begin - Current_Liabilities_Begin))
```

### Free Cash Flow to Firm (FCFF)
```excel
=EBIT * (1 - Tax_Rate) + Depreciation - CapEx - Change_in_NWC
```

### Free Cash Flow to Equity (FCFE)
```excel
=Net_Income + Depreciation - CapEx - Change_in_NWC + Net_Borrowing
```

### Weighted Average Cost of Capital (WACC)
```excel
=(E/(D+E))*Cost_of_Equity + (D/(D+E))*Cost_of_Debt*(1-Tax_Rate)
```
Where:
- E = Market value of equity
- D = Market value of debt

### Terminal Value (Perpetuity Growth)
```excel
=Final_Year_FCF * (1 + Terminal_Growth_Rate) / (WACC - Terminal_Growth_Rate)
```

### Terminal Value (Exit Multiple)
```excel
=Final_Year_EBITDA * Exit_Multiple
```

## Error Checking Formulas

### Sum Check (for Validation)
```excel
=IF(ABS(Expected - Actual) < 0.01, "PASS", "FAIL")
```

### Completeness Check
```excel
=IF(COUNTA(Required_Inputs) = ROWS(Required_Inputs), "Complete", "Missing Data")
```

### Range Check
```excel
=IF(AND(Value >= Min_Threshold, Value <= Max_Threshold), "Valid", "Out of Range")
```

### Circular Reference Detector
```excel
// In VBA:
Sub CheckCircularReferences()
    If Application.CircularReferences.Count > 0 Then
        MsgBox "Circular references detected: " & _
               Application.CircularReferences.Count
    End If
End Sub
```

## Audit Trail Formulas

### Version Control Cell
```excel
="v" & TEXT(VersionNumber, "0.0") & " - " & TEXT(UpdateDate, "yyyy-mm-dd")
```

### Last Modified Timestamp
```excel
// Requires VBA to update on change:
Private Sub Workbook_BeforeSave(ByVal SaveAsUI As Boolean, Cancel As Boolean)
    Sheets("Cover").Range("B7").Value = Now
End Sub
```

### Change Detection
```excel
=IF(Current_Value <> Prior_Value, "CHANGED", "Same")
```

## Critical Reminders

1. **Always wrap division in IFERROR** to prevent #DIV/0!
2. **Use FALSE for exact match** in VLOOKUP (or 0 in MATCH)
3. **Absolute references ($)** for assumption cells
4. **Named ranges** for readability and maintainability
5. **Avoid volatile functions** in calculation-heavy models
6. **Test formulas with edge cases**: zeros, negatives, very large numbers
7. **Document complex formulas** with cell comments
8. **Use consistent date formatting** throughout model
9. **Validate all inputs** with data validation rules
10. **Create validation checks** for critical calculations

This reference guide supports the creation of audit-ready, professional financial models with transparent, traceable calculations.
