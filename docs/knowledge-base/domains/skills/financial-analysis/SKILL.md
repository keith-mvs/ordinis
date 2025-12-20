---
name: financial-analysis
description: Comprehensive financial statement analysis including ratio calculation, trend analysis, and peer comparison. Evaluates liquidity, profitability, efficiency, and leverage metrics. Requires numpy>=1.24.0, pandas>=2.0.0, matplotlib>=3.7.0. Use when analyzing company fundamentals, comparing financial performance, or conducting equity research.
---

# Financial Analysis and Modeling Skill

## Purpose
Generate sophisticated financial models and analyses directly in Excel with comprehensive audit trails, validation logic, and professional formatting. This skill enables creation of production-ready financial models with full transparency, reproducibility, and enterprise-grade quality controls.

## Core Principles

### 1. Audit Trail Requirements

All formulas visible and documented, dedicated assumptions sheet with sources/dates, model version and modification log, clear data flow from inputs to outputs, change tracking with rationale, embedded validation checks with pass/fail indicators.

### 2. Model Architecture
Structure models for clarity, maintainability, and audit-readiness:

```
Standard Model Structure:
├── Cover Sheet (Model metadata, purpose, version)
├── Executive Summary (Key outputs and insights)
├── Assumptions (All inputs, sources, dates, sensitivity flags)
├── Input Data (Raw data with source documentation)
├── Calculations (Core model logic, clearly sectioned)
├── Outputs (Results, dashboards, charts)
├── Validation (Internal checks, reconciliations, error tests)
└── Documentation (Methodology, formulas explained, changelog)
```

### 3. Professional Standards

Consistent calculation methods, color-code hard-coded values, use named ranges for key cells, implement IFERROR/IFNA for robustness, build modular reusable blocks, support multiple scenarios (base/upside/downside).

## Excel Implementation Guidelines

### Python Library Requirements
```python
# Primary libraries for Excel manipulation
import openpyxl
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment, NumberFormat
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.chart import LineChart, BarChart, Reference

# For advanced calculations
import pandas as pd
import numpy as np
from datetime import datetime, date
```

### Standard Color Coding Scheme
```python
COLORS = {
    'input': 'FFFF00',        # Yellow - User inputs/assumptions
    'calculation': 'E0E0E0',  # Light gray - Formulas
    'output': '90EE90',       # Light green - Key results
    'validation': 'FFB6C1',   # Light pink - Check cells
    'header': '4472C4',       # Blue - Section headers
    'hardcoded': 'FFA500',    # Orange - Hard-coded constants
    'error': 'FF0000',        # Red - Error conditions
    'pass': '00FF00',         # Green - Validation passed
}
```

### Model Metadata Template
Always include on the Cover Sheet:

```python
def create_cover_sheet(ws, model_details):
    """Create cover sheet. model_details dict: title, purpose, version, author, created_date, modified_date, reviewer, assumptions_source."""
    ws['A1'] = model_details['title']
    ws['A1'].font = Font(size=18, bold=True)

    ws['A3'] = "Model Purpose:"
    ws['B3'] = model_details['purpose']

    ws['A4'] = "Version:"
    ws['B4'] = model_details['version']

    ws['A5'] = "Author:"
    ws['B5'] = model_details['author']

    ws['A6'] = "Created Date:"
    ws['B6'] = model_details['created_date']

    ws['A7'] = "Last Modified:"
    ws['B7'] = model_details['modified_date']

    ws['A8'] = "Reviewed By:"
    ws['B8'] = model_details.get('reviewer', 'Pending Review')

    ws['A10'] = "Assumptions Source:"
    ws['B10'] = model_details['assumptions_source']

    # Add audit note
    ws['A12'] = "Audit Trail:"
    ws['B12'] = "All assumptions, calculations, and validations are documented within this model."
```

### Assumptions Sheet Best Practices
The Assumptions sheet is the foundation of model credibility:

```python
def create_assumptions_sheet(ws, assumptions_data):
    """Create assumptions sheet. assumptions_data: list of dicts with category, parameter, value, unit, source, date, sensitivity, notes."""
    headers = ['Category', 'Parameter', 'Value', 'Unit', 'Source',
               'Date', 'Sensitivity', 'Notes']

    # Create header row with formatting
    for col_num, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.value = header
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        cell.alignment = Alignment(horizontal='center')

    # Add assumptions with color coding
    for row_num, assumption in enumerate(assumptions_data, 2):
        ws.cell(row=row_num, column=1, value=assumption['category'])
        ws.cell(row=row_num, column=2, value=assumption['parameter'])

        # Value cell - marked as input
        value_cell = ws.cell(row=row_num, column=3, value=assumption['value'])
        value_cell.fill = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')

        ws.cell(row=row_num, column=4, value=assumption['unit'])
        ws.cell(row=row_num, column=5, value=assumption['source'])
        ws.cell(row=row_num, column=6, value=assumption['date'])

        # Sensitivity indicator
        sensitivity_cell = ws.cell(row=row_num, column=7, value=assumption['sensitivity'])
        if assumption['sensitivity'] == 'High':
            sensitivity_cell.fill = PatternFill(start_color='FF6B6B', end_color='FF6B6B', fill_type='solid')

        ws.cell(row=row_num, column=8, value=assumption['notes'])

    # Add named ranges for key assumptions
    for row_num, assumption in enumerate(assumptions_data, 2):
        param_name = assumption['parameter'].replace(' ', '_').replace('%', 'pct')
        ws.define_name(param_name, f"Assumptions!${get_column_letter(3)}${row_num}")
```

### Validation Sheet Requirements
Every model must include comprehensive validation checks:

```python
def create_validation_sheet(ws):
    """Create validation with automated checks: balance, reconciliation, logical, completeness, formula, sensitivity."""
    headers = ['Check ID', 'Check Description', 'Expected Result',
               'Actual Result', 'Status', 'Tolerance', 'Notes']

    for col_num, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.value = header
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')

    # Example validation check row
    ws['A2'] = 'CHK-001'
    ws['B2'] = 'Balance Sheet Balance: Total Assets = Total Liabilities + Equity'
    ws['C2'] = '=Calculations!TotalAssets'  # Reference to expected
    ws['D2'] = '=Calculations!TotalLiabilitiesEquity'  # Reference to actual

    # Status formula with color coding
    ws['E2'] = '=IF(ABS(C2-D2)<=F2, "PASS", "FAIL")'

    # Conditional formatting for status
    ws['E2'].style = 'Good' if ws['E2'].value == 'PASS' else 'Bad'

    ws['F2'] = 0.01  # Tolerance threshold
    ws['G2'] = 'Critical balance check for accounting integrity'
```

### Advanced Formula Patterns

#### Time Series Calculations
```python
def create_projection_formulas(ws, start_row, periods):
    """
    Create time-series projection formulas with growth rates.

    Pattern: Next Period = Previous Period * (1 + Growth Rate)
    Includes error handling and circular reference prevention.
    """
    for period in range(periods):
        row = start_row + period
        if period == 0:
            # First period uses base assumption
            ws[f'C{row}'] = f'=Assumptions!Base_Value'
        else:
            # Subsequent periods use growth formula
            ws[f'C{row}'] = f'=IFERROR(C{row-1}*(1+Assumptions!Growth_Rate), C{row-1})'
```

#### NPV and IRR Calculations
```python
def create_npv_irr_section(ws, cash_flow_range, discount_rate_cell):
    """
    Create NPV and IRR calculations with proper documentation.

    Args:
        cash_flow_range: Range of cash flow cells (e.g., 'C10:C20')
        discount_rate_cell: Cell reference for discount rate
    """
    ws['B25'] = 'Net Present Value (NPV):'
    ws['C25'] = f'=NPV({discount_rate_cell}, {cash_flow_range})'
    ws['C25'].number_format = '$#,##0'
    ws['D25'] = 'Discount rate from Assumptions sheet'

    ws['B26'] = 'Internal Rate of Return (IRR):'
    ws['C26'] = f'=IRR({cash_flow_range})'
    ws['C26'].number_format = '0.00%'
    ws['D26'] = 'Iterative calculation'

    # Color code as output
    ws['C25'].fill = PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid')
    ws['C26'].fill = PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid')
```

#### Scenario Analysis Structure
```python
def create_scenario_section(ws, base_case_range):
    """
    Create scenario analysis with base, upside, and downside cases.
    """
    scenarios = ['Base Case', 'Upside (+20%)', 'Downside (-20%)']

    ws['A1'] = 'Scenario Analysis'
    ws['A1'].font = Font(size=14, bold=True)

    for col, scenario in enumerate(scenarios, 2):
        ws.cell(row=2, column=col, value=scenario)
        ws.cell(row=2, column=col).font = Font(bold=True)

    # Link calculations with scenario multipliers
    ws['B3'] = f'={base_case_range}'
    ws['C3'] = f'={base_case_range}*1.2'
    ws['D3'] = f'={base_case_range}*0.8'
```

### Data Visualization Standards
```python
def create_professional_chart(ws, data_range, chart_title, chart_type='line'):
    """
    Create professional chart with proper formatting.

    Args:
        ws: worksheet object
        data_range: Range containing data for chart
        chart_title: Descriptive chart title
        chart_type: 'line', 'bar', or 'column'
    """
    if chart_type == 'line':
        chart = LineChart()
    else:
        chart = BarChart()

    chart.title = chart_title
    chart.style = 10  # Professional style
    chart.y_axis.title = 'Value ($)'
    chart.x_axis.title = 'Period'

    # Add data
    data = Reference(ws, min_col=2, min_row=1, max_row=20, max_col=4)
    cats = Reference(ws, min_col=1, min_row=2, max_row=20)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)

    # Place chart
    ws.add_chart(chart, 'F2')
```

## Common Financial Model Types

**DCF Model**: Revenue projections, operating expenses, working capital, CapEx schedule, FCF calculation, terminal value, WACC, NPV/sensitivity | **Three-Statement Model**: Income statement, balance sheet, cash flow statement with automatic balancing, period analysis, ratios | **Budget vs Actual**: Budget assumptions, actual results, variance calculations ($ and %), commentary, rolling forecasts, YTD projections | **ROI Model**: Initial investment, projected returns, cost of capital, ROI/IRR/payback, sensitivity, breakeven | **Scenario Analysis**: Multiple scenarios, key variables, data tables, tornado charts, Monte Carlo simulation, risk-adjusted returns

## Workflow Process

**Requirements**: Define purpose, identify users/sophistication, determine scope (horizon/detail/frequency), list assumptions/sources, establish validation criteria | **Design**: Sketch flow, define sheets, establish naming, design inputs, plan validation | **Foundation**: Create cover (metadata/version), build assumptions (documented inputs), setup validation framework, create documentation | **Implementation**: Start simple, test incrementally, add complexity, implement error handling, document formulas | **Validation**: Run checks, stress test, reconcile totals, cross-check outputs, peer review | **Finishing**: Format consistently, create executive summary, add charts, write documentation, lock structure

## Best Practices for Production Models

### Formula Hygiene
- Never hide rows or columns with formulas
- Avoid merged cells in calculation areas
- Keep formulas short and readable (break complex calculations into steps)
- Use consistent reference styles (prefer absolute references for assumptions)
- Avoid volatile functions (NOW, TODAY, RAND) in large models

### Error Prevention
- Always use IFERROR to prevent #DIV/0, #N/A, #VALUE errors
- Implement input validation (Data Validation dropdowns, ranges)
- Add warning flags for unusual results
- Create boundary checks (negative values where impossible)
- Test with edge cases (zero values, very large numbers)

### Performance Optimization
- Minimize array formulas in large datasets
- Use manual calculation mode for very large models
- Replace complex formulas with helper columns
- Avoid entire column references (A:A)
- Turn off automatic chart updates during data entry

### Audit Trail Documentation
Every change should be logged:

```python
def create_change_log(ws):
    """Create a change log sheet for model modifications."""
    headers = ['Version', 'Date', 'Author', 'Section Modified',
               'Change Description', 'Rationale', 'Reviewed By']

    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col)
        cell.value = header
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color='4472C4',
                                end_color='4472C4', fill_type='solid')
```

## Example: Creating a Complete DCF Model

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from datetime import datetime

def create_dcf_model(company_name, projection_years=5):
    """Create complete DCF valuation model. Returns saved Excel workbook path."""
    wb = Workbook()
    wb.remove(wb.active)

    # 1. Cover Sheet: Title, purpose, version, created date, status
    cover = wb.create_sheet('Cover')
    cover['A1'] = f'{company_name} - DCF Valuation Model'
    cover['A1'].font = Font(size=18, bold=True)
    # ... metadata cells

    # 2. Assumptions Sheet: Revenue (Base, Growth %), Margins (EBITDA %), Cost of Capital (WACC, Terminal Growth)
    assumptions = wb.create_sheet('Assumptions')
    # Color-code inputs (yellow fill), add sources in column C
    # Define named ranges: BaseRevenue, RevenueGrowth, EBITDAMargin, WACC, TerminalGrowth

    # 3. Projections Sheet: Revenue, EBITDA, Free Cash Flow formulas
    projections = wb.create_sheet('Projections')
    # Revenue: =BaseRevenue*(1+RevenueGrowth)^year
    # EBITDA: =Revenue*EBITDAMargin
    # FCF: =EBITDA*0.7 (simplified)

    # 4. Valuation Sheet: PV calculations, Terminal Value, Enterprise Value
    valuation = wb.create_sheet('Valuation')
    # PV = FCF/(1+WACC)^year
    # Terminal Value = Final FCF*(1+g)/(WACC-g)
    # EV = Sum of PVs + PV of Terminal Value

    # 5. Validation Sheet: VAL-001 (all inputs populated), VAL-002 (growth <50%), VAL-003 (terminal growth < WACC)
    validation = wb.create_sheet('Validation')
    # Use IF formulas for PASS/FAIL status

    # 6. Documentation Sheet: Methodology, key formulas, changelog
    docs = wb.create_sheet('Documentation')

    wb.save(f'/mnt/user-data/outputs/{company_name.replace(" ", "_")}_DCF_Model.xlsx')
    return filename
```

## Quality Checklist

**Structure**: Cover sheet (metadata), assumptions (sources/dates), input/calculation/output separation, validation sheet, documentation | **Formulas**: Reference assumptions (no hard-coding), IFERROR wrapping, named ranges, auditable, no unintentional circular refs | **Formatting**: Color coding, professional fonts, clear headers, number formatting, charts | **Validation**: All checks pass/noted, balance checks, sense checks, sensitivity analysis, peer review | **Audit Trail**: Documented assumptions/sources, change log, version number, methodology explained, review sign-off

## Advanced Topics

### Scenario Management with Data Tables
Use Excel's Data Table feature for sophisticated scenario analysis:

```python
def create_sensitivity_table(ws, output_cell, variable_cells):
    """Create two-way data table. variable_cells: dict with 'row_variable' and 'col_variable' cell refs."""
    # Create table structure
    ws['A1'] = 'Sensitivity Analysis'
    ws['B1'] = output_cell

    # Column headers (first variable values)
    for i, val in enumerate([-20, -10, 0, 10, 20], 2):
        ws.cell(row=1, column=i, value=f'{val}%')

    # Row headers (second variable values)
    for i, val in enumerate([-20, -10, 0, 10, 20], 2):
        ws.cell(row=i, column=1, value=f'{val}%')

    # Excel data table will populate the interior cells automatically
    # User must manually: Select range → Data → What-If Analysis → Data Table
```

### Monte Carlo Simulation Integration
For risk analysis, integrate with Python's statistical libraries:

```python
import numpy as np
from scipy import stats

def monte_carlo_simulation(base_value, volatility, simulations=10000):
    """Run Monte Carlo for uncertain inputs. Returns array of simulated values."""
    np.random.seed(42)  # For reproducibility
    std_dev = base_value * volatility
    simulated_values = np.random.normal(base_value, std_dev, simulations)

    # Calculate statistics
    results = {
        'mean': np.mean(simulated_values),
        'median': np.median(simulated_values),
        'std': np.std(simulated_values),
        'p10': np.percentile(simulated_values, 10),
        'p90': np.percentile(simulated_values, 90),
    }

    return simulated_values, results
```

### VBA for Advanced Automation
For models requiring user forms or custom functions:

```vba
Function PV_Custom(rate As Double, nper As Integer, cashFlows As Range) As Double
    'Custom present value function with detailed audit trail
    Dim i As Integer
    Dim pv_sum As Double

    pv_sum = 0
    For i = 1 To nper
        pv_sum = pv_sum + cashFlows.Cells(i) / ((1 + rate) ^ i)
    Next i

    PV_Custom = pv_sum

    'Log calculation for audit trail
    ThisWorkbook.Sheets("Audit").Range("A1").Value = _
        "PV calculated: " & Format(Now, "yyyy-mm-dd hh:mm:ss")
End Function
```

## Integration with External Tools

### Power BI for Dashboard Creation
Export key model outputs to Power BI for executive dashboards:

```python
# Export summary data for Power BI consumption
def export_for_powerbi(workbook, output_path):
    """
    Export model outputs in Power BI friendly format.

    Creates a structured table with:
    - Metric name
    - Value
    - Unit
    - Scenario
    - Timestamp
    """
    import pandas as pd

    # Extract key metrics from model
    # (Implementation would read from Excel and create structured DataFrame)
    pass
```

### Python Analytics Integration
Link Excel models with Python for advanced analytics:

```python
# Read Excel model and run advanced analytics
def analyze_model_outputs(excel_path):
    """
    Read Excel model outputs and perform statistical analysis.

    Can perform:
    - Regression analysis
    - Time series forecasting
    - Correlation analysis
    - Risk metrics calculation
    """
    import pandas as pd

    # Read outputs
    df = pd.read_excel(excel_path, sheet_name='Outputs')

    # Perform analysis
    # (Implementation would include statistical tests, visualizations)
    pass
```

## Critical Reminders

Audit trail is non-negotiable (every number traceable), assumptions drive everything (document sources), validate ruthlessly (build error checks), format for clarity (color coding matters), document methodology (others must understand), test extensively (edge cases and stress scenarios), maintain version control, professional presentation (reflects credibility).

## Common Pitfalls

Avoid: circular references without purpose, hard-coded values in formulas, missing error handling (#DIV/0, #N/A), inconsistent time periods, hidden rows with calculations, entire column references (A:A), missing documentation, no validation checks, overly complex nested formulas, unprotected formulas.
