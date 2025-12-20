# Financial Analysis Scripts

## Overview
Production-ready Python scripts for automated financial model generation using openpyxl. These scripts create enterprise-grade Excel models with comprehensive audit trails, validation logic, and professional formatting.

## Available Scripts

### 1. dcf_model.py
**Purpose:** Generate complete Discounted Cash Flow (DCF) valuation models.

**Key Features:**
- Automated revenue and cash flow projections
- Terminal value calculation (perpetuity growth method)
- NPV and present value calculations
- Multiple scenario support
- Complete audit trail
- Validation checks

**Quick Start:**
```python
from dcf_model import create_dcf_model

model_path = create_dcf_model(
    company_name='ACME Corp',
    projection_years=5
)
```

**Generated Sheets:**
- Cover: Model metadata and summary
- Assumptions: All inputs with documentation
- Projections: Revenue and cash flow forecasts
- Valuation: NPV calculations and enterprise value
- Validation: Automated checks
- Documentation: Methodology explanation

**When to Use:**
- Company valuation for M&A
- Investment analysis
- Portfolio company assessment
- Fair value estimation
- Teaching DCF methodology

---

### 2. three_statement_model.py
**Purpose:** Create fully integrated three-statement financial models.

**Key Features:**
- Income statement projections
- Balance sheet with automatic balancing
- Cash flow statement (indirect method)
- Complete statement integration
- Working capital calculations
- Depreciation and CapEx tracking

**Quick Start:**
```python
from three_statement_model import create_three_statement_model

model_path = create_three_statement_model(
    company_name='Sample Corp',
    base_revenue=1000,
    initial_cash=100,
    projection_years=5
)
```

**Generated Sheets:**
- Cover: Model overview
- Assumptions: All operating assumptions
- Income Statement: Revenue through net income
- Balance Sheet: Assets, liabilities, equity
- Cash Flow: Operating, investing, financing
- Validation: Balance checks
- Documentation: Methodology

**When to Use:**
- Comprehensive financial projections
- Business planning
- Lending analysis
- Working capital modeling
- Scenario planning

---

### 3. budget_vs_actual.py
**Purpose:** Generate budget variance analysis workbooks.

**Key Features:**
- Automated variance calculations
- Favorable/unfavorable flagging
- Significant variance highlighting
- Commentary sections
- Rolling forecast capability
- Threshold analysis

**Quick Start:**
```python
from budget_vs_actual import create_budget_vs_actual_model

model_path = create_budget_vs_actual_model(
    company_name='Sample Corp',
    period='Q1 2024',
    revenue_items={'Product Sales': 5000000, 'Services': 2000000},
    expense_items={'COGS': 3000000, 'OpEx': 1500000},
    actual_revenue={'Product Sales': 5200000, 'Services': 1850000},
    actual_expenses={'COGS': 3100000, 'OpEx': 1450000}
)
```

**Generated Sheets:**
- Cover: Summary metrics
- Summary: High-level variance analysis
- Detail: Significant variances with commentary
- Variance Analysis: Threshold and trend analysis
- Charts: Visual comparisons

**When to Use:**
- Monthly/quarterly performance reviews
- Board reporting
- Management dashboards
- Operational analysis
- Forecast accuracy assessment

---

### 4. model_validation.py
**Purpose:** Automated validation of existing Excel financial models.

**Key Features:**
- Structural checks (required sheets, naming)
- Formula validation (hardcoded values, error handling)
- Formatting verification (color coding, number formats)
- Audit trail checks (assumptions, version control)
- Balance sheet verification
- Comprehensive reporting

**Quick Start:**
```python
from model_validation import validate_model

passed, report = validate_model(
    workbook_path='path/to/model.xlsx',
    report_path='validation_report.txt'
)

print(report)
```

**Command Line Usage:**
```bash
python model_validation.py path/to/model.xlsx
```

**Validation Categories:**
- Structural: Sheet organization and naming
- Formula: Hardcoded values, error handling, circular references
- Formatting: Color coding, number formats
- Audit: Documentation, version control
- Balance: Accounting identities

**When to Use:**
- Pre-delivery model review
- Quality assurance checks
- Training on model standards
- Periodic model audits
- Identifying improvement areas

---

### 5. excel_formatter.py
**Purpose:** Standardized formatting utilities for consistent model styling.

**Key Features:**
- Professional color schemes
- Standard fonts and alignments
- Pre-defined cell styles
- Table creation utilities
- Header formatting
- Number format application

**Quick Start:**
```python
from excel_formatter import ModelFormatter
from openpyxl import Workbook

wb = Workbook()
ws = wb.active

# Format title
ModelFormatter.format_title(ws, 'A1', 'Financial Model')

# Format header row
headers = ['Year', 'Revenue', 'COGS', 'Gross Profit']
ModelFormatter.format_header_row(ws, 3, headers)

# Format input cell
ModelFormatter.format_input_cell(ws, 'B4', 1000000, '$#,##0')

# Format calculation cell
ModelFormatter.format_calculation_cell(ws, 'C4', '=B4*0.6', '$#,##0')

# Format output cell
ModelFormatter.format_output_cell(ws, 'D4', '=B4-C4', '$#,##0')
```

**Available Formatters:**
- `format_title()`: Large title formatting
- `format_heading()`: Section headings
- `format_header_row()`: Blue header with white text
- `format_input_cell()`: Yellow background for inputs
- `format_calculation_cell()`: Gray background for formulas
- `format_output_cell()`: Green background for results
- `format_total_row()`: Bold with underline
- `apply_currency_format()`: Currency formatting
- `apply_percentage_format()`: Percentage formatting
- `set_column_widths()`: Set multiple column widths
- `freeze_panes()`: Freeze rows/columns

**When to Use:**
- Ensuring consistent formatting across models
- Quickly styling new sheets
- Applying corporate standards
- Creating professional presentations
- Teaching formatting best practices

---

## Common Workflows

### Workflow 1: Create New DCF Model
```python
from dcf_model import create_dcf_model

# Generate model with defaults
model_path = create_dcf_model(
    company_name='Target Company',
    projection_years=5
)

# Customize by opening and adjusting assumptions
# Model is ready for use with proper structure
```

### Workflow 2: Build Full Financial Projection
```python
from three_statement_model import ThreeStatementModel, CompanyProfile, ModelAssumptions

# Define company starting position
company = CompanyProfile(
    name='Growth Corp',
    fiscal_year_end='December 31',
    cash=50000,
    accounts_receivable=20000,
    inventory=15000,
    # ... other balance sheet items
)

# Define assumptions
assumptions = ModelAssumptions(
    base_revenue=100000,
    revenue_growth_rate=0.20,
    cogs_percent_revenue=0.55,
    # ... other assumptions
)

# Create model
model = ThreeStatementModel(company, assumptions, projection_years=5)
output_path = model.create_model('/path/to/output.xlsx')
```

### Workflow 3: Variance Analysis
```python
from budget_vs_actual import create_budget_vs_actual_model

# Define budget
revenue_budget = {
    'Product A Sales': 5000000,
    'Product B Sales': 3000000,
    'Service Revenue': 2000000,
}

expense_budget = {
    'COGS': 6000000,
    'Sales & Marketing': 2000000,
    'R&D': 1000000,
    'G&A': 800000,
}

# Add actuals (if available)
actual_revenue = {
    'Product A Sales': 5200000,
    'Product B Sales': 2800000,
    'Service Revenue': 2100000,
}

# Generate analysis
model_path = create_budget_vs_actual_model(
    company_name='My Company',
    period='Q1 2024',
    revenue_items=revenue_budget,
    expense_items=expense_budget,
    actual_revenue=actual_revenue
)
```

### Workflow 4: Validate Existing Model
```python
from model_validation import ModelValidator

# Load and validate
validator = ModelValidator('existing_model.xlsx')
results = validator.run_all_checks()

# Review results
for result in results:
    if result.status == 'FAIL':
        print(f'{result.check_id}: {result.details}')

# Generate report
report = validator.generate_report('validation_report.txt')
```

### Workflow 5: Create Formatted Template
```python
from excel_formatter import ModelFormatter, create_assumptions_sheet
from openpyxl import Workbook

wb = Workbook()

# Create standard sheets with formatting
assumptions_ws = create_assumptions_sheet(wb)
validation_ws = create_validation_sheet(wb)

# Add custom sheet with standard formatting
ws = wb.create_sheet('Analysis')
ModelFormatter.format_title(ws, 'A1', 'Financial Analysis')

headers = ['Metric', 'Value', 'Benchmark', 'Status']
ModelFormatter.format_header_row(ws, 3, headers)

# Set column widths
ModelFormatter.set_column_widths(ws, {
    'A': 30, 'B': 15, 'C': 15, 'D': 12
})

wb.save('formatted_template.xlsx')
```

## Installation and Setup

### Required Packages
```bash
pip install openpyxl pandas numpy --break-system-packages
```

### Import Patterns
```python
# For model generation
from dcf_model import create_dcf_model
from three_statement_model import create_three_statement_model
from budget_vs_actual import create_budget_vs_actual_model

# For validation
from model_validation import validate_model, ModelValidator

# For formatting
from excel_formatter import ModelFormatter
```

## Best Practices

### 1. Always Start with Defaults
Use the convenience functions (`create_*_model`) to generate models with sensible defaults, then customize.

### 2. Validate After Creation
```python
# Create model
model_path = create_dcf_model('Company', 5)

# Validate it
passed, report = validate_model(model_path)
print(report)
```

### 3. Use Type Hints
Scripts include dataclasses and type hints for clarity:
```python
@dataclass
class CompanyProfile:
    name: str
    fiscal_year_end: str
    cash: float = 0
```

### 4. Leverage Existing Formatting
Use `excel_formatter.py` functions consistently:
```python
# Don't manually create fills
ModelFormatter.format_input_cell(ws, 'B5', value, '$#,##0')

# Instead of:
ws['B5'] = value
ws['B5'].fill = PatternFill(...)  # Don't do this
```

### 5. Document Custom Models
When extending scripts:
```python
def create_custom_model(params):
    """
    Brief description of what this creates.

    Args:
        params: Description of parameters

    Returns:
        Path to created model
    """
    # Implementation
```

## Extending the Scripts

### Adding New Model Types
1. Follow the pattern from existing scripts:
   - Create dataclasses for inputs
   - Main generator class with `create_model()` method
   - Sheet creation methods (`_create_*_sheet()`)
   - Convenience function for quick generation

2. Example skeleton:
```python
from dataclasses import dataclass
from openpyxl import Workbook

@dataclass
class ModelInputs:
    """Define required inputs."""
    parameter1: float
    parameter2: str

class CustomModel:
    """Model generator."""

    def __init__(self, inputs: ModelInputs):
        self.inputs = inputs
        self.wb = Workbook()

    def create_model(self, output_path: str) -> str:
        self._create_cover_sheet()
        self._create_calculations()
        self.wb.save(output_path)
        return output_path

    def _create_cover_sheet(self):
        # Implementation
        pass

def create_custom_model(param1, param2, output_path=None):
    """Convenience function."""
    inputs = ModelInputs(param1, param2)
    model = CustomModel(inputs)
    if output_path is None:
        output_path = '/mnt/user-data/outputs/custom_model.xlsx'
    return model.create_model(output_path)
```

### Adding New Validation Checks
Extend `ModelValidator` class:
```python
def check_custom_requirement(self) -> ValidationResult:
    """Check for custom requirement."""
    # Implementation
    result = ValidationResult(
        check_id='CUS-001',
        check_description='Custom requirement check',
        status='PASS',
        details='Check passed'
    )
    self.results.append(result)
    return result
```

### Adding New Formatters
Add static methods to `ModelFormatter`:
```python
@staticmethod
def format_custom_style(ws, cell: str, value):
    """Apply custom formatting style."""
    ws[cell] = value
    ws[cell].font = Font(name='Calibri', size=11, italic=True)
    ws[cell].fill = PatternFill(start_color='...')
```

## Testing Scripts

### Manual Testing
```python
if __name__ == '__main__':
    # Test model creation
    output = create_dcf_model('Test Company', 3)
    print(f'Model created: {output}')

    # Validate it
    passed, report = validate_model(output)
    print('Validation:', 'PASSED' if passed else 'FAILED')
```

### Automated Testing
Create test files following this pattern:
```python
def test_dcf_model_creation():
    """Test DCF model generates successfully."""
    output = create_dcf_model('Test', 3, output_path='/tmp/test.xlsx')
    assert os.path.exists(output)

    wb = load_workbook(output)
    assert 'Cover' in wb.sheetnames
    assert 'Assumptions' in wb.sheetnames
    # More assertions...
```

## Performance Considerations

### Large Models
For models with many years or line items:
```python
# Use manual calculation mode in Excel after generation
# File → Options → Formulas → Manual calculation

# Optimize formulas
# Use single-cell references instead of ranges where possible
# Avoid volatile functions (TODAY, NOW, RAND)
```

### Memory Usage
```python
# For very large models, create in chunks
# Save intermediate progress
# Use data_only=True when reading for validation
wb = load_workbook(path, data_only=True)
```

## Common Issues and Solutions

### Issue 1: File Permissions
**Error:** `PermissionError: [Errno 13]`
**Solution:** Ensure output directory exists and file isn't open in Excel

### Issue 2: Missing Modules
**Error:** `ModuleNotFoundError: No module named 'openpyxl'`
**Solution:** `pip install openpyxl --break-system-packages`

### Issue 3: Formula Errors in Generated Model
**Problem:** #DIV/0! or #N/A errors appear
**Solution:** Check error handling in script, ensure IFERROR wrapping

### Issue 4: Incorrect Cell References
**Problem:** Formulas reference wrong cells
**Solution:** Verify `get_column_letter()` usage and row counting logic

### Issue 5: Formatting Not Applied
**Problem:** Colors or fonts don't appear
**Solution:** Ensure PatternFill and Font objects created correctly

## Version Control

### Script Versions
Each script includes version information:
```python
"""
Script Name
Version: 1.0.0
Last Updated: 2024-12-07
"""
```

### Model Versions
Generated models include version in Cover sheet:
```python
ws['A4'] = 'Version:'
ws['B4'] = '1.0'
```

## Support and Resources

### Internal Documentation
- See `../SKILL.md` for overall framework
- Check `../references/` for calculation formulas
- Review `../README.md` for skill overview

### External Resources
- openpyxl documentation: https://openpyxl.readthedocs.io/
- Python dataclasses: https://docs.python.org/3/library/dataclasses.html
- Financial modeling standards: See `../references/modeling-standards.md`

### Getting Help
1. Review script docstrings
2. Check examples in `if __name__ == '__main__':` sections
3. Run with test data first
4. Use validation script to identify issues

## Future Enhancements

### Planned Features
- ROI/IRR calculator model
- Scenario analysis automation
- Sensitivity analysis tables
- Monte Carlo simulation integration
- Power BI export functionality

### Contribution Guidelines
When adding new scripts:
1. Follow existing patterns
2. Include comprehensive docstrings
3. Add type hints
4. Create convenience functions
5. Include usage examples
6. Update this README

These scripts represent production-ready implementations of financial modeling best practices, enabling rapid generation of institutional-quality models with complete audit trails and validation.
