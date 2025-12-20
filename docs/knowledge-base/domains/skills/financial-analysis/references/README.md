# Financial Analysis and Modeling Skill - Complete Package

## Overview
Enterprise-grade financial analysis and modeling capability that generates sophisticated Excel-based models with comprehensive audit trails, validation logic, and professional formatting. This skill enables creation of production-ready financial models that meet institutional quality standards and can withstand external scrutiny.

## What This Skill Provides

### 1. Complete Framework (`SKILL.md`)
The main skill document provides:
- **Audit Trail Requirements:** Complete transparency and traceability standards
- **Model Architecture:** Standard structure for all financial models
- **Professional Standards:** Institutional-grade practices
- **Excel Implementation:** Detailed Python code examples using openpyxl
- **Model Types:** DCF, Three-Statement, Budget vs Actual, ROI, Scenario Analysis
- **Workflow Process:** Step-by-step model creation methodology
- **Quality Checklist:** Pre-delivery validation requirements
- **Advanced Topics:** Scenario management, Monte Carlo, VBA integration

### 2. Reference Materials (`references/`)
Four comprehensive reference documents:
- **Excel Formula Reference:** Complete guide to financial formulas
- **Modeling Standards:** Authoritative professional standards
- **Financial Ratios:** Calculation and interpretation guide
- **References README:** Integration guide and learning path
1
### 3. Production Scripts (`scripts/`)
Five battle-tested Python implementations:
- **dcf_model.py:** DCF valuation model generator
- **three_statement_model.py:** Integrated financial statement modeler
- **budget_vs_actual.py:** Variance analysis generator
- **model_validation.py:** Automated quality assurance
- **excel_formatter.py:** Standardized formatting utilities

## Quick Start

### For Claude Users
Simply ask Claude to create a financial model:

```
"Create a DCF model for ACME Corp with 5-year projections"
"Build a three-statement model for financial planning"
"Generate a budget vs actual analysis for Q1 2024"
```

Claude will automatically apply this skill's frameworks and generate production-ready models.

### For Direct Python Use

**Install Dependencies:**
```bash
pip install openpyxl pandas numpy --break-system-packages
```

**Create a DCF Model:**
```python
from scripts.dcf_model import create_dcf_model

model_path = create_dcf_model(
    company_name='Target Company',
    projection_years=5
)
# Output: Professional DCF model with all required components
```

**Create a Three-Statement Model:**
```python
from scripts.three_statement_model import create_three_statement_model

model_path = create_three_statement_model(
    company_name='Growth Corp',
    base_revenue=1000,
    projection_years=5
)
# Output: Fully integrated income statement, balance sheet, cash flow
```

**Validate Any Model:**
```python
from scripts.model_validation import validate_model

passed, report = validate_model('path/to/model.xlsx')
print(report)
# Output: Comprehensive validation report with all checks
```

## Directory Structure

```
financial-analysis/
├── SKILL.md                    # Main skill documentation (896 lines)
├── README.md                   # This file
├── references/                 # Reference materials
│   ├── README.md              # Reference guide and integration
│   ├── excel-formula-reference.md    # Formula patterns and best practices
│   ├── modeling-standards.md         # Professional standards and guidelines
│   └── financial-ratios.md          # Ratio calculations and interpretation
└── scripts/                    # Python implementation scripts
    ├── README.md              # Scripts documentation and workflows
    ├── dcf_model.py          # DCF valuation generator
    ├── three_statement_model.py     # Integrated statements
    ├── budget_vs_actual.py          # Variance analysis
    ├── model_validation.py          # Quality assurance
    └── excel_formatter.py           # Formatting utilities
```

## Core Principles

### 1. Audit Trail First
Every model maintains:
- Complete transparency: All formulas visible and documented
- Source documentation: Every assumption with source and date
- Version control: Model version, dates, change log
- Data lineage: Clear flow from inputs to outputs
- Validation results: Embedded checks with pass/fail indicators

### 2. Professional Quality
Models follow:
- **FAST Standard:** Flexible, Appropriate, Structured, Transparent
- **Color Coding:** Yellow (input), Gray (calc), Green (output)
- **Error Handling:** IFERROR wrapping on all risky operations
- **Named Ranges:** Descriptive names for key assumptions
- **Validation Framework:** Automated checks ensure accuracy

### 3. Enterprise-Grade Output
Generated models include:
- Cover sheet with metadata and executive summary
- Assumptions sheet with complete documentation
- Calculation sheets with clear sectioning
- Output sheets with visualizations
- Validation sheet with automated checks
- Documentation explaining methodology

## Use Cases

### Investment Analysis
- **DCF Valuations:** Enterprise and equity value estimation
- **Peer Benchmarking:** Comparative valuation analysis
- **Scenario Analysis:** Base, upside, downside cases
- **Sensitivity Testing:** Key driver impact assessment

### Financial Planning
- **Three-Statement Models:** Comprehensive projections
- **Budget Creation:** Detailed revenue and expense planning
- **Cash Flow Forecasting:** Working capital management
- **Capital Planning:** CapEx and financing requirements

### Performance Management
- **Budget vs Actual:** Variance analysis and explanation
- **KPI Dashboards:** Key metric tracking
- **Ratio Analysis:** Profitability, liquidity, efficiency
- **Trend Analysis:** Historical performance evaluation

### Due Diligence
- **Quality of Earnings:** Profitability analysis
- **Working Capital Assessment:** Cash conversion efficiency
- **Leverage Analysis:** Debt capacity and coverage
- **Growth Sustainability:** Historical and projected

### Credit Analysis
- **Coverage Ratios:** Interest and debt service coverage
- **Leverage Metrics:** Debt/EBITDA, debt/equity
- **Liquidity Assessment:** Current and quick ratios
- **Altman Z-Score:** Bankruptcy prediction

## Key Features

### Automated Generation
- Pre-built templates for common model types
- Consistent structure across all models
- Professional formatting applied automatically
- Standard validation checks included
- Complete audit trail established

### Comprehensive Validation
- Structural checks (sheets, naming, organization)
- Formula validation (hardcoded values, error handling)
- Balance verification (accounting identities)
- Format consistency (colors, fonts, numbers)
- Documentation completeness

### Professional Formatting
- Standard color schemes for inputs/calculations/outputs
- Corporate-quality fonts and alignments
- Proper number formatting (currency, percentages)
- Clear section headers and organization
- Print-ready layouts

### Full Audit Trail
- All assumptions sourced and dated
- Calculation methodology documented
- Formula explanations included
- Change log maintained
- Version control embedded

## Integration Points

### With Other Skills

**Portfolio Management:**
- Feed valuation outputs into investment decisions
- Monitor portfolio company performance
- Track against investment theses

**Due Diligence:**
- Generate models during company analysis
- Validate management projections
- Assess historical performance

**Benchmarking:**
- Compare financial metrics to peers
- Identify performance gaps
- Support investment recommendations

### With External Systems

**Power BI:**
- Export model outputs for dashboards
- Create executive visualizations
- Track performance metrics

**Python Analytics:**
- Statistical analysis of results
- Monte Carlo simulations
- Advanced forecasting

**Excel:**
- Direct user interaction and updates
- Scenario management
- Sensitivity analysis tables

## Learning Path

### Beginner (Getting Started)
1. Read `SKILL.md` sections:
   - Core Principles
   - Model Architecture
   - Excel Implementation Guidelines

2. Review `references/modeling-standards.md`:
   - Model Design Principles
   - Color Coding Standard
   - Sheet Organization

3. Run example scripts:
   ```python
   from scripts.dcf_model import create_dcf_model
   create_dcf_model('Example Co', 3)
   ```

### Intermediate (Building Models)
1. Study `SKILL.md` sections:
   - Common Financial Model Types
   - Workflow Process
   - Best Practices

2. Master `references/excel-formula-reference.md`:
   - Financial functions (NPV, IRR)
   - Lookup patterns (INDEX-MATCH)
   - Error handling requirements

3. Create custom models:
   ```python
   from scripts.three_statement_model import ThreeStatementModel
   # Customize with specific assumptions
   ```

### Advanced (Enterprise Standards)
1. Implement complete framework from `SKILL.md`:
   - Advanced formula patterns
   - Validation framework
   - Quality checklist

2. Apply all `references/modeling-standards.md`:
   - Complete validation framework
   - Model governance
   - Documentation standards

3. Extend scripts for specific needs:
   - Custom model types
   - Industry-specific metrics
   - Advanced scenarios

## Quality Standards

Every model generated by this skill:

✓ **Passes Validation Checks**
- Balance sheet balances
- All formulas have error handling
- No hardcoded values in calculation cells
- Color coding is consistent
- Documentation is complete

✓ **Meets Professional Standards**
- FAST Standard compliance
- Institutional-quality formatting
- Complete audit trail
- Version control
- Change log

✓ **Supports Decision-Making**
- Clear outputs and summaries
- Multiple scenarios
- Sensitivity analysis
- Professional visualizations
- Executive-ready presentation

## Best Practices Summary

### Before Building
- Define model purpose and decision it supports
- Identify users and sophistication level
- Determine scope and update frequency
- List all assumptions and data sources
- Establish validation criteria

### During Development
- Start simple with base case
- Test incrementally
- Add complexity gradually
- Implement error handling throughout
- Document as you build

### Before Delivery
- Run all validation checks
- Stress test with edge cases
- Verify color coding
- Complete documentation
- Peer review

## Support Resources

### Documentation
- **Main Skill:** `SKILL.md` - Complete implementation guide
- **References:** `references/` - Standards and formulas
- **Scripts:** `scripts/` - Code documentation and examples
- **READMEs:** Integration guides in each directory

### Standards Referenced
- FAST Standard (F1F9 consortium)
- Basel Committee guidelines
- CFA Institute standards
- GARP frameworks
- Microsoft Excel best practices

### External Resources
- Investopedia (definitions and benchmarks)
- SEC EDGAR (public company financials)
- Industry reports (sector benchmarks)
- Federal Reserve data (economic indicators)

## Version History

### Version 1.0.0 (December 2024)
- Initial comprehensive skill release
- Complete SKILL.md documentation
- Four reference documents
- Five production scripts
- Full integration documentation

## Contributing

### Adding New Features
1. Follow existing patterns and structure
2. Document thoroughly with examples
3. Include validation checks
4. Update relevant README files
5. Test with multiple scenarios

### Reporting Issues
Document:
- Specific use case or model type
- Expected vs actual behavior
- Steps to reproduce
- Suggested improvements

## Critical Success Factors

1. **Audit Trail:** Every number must be traceable to source
2. **Validation:** Build checks that catch errors automatically
3. **Documentation:** Others must understand your model
4. **Formatting:** Professional presentation matters
5. **Testing:** Try edge cases and stress scenarios
6. **Standards:** Follow institutional best practices
7. **Quality:** Models reflect your credibility

## Final Notes

This skill represents:
- Decades of financial modeling best practices
- Standards from major investment banks
- Big Four accounting firm requirements
- Professional certification knowledge (CFA, FRM)
- Condensed institutional wisdom

Use it as the foundation for all financial modeling work to ensure outputs that:
- **Stand up to scrutiny** from auditors, investors, regulators
- **Support confident decisions** with validated analysis
- **Communicate clearly** to diverse audiences
- **Maintain professional standards** throughout
- **Save time** through automation and templates
- **Ensure quality** with systematic validation

Every model generated using this skill can be delivered with confidence, knowing it meets institutional standards and includes the complete audit trail necessary for serious financial analysis.

---

**Ready to Start?**

1. **For Quick Models:** Use the Python convenience functions
2. **For Custom Models:** Extend the provided classes
3. **For Learning:** Follow the learning path in order
4. **For Standards:** Review the references first
5. **For Validation:** Run model_validation.py regularly

The skill is designed to accelerate model development from days to hours while maintaining enterprise-grade quality. Start with the examples, customize as needed, and build with confidence knowing every model includes the professional standards and audit trails required for critical financial decisions.
