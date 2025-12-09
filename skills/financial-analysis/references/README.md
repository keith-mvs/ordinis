# Financial Analysis References

## Overview
This directory contains comprehensive reference materials for professional financial modeling and analysis. These documents establish standards, provide formulas, and offer best practices for creating institutional-quality financial models.

## Reference Documents

### 1. Excel Formula Reference (`excel-formula-reference.md`)
**Purpose:** Comprehensive guide to Excel formulas commonly used in financial modeling.

**Contents:**
- Financial functions (NPV, IRR, PV, FV, PMT)
- Date and time functions
- Lookup and reference functions (VLOOKUP, INDEX-MATCH, XLOOKUP)
- Conditional logic (IF, AND, OR)
- Error handling (IFERROR, IFNA)
- Array formulas and SUMPRODUCT
- Text and formatting functions
- Statistical functions
- Financial ratio formulas

**Key Features:**
- Audit-friendly formula patterns
- Error handling requirements
- Best practices for each function
- Excel implementation examples
- Common calculation patterns

**When to Use:**
- Building formulas in financial models
- Implementing calculations with proper error handling
- Looking up syntax for specific functions
- Finding best practices for formula construction

---

### 2. Modeling Standards (`modeling-standards.md`)
**Purpose:** Authoritative standards for institutional-quality financial modeling.

**Contents:**
- Model design principles (transparency, simplicity, consistency)
- Structural standards (sheet organization, color coding)
- Formula standards (reference patterns, error handling)
- Validation framework requirements
- Documentation standards
- Quality assurance checklist
- Model governance procedures

**Key Features:**
- Industry-standard best practices
- FAST Standard alignment
- Comprehensive validation requirements
- Professional formatting guidelines
- Model approval process

**When to Use:**
- Starting a new financial model
- Reviewing existing models for compliance
- Establishing modeling standards for a team
- Preparing models for external review
- Training on professional modeling practices

---

### 3. Financial Ratios and Metrics (`financial-ratios.md`)
**Purpose:** Complete reference for calculating and interpreting financial ratios.

**Contents:**
- Profitability ratios (margins, ROE, ROA, ROIC)
- Liquidity ratios (current, quick, cash)
- Leverage ratios (debt/equity, interest coverage)
- Efficiency ratios (turnover, DSO, DIO, DPO)
- Valuation ratios (P/E, EV/EBITDA, P/B)
- Growth metrics
- Industry-specific metrics
- Credit analysis metrics

**Key Features:**
- Formula and Excel implementation for each ratio
- Interpretation guidelines
- Typical industry ranges
- Interrelationships between ratios
- Quality assessment framework

**When to Use:**
- Analyzing company financial performance
- Creating financial dashboards
- Benchmarking against peers
- Due diligence analysis
- Credit assessment
- Valuation exercises

---

## Quick Reference Guide

### For Building Financial Models

**Start Here:**
1. Review `modeling-standards.md` for structural requirements
2. Use `excel-formula-reference.md` for formula implementation
3. Apply `financial-ratios.md` for performance metrics

**Key Checklist:**
- [ ] Follow color coding standards (yellow=input, gray=calc, green=output)
- [ ] Wrap all divisions and lookups in error handling
- [ ] Document all assumptions with sources and dates
- [ ] Create validation checks for all major calculations
- [ ] Use named ranges for key assumptions
- [ ] Include version control information

### For Financial Analysis

**Start Here:**
1. Use `financial-ratios.md` to identify relevant metrics
2. Reference `excel-formula-reference.md` for calculation formulas
3. Apply `modeling-standards.md` for documentation requirements

**Key Checklist:**
- [ ] Calculate all relevant profitability metrics
- [ ] Assess liquidity and leverage ratios
- [ ] Analyze efficiency metrics
- [ ] Compare to industry benchmarks
- [ ] Examine trends over time
- [ ] Document interpretation and conclusions

### For Model Validation

**Start Here:**
1. Review `modeling-standards.md` validation framework
2. Check `excel-formula-reference.md` for proper formula patterns
3. Verify calculations using `financial-ratios.md`

**Key Checklist:**
- [ ] Balance sheet balances
- [ ] No hardcoded values in calculation cells
- [ ] All formulas have error handling
- [ ] Color coding is consistent
- [ ] Documentation is complete
- [ ] All validation checks pass

## Common Use Cases

### Use Case 1: Building a DCF Model
**References to Review:**
1. `modeling-standards.md` - Model structure and requirements
2. `excel-formula-reference.md` - NPV, IRR, growth calculations
3. `financial-ratios.md` - WACC calculation, working capital metrics

### Use Case 2: Creating a Budget vs Actual Report
**References to Review:**
1. `modeling-standards.md` - Dashboard design principles
2. `excel-formula-reference.md` - Variance calculations, conditional formatting
3. `financial-ratios.md` - Performance metrics

### Use Case 3: Peer Group Benchmarking
**References to Review:**
1. `financial-ratios.md` - All ratio categories for comparison
2. `excel-formula-reference.md` - Statistical functions for analysis
3. `modeling-standards.md` - Documentation standards

### Use Case 4: Credit Analysis
**References to Review:**
1. `financial-ratios.md` - Leverage, coverage, and credit metrics
2. `excel-formula-reference.md` - Calculation patterns
3. `modeling-standards.md` - Validation requirements

### Use Case 5: Company Valuation
**References to Review:**
1. `financial-ratios.md` - Valuation multiples, growth metrics
2. `excel-formula-reference.md` - DCF formulas, terminal value
3. `modeling-standards.md` - Scenario analysis structure

## Best Practices Summary

### Formula Best Practices
From `excel-formula-reference.md`:
- Always use IFERROR for division operations
- Use FALSE for exact match in lookups
- Prefer INDEX-MATCH over VLOOKUP
- Use named ranges for clarity
- Break complex formulas into steps

### Structure Best Practices
From `modeling-standards.md`:
- Separate inputs, calculations, and outputs
- Use consistent sheet organization
- Apply standard color coding
- Include validation checks
- Document all assumptions

### Analysis Best Practices
From `financial-ratios.md`:
- Calculate multiple ratio categories
- Compare to industry benchmarks
- Analyze trends over time
- Understand ratio interrelationships
- Verify data quality

## Integration with Skills

These references support the main skill (`SKILL.md`) and work with the Python scripts in the `../scripts/` directory:

**With `dcf_model.py`:**
- Use valuation ratios from `financial-ratios.md`
- Apply formula patterns from `excel-formula-reference.md`
- Follow standards from `modeling-standards.md`

**With `three_statement_model.py`:**
- Implement efficiency ratios from `financial-ratios.md`
- Use balance check formulas from `excel-formula-reference.md`
- Follow structural standards from `modeling-standards.md`

**With `budget_vs_actual.py`:**
- Calculate variance metrics per `excel-formula-reference.md`
- Apply performance ratios from `financial-ratios.md`
- Structure per `modeling-standards.md`

**With `model_validation.py`:**
- Check against standards in `modeling-standards.md`
- Verify formulas per `excel-formula-reference.md`
- Validate calculations per `financial-ratios.md`

## Learning Path

### Beginner (New to Financial Modeling)
1. Start with `modeling-standards.md` sections:
   - Model Design Principles
   - Sheet Organization
   - Color Coding Standard
2. Learn basic formulas from `excel-formula-reference.md`:
   - Basic math functions
   - Simple IF statements
   - SUM, AVERAGE
3. Understand core ratios from `financial-ratios.md`:
   - Profit margins
   - Current ratio
   - Basic growth rates

### Intermediate (Building Models)
1. Study `modeling-standards.md` sections:
   - Formula Standards
   - Validation Framework
   - Documentation Standards
2. Master `excel-formula-reference.md` functions:
   - VLOOKUP/INDEX-MATCH
   - NPV/IRR
   - Error handling
3. Apply `financial-ratios.md` categories:
   - All profitability ratios
   - Liquidity and leverage
   - Efficiency metrics

### Advanced (Professional Standards)
1. Implement `modeling-standards.md` fully:
   - Complete validation framework
   - Model governance
   - Quality assurance
2. Use advanced `excel-formula-reference.md` techniques:
   - Array formulas
   - Complex conditionals
   - Performance optimization
3. Analyze with `financial-ratios.md` comprehensively:
   - All ratio categories
   - Industry-specific metrics
   - Ratio interrelationships

## Maintenance and Updates

### Version Control
All reference documents include:
- Creation date
- Last updated date
- Version number (if applicable)

### Updates Needed When:
- Excel releases new functions
- Industry standards evolve
- New regulatory requirements emerge
- Best practices change

### Contributing
When updating references:
1. Document change rationale
2. Update version information
3. Add examples for new content
4. Verify cross-references remain valid
5. Update this README if structure changes

## Additional Resources

### External Standards Referenced
- FAST Standard (F1F9 consortium)
- Basel Committee guidelines
- CFA Institute standards
- GARP frameworks

### Recommended Reading
- "Financial Modeling" by Simon Benninga
- "Investment Banking: Valuation, LBOs, M&A" by Joshua Rosenbaum
- "Financial Statement Analysis" by Martin Fridson
- CFA Level 1 curriculum (financial ratios)

### Online Resources
- Investopedia (ratio definitions and benchmarks)
- SEC EDGAR (public company financials)
- Industry association reports (benchmarks)
- Federal Reserve data (economic indicators)

## Support

For questions about these references:
1. Check the specific reference document
2. Review examples in the document
3. Look at integration with Python scripts
4. Consult the main SKILL.md for context

These references represent distilled knowledge from:
- Major investment banks' modeling standards
- Big Four accounting firms' audit requirements
- Professional financial certifications (CFA, FRM)
- Decades of best practices

Use them as the foundation for all financial modeling work to ensure institutional-quality outputs that withstand scrutiny and support confident decision-making.
