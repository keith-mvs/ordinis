# Assets Folder Standards for Skill Packages

**Version:** 1.0
**Last Updated:** 2025-12-12
**Purpose:** Define standards for using the assets/ directory in skill packages

---

## Purpose and Philosophy

### What is the assets/ Folder?

The **assets/** directory contains files that are **used in generated output**, not files that are loaded into Claude's context for learning.

**Key Distinction:**
- **assets/**: Files used to **generate or template** outputs → Not loaded into context
- **references/**: Documentation files for **learning and understanding** → Loaded into context
- **scripts/**: Executable code for **calculations and processing** → Loaded when invoked

---

## When to Use assets/

### Use assets/ for:

1. **Output Templates**
   - Report templates (markdown, HTML)
   - Document boilerplates
   - Email templates
   - Analysis templates

2. **Sample/Example Data**
   - CSV files with example positions
   - JSON configuration examples
   - Sample datasets for learning
   - Example input files

3. **Reference Data**
   - Historical price data (if small)
   - Lookup tables
   - Static configuration data
   - Market calendars

4. **Generated Assets**
   - Placeholder images
   - Chart templates
   - Diagram templates

### Do NOT use assets/ for:

- Documentation (use references/)
- Executable scripts (use scripts/)
- Test files (use tests/)
- Package dependencies (use requirements.txt in root)
- Large datasets (store externally, reference by URL)

---

## File Organization

### Flat Structure (Preferred)

For skills with few assets (<5 files):

```
strategy-name/
└── assets/
    ├── sample_positions.csv
    ├── report-template.md
    └── lookup_table.json
```

### Nested Structure

For skills with many assets (5+ files):

```
strategy-name/
└── assets/
    ├── templates/
    │   ├── report-template.md
    │   └── analysis-template.md
    ├── samples/
    │   ├── sample_positions.csv
    │   └── example_config.json
    └── reference-data/
        ├── holidays.csv
        └── strike_increments.json
```

---

## File Naming Conventions

### General Rules

- **Lowercase** with underscores for spaces
- **Descriptive** names indicating purpose
- **Extension** matches file type

### Examples

**Good:**
- `report-template.md`
- `sample_positions.csv`
- `strike_selection_lookup.json`
- `earnings_calendar_example.csv`

**Bad:**
- `Template.md` (capital letter)
- `data.csv` (not descriptive)
- `file1.txt` (meaningless name)

---

## File Types and Usage

### 1. Templates (.md, .html, .txt)

**Purpose:** Boilerplate content for generated reports/outputs

**Example - Report Template:**
```markdown
# [STRATEGY] Analysis Report

**Date:** [DATE]
**Symbol:** [TICKER]
**Analysis:** Claude

## Position Summary
[POSITION_DETAILS]

## Risk Metrics
[RISK_ANALYSIS]
```

**Location:** `assets/report-template.md` or `assets/templates/report-template.md`

### 2. Sample Data (.csv, .json, .xml)

**Purpose:** Example data for learning and testing

**Example - Sample Positions CSV:**
```csv
ticker,stock_price,shares,put_strike,put_premium,days_to_expiration
AAPL,150.00,100,145.00,2.50,45
MSFT,350.00,50,340.00,5.25,60
```

**Location:** `assets/sample_positions.csv`

### 3. Reference Data (.csv, .json)

**Purpose:** Lookup tables and static configuration

**Example - Strike Increments:**
```json
{
  "stock_price_ranges": [
    {"min": 0, "max": 25, "increment": 2.5},
    {"min": 25, "max": 200, "increment": 5.0},
    {"min": 200, "max": null, "increment": 10.0}
  ]
}
```

**Location:** `assets/reference-data/strike_increments.json`

### 4. Configuration Examples (.json, .yaml, .toml)

**Purpose:** Example configuration files for users

**Example:**
```json
{
  "position_limits": {
    "max_portfolio_heat": 0.02,
    "max_single_position": 0.05
  }
}
```

**Location:** `assets/config_example.json`

---

## Current Usage Across Skills

### Audit Results (2025-12-12)

**Total Skills:** 21
**Skills with assets/ directory:** 21 (100%)
**Skills with asset files:** 2 (9.5%)

**Files Found:**

1. **due-diligence/assets/report-template.md** (7.0 KB)
   - Comprehensive due diligence report template
   - 300 lines with structured sections
   - Used to generate formatted analysis reports

2. **married-put/assets/sample_positions.csv** (0.6 KB)
   - 5 example positions with varied scenarios
   - Demonstrates different volatility regimes
   - Used for learning position structure

**File Type Breakdown:**
- `.md`: 1 file (template)
- `.csv`: 1 file (sample data)

---

## Best Practices

### 1. Keep Assets Small

- **Target:** <50 KB per file
- **Maximum:** 500 KB per file
- **Rationale:** Assets are distributed with skill packages

### 2. Document Asset Purpose

Add README.md in assets/ if you have >3 files:

```markdown
# Assets Directory

## Templates
- `report-template.md`: Due diligence report structure

## Sample Data
- `sample_positions.csv`: Example positions for learning

## Reference Data
- `holidays.csv`: Market holidays for expiration calculations
```

### 3. Version Templates

For templates that may change:

```
assets/templates/
├── report-template-v1.md
├── report-template-v2.md (current)
└── README.md (documents version differences)
```

### 4. Provide Examples

Sample data should cover:
- **Typical scenarios** (70%)
- **Edge cases** (20%)
- **Extreme cases** (10%)

### 5. Avoid Duplication

If multiple skills need the same asset:
- Put it in a master skill (e.g., options-strategies/assets/)
- Reference from individual skills
- Document the reference in SKILL.md

---

## Examples by Strategy Type

### Vertical Spreads (Bull Call, Bear Put)

**Typical Assets:**
```
assets/
├── sample_spreads.csv (example positions)
└── spread_comparison_template.md (analysis template)
```

### Stock + Option (Married Put, Covered Call)

**Typical Assets:**
```
assets/
├── sample_positions.csv (stock+option examples)
├── rolling_tracker_template.csv (position management)
└── portfolio_integration_example.json (allocation examples)
```

### Neutral Strategies (Straddles, Strangles)

**Typical Assets:**
```
assets/
├── iv_scenarios.csv (volatility examples)
├── earnings_calendar_example.csv (sample calendar)
└── volatility_analysis_template.md (IV report template)
```

### Complex Multi-Leg (Iron Condor, Butterfly)

**Typical Assets:**
```
assets/
├── wing_configurations.csv (structure examples)
├── adjustment_scenarios.csv (management examples)
└── probability_calculator_template.md (analysis template)
```

---

## Validation Checklist

Before finalizing assets/:

- [ ] All files have descriptive names
- [ ] File sizes <50 KB (or justified if larger)
- [ ] No duplicate content from references/
- [ ] README.md present if >3 files
- [ ] Sample data covers typical + edge cases
- [ ] Templates include placeholder markers clearly marked
- [ ] No sensitive/proprietary data included
- [ ] All file extensions match content type
- [ ] Files are actually used by the skill (no orphans)

---

## Migration Guide

### Moving Files TO assets/

**If you have:**
- Templates in root → Move to assets/templates/
- Sample data in scripts/ → Move to assets/samples/
- Example configs in references/ → Move to assets/

**Example Migration:**
```bash
# Move template from root
mv strategy-name/report_template.md strategy-name/assets/report-template.md

# Move sample data from scripts
mv strategy-name/scripts/sample_data.csv strategy-name/assets/sample_data.csv

# Update references in SKILL.md
# Old: See report_template.md
# New: See assets/report-template.md
```

### Moving Files FROM assets/

**If assets/ contains:**
- Documentation → Move to references/
- Executable code → Move to scripts/
- Test data → Move to tests/fixtures/

---

## Cross-Referencing Assets

### From SKILL.md

```markdown
## Using This Skill

To get started, see the [sample positions](assets/sample_positions.csv) for examples.

For report generation, use the [report template](assets/report-template.md).
```

### From Scripts

```python
from pathlib import Path

# Load asset file
SKILL_DIR = Path(__file__).parent.parent
SAMPLE_DATA = SKILL_DIR / "assets" / "sample_positions.csv"

if SAMPLE_DATA.exists():
    positions = pd.read_csv(SAMPLE_DATA)
```

### From References

```markdown
## Example Positions

The skill includes [sample positions](../assets/sample_positions.csv) demonstrating:
- Low volatility scenarios (IV < 25%)
- Medium volatility scenarios (IV 25-40%)
- High volatility scenarios (IV > 40%)
```

---

## Implementation Standards

### CSV Files

**Requirements:**
- Header row with descriptive column names
- Consistent data types per column
- No empty rows
- UTF-8 encoding

**Example:**
```csv
ticker,price,strike,premium,expiration,scenario
AAPL,150.00,145.00,2.50,2025-01-15,low_volatility
TSLA,250.00,240.00,12.00,2025-01-15,high_volatility
```

### JSON Files

**Requirements:**
- Valid JSON (use validator)
- Consistent key naming (snake_case)
- Commented version if needed (use JSON5 or add .md explanation)

**Example:**
```json
{
  "position_limits": {
    "max_portfolio_heat": 0.02,
    "max_single_position": 0.05
  },
  "greeks_thresholds": {
    "max_delta": 0.30,
    "max_theta": -50.0
  }
}
```

### Markdown Templates

**Requirements:**
- Use [PLACEHOLDER] for variable content
- Include usage instructions at top
- Document required vs optional sections

**Example:**
```markdown
<!--
TEMPLATE USAGE:
- Replace [TICKER] with stock symbol
- Replace [ANALYSIS] with generated content
- Optional sections marked with (Optional)
-->

# [TICKER] Analysis Report

**Generated:** [DATE]

## Analysis
[ANALYSIS]
```

---

## Maintenance

### Regular Review

**Quarterly:**
- Audit all assets/ directories (use scripts/audit_assets.py)
- Remove unused files
- Update templates based on user feedback
- Verify sample data still relevant

**When Adding New Asset:**
1. Confirm it belongs in assets/ (not references/ or scripts/)
2. Check file size (<50 KB preferred)
3. Use consistent naming
4. Update README.md if needed
5. Reference from SKILL.md or scripts/

**When Removing Asset:**
1. Search for references in SKILL.md
2. Search for references in scripts/
3. Search for references in references/
4. Remove or update all references
5. Update README.md if needed

---

## Related Documentation

- [CROSS_LINKING_SCHEME.md](CROSS_LINKING_SCHEME.md) - Reference linking standards
- [_template/SKILL.md](_template/SKILL.md) - Skill package template
- [REFERENCE_GENERATION_SUMMARY.md](../../REFERENCE_GENERATION_SUMMARY.md) - Reference documentation plan

---

## Summary

**Key Takeaways:**

1. **assets/ = Output files**, not learning files
2. Keep files **small** (<50 KB) and **well-named**
3. **Templates, samples, reference data** go in assets/
4. **Documentation** goes in references/, **code** goes in scripts/
5. Organize with **flat structure** unless >5 files
6. **Document purpose** with README.md if needed
7. **Cross-reference** from SKILL.md and scripts/

**Current State:** 19/21 skills have empty assets/ directories (opportunity to add valuable templates and samples)

**Next Steps:**
- Populate assets/ with strategy-specific templates
- Add sample data for each strategy
- Create reference data files where applicable
- Document assets with README.md files

---

**Compliance:** ✅ Aligned with skill package template v3.0
**Status:** Standard established, ready for implementation
