# Options Strategy Skill Package Template

**Version**: 3.0
**Last Updated**: 2025-12-12
**Conforms To**: Anthropic Skills Best Practices + Claude Code Agent Skills

---

## Overview

This template provides a standardized, compliant structure for creating Claude skills for options trading strategies. It follows progressive disclosure principles to keep SKILL.md files under 500 lines while preserving all necessary implementation details in reference files.

**Key Features**:
- ✅ Anthropic best practices compliant (<500 line SKILL.md)
- ✅ Claude Code Agent Skills compatible
- ✅ Progressive disclosure with references/
- ✅ Dual reference implementations (bull-call-spread + married-put)
- ✅ Complete documentation preserved (nothing deleted, just organized)

---

## Quick Start

**For template users**:
1. Read [references/quickstart-guide.md](references/quickstart-guide.md) for 5-minute overview
2. Choose strategy type from [references/strategy-variations.md](references/strategy-variations.md)
3. Follow structure patterns below
4. Review [references/developer-guide.md](references/developer-guide.md) for standards

**For skill developers**:
1. Copy this template directory
2. Customize SKILL.md frontmatter
3. Implement scripts/calculator.py
4. Create strategy-specific references/
5. Deploy (see [references/distribution-guide.md](references/distribution-guide.md))

---

## Standard Directory Structure

### Pattern 1: Simple Skill (Flat Files)

**Use for**: Vertical spreads, simple strategies, <5 supporting files

```
strategy-name/
├── SKILL.md (required, <500 lines)
├── reference.md (optional - detailed docs)
├── examples.md (optional - usage examples)
├── scripts/
│   └── calculator.py
└── assets/
    └── sample_data.csv
```

### Pattern 2: Complex Skill (Nested Directory)

**Use for**: Stock+option, multi-leg, complex strategies, 5+ supporting files

```
strategy-name/
├── SKILL.md (required, <500 lines, navigation hub)
├── scripts/
│   ├── calculator.py
│   ├── position_sizer.py
│   └── black_scholes.py (if standalone)
├── references/
│   ├── position-sizing.md
│   ├── strike-selection.md
│   ├── greeks.md
│   ├── examples.md
│   └── [strategy-specific topics]
└── assets/
    └── sample_positions.csv
```

**Progressive Disclosure**: SKILL.md provides overview + navigation, references/ contain detailed content loaded as needed by Claude.

---

## SKILL.md Template

### Frontmatter (Required)

```yaml
---
name: [strategy-name]
description: [What it does]. Requires [package1>=version, package2>=version]. Use when [specific triggers and scenarios].
allowed-tools: Read, Grep, Glob  # Optional, Claude Code only
---
```

**Field Requirements** (Validation):
- `name`: lowercase, numbers, hyphens only, max 64 characters
- `description`: what + when to use + packages, max 1024 characters, third-person only
- `allowed-tools`: optional, comma-separated tool names (Claude Code specific)

**Example - Bull-Put-Spread**:
```yaml
---
name: bull-put-spread
description: Analyzes bull-put-spread credit spreads with position sizing and risk management. Requires numpy>=1.24.0, pandas>=2.0.0, matplotlib>=3.7.0, scipy>=1.10.0. Use when evaluating put spreads, comparing strikes, assessing spread configurations, or analyzing credit spread opportunities on small to mid-cap stocks.
allowed-tools: Read, Grep, Glob
---
```

**Example - Married-Put**:
```yaml
---
name: married-put
description: Protective put strategy combining long stock with long put for downside protection. Requires numpy>=1.24.0, pandas>=2.0.0, matplotlib>=3.7.0. Use when protecting stock positions, analyzing downside hedges, comparing strike selections for protection, or evaluating portfolio insurance strategies.
---
```

### Body Structure (Recommended)

See [references/developer-guide.md](references/developer-guide.md) for complete section templates. Typical structure:

1. **Overview** - Version history, 2-3 sentence description, quick reference
2. **Core Workflow** - 3-5 major steps with links to references/
3. **Scripts** - CLI usage and quick example
4. **References** - Navigation to all detailed docs
5. **Dependencies** - Required packages list

**Estimated length**: 300-400 lines

---

## Reference Files Organization

### Core References (Every Skill)

**quickstart-guide.md**:
- 5-minute getting started
- Basic usage examples
- Common commands
- Troubleshooting

**installation-guide.md**:
- Prerequisites
- Installation methods (Claude Code, claude.ai, API)
- Environment setup
- Verification steps

**developer-guide.md**:
- Naming conventions (kebab-case, snake_case, PascalCase)
- Documentation standards (Google docstrings, type hints)
- Testing standards (import, calculation, edge case, integration tests)
- Deployment checklist

**distribution-guide.md**:
- Personal Skills (~/.claude/skills/)
- Project Skills (.claude/skills/)
- Plugin Skills (marketplace)

### Strategy-Specific References

**Choose based on strategy type** (see [references/strategy-variations.md](references/strategy-variations.md)):

**For Vertical Spreads** (Bull Call, Bear Put, Credit Spreads):
- `greeks.md` - Greeks calculations and formulas
- `spread-width-analysis.md` - Compare $2.50, $5, $10 widths
- `strike-selection.md` - Delta-based strike framework
- `examples.md` - Real-world scenarios

**For Stock + Option** (Covered Call, Married Put, Collar):
- `position-sizing.md` - Portfolio heat calculations
- `rolling-strategies.md` - When and how to roll
- `dividend-considerations.md` - Ex-div dates, early assignment
- `examples.md` - Portfolio positions

**For Neutral** (Straddle, Strangle):
- `iv-analysis.md` - IV rank, percentile, expected move
- `volatility-management.md` - Vega exposure
- `earnings-plays.md` - Earnings opportunities
- `examples.md` - Volatility scenarios

**For Complex Multi-Leg** (Iron Condor, Butterfly):
- `payoff-analysis.md` - Understanding complex payoffs
- `leg-optimization.md` - Multi-strike selection
- `adjustment-strategies.md` - Managing underwater positions
- `examples.md` - Wing configurations

---

## Scripts Templates

See `scripts/template_calculator.py` for complete calculator template with:
- Dataclass-based position model
- Properties for max_profit, max_loss, breakeven
- Greeks calculations
- CLI with argparse
- Comprehensive analysis output

See [references/developer-guide.md](references/developer-guide.md) for code quality standards and testing requirements.

---

## Assets / Templates

**assets/** directory contains files used in output, not loaded into context:

- `sample_data.csv` - Example positions for learning
- `template.xlsx` - Spreadsheet template (if applicable)
- Historical data files (if applicable)

**templates/** directory (alternative naming for document generation):

- Document templates
- Boilerplate files
- Output templates

**Note**: Choose `assets/` or `templates/` based on your skill's purpose. Both are valid.

---

## Strategy-Specific Customization

See [references/strategy-variations.md](references/strategy-variations.md) for detailed guidance on:

- **Vertical Spreads** - Emphasize mathematical rigor, spread optimization
- **Stock + Option** - Emphasize portfolio integration, rolling strategies
- **Neutral Strategies** - Emphasize volatility analysis, IV focus
- **Complex Multi-Leg** - Emphasize visualization, multi-strike optimization

**Reference Implementations**:
- **bull-call-spread**: `C:\Users\kjfle\Workspace\ordinis\skills\bull-call-spread\`
- **married-put**: `C:\Users\kjfle\Workspace\ordinis\skills\married-put\`

---

## Deployment Workflow

### 1. Choose Distribution Method

See [references/distribution-guide.md](references/distribution-guide.md) for:
- **Personal Skills** (~/.claude/skills/) - Individual use, all projects
- **Project Skills** (.claude/skills/) - Team sharing via git
- **Plugin Skills** (plugins) - Marketplace distribution

### 2. Development Checklist

**Before starting**:
- [ ] Choose strategy type (vertical, stock+option, neutral, complex)
- [ ] Select reference model (bull-call-spread or married-put)
- [ ] Identify required calculations and formulas
- [ ] Plan progressive disclosure structure

**During development**:
- [ ] Create SKILL.md with frontmatter (<500 lines)
- [ ] Implement scripts/calculator.py with type hints and docstrings
- [ ] Create references/ files for detailed content
- [ ] Add assets/ or templates/ if needed
- [ ] Write tests (import, calculation, edge case, integration)
- [ ] Create requirements.txt with pinned versions

**Before release**:
- [ ] Verify SKILL.md <500 lines (`wc -l SKILL.md`)
- [ ] Test all imports (`pytest tests/`)
- [ ] Validate YAML frontmatter (no tabs, proper indentation)
- [ ] Check all references links work
- [ ] Run linter (`ruff check scripts/`)
- [ ] Test with Claude Code (skill triggers correctly?)

See complete checklist in [references/developer-guide.md](references/developer-guide.md).

### 3. Quality Standards

**Code Quality**:
- Linting: Pass ruff/pylint with no errors
- Type hints: 100% coverage on public APIs
- Documentation: All public functions documented
- Error handling: Graceful failure with clear messages

**Documentation Quality**:
- Clarity: Understandable by intermediate Python users
- Completeness: All features documented with examples
- Accuracy: Calculations match financial formulas
- Consistency: Terminology consistent across files

See standards in [references/developer-guide.md](references/developer-guide.md).

---

## Progressive Disclosure Best Practices

### Keep SKILL.md Lean

**Do**:
- ✅ Provide overview and navigation
- ✅ Link to detailed references
- ✅ Show minimal code examples
- ✅ Keep under 500 lines

**Don't**:
- ❌ Embed full implementations
- ❌ Duplicate content from references
- ❌ Include lengthy explanations
- ❌ Add meta-documentation

### Organize References Logically

**One level deep**:
```markdown
## Position Sizing
See [references/position-sizing.md](references/position-sizing.md)
```

**Not nested**:
```markdown
## Advanced Topics
See [references/advanced.md](references/advanced.md)
  → Which links to [references/details.md](references/details.md)  ❌ Too deep
```

### Use Table of Contents

**For long reference files** (>100 lines):
```markdown
# Position Sizing Guide

## Contents
- Capital Requirements
- Contract Calculations
- Portfolio Allocation
- Risk Limits

## Capital Requirements
[Content...]
```

---

## Compliance Summary

### Anthropic Best Practices ✅

- [x] SKILL.md <500 lines
- [x] Progressive disclosure with references/
- [x] Third-person description
- [x] One-level reference depth
- [x] Concise and focused
- [x] Forward slashes for paths
- [x] No deeply nested references

### Claude Code Agent Skills ✅

- [x] YAML frontmatter (name, description)
- [x] Description includes what + when + packages
- [x] allowed-tools field (optional)
- [x] Version history in body
- [x] Forward slashes for paths
- [x] Progressive disclosure

**Overall Compliance**: 95/100 ✅

---

## Example: Bull-Put-Spread

See [references/bull-put-spread-example.md](references/bull-put-spread-example.md) for complete implementation example showing:
- How to split 1,387-line implementation into 10 focused reference files
- SKILL.md structure (~80 lines)
- Progressive disclosure pattern
- Real-world code examples

---

## Resources

### Documentation

- [quickstart-guide.md](references/quickstart-guide.md) - 5-minute start
- [installation-guide.md](references/installation-guide.md) - Comprehensive setup
- [developer-guide.md](references/developer-guide.md) - Standards and checklists
- [strategy-variations.md](references/strategy-variations.md) - Strategy-specific guidance
- [distribution-guide.md](references/distribution-guide.md) - Personal/project/plugin
- [bull-put-spread-example.md](references/bull-put-spread-example.md) - Full implementation example

### Analysis Documents

- [REFACTORING_PLAN.md](REFACTORING_PLAN.md) - Migration roadmap
- [CLAUDE_CODE_COMPLIANCE.md](CLAUDE_CODE_COMPLIANCE.md) - Claude Code alignment
- [TEMPLATE_COMPLIANCE_REPORT.md](../TEMPLATE_COMPLIANCE_REPORT.md) - Anthropic compliance

### Reference Implementations

- **bull-call-spread**: Comprehensive single-file approach
- **married-put**: Modular multi-utility approach

**Location**: `C:\Users\kjfle\Workspace\ordinis\skills\`

---

## Next Steps

1. **Choose your strategy** from [references/strategy-variations.md](references/strategy-variations.md)
2. **Copy this template** directory
3. **Customize** SKILL.md frontmatter and body
4. **Implement** scripts/calculator.py
5. **Create** strategy-specific references/
6. **Test** with Claude Code
7. **Deploy** using [references/distribution-guide.md](references/distribution-guide.md)

---

**Template Version**: 3.0
**Conforms**: Anthropic + Claude Code Best Practices
**Ready**: To generate compliant options strategy skills

**Total Template Size**: ~420 lines (under 500-line limit ✅)
