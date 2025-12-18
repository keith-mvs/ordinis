# Developer Guide for Options Strategy Skills

Comprehensive standards for creating, testing, and deploying skill packages.

---

## Table of Contents

- [Naming Conventions](#naming-conventions)
- [Documentation Standards](#documentation-standards)
- [Testing Standards](#testing-standards)
- [Deployment Checklist](#deployment-checklist)
- [Quality Standards](#quality-standards)
- [Integration with Ordinis Project](#integration-with-ordinis-project)
- [Version Control](#version-control)

---

## Naming Conventions

### File Names

**Strategy directories**: kebab-case
```
bull-call-spread/
iron-condor/
married-put/
```

**Python modules**: snake_case
```python
bull_call_calculator.py
strike_comparison.py
position_sizer.py
black_scholes.py
```

**Python classes**: PascalCase
```python
class BullCallSpread:
class IronCondor:
class PositionSizer:
class BlackScholesCalculator:
```

### YAML Frontmatter (SKILL.md)

**Format**:
```yaml
---
name: [kebab-case-name]
description: [What it does]. Requires [packages if any]. Use when [specific triggers].
allowed-tools: Read, Grep, Glob  # Optional, Claude Code only
---
```

**Example**:
```yaml
---
name: bull-put-spread
description: Analyzes bull-put-spread credit spreads with position sizing and risk management. Requires numpy>=1.24.0, pandas>=2.0.0, matplotlib>=3.7.0, scipy>=1.10.0. Use when evaluating put spreads, comparing strikes, or assessing spread configurations.
allowed-tools: Read, Grep, Glob
---
```

**Field requirements**:
- `name`: lowercase, numbers, hyphens only, max 64 characters
- `description`: what + when to use, list packages, max 1024 characters
- `allowed-tools`: optional, Claude Code specific

### Directory Names

**Standard subdirectories**: lowercase
```
scripts/
references/
assets/
templates/
```

**Strategy-specific references**: kebab-case
```
references/bull-put-spread/
references/iron-condor/
```

---

## Documentation Standards

### Code Documentation

**Docstrings**: Google style for all public functions and classes

**Example**:
```python
def calculate_collateral(spread_width: float, contracts: int) -> float:
    """Calculate margin requirement for bull-put-spread.

    Collateral = (Spread Width × 100 × Contracts)

    Args:
        spread_width: Difference between short and long strikes
        contracts: Number of contracts

    Returns:
        Required collateral in dollars

    Raises:
        ValueError: If spread_width or contracts are non-positive

    Example:
        >>> calculate_collateral(5.0, 2)
        1000.0
    """
    if spread_width <= 0 or contracts <= 0:
        raise ValueError("Spread width and contracts must be positive")
    return spread_width * 100 * contracts
```

**Type hints**: Required for all function parameters and returns

**Good**:
```python
def calculate_ror(max_profit: float, max_loss: float) -> float:
    """Calculate return on risk."""
    return (max_profit / max_loss) * 100
```

**Bad** (missing type hints):
```python
def calculate_ror(max_profit, max_loss):
    """Calculate return on risk."""
    return (max_profit / max_loss) * 100
```

**Comments**: Only for complex logic, not obvious operations

**Good** (complex logic):
```python
# Adjust for early assignment risk when ITM and near ex-div date
if is_itm and days_to_ex_div < 7:
    risk_premium = intrinsic_value * 0.1  # 10% buffer for early assignment
```

**Bad** (obvious operation):
```python
# Calculate the maximum profit
max_profit = net_credit * 100  # Multiply by 100 for per-contract basis
```

### Markdown Documentation

**Headers**: Use ATX-style (#, ##, ###)

**Good**:
```markdown
# Main Title
## Section
### Subsection
```

**Avoid**: Setext-style (underlines)

**Code blocks**: Always specify language

**Good**:
````markdown
```python
import numpy as np
```

```bash
pip install numpy
```
````

**Bad** (no language):
````markdown
```
import numpy as np
```
````

**Links**: Use reference-style for repeated URLs

**Good**:
```markdown
See the [Alpaca API][alpaca] for details.
More info in [Alpaca documentation][alpaca].

[alpaca]: https://alpaca.markets/docs/api-references/
```

**Tables**: Use for comparisons and metrics

**Example**:
```markdown
| Strike | Delta | Premium | Risk Level |
|--------|-------|---------|------------|
| 145    | -0.10 | $0.90   | Low        |
| 150    | -0.20 | $1.50   | Moderate   |
| 155    | -0.30 | $2.20   | High       |
```

---

## Testing Standards

### Required Tests

**1. Import test**: Verify all modules import successfully

```python
def test_imports():
    """Test that all modules can be imported."""
    from scripts.bull_call_calculator import BullCallSpread
    from scripts.black_scholes import BlackScholesCalculator
    from scripts.position_sizer import calculate_contracts
    assert True  # If we get here, imports worked
```

**2. Calculation test**: Validate core calculations with known values

```python
def test_bull_call_spread_calculations():
    """Test basic bull call spread calculations."""
    position = BullCallSpread(
        underlying_symbol="SPY",
        underlying_price=450.00,
        long_strike=445.00,
        short_strike=455.00,
        long_premium=8.50,
        short_premium=3.20,
        expiration_date=datetime.now() + timedelta(days=45)
    )

    # Test calculations
    assert position.net_debit == 5.30
    assert position.spread_width == 10.00
    assert abs(position.breakeven_price - 450.30) < 0.01
    assert position.max_profit == 470.00
    assert position.max_loss == 530.00
```

**3. Edge case test**: Handle T=0, invalid parameters

```python
def test_edge_cases():
    """Test edge cases and error handling."""

    # Test zero time to expiration
    position = BullCallSpread(
        underlying_symbol="SPY",
        underlying_price=450.00,
        long_strike=445.00,
        short_strike=455.00,
        long_premium=8.50,
        short_premium=3.20,
        expiration_date=datetime.now()  # T=0
    )
    assert position.theta == 0  # No time decay at expiration

    # Test invalid parameters
    with pytest.raises(ValueError):
        BullCallSpread(
            underlying_symbol="SPY",
            underlying_price=-100,  # Invalid negative price
            long_strike=445.00,
            short_strike=455.00,
            long_premium=8.50,
            short_premium=3.20,
            expiration_date=datetime.now() + timedelta(days=45)
        )
```

**4. Integration test**: Full workflow from position creation to analysis

```python
def test_full_workflow():
    """Test complete workflow: create → analyze → visualize."""

    # Step 1: Create position
    position = BullCallSpread(
        underlying_symbol="SPY",
        underlying_price=450.00,
        long_strike=445.00,
        short_strike=455.00,
        long_premium=8.50,
        short_premium=3.20,
        expiration_date=datetime.now() + timedelta(days=45)
    )

    # Step 2: Calculate Greeks
    greeks = position.calculate_greeks()
    assert 'delta' in greeks
    assert 'gamma' in greeks
    assert 'theta' in greeks
    assert 'vega' in greeks

    # Step 3: Generate analysis
    analysis = position.get_full_analysis()
    assert 'breakeven' in analysis
    assert 'max_profit' in analysis
    assert 'max_loss' in analysis

    # Step 4: Create visualization (check it doesn't crash)
    from scripts.visualizations import plot_payoff_diagram
    plot_payoff_diagram(position, save_path="test_payoff.png")
    assert os.path.exists("test_payoff.png")
```

### Test Organization

**File structure**:
```
strategy-name/
├── scripts/
│   └── strategy_calculator.py
└── tests/
    ├── __init__.py
    ├── test_calculator.py
    ├── test_greeks.py
    └── test_integration.py
```

**Running tests**:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=scripts tests/

# Run specific test
pytest tests/test_calculator.py::test_bull_call_spread_calculations
```

---

## Deployment Checklist

### Before Creating Skill Package

- [ ] Review strategy fundamentals (payoff, risks, ideal market conditions)
- [ ] Identify key calculations and formulas (max profit/loss, breakeven, Greeks)
- [ ] Determine required Python dependencies (numpy, pandas, matplotlib, scipy)
- [ ] Plan file structure based on complexity (simple vs complex skill)
- [ ] Choose reference implementation (bull-call-spread vs married-put model)

### During Development

- [ ] Create SKILL.md with YAML frontmatter (name, description, allowed-tools)
- [ ] Implement calculator class with full docstrings and type hints
- [ ] Add CLI argument parsing with `argparse`
- [ ] Implement Greeks calculations (if applicable)
- [ ] Create references/ files following progressive disclosure
- [ ] Write comprehensive tests (import, calculation, edge case, integration)
- [ ] Generate example positions or sample data
- [ ] Create requirements.txt with pinned versions

### Before Release

- [ ] Test all imports and calculations (`pytest tests/`)
- [ ] Verify CLI works with various parameters (`python scripts/calculator.py --help`)
- [ ] Validate YAML frontmatter syntax (no tabs, proper indentation)
- [ ] Check all internal links in SKILL.md and references
- [ ] Run linter on Python code (`ruff check scripts/` or `pylint scripts/`)
- [ ] Create .gitignore (exclude __pycache__, *.pyc, .pytest_cache, *.png)
- [ ] Test with Claude Code (does skill trigger correctly?)
- [ ] Verify package dependencies install cleanly (`pip install -r requirements.txt`)

### Post-Release

- [ ] Create version history section in SKILL.md
- [ ] Test skill invocation in Claude with various queries
- [ ] Verify all examples work as documented
- [ ] Document any issues or limitations in references/
- [ ] Get team feedback on skill effectiveness

---

## Quality Standards

### Code Quality

**Linting**: Pass ruff/pylint with no errors

```bash
# Using ruff (recommended - faster)
ruff check scripts/

# Using pylint
pylint scripts/
```

**Type hints**: 100% coverage on public APIs

```python
# Good - complete type hints
def calculate_spread_width(
    short_strike: float,
    long_strike: float
) -> float:
    """Calculate spread width."""
    return abs(short_strike - long_strike)

# Bad - missing type hints
def calculate_spread_width(short_strike, long_strike):
    """Calculate spread width."""
    return abs(short_strike - long_strike)
```

**Documentation**: All public functions documented

```python
# Good - documented
def calculate_ror(max_profit: float, max_loss: float) -> float:
    """Calculate return on risk percentage.

    Args:
        max_profit: Maximum profit in dollars
        max_loss: Maximum loss in dollars

    Returns:
        Return on risk as percentage
    """
    return (max_profit / max_loss) * 100

# Bad - no docstring
def calculate_ror(max_profit: float, max_loss: float) -> float:
    return (max_profit / max_loss) * 100
```

**Error handling**: Graceful failure with clear messages

```python
# Good - clear error messages
def calculate_contracts(capital: float, spread_width: float) -> int:
    """Calculate number of contracts."""
    if capital <= 0:
        raise ValueError(
            f"Capital must be positive, got {capital}. "
            "Please provide available capital in dollars."
        )
    if spread_width <= 0:
        raise ValueError(
            f"Spread width must be positive, got {spread_width}. "
            "Check that short_strike > long_strike for bull spreads."
        )
    return int(capital / (spread_width * 100))

# Bad - cryptic error
def calculate_contracts(capital: float, spread_width: float) -> int:
    """Calculate number of contracts."""
    assert capital > 0 and spread_width > 0
    return int(capital / (spread_width * 100))
```

### Documentation Quality

**Clarity**: Understandable by intermediate Python users

- Explain domain-specific concepts (delta, theta, credit spread)
- Provide examples for complex calculations
- Use consistent terminology throughout

**Completeness**: All features documented with examples

- Every public function/class in SKILL.md or references/
- Every CLI flag in quickstart or reference
- Common use cases in examples

**Accuracy**: Calculations match financial formulas

- Verify against authoritative sources (CBOE, OIC)
- Cross-check with existing implementations
- Include references to formulas used

**Consistency**: Terminology consistent across files

- Choose one term and stick with it (e.g., "net debit" not "cost basis")
- Maintain consistent variable names (e.g., always `underlying_price` not mixing with `stock_price`)
- Use same examples across documentation

---

## Integration with Ordinis Project

### Engine Integration

If using shared engines (e.g., `src/engines/optionscore/`):

**Reference in documentation**:
```markdown
## Shared Components

This skill uses the Ordinis OptionsCore engine:
- Black-Scholes calculations: `src/engines/optionscore/black_scholes.py`
- Greeks calculations: `src/engines/optionscore/greeks.py`

**Import paths**:
```python
from src.engines.optionscore.black_scholes import BlackScholesCalculator
from src.engines.optionscore.greeks import calculate_greeks
```

**Version compatibility**: OptionsCore v2.0+
```

**Note version compatibility** to prevent breaking changes

### Standalone vs. Collection

**Standalone skill** (bull-call-spread model):
- Include all dependencies within skill directory
- No external imports from Ordinis project
- Fully self-contained and portable
- **Use when**: Skill should work independently, be shareable outside project

**Collection member** (options-strategies model):
- Reference shared utilities from project
- Import from `src/` or other skill directories
- Requires Ordinis project structure
- **Use when**: Skill is part of larger ecosystem, shares common code

---

## Version Control

### Git Workflow

**1. Create feature branch**:
```bash
git checkout -b features/bull-put-spread
```

**2. Develop skill package**:
```bash
# Create directory structure
mkdir -p .claude/skills/bull-put-spread/{scripts,references,assets}

# Develop files
code .claude/skills/bull-put-spread/SKILL.md
code .claude/skills/bull-put-spread/scripts/calculator.py

# Test
pytest .claude/skills/bull-put-spread/tests/
```

**3. Test thoroughly** before committing

**4. Commit and push**:
```bash
git add .claude/skills/bull-put-spread/
git commit -m "Add bull-put-spread skill package

- Implement calculator with position sizing
- Add strike selection framework
- Create 5 real-world examples
- Full test coverage"
git push origin features/bull-put-spread
```

**5. Create pull request to main**

**6. Tag release after merge**:
```bash
git checkout main
git pull
git tag v1.0.0-bull-put-spread
git push --tags
```

### Changelog Format

**Keep version history in SKILL.md**:

```markdown
# Bull-Put-Spread

## Version History

### [2.0.0] - 2025-12-15

**Breaking Changes**:
- Changed `calculate_contracts()` signature to include max_position_pct
- Renamed `collateral_required` property to `margin_requirement`

**Added**:
- IV rank analysis
- Expected move calculations
- Liquidity checks (volume, OI, bid/ask)

**Changed**:
- Improved strike selection framework with delta classifications
- Enhanced real-world examples with varying IV levels

**Fixed**:
- Corrected ROR calculation for credit spreads
- Fixed early assignment risk assessment logic

### [1.1.0] - 2025-12-01

**Added**:
- Risk assessment for early assignment
- Pin risk evaluation
- Management triggers (take-profit, stop-loss)

**Changed**:
- Updated examples with transaction costs

**Fixed**:
- None

### [1.0.0] - 2025-11-15

**Added**:
- Initial release of bull-put-spread skill package
- Core calculator with Greeks
- CLI interface
- 5 example positions
- Position sizing logic

**Changed**:
- N/A

**Fixed**:
- N/A
```

**Follow semantic versioning**:
- **MAJOR** (2.0.0): Breaking changes
- **MINOR** (1.1.0): New features, backward compatible
- **PATCH** (1.0.1): Bug fixes, backward compatible

---

## Reference Implementations

### Available Models

**Location**: `C:\Users\kjfle\Workspace\ordinis\skills\`

**1. bull-call-spread** (Simple model)
- **Files**: 6 core files
- **Scripts**: 2 (calculator + Black-Scholes)
- **Examples**: None
- **Documentation**: 3 files (SKILL, QUICKSTART, INSTALLATION)
- **Focus**: Single strategy, mathematical rigor

**2. married-put** (Complex model)
- **Files**: 13 files
- **Scripts**: 5 (calculator + 4 utilities)
- **Examples**: CSV with 5 positions
- **Documentation**: 5 files
- **Focus**: Portfolio integration, multiple utilities

### Comparison Matrix

| Feature | bull-call-spread | married-put |
|---------|------------------|-------------|
| **Complexity** | Simple | Complex |
| **Total Files** | 6 | 13 |
| **Python Scripts** | 2 | 5 |
| **Example Data** | None | CSV with 5 positions |
| **Documentation Files** | 3 | 5 |
| **SKILL.md Size** | 31KB | 8KB |
| **CLI Interface** | Comprehensive | Modular utilities |
| **Greeks Calculation** | Integrated | Separate module |
| **Payoff Diagrams** | In calculator | Separate visualizations.py |
| **Portfolio Tools** | No | Yes (position_sizer.py) |
| **Best For** | Vertical spreads, quantitative | Stock+option, portfolio-level |

### When to Use Each Model

**Use bull-call-spread model for**:
- Vertical spreads (bull call, bear put, credit spreads)
- Single-strategy focus
- Quantitative analysis emphasis
- Professional users
- Mathematical rigor required
- Single comprehensive CLI tool

**Use married-put model for**:
- Stock + option strategies (covered call, married put, collar)
- Portfolio-level analysis
- Multiple utility scripts
- Beginner-friendly approach
- Modular tools
- Team collaboration (each person works on different utility)

---

## Best Practices Summary

**DO**:
- ✅ Use type hints on all public functions
- ✅ Write Google-style docstrings
- ✅ Test edge cases (T=0, invalid params)
- ✅ Handle errors gracefully with clear messages
- ✅ Keep SKILL.md under 500 lines
- ✅ Use progressive disclosure (references/)
- ✅ List package dependencies in description
- ✅ Follow semantic versioning
- ✅ Include version history in SKILL.md

**DON'T**:
- ❌ Use magic numbers without comments
- ❌ Mix terminology (pick one term, use consistently)
- ❌ Skip error handling
- ❌ Put everything in SKILL.md (use references/)
- ❌ Forget to test imports
- ❌ Use Windows-style paths (\)
- ❌ Commit without testing
- ❌ Break semantic versioning

---

**This guide ensures high-quality, maintainable skill packages that integrate seamlessly with Claude Code and the Ordinis project.**
