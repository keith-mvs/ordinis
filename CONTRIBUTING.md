# Contributing to Ordinis

Thank you for your interest in contributing to Ordinis, an AI-driven quantitative trading system!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/keith-mvs/ordinis.git
   cd ordinis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

5. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Standards

### Python Code Style
- **Python Version**: 3.11+
- **Type Hints**: Required for all functions and methods
- **Docstrings**: Google-style docstrings for all public APIs
- **Line Length**: 100 characters maximum
- **Imports**: Organized with `isort`
- **Formatting**: Automated with `ruff format`
- **Linting**: Enforced with `ruff check`

### Function Guidelines
- Keep functions under 50 lines when possible
- Single responsibility principle
- Descriptive names (avoid abbreviations)
- Return type annotations required

### Testing Requirements
- Write tests for all new features
- Maintain >80% code coverage on business logic
- Use pytest for all tests
- Place tests in `tests/` mirroring `src/` structure

### Example Code
```python
"""Module docstring describing purpose."""

from typing import Optional

import pandas as pd


def calculate_moving_average(
    data: pd.Series,
    window: int = 20,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate simple moving average.

    Args:
        data: Time series data
        window: Number of periods for average
        min_periods: Minimum observations required

    Returns:
        Moving average series

    Raises:
        ValueError: If window < 1
    """
    if window < 1:
        raise ValueError("Window must be positive")

    return data.rolling(window=window, min_periods=min_periods).mean()
```

## Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

Use branch prefixes:
- `feature/` - New features
- `fix/` - Bug fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation updates
- `test/` - Test additions/fixes

### 2. Make Changes
- Write code following style guidelines
- Add/update tests
- Update documentation
- Run linters and formatters

### 3. Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_strategies/test_bollinger_bands.py
```

### 4. Run Code Quality Checks
```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type check
mypy src/

# Run all pre-commit hooks
pre-commit run --all-files
```

### 5. Commit Changes
```bash
git add .
git commit -m "Add feature: brief description"
```

**Commit Message Format**:
- Start with verb: Add, Fix, Update, Refactor, Remove
- Keep first line under 72 characters
- Add detailed description if needed
- Reference issues: "Fix #123: ..."

### 6. Push and Create PR
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots if UI changes
- Test results/coverage

## Pull Request Guidelines

### PR Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] No linting errors
- [ ] Coverage maintained/improved
- [ ] Commit messages follow conventions
- [ ] PR description is clear

### Review Process
1. Automated checks must pass (CI/CD)
2. At least one maintainer approval required
3. All review comments addressed
4. Up to date with master branch

## Project Structure

```
ordinis/
├── src/              # Source code
├── tests/            # Test suite
├── scripts/          # Utility scripts (organized by category)
├── docs/             # Documentation
├── examples/         # Example usage
├── data/             # Data files (gitignored)
└── .claude/skills/   # Claude Code skills
```

See `ARCHITECTURE.md` for system design details.

## Testing Guidelines

### Unit Tests
- Test individual functions/methods
- Mock external dependencies
- Fast execution (<1s per test)
- Place in `tests/unit/`

### Integration Tests
- Test component interactions
- Use real dependencies when possible
- Place in `tests/integration/`

### Test Fixtures
- Shared fixtures in `tests/fixtures/`
- Use `conftest.py` for pytest fixtures

### Example Test
```python
"""Tests for moving average calculations."""

import pandas as pd
import pytest

from src.indicators import calculate_moving_average


def test_simple_moving_average():
    """Test SMA calculation with valid data."""
    data = pd.Series([1, 2, 3, 4, 5])
    result = calculate_moving_average(data, window=3)

    expected = pd.Series([None, None, 2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(result, expected)


def test_moving_average_invalid_window():
    """Test SMA raises error for invalid window."""
    data = pd.Series([1, 2, 3])

    with pytest.raises(ValueError, match="Window must be positive"):
        calculate_moving_average(data, window=0)
```

## Documentation

### Code Documentation
- All public functions/classes require docstrings
- Use Google-style docstrings
- Include examples in docstrings when helpful

### User Documentation
- Update `docs/` for user-facing changes
- Keep README.md current
- Add examples to `examples/`

### Building Documentation
```bash
# Build MkDocs site
mkdocs serve

# View at http://localhost:8000
```

## Getting Help

- **Questions**: Open a discussion on GitHub
- **Bugs**: Open an issue with reproduction steps
- **Features**: Open an issue with use case description
- **Security**: Email security concerns privately

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the project
- Show empathy towards others

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for each release
- GitHub contributors page
- Project documentation

Thank you for contributing to Ordinis!
