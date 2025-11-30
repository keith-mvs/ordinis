# Branch Workflow

## Overview

This document outlines the branch strategy for the Intelligent Investor project, defining stability levels, testing expectations, and merge policies.

## Branch Stability Levels

### `main` - Production Stable
**Status:** Production-ready, thoroughly tested, stable

**Characteristics:**
- All tests passing (100%)
- Test coverage ≥50% (currently 67%)
- All pre-commit hooks passing
- Documentation complete
- Code reviewed
- Features fully implemented

**Use When:**
- Running production trading strategies
- Building on stable foundation
- Contributing to established features
- Creating releases

**Merge Policy:**
- Requires user authorization
- All tests must pass
- Coverage threshold maintained
- No regressions
- Complete testing checklist

### `user/interface` - User Testing Branch
**Status:** User testing, in-development features, may have rough edges

**Characteristics:**
- New features available for testing
- Tests mostly passing (minor issues acceptable)
- Documentation may be incomplete
- Some features experimental
- Active development

**Use When:**
- Testing new features before production
- Providing feedback on functionality
- Validating user workflows
- Exploring experimental capabilities

**Merge Policy:**
- Must obtain user authorization before merging to `main`
- Complete testing checklist required
- Known issues documented
- Test coverage maintained
- All critical tests passing

### Other Development Branches
**Examples:** `features/*`, `research/*`, `claude/*`

**Characteristics:**
- Experimental code
- Breaking changes possible
- Tests may fail
- Documentation optional
- Individual developer/AI work

**Use When:**
- Developing new features
- Research and experimentation
- Prototyping
- Isolated development work

**Merge Policy:**
- No direct merge to `main`
- Must merge to `user/interface` first for testing
- Code review recommended

## Branch Workflow

### Standard Development Flow

```
main (stable)
  └─> user/interface (testing)
       └─> features/new-feature (development)
```

**Steps:**
1. Create feature branch from `user/interface`
2. Develop and test locally
3. Merge to `user/interface` for user testing
4. After user validation, merge to `main`

### Working on User Testing Branch

**Checkout:**
```powershell
git checkout user/interface
git pull origin user/interface
```

**Testing:**
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run tests
pytest

# Run specific test suite
pytest tests/test_strategies/

# Check coverage
pytest --cov
```

**Reporting Issues:**
- Create GitHub issue with `[user/interface]` prefix
- Include steps to reproduce
- Note expected vs actual behavior
- Include test output if applicable

## Current Branch Status

### What's in `main`
- Core trading engine architecture
- Basic strategy implementations (RSI, MA, Momentum)
- Market data plugins (Polygon.io, IEX Cloud)
- ProofBench backtesting engine
- CLI interface
- Monitoring and logging system
- 413 passing tests, 67% coverage

### What's in `user/interface`
- All features from `main`
- RAG system for knowledge base integration (experimental)
- Enhanced testing suites
- Latest bug fixes and improvements
- Experimental features in development

### Known Issues in `user/interface`
- RAG system not fully tested
- 14 test failures in experimental Bollinger Bands and Momentum strategies
- Some experimental features may have edge cases
- Documentation updates in progress

## Merge Requirements

### Before Merging to `main`

**Required Checks:**
- [ ] All critical tests passing: `pytest`
- [ ] Coverage ≥50%: `pytest --cov`
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] No regressions in core workflows
- [ ] Documentation updated
- [ ] Testing checklist complete (see `.github/TESTING_CHECKLIST.md`)
- [ ] User authorization obtained

**Recommended:**
- [ ] Code review completed
- [ ] Integration tests pass
- [ ] Manual testing of affected features
- [ ] Performance impact assessed

## Quick Reference

| Branch | Stability | Testing Required | Best For |
|--------|-----------|------------------|----------|
| `main` | Production | 100% critical pass | Stable trading |
| `user/interface` | Testing | Most pass | Feature validation |
| `features/*` | Development | Optional | New development |
| `research/*` | Experimental | Optional | Research work |

## Guidelines

**DO:**
- Test thoroughly on `user/interface` before merging to `main`
- Document known issues
- Keep `main` stable at all times
- Request user authorization for main merges
- Update documentation with changes

**DON'T:**
- Push breaking changes to `main`
- Merge untested code to `main`
- Skip testing checklist
- Merge without user approval
- Leave failing critical tests in `user/interface`

## Getting Help

**Questions about branches:**
- Check this document first
- Review `docs/USER_TESTING_GUIDE.md`
- Create GitHub issue
- Check recent commit history

**Testing issues:**
- See `docs/PHASE_1_TESTING_SETUP.md`
- Review pytest configuration in `pyproject.toml`
- Check test logs in `htmlcov/` directory

**Documentation:**
- Architecture: `docs/architecture/`
- CLI Usage: `docs/CLI_USAGE.md`
- Current Status: `docs/CURRENT_STATUS_AND_NEXT_STEPS.md`
