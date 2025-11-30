# User Testing Guide

## Overview

This guide provides instructions for testing features on the `user/interface` branch of the Intelligent Investor project. This branch contains new features and improvements that are ready for user validation before merging to the stable `main` branch.

## Prerequisites

**Required:**
- Python 3.11 or higher
- Git installed and configured
- Virtual environment setup

**Recommended:**
- VS Code or preferred IDE
- PowerShell 7.x (Windows)
- Basic understanding of trading concepts

## Getting Started

### 1. Checkout the Testing Branch

```powershell
# Navigate to project directory
cd C:\Users\kjfle\.projects\intelligent-investor

# Checkout user testing branch
git checkout user/interface
git pull origin user/interface
```

### 2. Set Up Python Environment

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Verify Python version (should be 3.11+)
python --version

# Install/update dependencies
python -m pip install -e ".[dev]"
```

## Testing Workflows

### Quick Health Check

```powershell
# Run all tests
pytest

# Expected: 413+ tests passing, ~67% coverage
```

### Test Individual Components

**Core Utilities:**
```powershell
pytest tests/test_core/ -v
```

**Trading Strategies:**
```powershell
pytest tests/test_strategies/ -v
```

**Market Data Plugins:**
```powershell
pytest tests/test_plugins/ -v
```

## Features Available for Testing

### Stable Features
✅ Core Infrastructure
✅ Trading Strategies (RSI, MA, Momentum)
✅ Market Data Integration (Polygon.io, IEX Cloud)
✅ Backtesting System
✅ CLI Interface
✅ Monitoring

### Experimental Features
🔬 RAG System (src/rag/)
🔬 Enhanced Testing
🔬 Bollinger Bands Strategy (known test failures)

## Reporting Issues

When you find an issue:

1. Check if already reported
2. Create detailed bug report with:
   - What you were doing
   - Expected vs actual behavior
   - Steps to reproduce
   - Test output

3. Label as `bug` and `user-testing`

## Common Issues & Solutions

**Tests Failing:**
```powershell
# Ensure virtual environment activated
.\.venv\Scripts\Activate.ps1

# Reinstall dependencies
python -m pip install -e ".[dev]"
```

**CLI Not Working:**
```powershell
# Use module syntax
python -m src.cli --help
```

## Resources

**Documentation:**
- Architecture: `docs/architecture/`
- CLI Usage: `docs/CLI_USAGE.md`
- Branch Workflow: `docs/BRANCH_WORKFLOW.md`

**Testing:**
- Phase 1 Setup: `docs/PHASE_1_TESTING_SETUP.md`
- Testing Checklist: `.github/TESTING_CHECKLIST.md`

---

**Last Updated:** 2025-11-30
**Branch:** user/interface
**Test Coverage:** 67%
**Total Tests:** 413 passing
