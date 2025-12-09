# 4. User Guides

**Last Updated:** {{ git_revision_date_localized }}

---

## 4.1 Overview

User guides and tutorials for working with the Ordinis trading system.

## 4.2 Available Guides

| Guide | Description |
|-------|-------------|
| [CLI Usage](CLI_USAGE.md) | Command-line interface guide |
| [Due Diligence Skill](DUE_DILIGENCE_SKILL.md) | Using the due diligence analysis skill |
| [Recommended Skills](RECOMMENDED_SKILLS.md) | Recommended Claude Code skills |
| [UI Improvement Proposal](UI_IMPROVEMENT_PROPOSAL.md) | Documentation UI/UX enhancement proposal |

## 4.3 Quick Start

### 4.3.1 Running the System

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run with paper trading
python -m src.main --mode paper

# Run dashboard
streamlit run src/dashboard/app.py
```

### 4.3.2 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_engines/test_governance/ -v
```

## 4.4 Configuration

Key configuration files:

| File | Purpose |
|------|---------|
| `config/config.yaml` | Main configuration |
| `config/strategies/` | Strategy definitions |
| `.env` | Environment variables (API keys) |
