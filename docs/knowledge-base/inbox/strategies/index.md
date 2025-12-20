# 7. Strategy Documentation

**Last Updated:** {{ git_revision_date_localized }}

---

## 7.1 Overview

Trading strategy templates and implementation guides.

## 7.2 Documents

| Document | Description |
|----------|-------------|
| [Strategy Template](strategy-template.md) | Base template for new strategies |

## 7.3 Strategy Development Process

1. **Hypothesis Generation** - Define market inefficiency
2. **Signal Design** - Implement entry/exit logic
3. **Risk Integration** - Add position sizing and stops
4. **Backtesting** - Validate historical performance
5. **Paper Trading** - Live validation without capital
6. **Deployment** - Production with governance checks

## 7.4 Available Strategies

| Strategy | Type | Status |
|----------|------|--------|
| SMA Crossover | Technical | Active |
| Momentum | Technical | Active |
| Mean Reversion | Statistical | Testing |
