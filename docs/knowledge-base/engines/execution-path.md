---
title: Execution Path (Research → Live Trading)
date: 2025-12-18
version: 1.0
type: knowledge base
description: >
  Phased approach for progressing from research and backtesting to paper and live trading.
source_of_truth: ../inbox/documents/system-specification.md
---

# Path from Research to Automated Trade Execution

## Overview

This document outlines the phased approach to deploying an automated trading system, from initial research through live trading with full autonomy.

**Guiding Principle**: Progress through phases only when previous phase criteria are met. Never skip phases.

---

## Phase Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT PHASES                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1        Phase 2        Phase 3        Phase 4        Phase 5│
│  ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐ │
│  │ KB & │  ──▶ │ Code │  ──▶ │Paper │  ──▶ │ Live │  ──▶ │ Full │ │
│  │Design│      │ & BT │      │Trade │      │ Pilot│      │ Auto │ │
│  └──────┘      └──────┘      └──────┘      └──────┘      └──────┘ │
│                                                                     │
│  Duration:     Duration:     Duration:     Duration:     Duration: │
│  2-4 weeks     4-8 weeks     4-12 weeks    4-12 weeks    Ongoing   │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Knowledge Base & Strategy Design

### Objectives
- Build comprehensive knowledge base
- Design strategy specifications
- Define risk parameters
- Establish data requirements

### Deliverables
- [ ] Complete KB documentation
- [ ] 3-5 fully specified strategies (YAML format)
- [ ] Risk management framework
- [ ] Data source selection
- [ ] Technology stack decision

### Exit Criteria
- All KB sections documented
- Strategy specs are complete and unambiguous
- Risk parameters defined and approved
- Data sources identified and accessible

### Current Status: **IN PROGRESS**

---

## Phase 2: Code Implementation & Backtesting

### Objectives
- Implement AnalyticsEngine (ProofBench) simulation/backtesting
- Code strategies from specifications
- Run comprehensive backtests
- Validate and refine strategies

### 2.1 AnalyticsEngine (ProofBench) Development

**Tasks**:
1. Build event-driven simulation core
2. Implement execution simulator with slippage/costs
3. Build portfolio tracking module
4. Implement performance analytics
5. Create walk-forward testing framework

**Milestones**:
- [ ] Core engine passing unit tests
- [ ] Execution model validated against known scenarios
- [ ] Performance metrics matching manual calculations

### 2.2 Strategy Implementation

**Tasks**:
1. Translate YAML specs to code
2. Implement indicator calculations
3. Build signal generation logic
4. Implement entry/exit rules
5. Add position sizing logic

**Code Structure**:
```python
# Each strategy is a module
strategies/
├── momentum_breakout/
│   ├── __init__.py
│   ├── signals.py      # Signal generation
│   ├── entry.py        # Entry logic
│   ├── exit.py         # Exit logic
│   ├── sizing.py       # Position sizing
│   └── config.yaml     # Parameters
├── mean_reversion/
│   └── ...
└── base/
    ├── strategy.py     # Abstract base class
    └── indicators.py   # Shared indicators
```

### 2.3 Backtesting Protocol

**Process**:
1. In-sample optimization
2. Out-of-sample validation
3. Walk-forward analysis
4. Monte Carlo simulation
5. Parameter sensitivity testing
6. Regime testing

**Required Documentation**:
- Backtest report for each strategy
- Parameter sensitivity analysis
- Walk-forward results
- Comparison to benchmarks

### Exit Criteria
- Strategies pass minimum validation criteria
- Out-of-sample Sharpe > 1.0
- Walk-forward degradation < 30%
- Monte Carlo 5th percentile profitable
- Full documentation complete

---

## Phase 3: Paper Trading / Simulation

### Objectives
- Validate strategies in real-time conditions
- Test system infrastructure
- Identify operational issues
- Build confidence before risking capital

### 3.1 Paper Trading Setup

**Requirements**:
- Real-time market data feed
- Paper trading account (broker or simulated)
- Full logging infrastructure
- Monitoring and alerting
- Daily reconciliation process

**Broker Paper Trading Options**:
| Broker | Paper Trading | API Access | Notes |
|--------|---------------|------------|-------|
| TD Ameritrade/Schwab | Yes | Yes | Transitioning APIs |
| Interactive Brokers | Yes | Yes | Best API, complex |
| Alpaca | Yes | Yes | Modern API, stocks only |
| Tradier | Yes | Yes | Good for options |

### 3.2 Paper Trading Protocol

**Duration**: Minimum 3 months (covering different market conditions)

**Daily Process**:
1. System generates signals before market open
2. Paper orders placed automatically
3. Fills recorded and positions tracked
4. End-of-day reconciliation
5. Performance tracking and logging

**Monitoring Requirements**:
```python
PAPER_TRADE_MONITORING = {
    'signal_accuracy': 'Compare signals to expected',
    'execution_timing': 'Log all order timestamps',
    'fill_quality': 'Compare paper fills to actual market',
    'position_tracking': 'Verify positions match expectations',
    'risk_compliance': 'Check all limits respected',
    'system_health': 'Monitor uptime, errors, latency'
}
```

### 3.3 Paper Trading Metrics

**Track and Compare**:
- Signals generated vs executed
- Paper PnL vs backtest expectations
- Slippage estimates vs actual
- System uptime and reliability
- Error rates and types

### Exit Criteria
- 3+ months of paper trading
- Paper results within 20% of backtest expectations
- System uptime > 99.5%
- Zero critical errors in final month
- All operational processes working smoothly
- Risk limits never breached

---

## Phase 4: Limited Live Deployment

### Objectives
- Begin real money trading with minimal risk
- Validate execution in live markets
- Build operational experience
- Prove profitability with real capital

### 4.1 Initial Deployment Parameters

**Capital Constraints**:
```python
PILOT_PHASE_LIMITS = {
    'initial_capital': min(account_size, 5000),  # Start small
    'max_position_size': 0.05,  # 5% max per position
    'max_positions': 3,
    'risk_per_trade': 0.005,  # 0.5% (half normal)
    'daily_loss_limit': 0.01,  # 1%
    'max_drawdown': 0.05  # 5% halt
}
```

**Scaling Schedule**:
| Week | Capital % | Max Positions | Risk/Trade |
|------|-----------|---------------|------------|
| 1-2 | 25% | 2 | 0.5% |
| 3-4 | 50% | 3 | 0.75% |
| 5-8 | 75% | 4 | 0.75% |
| 9-12 | 100% | Full | 1.0% |

### 4.2 Oversight Requirements

**Human Oversight**:
- Daily review of all trades and positions
- Manual approval for any trade > $X
- Kill switch accessible at all times
- Weekly strategy review meetings

**Automated Oversight**:
- Real-time PnL monitoring
- Position limit enforcement
- Automatic halt on daily loss limit
- Error alerting (SMS, email)

### 4.3 Live Performance Tracking

```python
class LivePerformanceTracker:
    """
    Track live performance vs expectations.
    """
    def __init__(self, backtest_baseline: BacktestResults):
        self.baseline = backtest_baseline
        self.live_trades = []

    def add_trade(self, trade: Trade):
        self.live_trades.append(trade)
        self._compare_to_baseline()

    def _compare_to_baseline(self):
        """
        Alert if live performance deviates significantly.
        """
        live_metrics = calculate_metrics(self.live_trades)

        if live_metrics.sharpe < self.baseline.sharpe * 0.5:
            alert("Sharpe significantly below baseline")

        if live_metrics.win_rate < self.baseline.win_rate * 0.7:
            alert("Win rate significantly below baseline")

        if live_metrics.max_drawdown > self.baseline.max_drawdown * 1.5:
            alert("Drawdown exceeding baseline")
```

### Exit Criteria
- 3+ months of profitable live trading
- Live Sharpe within 30% of backtest
- No critical system failures
- All operational procedures documented
- Consistent risk limit compliance

---

## Phase 5: Full Autonomous Operation

### Objectives
- Full capital deployment
- Reduced manual oversight
- Strategy expansion
- Continuous improvement

### 5.1 Full Deployment Configuration

```python
FULL_DEPLOYMENT = {
    'capital': 'full_account',
    'strategies': 'all_validated',
    'max_positions': 10,
    'risk_per_trade': 0.01,
    'daily_loss_limit': 0.03,
    'max_drawdown': 0.15,
    'oversight': 'weekly_review',
    'intervention': 'exception_only'
}
```

### 5.2 Ongoing Operations

**Daily Automated**:
- Pre-market: Generate signals, prepare orders
- Market hours: Execute, manage positions
- Post-market: Reconciliation, reporting
- Overnight: Risk checks, system health

**Weekly Human Review**:
- Performance vs baseline
- Risk metric review
- Strategy health assessment
- System health assessment

**Monthly**:
- Detailed performance analysis
- Strategy refinement decisions
- Parameter updates if needed
- New strategy pipeline review

### 5.3 Strategy Lifecycle Management

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│Research  │───▶│Validation│───▶│Production│───▶│Retirement│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │
     │               │               │               │
  New ideas    Backtest/Paper   Live trading    Performance
  developed    validation      with capital    degradation
```

**Retirement Triggers**:
```python
RETIREMENT_CRITERIA = {
    'sustained_loss': 'Negative returns for 6 months',
    'sharpe_degradation': 'Sharpe < 0.5 for 3 months',
    'drawdown_breach': 'Max drawdown exceeded',
    'market_change': 'Strategy edge eliminated',
    'capacity_hit': 'Strategy returns diminish with size'
}
```

---

## System Architecture (High-Level)

```
┌────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION ARCHITECTURE                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐         │
│   │   Market    │────▶│   Signal    │────▶│    Risk     │         │
│   │ Data Feed   │     │  Generator  │     │   Engine    │         │
│   └─────────────┘     └─────────────┘     └─────────────┘         │
│          │                   │                   │                 │
│          ▼                   ▼                   ▼                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐         │
│   │    Data     │     │  Strategy   │     │   Order     │         │
│   │   Storage   │     │   Engine    │     │  Manager    │         │
│   └─────────────┘     └─────────────┘     └─────────────┘         │
│                                                  │                 │
│                                                  ▼                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐         │
│   │  Monitoring │◀────│  Portfolio  │◀────│   Broker    │         │
│   │  & Alerts   │     │   Manager   │     │    API      │         │
│   └─────────────┘     └─────────────┘     └─────────────┘         │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

**Market Data Feed → StreamingBus**:
- Real-time price data
- Options chains
- News and events
- Economic data

**SignalEngine (SignalCore)**:
- Indicator calculations
- Pattern detection
- Signal scoring
- Universe filtering

**RiskEngine (RiskGuard)**:
- Pre-trade validation
- Position limits
- Exposure checks
- Kill switch logic

**ExecutionEngine (FlowRoute)**:
- Order creation
- Smart routing
- Fill tracking
- Order lifecycle

**Broker API**:
- Connection to broker
- Order submission
- Fill confirmation
- Position sync

**PortfolioEngine**:
- Position tracking
- PnL calculation
- Margin management
- Reconciliation

**Monitoring & Alerts**:
- System health
- Performance tracking
- Error alerting
- Operational dashboards

---

## Safety Mechanisms

### Kill Switches

```python
class KillSwitch:
    """
    Emergency shutdown system.
    """
    triggers = {
        'daily_loss': 0.03,      # 3% daily loss
        'max_drawdown': 0.15,    # 15% drawdown
        'system_error': True,    # Any critical error
        'connectivity': True,    # Lost broker connection
        'manual': True           # Human trigger
    }

    actions = [
        'cancel_all_pending_orders',
        'close_all_positions',  # Optional
        'disable_new_orders',
        'send_alerts',
        'log_full_state',
        'require_manual_restart'
    ]
```

### Order Validation

```python
class OrderValidator:
    """
    Pre-flight checks for all orders.
    """
    def validate(self, order: Order) -> Tuple[bool, List[str]]:
        checks = []

        # Price sanity
        if self._price_out_of_range(order):
            checks.append("Price deviation too large")

        # Size sanity
        if order.quantity > self.max_shares:
            checks.append("Order size exceeds limit")

        # Risk check
        if self._exceeds_risk_limit(order):
            checks.append("Risk limit exceeded")

        # Portfolio check
        if self._exceeds_concentration(order):
            checks.append("Concentration limit exceeded")

        return len(checks) == 0, checks
```

### Monitoring Dashboard

**Required Displays**:
- Real-time PnL (daily, MTD, YTD)
- Open positions with risk metrics
- Pending orders
- System health status
- Recent alerts and errors
- Strategy performance breakdown

---

## Broker Integration Notes

### TD Ameritrade / Charles Schwab

**Status**: TD Ameritrade API deprecated; Schwab API in development

**Considerations**:
- Check latest API availability
- May need to transition during project
- Consider alternatives for now

### Recommended Alternatives

**Interactive Brokers**:
- Pros: Robust API, global markets, options support
- Cons: Complex interface, learning curve
- Best for: Serious algorithmic trading

**Alpaca**:
- Pros: Modern REST API, commission-free, paper trading
- Cons: US stocks only (no options yet)
- Best for: Getting started, stocks-only strategies

**Tradier**:
- Pros: Good options API, reasonable costs
- Cons: Smaller, less mature
- Best for: Options-focused strategies

---

## Timeline Summary

| Phase | Duration | Key Milestone |
|-------|----------|---------------|
| 1 | 2-4 weeks | KB and specs complete |
| 2 | 4-8 weeks | Backtest validation complete |
| 3 | 3+ months | Paper trading profitable |
| 4 | 3+ months | Live pilot profitable |
| 5 | Ongoing | Full autonomous operation |

**Minimum Total**: ~9-12 months to full deployment

**Note**: These are minimums. Rushing leads to costly mistakes. Add time for setbacks and learning.

---

## Disclaimers

1. **No Profit Guarantees**: Past performance (backtests) does not predict future results
2. **Risk of Loss**: All trading involves risk of capital loss
3. **Not Financial Advice**: This is an engineering/research project
4. **Regulatory Compliance**: User responsible for compliance with applicable regulations
5. **Technology Risks**: Systems can fail; have contingency plans
