# Ordinis v0.53 Portfolio Integration Plan

## Executive Summary

Integrate v0.53 Alpaca Live Trading System with existing PortfolioEngine, PositionSizer, and PortfolioOptimizer components to replace hardcoded position sizing with dynamic, account-adaptive portfolio management.

---

## Current Issues

### 1. Low Bandwidth Observation
- **Root Cause**: System processes minute bars (1 bar/min/symbol), not tick data
- **Data Rate**: ~5.2 bars/second across 16 symbols (expected behavior)
- **Not an Issue**: This is actually efficient for the strategy

### 2. Pattern Day Trading (PDT) Limitations
- **Account Equity**: $14,999 (below $25k threshold)
- **Impact**: 325+ order rejections, limiting to 3 day trades per 5 days
- **Current Positions**: 10/10 maximum reached, blocking new entries

### 3. Hardcoded Position Sizing
- **Problem**: Fixed parameters don't adapt to account conditions
- **Missing**: Dynamic sizing based on volatility, win rates, and correlations
- **Solution**: Leverage existing PortfolioEngine infrastructure

---

## Integration Architecture

```
┌─────────────────────┐
│   Market Data       │
│  (Minute Bars)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   SignalCore        │
│  (Signal Generation)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐        ┌──────────────────────┐
│   PositionSizer     │◄───────┤  PortfolioOptimizer  │
│  (Dynamic Sizing)   │        │  (GPU-Accelerated)   │
└──────────┬──────────┘        └──────────────────────┘
           │
           ▼
┌─────────────────────┐
│    RiskGuard        │
│  (Risk Validation)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Alpaca Executor   │
│  (Order Placement)  │
└─────────────────────┘
```

---

## Implementation Plan

### Phase 1: Position Sizing Integration (Immediate)

#### 1.1 Replace Hardcoded Logic
**File**: `exe/053/ordinis-v053-alpaca-live.py`

```python
# REMOVE hardcoded parameters
MAX_POSITIONS = 10  # DELETE
POSITION_SIZE_PCT = 0.15  # DELETE

# ADD dynamic sizing
from ordinis.engines.portfolio.sizing import PositionSizer, SizingConfig, SizingMethod

class AlpacaOrchestrator:
    def __init__(self, account_info):
        # Dynamic configuration based on account
        self.sizing_config = self._determine_sizing_config(account_info)
        self.position_sizer = PositionSizer(self.sizing_config)
        self.position_sizer.set_portfolio_value(account_info['equity'])
```

#### 1.2 Account-Adaptive Configuration
```python
def _determine_sizing_config(self, account_info) -> SizingConfig:
    """Adapt sizing to account conditions."""
    equity = float(account_info['equity'])

    if equity < 25000:  # PDT Protection Active
        return SizingConfig(
            method=SizingMethod.VOLATILITY_ADJUSTED,
            max_position_pct=0.20,      # Larger positions, fewer trades
            max_portfolio_leverage=1.0,  # No leverage
            max_positions=5,             # Reduce position count
            target_volatility=0.10,      # Conservative 10% vol
            stop_loss_pct=0.02           # Tight 2% stops
        )
    else:  # No PDT Restrictions
        return SizingConfig(
            method=SizingMethod.HALF_KELLY,  # Optimal sizing
            max_position_pct=0.15,
            max_portfolio_leverage=1.5,
            max_positions=10,
            target_volatility=0.15,
            stop_loss_pct=0.03
        )
```

#### 1.3 Signal Processing Enhancement
```python
async def execute_trade(self, signal: Dict):
    """Execute with dynamic position sizing."""

    # Calculate position size
    size_result = self.position_sizer.calculate_size(
        symbol=signal['symbol'],
        current_price=signal['price'],
        signal_strength=signal['confidence'],
        win_rate=self.metrics.get_win_rate(signal['symbol']),
        avg_win_loss_ratio=self.metrics.get_ratio(signal['symbol'])
    )

    # Apply portfolio constraints
    if self._check_portfolio_constraints(size_result):
        await self._place_alpaca_order(size_result)
```

### Phase 2: Portfolio Optimization (Week 1)

#### 2.1 GPU-Accelerated Rebalancing
```python
from ordinis.engines.portfolioopt.optimizer import PortfolioOptimizer

class AlpacaOrchestrator:
    def __init__(self):
        self.portfolio_optimizer = PortfolioOptimizer(method="mean_variance")
        self.rebalance_schedule = "1H"  # Hourly for PDT accounts
```

#### 2.2 Periodic Rebalancing Task
```python
async def rebalance_portfolio(self):
    """Optimize portfolio weights hourly."""
    while self.running:
        await asyncio.sleep(3600)  # 1 hour

        # Get current positions and returns
        positions = await self.get_positions()
        returns_data = await self.get_returns_matrix(positions.keys())

        # Optimize weights (GPU if available)
        optimal_weights = self.portfolio_optimizer.optimize(
            returns_data,
            constraints={'max_weight': self.sizing_config.max_position_pct}
        )

        # Execute rebalancing trades
        await self._execute_rebalance(optimal_weights, positions)
```

### Phase 3: Performance Tracking (Week 2)

#### 3.1 Kelly Criterion Updates
```python
class PerformanceTracker:
    """Track metrics for Kelly sizing."""

    def update_trade_result(self, symbol: str, pnl: float, entry: float, exit: float):
        """Update win rate and payoff ratios."""
        self.total_trades[symbol] += 1

        if pnl > 0:
            self.wins[symbol] += 1
            self.total_win_amount[symbol] += pnl
        else:
            self.losses[symbol] += 1
            self.total_loss_amount[symbol] += abs(pnl)

        # Update metrics for Kelly
        self.win_rate[symbol] = self.wins[symbol] / self.total_trades[symbol]
        if self.losses[symbol] > 0:
            avg_win = self.total_win_amount[symbol] / max(1, self.wins[symbol])
            avg_loss = self.total_loss_amount[symbol] / max(1, self.losses[symbol])
            self.win_loss_ratio[symbol] = avg_win / avg_loss
```

#### 3.2 Integration with LearningEngine
```python
# Feed performance data to LearningEngine
feedback_collector.record_trade_outcome({
    'symbol': symbol,
    'win_rate': tracker.win_rate[symbol],
    'ratio': tracker.win_loss_ratio[symbol],
    'strategy': 'v053_live'
})
```

---

## PDT-Specific Optimizations

### For Accounts Under $25k

1. **Swing Trading Mode**
   ```python
   if self.pdt_protection:
       self.min_holding_period = timedelta(days=2)  # Hold 2+ days
       self.exit_timeframe = "1D"  # Daily exit signals
       self.max_day_trades_per_week = 3
   ```

2. **Position Concentration**
   ```python
   # Fewer, larger positions for PDT accounts
   self.sizing_config.max_positions = 5
   self.sizing_config.max_position_pct = 0.20  # 20% per position
   ```

3. **Signal Filtering**
   ```python
   # Higher confidence threshold for PDT accounts
   self.min_confidence = 0.70 if self.pdt_protection else 0.60
   self.signal_cooldown_minutes = 120  # 2-hour cooldown
   ```

---

## Context Management

### Minimize Claude Code Context Usage

1. **Silent Launch Mode**
   ```python
   # launch_v053_minimal_output.py
   - Redirects all output to log files
   - No console streaming to preserve context
   - Returns only PID for process management
   ```

2. **Lightweight Monitoring**
   ```python
   # monitor_v053.py
   - Shows only essential metrics
   - Reads last N lines efficiently
   - No full log loading
   ```

3. **Process Management**
   ```python
   # manage_v053.py
   - Start/stop/restart without output
   - Status checks via process info only
   - PID-based management
   ```

---

## Success Metrics

### Week 1 Targets
- [ ] Zero hardcoded position parameters
- [ ] Dynamic sizing based on account equity
- [ ] PDT-compliant trade execution
- [ ] GPU-optimized portfolio rebalancing

### Week 2 Targets
- [ ] Kelly criterion using actual win rates
- [ ] Correlation-aware position sizing
- [ ] Automated rebalancing triggers
- [ ] Performance metrics feeding LearningEngine

### Month 1 Targets
- [ ] Sharpe ratio > 1.5
- [ ] Max drawdown < 10%
- [ ] Win rate > 55%
- [ ] Zero PDT violations

---

## Risk Mitigations

1. **PDT Protection**
   - Track day trades in rolling 5-day window
   - Block 4th day trade automatically
   - Switch to swing mode when limit approached

2. **Position Sizing Limits**
   - Never exceed account buying power
   - Maintain 10% cash buffer
   - Scale down during high volatility

3. **Correlation Management**
   - Limit sector concentration
   - Monitor portfolio beta
   - Reduce correlated positions

---

## Testing Strategy

### 1. Paper Trading Validation
```bash
# Run with paper account first
python exe/053/launch_v053_minimal_output.py --paper
```

### 2. Small Capital Test
- Start with $1,000 allocation
- Validate PDT logic
- Confirm position sizing scales correctly

### 3. Full Deployment
- Scale to full $15k account
- Monitor metrics for 1 week
- Adjust parameters based on performance

---

## Monitoring Dashboard

### Key Metrics to Track
```python
{
    "account": {
        "equity": 14999.41,
        "positions": 5,  # Reduced from 10
        "day_trades_used": 2,  # Track PDT
        "day_trades_remaining": 1
    },
    "performance": {
        "win_rate": 0.55,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.08,
        "profit_factor": 1.4
    },
    "sizing": {
        "method": "VOLATILITY_ADJUSTED",
        "avg_position_size": 0.18,
        "portfolio_volatility": 0.09
    }
}
```

---

## Implementation Timeline

| Phase | Task | Duration | Priority |
|-------|------|----------|----------|
| 1 | Remove hardcoded parameters | 1 day | Critical |
| 1 | Integrate PositionSizer | 1 day | Critical |
| 1 | Add PDT protection logic | 1 day | Critical |
| 2 | Add PortfolioOptimizer | 2 days | High |
| 2 | Implement rebalancing | 2 days | High |
| 3 | Performance tracking | 3 days | Medium |
| 3 | LearningEngine integration | 3 days | Medium |
| 4 | Testing & validation | 1 week | Critical |

---

## Next Steps

1. **Immediate Action** (Today)
   - Implement position_sizing_integration.py
   - Remove hardcoded parameters from v053
   - Test with paper account

2. **This Week**
   - Add portfolio optimization
   - Implement PDT tracking
   - Deploy to live account with minimal capital

3. **Next Week**
   - Full capital deployment
   - Performance monitoring
   - Parameter tuning based on results

---

## Conclusion

This integration plan leverages Ordinis's existing sophisticated portfolio management infrastructure while addressing the specific constraints of a PDT-limited account. By replacing hardcoded parameters with dynamic, GPU-accelerated portfolio optimization, the system will adapt to market conditions and account restrictions automatically, maximizing returns within regulatory constraints.