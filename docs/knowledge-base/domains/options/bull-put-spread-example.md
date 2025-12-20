# Bull-Put-Spread Implementation Example

This file demonstrates how to structure a complex strategy implementation using progressive disclosure. The full bull-put-spread implementation (1,387 lines) from the original template has been organized into separate focused files.

---

## Overview

**Bull-Put-Spread (Credit Put Spread)**: A bullish income strategy that involves selling a higher-strike put (receiving premium) and buying a lower-strike put (paying premium) with the same expiration date.

**Position Structure**:
- Sell 1 OTM put (higher strike) - **Collect premium**
- Buy 1 OTM put (lower strike) - **Pay premium**
- Net Credit = Premium Received - Premium Paid

**Target Market**: Small to mid-cap stocks with position sizes $5,000 to $50,000

---

## Content Organization (Progressive Disclosure)

The original 1,387-line implementation would be split into these focused reference files:

### 1. position-sizing.md
**Lines**: ~150 lines
**Content**:
- Investment range ($5,000 - $50,000)
- Collateral/margin calculation formulas
- Position sizing logic with capital allocation
- Contract calculation with max_position_pct
- Example code for `calculate_collateral()` and `calculate_contracts()`

**Key Code**:
```python
def calculate_collateral(spread_width: float, contracts: int) -> float:
    """Calculate margin requirement for bull-put-spread."""
    return spread_width * 100 * contracts

def calculate_contracts(
    available_capital: float,
    spread_width: float,
    net_credit: float,
    max_position_pct: float = 0.25
) -> int:
    """Determine optimal number of contracts."""
    max_allocation = available_capital * max_position_pct
    collateral_per_contract = (spread_width * 100) - (net_credit * 100)
    contracts = int(max_allocation / collateral_per_contract)
    return max(1, contracts)
```

---

### 2. strike-selection.md
**Lines**: ~200 lines
**Content**:
- Short put selection framework
- Delta-based classification (-0.10 to -0.30)
- Conservative vs aggressive strike levels
- Distance from current price analysis
- Probability of profit estimates

**Key Code**:
```python
@dataclass
class ShortPutOption:
    """Short put leg configuration."""
    strike: float
    premium: float
    delta: float
    distance_from_current: float

    @property
    def conservative_level(self) -> str:
        """Classify strike selection aggressiveness."""
        if self.delta >= -0.10:
            return "Very Conservative (Deep OTM)"
        elif self.delta >= -0.15:
            return "Conservative (Moderate OTM)"
        elif self.delta >= -0.20:
            return "Moderate (Slightly OTM)"
        elif self.delta >= -0.30:
            return "Aggressive (Near ATM)"
        else:
            return "Very Aggressive (ITM)"
```

**Guidelines**:
- Very Conservative: Delta -0.05 to -0.10 (5-10% OTM)
- Conservative: Delta -0.10 to -0.15 (7-12% OTM)
- Moderate: Delta -0.15 to -0.20 (5-8% OTM)
- Aggressive: Delta -0.20 to -0.30 (2-5% OTM)

---

### 3. spread-width-analysis.md
**Lines**: ~250 lines
**Content**:
- Spread width options ($2.50, $5, $10)
- SpreadConfiguration dataclass
- Max profit/loss calculations
- Return on risk (ROR) analysis
- Return on capital (ROC) analysis
- Comparison function for different widths

**Key Code**:
```python
@dataclass
class SpreadConfiguration:
    short_strike: float
    long_strike: float
    short_premium: float
    long_premium: float

    @property
    def spread_width(self) -> float:
        return self.short_strike - self.long_strike

    @property
    def net_credit(self) -> float:
        return self.short_premium - self.long_premium

    @property
    def max_profit(self) -> float:
        return self.net_credit * 100

    @property
    def max_loss(self) -> float:
        return (self.spread_width - self.net_credit) * 100

    @property
    def return_on_risk(self) -> float:
        return (self.max_profit / self.max_loss) * 100 if self.max_loss else 0

def compare_spread_widths(
    short_strike: float,
    short_premium: float,
    long_strikes: List[float],
    long_premiums: List[float]
) -> pd.DataFrame:
    """Compare $2.50, $5, $10 spreads for same short strike."""
    # Returns comparison DataFrame
```

---

### 4. expiration-cycles.md
**Lines**: ~180 lines
**Content**:
- 30/45/60/90 day expiration comparison
- Time decay (theta) analysis
- Premium collection vs time tradeoff
- Annualized return calculations
- Expiration cycle recommendations

**Key Code**:
```python
@dataclass
class ExpirationCycle:
    days_to_expiration: int
    net_credit: float
    theta_per_day: float

    @property
    def annualized_return(self) -> float:
        """Annualize the return for comparison."""
        period_return = self.net_credit  # Assume per contract
        periods_per_year = 365 / self.days_to_expiration
        return period_return * periods_per_year

    @property
    def time_efficiency(self) -> float:
        """Credit per day of time."""
        return self.net_credit / self.days_to_expiration

def compare_expiration_cycles(
    spread_config: SpreadConfiguration,
    cycles: List[int] = [30, 45, 60, 90]
) -> pd.DataFrame:
    """Compare returns across different expiration cycles."""
    # Returns comparison DataFrame
```

---

### 5. iv-analysis.md
**Lines**: ~220 lines
**Content**:
- IV rank calculation (0-100 scale)
- IV percentile calculation
- Expected move formulas
- Probability of profit (delta-based)
- IV decision support (when to sell spreads)

**Key Code**:
```python
def calculate_iv_rank(
    current_iv: float,
    iv_52w_high: float,
    iv_52w_low: float
) -> float:
    """Calculate IV Rank on 0-100 scale."""
    iv_rank = ((current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low)) * 100
    return max(0, min(100, iv_rank))

def expected_move(
    stock_price: float,
    implied_volatility: float,
    days_to_expiration: int
) -> Dict[str, float]:
    """Calculate expected stock move using IV."""
    time_factor = np.sqrt(days_to_expiration / 365)
    one_sd_move = stock_price * implied_volatility * time_factor

    return {
        'expected_move_1sd': one_sd_move,
        'expected_move_2sd': one_sd_move * 2,
        'upper_1sd': stock_price + one_sd_move,
        'lower_1sd': stock_price - one_sd_move
    }

def probability_of_profit(short_put_delta: float) -> float:
    """Approximate POP using delta. POP ≈ 100 - |Delta| × 100"""
    return (1 - abs(short_put_delta)) * 100
```

---

### 6. risk-assessment.md
**Lines**: ~200 lines
**Content**:
- Early assignment risk evaluation
- Ex-dividend date considerations
- Pin risk assessment (stock near short strike at expiration)
- ITM scenario analysis
- Risk level classification

**Key Code**:
```python
def check_early_assignment_risk(
    short_put: ShortPutOption,
    current_stock_price: float,
    ex_dividend_date: Optional[datetime],
    dividend_amount: float,
    days_to_expiration: int
) -> Dict[str, Any]:
    """Assess early assignment risk for short put."""
    is_itm = current_stock_price < short_put.strike
    intrinsic_value = max(short_put.strike - current_stock_price, 0)
    extrinsic_value = short_put.premium - intrinsic_value

    # High risk if: ITM + near ex-div + dividend > extrinsic value
    if is_itm and ex_dividend_date:
        days_to_ex_div = (ex_dividend_date - datetime.now()).days
        if days_to_ex_div < 7 and dividend_amount > extrinsic_value:
            risk_level = "High"
        else:
            risk_level = "Moderate" if is_itm else "Low"
    else:
        risk_level = "Low"

    return {
        'risk_level': risk_level,
        'is_itm': is_itm,
        'intrinsic_value': intrinsic_value,
        'extrinsic_value': extrinsic_value,
        'recommendation': "Monitor closely" if risk_level == "High" else "Normal"
    }

def assess_pin_risk(
    stock_price: float,
    short_strike: float,
    days_to_expiration: int
) -> Dict[str, Any]:
    """Evaluate pin risk (stock near short strike at expiration)."""
    distance_pct = abs(stock_price - short_strike) / stock_price * 100

    if days_to_expiration <= 3:
        if distance_pct < 1.0:
            risk = "High"
        elif distance_pct < 2.0:
            risk = "Moderate"
        else:
            risk = "Low"
    else:
        risk = "Low"

    return {
        'pin_risk': risk,
        'distance_pct': distance_pct,
        'recommendation': "Consider closing" if risk == "High" else "Monitor"
    }
```

---

### 7. liquidity-execution.md
**Lines**: ~150 lines
**Content**:
- Volume and Open Interest requirements
- Bid/ask spread analysis
- Slippage estimation
- Liquidity scoring
- Execution recommendations

**Key Code**:
```python
@dataclass
class LiquidityCheck:
    ticker: str
    option_symbol: str
    bid: float
    ask: float
    volume: int
    open_interest: int

    @property
    def bid_ask_spread_pct(self) -> float:
        mid = (self.bid + self.ask) / 2
        return ((self.ask - self.bid) / mid) * 100

    @property
    def liquidity_score(self) -> str:
        """Classify liquidity quality."""
        if self.volume > 500 and self.open_interest > 1000 and self.bid_ask_spread_pct < 5:
            return "Excellent"
        elif self.volume > 100 and self.open_interest > 500 and self.bid_ask_spread_pct < 10:
            return "Good"
        elif self.volume > 50 and self.open_interest > 200:
            return "Fair"
        else:
            return "Poor"

    @property
    def estimated_slippage(self) -> float:
        """Estimate slippage based on spread and liquidity."""
        base_slippage = (self.ask - self.bid) / 2

        # Adjust for liquidity
        if self.liquidity_score == "Poor":
            return base_slippage * 2
        elif self.liquidity_score == "Fair":
            return base_slippage * 1.5
        else:
            return base_slippage
```

---

### 8. risk-controls.md
**Lines**: ~120 lines
**Content**:
- Per-trade max loss limits
- Portfolio allocation caps
- Correlation/sector concentration limits
- Position validation logic
- Risk control framework

**Key Code**:
```python
@dataclass
class RiskControls:
    max_loss_per_trade: float
    max_allocation_pct: float
    max_correlation: float
    max_sector_exposure_pct: float

    def validate_position(
        self,
        spread_config: SpreadConfiguration,
        contracts: int,
        portfolio_value: float,
        existing_positions: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """Validate position against risk controls."""
        violations = []

        # Check max loss
        position_max_loss = spread_config.max_loss * contracts
        if position_max_loss > self.max_loss_per_trade:
            violations.append(f"Max loss ${position_max_loss:.0f} exceeds limit ${self.max_loss_per_trade:.0f}")

        # Check portfolio allocation
        position_value = spread_config.collateral_required * contracts
        allocation_pct = position_value / portfolio_value
        if allocation_pct > self.max_allocation_pct:
            violations.append(f"Allocation {allocation_pct:.1%} exceeds limit {self.max_allocation_pct:.1%}")

        # Check sector exposure
        # ... (sector concentration logic)

        approved = len(violations) == 0
        return (approved, violations)
```

---

### 9. management-triggers.md
**Lines**: ~100 lines
**Content**:
- Take-profit thresholds (50% of max profit)
- Stop-loss levels (200% of max profit)
- Delta threshold monitoring
- Days-to-close rules
- Management action recommendations

**Key Code**:
```python
@dataclass
class ManagementTriggers:
    take_profit_pct: float = 50.0    # Close at 50% of max profit
    stop_loss_pct: float = 200.0     # Stop at 200% of max profit
    delta_threshold: float = -0.40   # Close if delta exceeds this
    days_to_close: int = 7           # Close within 7 days of expiration

    def check_management_action(
        self,
        current_pl: float,
        max_profit: float,
        max_loss: float,
        short_put_delta: float,
        days_to_expiration: int
    ) -> Dict[str, Any]:
        """Determine if position requires management action."""

        actions = []
        priority = "None"

        # Check take-profit
        if current_pl >= (max_profit * self.take_profit_pct / 100):
            actions.append("TAKE PROFIT - Achieved 50% of max profit")
            priority = "High"

        # Check stop-loss
        if current_pl <= -(max_profit * self.stop_loss_pct / 100):
            actions.append("STOP LOSS - Loss exceeds 200% of max profit")
            priority = "Critical"

        # Check delta
        if abs(short_put_delta) > abs(self.delta_threshold):
            actions.append(f"DELTA ALERT - Short put delta {short_put_delta:.2f} exceeds threshold")
            priority = "High" if priority != "Critical" else priority

        # Check expiration proximity
        if days_to_expiration <= self.days_to_close:
            actions.append(f"EXPIRATION - {days_to_expiration} days remaining, consider closing")
            priority = "Medium" if priority == "None" else priority

        return {
            'action_required': len(actions) > 0,
            'priority': priority,
            'actions': actions,
            'recommendation': actions[0] if actions else "Hold position"
        }
```

---

### 10. examples.md
**Lines**: ~150 lines
**Content**: 5 real-world scenarios with full configurations

**Example 1 - TGT (Low IV 18%)**:
```python
{
    'ticker': 'TGT',
    'scenario': 'Low IV Environment',
    'stock_price': 152.50,
    'iv_current': 0.18,
    'iv_rank': 25,
    'position_size': 15000,
    'recommended_configuration': {
        'short_strike': 145,        # ~5% OTM
        'short_delta': -0.12,
        'short_premium': 1.20,
        'long_strike': 140,         # $5 width
        'long_premium': 0.30,
        'net_credit': 0.90,
        'dte': 60,
        'contracts': 3,
        'max_profit': 270,
        'max_loss': 1230,
        'collateral': 1500,
        'ror_pct': 21.9,
        'roc_pct': 18.0,
        'pop': 88
    },
    'rationale': 'Conservative approach in low IV. Longer DTE (60 days) for better time value collection.'
}
```

**Example 2 - TDOC (Medium IV 32%)**:
```python
{
    'ticker': 'TDOC',
    'scenario': 'Medium IV Environment',
    'stock_price': 38.50,
    'iv_current': 0.32,
    'iv_rank': 45,
    'recommended_configuration': {
        'short_strike': 36,         # ~6.5% OTM
        'short_delta': -0.18,
        'long_strike': 33.50,       # $2.50 width
        'net_credit': 1.10,
        'dte': 45,
        'contracts': 4,
        'max_profit': 440,
        'max_loss': 560,
        'ror_pct': 78.6
    }
}
```

**Example 3 - SNAP (High IV 58%)**:
```python
{
    'ticker': 'SNAP',
    'scenario': 'High IV Environment',
    'stock_price': 23.80,
    'iv_current': 0.58,
    'iv_rank': 78,
    'recommended_configuration': {
        'short_strike': 21,         # ~12% OTM
        'short_delta': -0.15,
        'long_strike': 19,          # $2 width
        'net_credit': 0.90,
        'dte': 30,                  # Shorter DTE to avoid IV crush
        'contracts': 5,
        'special_notes': 'High IV = higher premium but greater assignment risk'
    }
}
```

(Plus examples for PLTR and CAT...)

---

## How to Use This Pattern

When creating an actual bull-put-spread skill:

**1. Create directory structure**:
```
bull-put-spread/
├── SKILL.md                    (~80 lines - overview + navigation)
├── scripts/
│   └── bull_put_calculator.py
├── references/
│   ├── position-sizing.md
│   ├── strike-selection.md
│   ├── spread-width-analysis.md
│   ├── expiration-cycles.md
│   ├── iv-analysis.md
│   ├── risk-assessment.md
│   ├── liquidity-execution.md
│   ├── risk-controls.md
│   ├── management-triggers.md
│   └── examples.md
└── assets/
    └── sample_positions.csv
```

**2. SKILL.md content** (~80 lines):
```markdown
---
name: bull-put-spread
description: Analyzes bull-put-spread credit spreads with position sizing and risk management. Requires numpy>=1.24.0, pandas>=2.0.0, matplotlib>=3.7.0, scipy>=1.10.0. Use when evaluating put spreads, comparing strikes, or assessing spread configurations.
---

# Bull-Put-Spread

## Overview
[2-3 sentences]

## Quick Reference
- Type: Credit spread
- Risk: Defined
- Best For: Neutral to bullish

## Core Workflow
1. **Position Sizing** → [references/position-sizing.md](references/position-sizing.md)
2. **Strike Selection** → [references/strike-selection.md](references/strike-selection.md)
3. **Spread Width** → [references/spread-width-analysis.md](references/spread-width-analysis.md)
... (etc for all 10 reference files)

## Scripts
- `scripts/bull_put_calculator.py` - Calculate metrics
- Run: `python scripts/bull_put_calculator.py --help`

## Examples
See [references/examples.md](references/examples.md) for 5 real-world scenarios.
```

**3. Each reference file** focuses on ONE topic from the list above

**Result**: Clean progressive disclosure - Claude loads only what's needed for each query.

---

## Benefits of This Approach

✅ **Token Efficient**: SKILL.md is 80 lines (vs 1,387)
✅ **Progressive Loading**: Claude reads only relevant references
✅ **Maintainable**: Update one topic without touching others
✅ **Discoverable**: Clear navigation structure
✅ **Compliant**: Meets both Anthropic + Claude Code guidelines

---

**This example demonstrates the progressive disclosure pattern for complex strategy implementations.**
