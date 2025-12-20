# Strategy-Specific Variations Guide

Guidance for implementing different types of options strategies based on their characteristics.

---

## Overview

Options strategies fall into four main categories, each with unique implementation requirements:

1. **Vertical Spreads** - Directional, defined risk/reward
2. **Stock + Option** - Stock protection or income generation
3. **Neutral Strategies** - Volatility plays
4. **Complex Multi-Leg** - Precise payoff profiles

This guide shows which files and features to emphasize for each category.

---

## Vertical Spreads

**Strategies**: Bull Call Spread, Bear Put Spread, Bull Put Spread (Credit Spread), Bear Call Spread

### Characteristics

- **Directional**: Require bullish or bearish outlook
- **Defined Risk**: Max loss and max profit are predetermined
- **Two Legs**: One long, one short option at different strikes
- **Same Expiration**: Both legs expire on the same date

### Focus Areas

**Primary Focus**:
- Directional strategies with defined risk/reward
- Mathematical rigor (Black-Scholes, Greeks)
- Spread width optimization
- Strike selection based on outlook

### Key Files

**Required**:
- `SKILL.md` - Strategy definition and overview
- `scripts/calculator.py` - Core calculator with spread calculations
- `scripts/black_scholes.py` - Options pricing (if standalone)
- `references/greeks.md` - Greeks calculations and explanations

**Recommended**:
- `scripts/strike_comparison.py` - Compare different strike selections
- `references/spread-width-analysis.md` - Analyze different widths
- `assets/examples.csv` - Sample positions

### Unique Features

**Spread Width Optimization**:
```python
def compare_spread_widths(
    short_strike: float,
    short_premium: float,
    long_strikes: List[float],
    long_premiums: List[float]
) -> pd.DataFrame:
    """Compare spread widths ($2.50, $5, $10) for same short strike."""
    # Returns comparison of max profit, max loss, ROR, collateral
```

**Strike Selection Logic**:
- Delta-based classification (conservative OTM to aggressive near-ATM)
- Probability of profit estimates
- Risk/reward ratios by strike distance

### Reference Implementation

**bull-call-spread**: `C:\Users\kjfle\Workspace\ordinis\skills\bull-call-spread\`

**Key characteristics**:
- Comprehensive SKILL.md (31KB) with all details in one file
- Black-Scholes calculator integrated
- CLI with full analysis output
- Mathematical rigor emphasis

---

## Stock + Option Strategies

**Strategies**: Covered Call, Married Put, Protective Collar, Cash-Secured Put

### Characteristics

- **Stock Component**: Includes actual stock position (100 shares per contract)
- **Protection or Income**: Either protect downside or generate income
- **Stock Price Sensitivity**: More sensitive to stock movement than spreads
- **Assignment Considerations**: May result in stock delivery

### Focus Areas

**Primary Focus**:
- Stock protection or income generation
- Portfolio-level analysis
- Position sizing with stock component
- Rolling strategies (close and reopen)
- Dividend considerations

### Key Files

**Required**:
- `SKILL.md` - Strategy definition with stock integration
- `scripts/calculator.py` - Position calculator (stock + option)
- `scripts/position_sizer.py` - Position sizing with portfolio heat
- `references/position-sizing.md` - Capital allocation, portfolio limits

**Recommended**:
- `scripts/expiration_analysis.py` - Compare expiration cycles
- `scripts/strike_comparison.py` - Strike selection for covered calls/puts
- `references/rolling-strategies.md` - When and how to roll
- `references/dividend-considerations.md` - Ex-div dates, early assignment
- `assets/sample_positions.csv` - Example portfolio positions

### Unique Features

**Stock Position Integration**:
```python
@dataclass
class MarriedPut:
    """Married put: Long stock + Long put protection."""
    stock_symbol: str
    stock_price: float
    stock_shares: int  # Typically 100 per contract
    put_strike: float
    put_premium: float
    put_contracts: int

    @property
    def total_cost(self) -> float:
        """Total capital: stock cost + put premium."""
        return (self.stock_price * self.stock_shares) + (self.put_premium * 100 * self.put_contracts)
```

**Portfolio Heat Calculation**:
```python
def calculate_portfolio_heat(
    positions: List[Position],
    portfolio_value: float,
    max_heat_pct: float = 0.02  # 2% max portfolio risk
) -> float:
    """Calculate total risk across all positions."""
    total_risk = sum(pos.max_loss for pos in positions)
    return total_risk / portfolio_value
```

**Rolling Logic**:
```python
def should_roll_covered_call(
    current_pl: float,
    max_profit: float,
    days_to_expiration: int,
    underlying_price: float,
    strike: float
) -> Tuple[bool, str]:
    """Determine if covered call should be rolled.

    Roll when:
    - Captured 80% of max profit with >7 days left
    - Stock approaching strike with >14 days left
    - Want to extend duration for more premium
    """
```

### Reference Implementation

**married-put**: `C:\Users\kjfle\Workspace\ordinis\skills\married-put\`

**Key characteristics**:
- Modular structure (13 files)
- Multiple utility scripts (5 Python files)
- Sample CSV with 5 positions
- QUICKSTART.md for fast onboarding
- Portfolio-level tools (position_sizer.py)
- Emphasis on practical execution

---

## Neutral Strategies

**Strategies**: Long Straddle, Long Strangle, Short Straddle, Short Strangle

### Characteristics

- **Non-Directional**: Profit from volatility movement, not direction
- **Volatility Sensitive**: Highly dependent on IV changes
- **Two Legs**: Call + Put at same (straddle) or different (strangle) strikes
- **Vega Dominant**: Vega (volatility sensitivity) is key Greek

### Focus Areas

**Primary Focus**:
- Volatility plays and IV changes
- Expected move calculations
- IV rank/percentile analysis
- Earnings plays and event-driven setups
- Time decay management

### Key Files

**Required**:
- `SKILL.md` - Strategy with volatility emphasis
- `scripts/calculator.py` - Straddle/strangle calculator
- `scripts/volatility_analysis.py` - IV rank, percentile, expected move
- `references/iv-analysis.md` - Volatility concepts and calculations

**Recommended**:
- `scripts/earnings_analyzer.py` - Identify earnings opportunities
- `references/expected-move.md` - Expected move formulas and examples
- `references/vega-management.md` - Managing volatility exposure
- `assets/historical_iv_data.csv` - IV history for backtesting

### Unique Features

**IV Rank and Percentile**:
```python
def calculate_iv_rank(
    current_iv: float,
    iv_52w_high: float,
    iv_52w_low: float
) -> float:
    """Calculate IV Rank (0-100 scale).

    IV Rank = (Current IV - 52W Low) / (52W High - 52W Low) × 100
    """
    iv_rank = ((current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low)) * 100
    return max(0, min(100, iv_rank))
```

**Expected Move Calculation**:
```python
def expected_move(
    stock_price: float,
    implied_volatility: float,
    days_to_expiration: int
) -> Dict[str, float]:
    """Calculate expected stock move using IV.

    Expected Move (1 SD) = Stock Price × IV × √(DTE / 365)
    """
    time_factor = np.sqrt(days_to_expiration / 365)
    one_sd_move = stock_price * implied_volatility * time_factor

    return {
        'expected_move_1sd': one_sd_move,
        'expected_move_2sd': one_sd_move * 2,
        'upper_1sd': stock_price + one_sd_move,
        'lower_1sd': stock_price - one_sd_move,
        'upper_2sd': stock_price + (one_sd_move * 2),
        'lower_2sd': stock_price - (one_sd_move * 2)
    }
```

**Earnings Play Analysis**:
```python
@dataclass
class EarningsPlay:
    """Straddle/strangle for earnings announcement."""
    ticker: str
    stock_price: float
    earnings_date: datetime
    current_iv: float
    iv_rank: float
    expected_move: float
    straddle_cost: float

    @property
    def breakeven_move_pct(self) -> float:
        """% move needed to breakeven."""
        return (self.straddle_cost / self.stock_price) * 100

    @property
    def iv_crush_risk(self) -> str:
        """Assess post-earnings IV crush risk."""
        if self.iv_rank > 75:
            return "High - IV likely to drop significantly"
        elif self.iv_rank > 50:
            return "Moderate - Some IV contraction expected"
        else:
            return "Low - IV already relatively low"
```

### Comparison: Long vs Short

**Long Straddle/Strangle**:
- Pay debit (long both call and put)
- Profit from large move in either direction
- High IV crush risk after earnings
- Vega positive (benefit from IV increase)

**Short Straddle/Strangle**:
- Receive credit (short both call and put)
- Profit from minimal movement
- Undefined risk (both sides can move against you)
- Vega negative (benefit from IV decrease)

---

## Complex Multi-Leg Strategies

**Strategies**: Iron Condor, Iron Butterfly, Long Call Butterfly, Long Put Butterfly

### Characteristics

- **Four Legs**: Two spreads combined
- **Precise Payoff**: Specific profit zones with multiple breakevens
- **Limited Risk**: All legs define maximum loss
- **Wing Configuration**: Distance between strikes affects P&L

### Focus Areas

**Primary Focus**:
- Precise payoff profiles
- Multiple strike optimization
- Wing width selection (narrow vs wide)
- Risk graph visualization
- Advanced Greeks analysis

### Key Files

**Required**:
- `SKILL.md` - Multi-leg strategy definition
- `scripts/calculator.py` - Four-leg calculator
- `scripts/leg_optimizer.py` - Optimize strike selections
- `scripts/visualizations.py` - Detailed payoff diagrams
- `references/payoff-analysis.md` - Understanding complex payoffs

**Recommended**:
- `scripts/wing_width_analyzer.py` - Compare wing configurations
- `references/adjustment-strategies.md` - Managing underwater positions
- `references/greeks-multi-leg.md` - Net Greeks for complex positions
- `assets/example_condors.csv` - Sample configurations

### Unique Features

**Multi-Leg Strike Optimization**:
```python
@dataclass
class IronCondor:
    """Iron condor: Short put spread + Short call spread."""
    # Put spread (lower)
    long_put_strike: float      # Lowest strike (protection)
    short_put_strike: float     # Higher put strike (premium collected)

    # Call spread (upper)
    short_call_strike: float    # Lower call strike (premium collected)
    long_call_strike: float     # Highest strike (protection)

    # Premiums
    long_put_premium: float
    short_put_premium: float
    short_call_premium: float
    long_call_premium: float

    @property
    def net_credit(self) -> float:
        """Total credit received."""
        return (
            self.short_put_premium +
            self.short_call_premium -
            self.long_put_premium -
            self.long_call_premium
        )

    @property
    def put_spread_width(self) -> float:
        return self.short_put_strike - self.long_put_strike

    @property
    def call_spread_width(self) -> float:
        return self.long_call_strike - self.short_call_strike

    @property
    def max_profit(self) -> float:
        """Maximum profit = Net credit."""
        return self.net_credit * 100

    @property
    def max_loss(self) -> float:
        """Maximum loss = Spread width - Net credit."""
        spread_width = max(self.put_spread_width, self.call_spread_width)
        return (spread_width - self.net_credit) * 100

    @property
    def profit_zone(self) -> Tuple[float, float]:
        """Price range for profit (both breakevens)."""
        lower_breakeven = self.short_put_strike - self.net_credit
        upper_breakeven = self.short_call_strike + self.net_credit
        return (lower_breakeven, upper_breakeven)
```

**Wing Width Analysis**:
```python
def compare_wing_widths(
    short_put: float,
    short_call: float,
    wing_widths: List[float]  # [5, 10, 15]
) -> pd.DataFrame:
    """Compare narrow vs wide wings for iron condor.

    Narrow wings ($5):
    - Less credit collected
    - Lower max loss
    - Higher probability of profit (wider profit zone)

    Wide wings ($15):
    - More credit collected
    - Higher max loss
    - Lower probability of profit (narrower profit zone relative to wing width)
    """
```

**Payoff Visualization**:
```python
def plot_iron_condor_payoff(
    condor: IronCondor,
    price_range: Optional[np.ndarray] = None
) -> None:
    """Generate detailed payoff diagram with:
    - Breakeven points marked
    - Profit zone highlighted
    - Max profit/loss lines
    - Current stock price
    - Greeks at different prices
    """
```

---

## Implementation Decision Matrix

| Strategy Type | File Count | Complexity | Greeks Focus | Reference Model |
|---------------|------------|------------|--------------|-----------------|
| **Vertical Spreads** | 6-8 | Low-Medium | High | bull-call-spread |
| **Stock + Option** | 10-13 | Medium-High | Medium | married-put |
| **Neutral** | 8-10 | Medium | Very High (Vega) | New (create) |
| **Complex Multi-Leg** | 10-15 | High | Very High | New (create) |

---

## Common Files Across All Strategies

**Every strategy should have**:
1. `SKILL.md` - Core documentation (<500 lines)
2. `scripts/calculator.py` - Main calculator
3. `references/examples.md` - Real-world scenarios
4. `requirements.txt` - Package dependencies

**Most strategies should have**:
5. `scripts/black_scholes.py` or reference to shared engine
6. `scripts/visualizations.py` - Payoff diagrams
7. `references/greeks.md` - Greeks explanations
8. `assets/sample_data.csv` - Example positions

**Optional based on complexity**:
9. `scripts/position_sizer.py` - Portfolio-level sizing
10. `scripts/strike_comparison.py` - Compare strikes
11. `scripts/expiration_analysis.py` - Compare expirations
12. `references/risk-management.md` - Risk rules and limits

---

## Strategy-Specific Customization Guide

### When Creating a Vertical Spread Skill

**Emphasize**:
- Mathematical precision (Black-Scholes, Greeks)
- Spread width optimization
- Delta-based strike selection
- ROR (Return on Risk) calculations

**De-emphasize**:
- Portfolio heat calculations
- Rolling strategies
- Stock-specific considerations

**Files to include**: calculator, black_scholes, strike_comparison, greeks reference

---

### When Creating a Stock + Option Skill

**Emphasize**:
- Stock position integration (100 shares)
- Portfolio heat and allocation
- Rolling strategies and timing
- Dividend and early assignment considerations

**De-emphasize**:
- Advanced Greeks calculations
- Spread width optimization
- Multiple strike comparisons

**Files to include**: calculator, position_sizer, expiration_analysis, rolling-strategies reference

---

### When Creating a Neutral Strategy Skill

**Emphasize**:
- IV rank and percentile calculations
- Expected move analysis
- Vega exposure and management
- Earnings play opportunities
- Time decay (theta) management

**De-emphasize**:
- Directional outlook
- Stock fundamentals
- Dividend considerations

**Files to include**: calculator, volatility_analysis, iv-analysis reference, expected-move reference

---

### When Creating a Complex Multi-Leg Skill

**Emphasize**:
- Precise payoff visualization
- Multi-leg strike optimization
- Wing width configuration
- Net Greeks across all legs
- Adjustment strategies

**De-emphasize**:
- Single-leg analysis
- Simple breakeven calculations

**Files to include**: calculator, leg_optimizer, visualizations, payoff-analysis reference, adjustment-strategies reference

---

## Summary

**Key Principle**: Match your skill's files and features to the strategy's natural characteristics.

- **Vertical spreads** → Mathematical rigor, spread optimization
- **Stock + option** → Portfolio integration, rolling strategies
- **Neutral strategies** → Volatility analysis, IV focus
- **Complex multi-leg** → Visualization, multi-strike optimization

Use the reference implementations (bull-call-spread for vertical, married-put for stock+option) as starting points, then customize based on the strategy category.

---

**This guide helps you create strategy skills that emphasize the right features for each type of options strategy.**
