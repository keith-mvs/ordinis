# Market Fundamentals & Microstructure - Knowledge Base

## Purpose

Understanding market microstructure is essential for building an automated trading system that can execute effectively and avoid common pitfalls. This section covers how markets actually work at a mechanical level.

---

## 1. Market Structure

### 1.1 Exchange Types

| Exchange Type | Description | Examples |
|---------------|-------------|----------|
| **Primary Exchange** | Listed securities trade here | NYSE, NASDAQ |
| **ECN** | Electronic Communication Networks | ARCA, BATS |
| **Dark Pools** | Anonymous trading venues | Crossfinder, SIGMA X |
| **OTC** | Over-the-counter, less regulated | OTC Markets |

**Rule Templates**:
```python
# Exchange preferences for routing
PREFERRED_EXCHANGES = ['NYSE', 'NASDAQ', 'ARCA', 'BATS']
AVOID_EXCHANGES = ['OTC']  # Less liquidity, wider spreads

# Exchange-specific considerations
def select_exchange(symbol: str, order_type: str) -> str:
    if symbol.primary_exchange == 'NYSE':
        if order_type == 'MARKET':
            return 'NYSE'  # Use primary for market orders
        else:
            return 'SMART'  # Smart routing for limits
    return 'SMART'  # Default to smart order routing
```

---

### 1.2 Market Participants

| Participant | Role | Impact |
|-------------|------|--------|
| **Market Makers** | Provide liquidity, quote bid/ask | Tighten spreads |
| **Institutional** | Large orders, long horizons | Move markets |
| **Retail** | Small orders, various horizons | Minimal impact |
| **HFT** | High-frequency, arbitrage | Sub-second liquidity |
| **Algorithms** | Automated execution | Various strategies |

**Rule Templates**:
```python
# Detecting institutional activity
INSTITUTIONAL_VOLUME = volume > avg_volume * 3 AND large_block_prints
RETAIL_DOMINATED = small_order_flow AND low_block_activity

# Implications
IF INSTITUTIONAL_VOLUME:
    expect = "momentum_continuation"  # Follow the flow
IF RETAIL_DOMINATED:
    expect = "mean_reversion_possible"  # Retail often wrong at extremes
```

---

## 2. Order Types

### 2.1 Basic Order Types

| Order Type | Description | Use Case |
|------------|-------------|----------|
| **Market** | Execute immediately at best available | Urgent entry/exit |
| **Limit** | Execute at specified price or better | Controlled entry |
| **Stop** | Becomes market when trigger hit | Stop loss |
| **Stop-Limit** | Becomes limit when trigger hit | Controlled stop |

**Rule Templates**:
```python
# Order type selection
def select_order_type(urgency: str, liquidity: str, direction: str) -> str:
    if urgency == 'high':
        return 'MARKET'
    elif liquidity == 'low':
        return 'LIMIT'  # Avoid slippage in illiquid names
    elif direction == 'entry':
        return 'LIMIT'  # Control entry price
    elif direction == 'stop_loss':
        return 'STOP'  # Ensure execution
    return 'LIMIT'

# Limit order pricing
def calculate_limit_price(side: str, current_price: float, urgency: str) -> float:
    if urgency == 'high':
        buffer = 0.001  # 0.1% buffer
    else:
        buffer = 0.0  # At current price

    if side == 'BUY':
        return current_price * (1 + buffer)
    else:
        return current_price * (1 - buffer)
```

---

### 2.2 Advanced Order Types

| Order Type | Description | Use Case |
|------------|-------------|----------|
| **Trailing Stop** | Stop that follows price | Lock in profits |
| **MOO/MOC** | Market on Open/Close | Execution at auction |
| **LOO/LOC** | Limit on Open/Close | Controlled auction |
| **IOC** | Immediate or Cancel | Avoid partial fills |
| **FOK** | Fill or Kill | All or nothing |
| **GTC** | Good Till Cancelled | Multi-day orders |

**Rule Templates**:
```python
# Trailing stop implementation
def update_trailing_stop(position: Position, current_price: float) -> float:
    trail_pct = 0.05  # 5% trailing

    if position.side == 'LONG':
        new_stop = current_price * (1 - trail_pct)
        return max(position.stop_price, new_stop)  # Only trail up
    else:
        new_stop = current_price * (1 + trail_pct)
        return min(position.stop_price, new_stop)  # Only trail down

# Time-in-force selection
def select_time_in_force(strategy: str) -> str:
    if strategy == 'intraday':
        return 'DAY'
    elif strategy == 'swing':
        return 'GTC'
    elif strategy == 'auction_only':
        return 'OPG'  # Opening only
    return 'DAY'
```

---

## 3. Price Formation & Order Book

### 3.1 Bid-Ask Spread

**Definition**: Difference between best bid and best ask.

```python
# Spread calculations
SPREAD = ask - bid
MIDPOINT = (bid + ask) / 2
SPREAD_PCT = SPREAD / MIDPOINT * 100

# Spread thresholds
TIGHT_SPREAD = SPREAD_PCT < 0.05  # < 5 bps
NORMAL_SPREAD = 0.05 <= SPREAD_PCT < 0.15
WIDE_SPREAD = SPREAD_PCT >= 0.15

# Trading implications
IF WIDE_SPREAD:
    action = "use_limit_orders_only"
    size_reduction = 0.5  # Reduce size
IF TIGHT_SPREAD:
    action = "market_orders_acceptable"
```

---

### 3.2 Order Book Dynamics

**Key Concepts**:
- **Depth**: Quantity available at each price level
- **Imbalance**: Ratio of bid to ask volume
- **Queue Position**: Priority in price-time queue

```python
# Order book imbalance
def calculate_imbalance(bid_volume: float, ask_volume: float) -> float:
    """
    Positive = more bids (buying pressure)
    Negative = more asks (selling pressure)
    """
    total = bid_volume + ask_volume
    if total == 0:
        return 0
    return (bid_volume - ask_volume) / total

# Imbalance signals
STRONG_BID_IMBALANCE = imbalance > 0.3  # 65%+ bid side
STRONG_ASK_IMBALANCE = imbalance < -0.3  # 65%+ ask side

# Order book depth check
def has_sufficient_depth(order_size: int, book_depth: List[Level]) -> bool:
    """Check if order can fill without walking the book too far."""
    cumulative_size = 0
    price_impact = 0

    for level in book_depth:
        cumulative_size += level.size
        if cumulative_size >= order_size:
            price_impact = (level.price - book_depth[0].price) / book_depth[0].price
            break

    return price_impact < 0.002  # Less than 20 bps impact
```

---

### 3.3 Price Discovery

**Mechanisms**:
- Continuous trading during market hours
- Auction at open and close
- Price improvement on limit orders

```python
# Auction participation
PARTICIPATE_IN_AUCTION = (
    order_size_significant AND
    (want_open_price OR want_close_price)
)

# Fair value estimation
def estimate_fair_value(bid: float, ask: float, last: float) -> float:
    """
    Weighted estimate of fair value.
    """
    midpoint = (bid + ask) / 2

    # If last is near midpoint, use midpoint
    if abs(last - midpoint) / midpoint < 0.001:
        return midpoint

    # Otherwise, weight toward last trade
    return 0.7 * midpoint + 0.3 * last
```

---

## 4. Trade Execution

### 4.1 Fill Quality Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Slippage** | Difference from expected price | < 10 bps |
| **Fill Rate** | % of order filled | > 95% |
| **Execution Speed** | Time to fill | < 1 second |
| **Price Improvement** | Better than quoted | Any positive |

```python
# Slippage calculation
def calculate_slippage(expected_price: float, fill_price: float, side: str) -> float:
    """
    Returns slippage in basis points.
    Positive = worse than expected, Negative = better.
    """
    if side == 'BUY':
        return (fill_price - expected_price) / expected_price * 10000
    else:
        return (expected_price - fill_price) / expected_price * 10000

# Execution quality logging
@dataclass
class ExecutionQuality:
    order_id: str
    expected_price: float
    fill_price: float
    slippage_bps: float
    fill_time_ms: int
    fill_rate_pct: float
    venue: str
```

---

### 4.2 Partial Fills

```python
# Handling partial fills
def handle_partial_fill(order: Order, fill: Fill) -> Action:
    remaining = order.quantity - fill.quantity
    fill_pct = fill.quantity / order.quantity

    if fill_pct >= 0.90:
        # Mostly filled, cancel remainder
        return Action('CANCEL_REMAINING')
    elif fill_pct >= 0.50:
        # Half filled, adjust remaining to market
        return Action('MODIFY_TO_MARKET')
    else:
        # Poorly filled, wait or cancel
        if order.urgency == 'low':
            return Action('WAIT')
        else:
            return Action('MODIFY_TO_MARKET')

# Partial fill position management
IF PARTIAL_FILL:
    adjust_stop_to_partial_position()
    recalculate_risk()
```

---

## 5. Market Sessions

### 5.1 US Equity Sessions

| Session | Time (ET) | Characteristics |
|---------|-----------|-----------------|
| **Pre-Market** | 4:00 - 9:30 | Low liquidity, wide spreads |
| **Open Auction** | 9:30 | Price discovery |
| **Regular Hours** | 9:30 - 16:00 | Full liquidity |
| **Close Auction** | 16:00 | Significant volume |
| **After-Hours** | 16:00 - 20:00 | Low liquidity |

**Rule Templates**:
```python
# Session-based trading rules
def session_rules(current_time: datetime) -> dict:
    hour = current_time.hour
    minute = current_time.minute

    # Pre-market
    if 4 <= hour < 9 or (hour == 9 and minute < 30):
        return {
            'trade': False,  # Avoid pre-market
            'reason': 'low_liquidity'
        }

    # First 15 minutes (opening volatility)
    if hour == 9 and 30 <= minute < 45:
        return {
            'trade': 'limited',
            'size_reduction': 0.5,
            'reason': 'opening_volatility'
        }

    # Last 15 minutes (closing volatility)
    if hour == 15 and minute >= 45:
        return {
            'trade': 'limited',
            'new_positions': False,
            'reason': 'closing_auction'
        }

    # Regular hours
    if 9 <= hour < 16:
        return {
            'trade': True,
            'size_reduction': 1.0
        }

    # After hours
    return {
        'trade': False,
        'reason': 'after_hours'
    }
```

---

### 5.2 Intraday Volume Patterns

```python
# Expected volume distribution (US equities)
VOLUME_DISTRIBUTION = {
    '09:30-10:00': 0.15,  # 15% of daily volume
    '10:00-11:00': 0.12,
    '11:00-12:00': 0.08,
    '12:00-13:00': 0.07,
    '13:00-14:00': 0.08,
    '14:00-15:00': 0.12,
    '15:00-15:30': 0.13,
    '15:30-16:00': 0.25   # 25% in final 30 min
}

# Time-adjusted RVOL
def time_adjusted_rvol(current_volume: float, time: datetime, avg_daily: float) -> float:
    """
    Calculate relative volume adjusted for time of day.
    """
    expected_pct = get_expected_volume_pct(time)
    expected_volume = avg_daily * expected_pct

    if expected_volume == 0:
        return 0

    return current_volume / expected_volume
```

---

## 6. Settlement & Clearing

### 6.1 Settlement Cycles

| Instrument | Settlement | Notes |
|------------|------------|-------|
| **US Equities** | T+1 | Trade date + 1 business day |
| **US Options** | T+1 | Same as equities |
| **Government Bonds** | T+1 | |
| **Mutual Funds** | T+1 or T+2 | Fund dependent |
| **Crypto** | Variable | Often immediate |

```python
# Settlement date calculation
def calculate_settlement_date(trade_date: date, instrument: str) -> date:
    settlement_days = {
        'equity': 1,
        'option': 1,
        'bond': 1,
        'mutual_fund': 2
    }

    days = settlement_days.get(instrument, 1)
    settlement = trade_date

    while days > 0:
        settlement += timedelta(days=1)
        if is_business_day(settlement):
            days -= 1

    return settlement

# Settlement implications
IF account_type == 'CASH':
    available_funds = settled_cash  # Can only use settled funds
    warn_if_day_trading()  # Violation risk
IF account_type == 'MARGIN':
    available_funds = buying_power  # Can use unsettled
```

---

### 6.2 Margin Requirements

```python
# Regulation T margin
REG_T_INITIAL = 0.50  # 50% initial margin
REG_T_MAINTENANCE = 0.25  # 25% maintenance margin

def calculate_margin_requirement(
    position_value: float,
    position_type: str
) -> float:
    """Calculate margin required for position."""

    if position_type == 'LONG_STOCK':
        return position_value * REG_T_INITIAL
    elif position_type == 'SHORT_STOCK':
        return position_value * 1.50  # 150% for short
    elif position_type == 'LONG_OPTION':
        return position_value  # 100% (premium paid)
    elif position_type == 'SHORT_OPTION':
        return calculate_option_margin(position_value)

    return position_value

# Margin call handling
def check_margin_call(equity: float, margin_requirement: float) -> bool:
    margin_pct = equity / margin_requirement

    if margin_pct < REG_T_MAINTENANCE:
        return True  # Margin call triggered
    return False
```

---

## 7. Corporate Actions

### 7.1 Types & Adjustments

| Action | Price Adjustment | Position Effect |
|--------|------------------|-----------------|
| **Cash Dividend** | Price drops by dividend | No change |
| **Stock Split** | Price / split ratio | Shares × split ratio |
| **Reverse Split** | Price × ratio | Shares / ratio |
| **Merger** | Per deal terms | Converted |
| **Spinoff** | Cost basis allocated | New position |

**Rule Templates**:
```python
# Dividend handling
def handle_ex_dividend(position: Position, dividend: float) -> None:
    """
    On ex-date, stock typically drops by dividend amount.
    Adjust expectations and stops accordingly.
    """
    expected_drop = dividend

    # Adjust stop loss for expected drop
    if position.stop_price:
        position.stop_price -= expected_drop

    # Log expected dividend income
    position.expected_dividend = dividend * position.shares

# Split adjustment
def adjust_for_split(position: Position, split_ratio: float) -> Position:
    """
    Adjust position for stock split.
    Example: 4:1 split, ratio = 4.0
    """
    position.shares = int(position.shares * split_ratio)
    position.avg_cost = position.avg_cost / split_ratio
    position.stop_price = position.stop_price / split_ratio if position.stop_price else None
    position.target_price = position.target_price / split_ratio if position.target_price else None

    return position
```

---

### 7.2 Corporate Action Calendar

```python
# Event monitoring
CORPORATE_EVENTS = [
    'earnings_date',
    'ex_dividend_date',
    'split_date',
    'merger_vote_date',
    'tender_offer_deadline'
]

def check_corporate_events(ticker: str, days_ahead: int = 5) -> List[Event]:
    """
    Check for upcoming corporate events that may affect positions.
    """
    events = []

    for event_type in CORPORATE_EVENTS:
        event = get_event(ticker, event_type)
        if event and event.date <= today() + timedelta(days=days_ahead):
            events.append(event)

    return events

# Position rules around events
IF upcoming_event('ex_dividend') AND position.type == 'SHORT':
    action = 'close_before_ex_date'  # Avoid paying dividend
IF upcoming_event('earnings') AND days_until < 3:
    action = 'no_new_positions'  # Avoid binary event
```

---

## 8. Circuit Breakers & Halts

### 8.1 Market-Wide Circuit Breakers (S&P 500)

| Level | Decline | Halt Duration |
|-------|---------|---------------|
| Level 1 | 7% | 15 minutes |
| Level 2 | 13% | 15 minutes |
| Level 3 | 20% | Remainder of day |

```python
# Circuit breaker detection
def check_market_circuit_breaker(sp500_change: float) -> str:
    if sp500_change <= -0.20:
        return 'LEVEL_3_HALT'
    elif sp500_change <= -0.13:
        return 'LEVEL_2_HALT'
    elif sp500_change <= -0.07:
        return 'LEVEL_1_HALT'
    return 'NORMAL'

# Response
IF circuit_breaker_triggered:
    action = 'pause_all_trading'
    wait_for = 'halt_end_notification'
```

---

### 8.2 Single-Stock Halts

| Halt Type | Cause | Duration |
|-----------|-------|----------|
| **LULD** | Limit Up/Limit Down bands | 5-10 minutes |
| **News** | Material news pending | Until resolved |
| **Regulatory** | SEC/exchange action | Variable |
| **Volatility** | Extreme price move | Minutes |

```python
# LULD band monitoring
def check_luld_bands(price: float, upper_band: float, lower_band: float) -> str:
    if price >= upper_band:
        return 'LIMIT_UP'
    elif price <= lower_band:
        return 'LIMIT_DOWN'
    return 'NORMAL'

# Halt handling
def handle_trading_halt(ticker: str, halt_type: str) -> Action:
    if halt_type == 'NEWS':
        return Action('WAIT_FOR_NEWS', 'monitor_feed')
    elif halt_type == 'LULD':
        return Action('CANCEL_OPEN_ORDERS', 'wait_for_reopen')
    elif halt_type == 'REGULATORY':
        return Action('CLOSE_POSITION_ON_REOPEN', 'high_risk')

    return Action('MONITOR')
```

---

## 9. Instrument Classes

### 9.1 Equities (Stocks)

```python
EQUITY_CHARACTERISTICS = {
    'settlement': 'T+1',
    'trading_hours': '09:30-16:00 ET',
    'minimum_tick': 0.01,  # 1 cent
    'margin_eligible': True,
    'short_sellable': 'if_borrowable',
    'dividend_eligible': True
}

# Stock classification
def classify_stock(market_cap: float, sector: str) -> dict:
    if market_cap > 200e9:
        size = 'mega_cap'
    elif market_cap > 10e9:
        size = 'large_cap'
    elif market_cap > 2e9:
        size = 'mid_cap'
    elif market_cap > 300e6:
        size = 'small_cap'
    else:
        size = 'micro_cap'

    return {
        'size': size,
        'sector': sector,
        'volatility_expectation': 'higher' if size in ['small_cap', 'micro_cap'] else 'normal'
    }
```

---

### 9.2 ETFs

```python
ETF_CHARACTERISTICS = {
    'settlement': 'T+1',
    'trading_hours': 'same_as_stocks',
    'intraday_nav': 'available',
    'premium_discount': 'possible',
    'creation_redemption': 'authorized_participants'
}

# ETF premium/discount
def calculate_etf_premium(etf_price: float, nav: float) -> float:
    """
    Calculate premium (positive) or discount (negative) to NAV.
    """
    return (etf_price - nav) / nav * 100

# ETF trading rules
IF abs(etf_premium) > 1.0:  # >1% premium/discount
    caution = True
    reason = 'trading_away_from_fair_value'
```

---

### 9.3 Options Specifics

```python
OPTIONS_CHARACTERISTICS = {
    'settlement': 'T+1',
    'exercise_style': 'american' or 'european',
    'multiplier': 100,  # Shares per contract
    'expiration': 'third_friday' or 'daily/weekly',
    'assignment_risk': 'for_sellers'
}

# See 06_options_derivatives for full options knowledge
```

---

## 10. Regulatory Framework

### 10.1 Key Regulators

| Regulator | Jurisdiction | Focus |
|-----------|--------------|-------|
| **SEC** | Securities | Investor protection, market integrity |
| **FINRA** | Broker-dealers | Trading rules, compliance |
| **CFTC** | Futures/derivatives | Commodity trading |
| **OCC** | Options clearing | Clearinghouse |

---

### 10.2 Key Rules for Automated Trading

```python
# Pattern Day Trader rule
PDT_RULES = {
    'threshold': 4,  # 4+ day trades in 5 days
    'account_minimum': 25000,
    'applies_to': 'margin_accounts',
    'consequence': 'account_restricted'
}

def check_pdt_status(account: Account, day_trades_5d: int) -> bool:
    if account.type != 'MARGIN':
        return False  # PDT only applies to margin

    if account.equity < PDT_RULES['account_minimum']:
        if day_trades_5d >= PDT_RULES['threshold']:
            return True  # PDT flagged

    return False

# Wash sale rule (tax)
WASH_SALE = {
    'window': 30,  # days before or after
    'consequence': 'loss_disallowed',
    'adjustment': 'add_to_cost_basis'
}

def check_wash_sale(sell: Trade, buys: List[Trade]) -> bool:
    for buy in buys:
        days_apart = abs((sell.date - buy.date).days)
        if days_apart <= WASH_SALE['window']:
            if sell.ticker == buy.ticker:
                return True
    return False
```

---

## Academic References

1. **O'Hara, M. (1995)**: "Market Microstructure Theory" - Foundational text
2. **Harris, L. (2003)**: "Trading and Exchanges" - Comprehensive market mechanics
3. **Madhavan, A. (2000)**: "Market Microstructure: A Survey" - Academic overview
4. **Hasbrouck, J. (2007)**: "Empirical Market Microstructure" - Quantitative approach
5. **SEC Market Structure Resources**: Official regulatory guidance

---

## Key Takeaways

1. **Understand order types**: Use the right order for the situation
2. **Respect liquidity**: Wide spreads = higher costs
3. **Mind the clock**: Trading sessions matter
4. **Watch for halts**: Have procedures for interruptions
5. **Know settlement**: Cash vs margin implications
6. **Corporate actions**: Adjust positions accordingly
7. **Regulatory compliance**: PDT, wash sales, and more
