# Corporate Actions Trading

## Overview

Corporate actions include mergers, acquisitions, spinoffs, buybacks, and special dividends. These events create discrete trading opportunities with defined risk/reward profiles distinct from continuous trading strategies.

---

## Merger Arbitrage

### Deal Types

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class DealType(Enum):
    CASH = "cash"           # Fixed cash per share
    STOCK = "stock"         # Fixed exchange ratio
    MIXED = "mixed"         # Cash + stock
    COLLAR = "collar"       # Exchange ratio with price bounds
    CVR = "cvr"             # Contingent value rights

@dataclass
class MergerDeal:
    # Parties
    target: str
    acquirer: str

    # Terms
    deal_type: DealType
    cash_per_share: Optional[float]
    exchange_ratio: Optional[float]
    collar_floor: Optional[float]
    collar_cap: Optional[float]

    # Timeline
    announcement_date: date
    expected_close: date
    drop_dead_date: date  # Deal expires if not closed

    # Conditions
    regulatory_approval: list  # ['DOJ', 'FTC', 'CFIUS', etc.]
    shareholder_vote: bool
    financing_condition: bool
    mac_clause: bool  # Material adverse change
```

### Spread Calculation

```python
def calculate_deal_spread(
    deal: MergerDeal,
    target_price: float,
    acquirer_price: float = None
) -> dict:
    """
    Calculate merger arbitrage spread for different deal types.
    """
    if deal.deal_type == DealType.CASH:
        implied_value = deal.cash_per_share
        spread = (implied_value - target_price) / target_price

    elif deal.deal_type == DealType.STOCK:
        implied_value = deal.exchange_ratio * acquirer_price
        spread = (implied_value - target_price) / target_price

    elif deal.deal_type == DealType.MIXED:
        implied_value = deal.cash_per_share + (deal.exchange_ratio * acquirer_price)
        spread = (implied_value - target_price) / target_price

    elif deal.deal_type == DealType.COLLAR:
        # Collar bounds the exchange ratio
        if acquirer_price < deal.collar_floor:
            effective_ratio = deal.exchange_ratio * (deal.collar_floor / acquirer_price)
        elif acquirer_price > deal.collar_cap:
            effective_ratio = deal.exchange_ratio * (deal.collar_cap / acquirer_price)
        else:
            effective_ratio = deal.exchange_ratio

        implied_value = effective_ratio * acquirer_price
        spread = (implied_value - target_price) / target_price

    # Annualize
    days_to_close = (deal.expected_close - date.today()).days
    annual_spread = spread * (365 / max(days_to_close, 1))

    return {
        'current_spread': spread,
        'annualized_spread': annual_spread,
        'implied_value': implied_value,
        'days_to_close': days_to_close
    }
```

### Deal Risk Assessment

```python
def assess_deal_risk(deal: MergerDeal, market_conditions: dict) -> dict:
    """
    Comprehensive deal failure risk assessment.
    """
    risk_score = 0.0
    risk_factors = []

    # Regulatory risk
    if 'DOJ' in deal.regulatory_approval or 'FTC' in deal.regulatory_approval:
        risk_score += 0.15
        risk_factors.append('antitrust_review')

    if 'CFIUS' in deal.regulatory_approval:
        risk_score += 0.20
        risk_factors.append('national_security_review')

    # Shareholder risk
    if deal.shareholder_vote:
        risk_score += 0.05
        risk_factors.append('shareholder_approval')

    # Financing risk
    if deal.financing_condition:
        risk_score += 0.25
        if market_conditions.get('credit_spreads_widening'):
            risk_score += 0.10
        risk_factors.append('financing_contingent')

    # Premium assessment (high premium = more scrutiny)
    if deal.cash_per_share:
        # Would need historical price
        pass

    # Timeline risk
    days_remaining = (deal.expected_close - date.today()).days
    if days_remaining > 365:
        risk_score += 0.10
        risk_factors.append('extended_timeline')

    # MAC risk
    if deal.mac_clause:
        risk_score += 0.05
        risk_factors.append('mac_clause')

    return {
        'risk_score': min(risk_score, 1.0),
        'risk_factors': risk_factors,
        'risk_level': 'HIGH' if risk_score > 0.4 else 'MEDIUM' if risk_score > 0.2 else 'LOW'
    }
```

### Merger Arb Position Sizing

```python
def merger_arb_position_size(
    spread_info: dict,
    risk_info: dict,
    base_allocation: float = 0.05
) -> dict:
    """
    Size merger arb position based on spread and risk.
    """
    # Minimum spread requirements
    MIN_SPREAD = 0.02
    MIN_ANNUALIZED = 0.06

    if spread_info['current_spread'] < MIN_SPREAD:
        return {'size': 0, 'reason': 'spread_too_tight'}

    if spread_info['annualized_spread'] < MIN_ANNUALIZED:
        return {'size': 0, 'reason': 'annualized_return_insufficient'}

    # Risk adjustment
    risk_mult = {
        'LOW': 1.0,
        'MEDIUM': 0.6,
        'HIGH': 0.3
    }.get(risk_info['risk_level'], 0.5)

    # Spread reward adjustment
    spread_mult = min(spread_info['annualized_spread'] / 0.10, 1.5)

    final_size = base_allocation * risk_mult * spread_mult

    return {
        'size': final_size,
        'risk_adjusted': True,
        'risk_multiplier': risk_mult,
        'spread_multiplier': spread_mult
    }
```

---

## Spinoff Strategies

### Spinoff Opportunities

```python
@dataclass
class SpinoffEvent:
    parent: str
    spinoff: str
    record_date: date
    distribution_date: date
    distribution_ratio: float  # Shares of spinoff per parent share

    # Characteristics
    spinoff_size_pct: float  # % of parent value
    forced_selling_expected: bool  # Index funds, mandates
    management_retention: bool

def evaluate_spinoff(event: SpinoffEvent, post_spin_days: int = 0) -> dict:
    """
    Evaluate spinoff trading opportunity.
    """
    opportunities = []

    # Pre-spin: Parent often sells off due to uncertainty
    if post_spin_days < 0:
        opportunities.append({
            'trade': 'long_parent_pre_spin',
            'rationale': 'temporary_selloff_pre_distribution',
            'timing': '1-2_weeks_before_record'
        })

    # Immediately post-spin: Forced selling creates opportunity
    if 0 <= post_spin_days <= 30 and event.forced_selling_expected:
        opportunities.append({
            'trade': 'long_spinoff_post_distribution',
            'rationale': 'forced_selling_by_index_funds',
            'timing': 'first_30_days',
            'historical_avg_return': '10-15%'  # Academic finding
        })

    # Post-spin information arbitrage
    if 0 <= post_spin_days <= 90:
        opportunities.append({
            'trade': 'fundamental_analysis_spinoff',
            'rationale': 'analyst_coverage_gap',
            'timing': 'first_90_days'
        })

    return {
        'opportunities': opportunities,
        'forced_selling': event.forced_selling_expected,
        'size_factor': 'small' if event.spinoff_size_pct < 0.20 else 'large'
    }
```

### Spinoff Academic Findings

```
Historical spinoff performance (academic research):

1. Spinoffs outperform market by ~10% in first year
2. Forced selling creates 30-day buying opportunity
3. Small spinoffs outperform large spinoffs
4. Management-retained spinoffs outperform
5. Parent companies also tend to outperform post-spin
```

---

## Share Buybacks

### Buyback Signal Analysis

```python
@dataclass
class BuybackAnnouncement:
    ticker: str
    announcement_date: date
    authorization_amount: float  # Dollar amount authorized
    shares_authorized: int       # Shares authorized
    expiration_date: date
    buyback_type: str           # 'open_market', 'tender', 'asr'

def analyze_buyback(
    announcement: BuybackAnnouncement,
    market_cap: float,
    shares_outstanding: int,
    cash_on_hand: float
) -> dict:
    """
    Analyze buyback announcement quality.
    """
    # Authorization as % of market cap
    auth_pct = announcement.authorization_amount / market_cap

    # Affordability
    cash_coverage = cash_on_hand / announcement.authorization_amount

    # Buyback intensity
    if announcement.buyback_type == 'tender':
        intensity = 'HIGH'  # Immediate, committed
    elif announcement.buyback_type == 'asr':
        intensity = 'HIGH'  # Accelerated share repurchase
    else:
        intensity = 'MODERATE'  # Open market, discretionary

    # Signal strength
    if auth_pct > 0.10 and cash_coverage > 1.0:
        signal = 'STRONG_POSITIVE'
    elif auth_pct > 0.05:
        signal = 'POSITIVE'
    elif auth_pct < 0.02:
        signal = 'WEAK'  # Token announcement
    else:
        signal = 'MODERATE'

    return {
        'signal': signal,
        'authorization_pct': auth_pct,
        'cash_coverage': cash_coverage,
        'buyback_type': announcement.buyback_type,
        'intensity': intensity
    }

# Buyback execution tracking
def track_buyback_execution(
    ticker: str,
    quarterly_repurchases: list,
    total_authorization: float
) -> dict:
    """
    Track actual buyback execution vs authorization.
    """
    total_executed = sum(quarterly_repurchases)
    execution_rate = total_executed / total_authorization

    # Companies that execute aggressively signal more conviction
    if execution_rate > 0.80:
        credibility = 'HIGH'
    elif execution_rate > 0.50:
        credibility = 'MODERATE'
    else:
        credibility = 'LOW'  # All talk, no action

    return {
        'execution_rate': execution_rate,
        'credibility': credibility,
        'remaining_authorization': total_authorization - total_executed
    }
```

---

## Special Dividends

### Special Dividend Trading

```python
def evaluate_special_dividend(
    ticker: str,
    dividend_amount: float,
    current_price: float,
    ex_date: date,
    payment_date: date
) -> dict:
    """
    Evaluate special dividend trading opportunity.
    """
    yield_pct = dividend_amount / current_price

    # Days between ex-date and payment
    settlement_days = (payment_date - ex_date).days

    # Large special dividends often create opportunities
    if yield_pct > 0.05:  # >5% yield
        # Stock may not drop full dividend amount on ex-date
        opportunity = 'potential_incomplete_adjustment'
    else:
        opportunity = 'limited'

    # Tax considerations affect drop
    # Qualified dividends taxed less, so drop may be less than dividend

    return {
        'yield_pct': yield_pct,
        'opportunity': opportunity,
        'ex_date': ex_date,
        'expected_drop': dividend_amount * 0.85,  # Typical 85% drop
        'note': 'Tax effects may reduce actual drop'
    }
```

---

## Tender Offers

### Tender Offer Arbitrage

```python
@dataclass
class TenderOffer:
    ticker: str
    offer_price: float
    offer_type: str  # 'any_and_all', 'partial', 'dutch_auction'
    shares_sought: int
    minimum_condition: int
    expiration_date: date
    financing_condition: bool

def evaluate_tender_offer(
    offer: TenderOffer,
    current_price: float,
    shares_outstanding: int
) -> dict:
    """
    Evaluate tender offer arbitrage opportunity.
    """
    spread = (offer.offer_price - current_price) / current_price

    # Proration risk for partial tenders
    if offer.offer_type == 'partial':
        proration = offer.shares_sought / shares_outstanding
        expected_acceptance = spread * proration
    else:
        proration = 1.0
        expected_acceptance = spread

    # Days to expiration
    days = (offer.expiration_date - date.today()).days

    # Risk assessment
    if offer.financing_condition:
        risk = 'ELEVATED'
    elif offer.minimum_condition > 0:
        risk = 'MODERATE'
    else:
        risk = 'LOW'

    return {
        'spread': spread,
        'expected_return': expected_acceptance,
        'proration_factor': proration,
        'days_to_expiration': days,
        'annualized': expected_acceptance * (365 / max(days, 1)),
        'risk_level': risk
    }
```

---

## Risk Management

### Corporate Action Stops

```python
def corporate_action_risk_management(
    position: dict,
    action_type: str,
    expected_close_date: date
) -> dict:
    """
    Risk management for corporate action positions.
    """
    # Deal break stop
    if action_type == 'merger_arb':
        stop_loss = 0.50  # 50% of spread - deal break indicator
        position_max = 0.05  # 5% of portfolio max

    elif action_type == 'spinoff':
        stop_loss = 0.15  # 15% from entry
        position_max = 0.03

    elif action_type == 'tender':
        stop_loss = 0.30  # Below offer = deal risk
        position_max = 0.04

    # Time-based exit
    days_to_deadline = (expected_close_date - date.today()).days
    if days_to_deadline < 0:
        action = 'REVIEW_IMMEDIATELY'  # Past expected close

    return {
        'stop_loss_pct': stop_loss,
        'max_position': position_max,
        'days_remaining': days_to_deadline,
        'note': 'Review if deal timeline extends'
    }
```

---

## Academic References

- Mitchell & Pulvino (2001): "Characteristics of Risk and Return in Risk Arbitrage"
- Cusatis, Miles & Woolridge (1993): "Restructuring through Spinoffs"
- Lakonishok & Vermaelen (1990): "Anomalous Price Behavior Around Repurchase Tender Offers"
- Ikenberry, Lakonishok & Vermaelen (1995): "Market Underreaction to Open Market Share Repurchases"
