# Macro Events Trading

## Overview

Macro events—Federal Reserve decisions, economic data releases, and geopolitical events—move entire markets. Trading around these events requires understanding both the event mechanics and market positioning.

---

## Economic Calendar

### Key Scheduled Events

```python
MACRO_CALENDAR = {
    'FOMC_DECISION': {
        'frequency': '8x_annually',
        'schedule': 'predetermined_dates',
        'time': '14:00_ET',
        'impact': 'very_high',
        'affects': ['rates', 'equities', 'fx', 'commodities'],
        'volatility_window': '-24h_to_+48h'
    },
    'FOMC_MINUTES': {
        'frequency': '8x_annually',
        'schedule': '3_weeks_after_decision',
        'time': '14:00_ET',
        'impact': 'medium',
        'affects': ['rates', 'equities'],
        'volatility_window': '-1h_to_+4h'
    },
    'NFP_REPORT': {
        'frequency': 'monthly',
        'schedule': 'first_friday',
        'time': '08:30_ET',
        'impact': 'high',
        'affects': ['rates', 'equities', 'fx'],
        'volatility_window': '-30m_to_+4h'
    },
    'CPI_RELEASE': {
        'frequency': 'monthly',
        'schedule': 'mid_month',
        'time': '08:30_ET',
        'impact': 'high',
        'affects': ['rates', 'equities', 'fx', 'tips'],
        'volatility_window': '-30m_to_+4h'
    },
    'GDP_ADVANCE': {
        'frequency': 'quarterly',
        'schedule': 'month_after_quarter_end',
        'time': '08:30_ET',
        'impact': 'medium',
        'affects': ['equities', 'fx'],
        'volatility_window': '-30m_to_+2h'
    },
    'ISM_MANUFACTURING': {
        'frequency': 'monthly',
        'schedule': 'first_business_day',
        'time': '10:00_ET',
        'impact': 'medium',
        'affects': ['equities', 'industrials'],
        'volatility_window': '-15m_to_+1h'
    },
    'RETAIL_SALES': {
        'frequency': 'monthly',
        'schedule': 'mid_month',
        'time': '08:30_ET',
        'impact': 'medium',
        'affects': ['consumer_discretionary', 'retail'],
        'volatility_window': '-15m_to_+1h'
    }
}
```

---

## FOMC Decision Trading

### Pre-FOMC Framework

```python
class FOMCStrategy:
    """
    Trading strategy around Federal Reserve decisions.
    """

    # Pre-FOMC announcement drift (academic finding)
    PRE_FOMC_DRIFT = {
        'start': '-24_hours',
        'historical_bias': 'bullish',
        'avg_excess_return': '0.25%',  # Per Lucca & Moench (2015)
        'note': 'Drift has weakened since publication'
    }

    def pre_fomc_positioning(
        self,
        hours_until: float,
        fed_funds_futures: float,  # Market-implied probability
        dot_plot_median: float,
        economic_data_trend: str
    ) -> dict:
        """
        Position before FOMC decision.
        """
        # No positions within 1 hour
        if hours_until < 1:
            return {
                'action': 'FLAT',
                'reason': 'binary_event_imminent'
            }

        # Calculate market pricing
        hike_priced = fed_funds_futures > 0.50
        cut_priced = fed_funds_futures < -0.25

        # If market is strongly positioned one way
        if fed_funds_futures > 0.90:  # >90% priced
            return {
                'action': 'CAUTIOUS',
                'reason': 'crowded_positioning',
                'contrarian_risk': 'HIGH'
            }

        # Pre-FOMC drift opportunity
        if 4 < hours_until < 24:
            return {
                'action': 'CONSIDER_LONG_SPY',
                'size': 'SMALL',
                'reason': 'pre_fomc_drift',
                'exit': 'before_announcement'
            }

        return {'action': 'WAIT'}

    def post_fomc_signal(
        self,
        decision: str,         # 'hike', 'cut', 'hold'
        expected: str,         # Market expectation
        statement_tone: str,   # 'hawkish', 'dovish', 'neutral'
        dot_plot_shift: str,   # 'higher', 'lower', 'unchanged'
        initial_reaction: dict # {'spx': 0.01, 'tlt': -0.02}
    ) -> dict:
        """
        Generate signal after FOMC decision.
        """
        surprise = decision != expected

        # Pure decision surprise
        if surprise:
            if decision == 'hike' and expected != 'hike':
                return {
                    'signal': 'BEARISH',
                    'equities': 'SELL',
                    'duration': 'SELL',
                    'dollar': 'BUY',
                    'confidence': 'HIGH',
                    'reason': 'hawkish_surprise'
                }
            elif decision == 'cut' and expected != 'cut':
                return {
                    'signal': 'BULLISH',
                    'equities': 'BUY',
                    'duration': 'BUY',
                    'dollar': 'SELL',
                    'confidence': 'HIGH',
                    'reason': 'dovish_surprise'
                }

        # Statement tone matters when decision is expected
        if not surprise:
            if statement_tone == 'hawkish' and initial_reaction['spx'] > 0:
                return {
                    'signal': 'FADE_RALLY',
                    'reason': 'hawkish_tone_ignored',
                    'confidence': 'MEDIUM'
                }
            elif statement_tone == 'dovish' and initial_reaction['spx'] < 0:
                return {
                    'signal': 'FADE_SELLOFF',
                    'reason': 'dovish_tone_ignored',
                    'confidence': 'MEDIUM'
                }

        # Dot plot revision matters for forward guidance
        if dot_plot_shift == 'higher':
            return {
                'signal': 'DURATION_NEGATIVE',
                'rates': 'HIGHER_FOR_LONGER',
                'confidence': 'MEDIUM'
            }

        return {'signal': 'NEUTRAL', 'reason': 'as_expected'}
```

---

## Employment Data Trading

### Non-Farm Payrolls (NFP)

```python
def trade_nfp_release(
    actual: int,           # Actual payrolls change
    consensus: int,        # Consensus estimate
    prior: int,            # Prior month (may be revised)
    prior_revision: int,   # Revision to prior
    unemployment_rate: float,
    expected_unemployment: float,
    avg_hourly_earnings: float,
    expected_earnings: float
) -> dict:
    """
    Generate trading signal from NFP release.
    """
    # Headline surprise
    headline_surprise = actual - consensus

    # Unemployment surprise (lower = stronger)
    unemployment_surprise = expected_unemployment - unemployment_rate

    # Wages surprise (higher = inflationary)
    wage_surprise = avg_hourly_earnings - expected_earnings

    # Revision adjustment (often overlooked)
    revision_factor = prior_revision / 50000  # Normalize

    # Composite score
    # Strong jobs + low unemployment + high wages = hawkish for Fed
    strength_score = (
        (headline_surprise / 100000) +       # Per 100k jobs
        (unemployment_surprise * 5) +        # Per 0.1% unemployment
        (wage_surprise * 10)                 # Per 0.1% wage growth
    )

    # Trading signals
    if strength_score > 0.5:
        # Strong report = Fed stays hawkish
        return {
            'signal': 'HAWKISH',
            'rates': 'HIGHER',
            'equities': 'MIXED_TO_NEGATIVE',
            'dollar': 'STRONGER',
            'confidence': abs(strength_score),
            'rationale': 'strong_labor_market_keeps_fed_hawkish'
        }
    elif strength_score < -0.5:
        # Weak report = Fed may ease
        return {
            'signal': 'DOVISH',
            'rates': 'LOWER',
            'equities': 'POSITIVE',
            'dollar': 'WEAKER',
            'confidence': abs(strength_score),
            'rationale': 'weak_labor_market_opens_door_to_easing'
        }

    return {
        'signal': 'NEUTRAL',
        'confidence': 0.3,
        'rationale': 'mixed_signals'
    }
```

---

## Inflation Data Trading

### CPI Release Strategy

```python
def trade_cpi_release(
    headline_actual: float,
    headline_expected: float,
    core_actual: float,
    core_expected: float,
    yoy_headline: float,
    yoy_core: float,
    fed_target: float = 2.0
) -> dict:
    """
    Generate trading signal from CPI release.
    """
    # Monthly surprises
    headline_surprise = headline_actual - headline_expected
    core_surprise = core_actual - core_expected

    # Year-over-year trend context
    inflation_trend = 'above_target' if yoy_core > fed_target else 'below_target'

    # Core matters more than headline
    if core_surprise > 0.1:  # +0.1% surprise on monthly
        signal = 'HOT_INFLATION'
        return {
            'signal': signal,
            'rates': 'SIGNIFICANTLY_HIGHER',
            'equities': 'SELL',
            'duration': 'SELL',
            'tips': 'BUY',
            'confidence': 'HIGH',
            'magnitude': 'LARGE',
            'rationale': 'inflation_surprising_higher_pressures_fed'
        }
    elif core_surprise < -0.1:
        signal = 'COOL_INFLATION'
        return {
            'signal': signal,
            'rates': 'LOWER',
            'equities': 'BUY',
            'duration': 'BUY',
            'confidence': 'HIGH',
            'rationale': 'inflation_cooling_reduces_fed_pressure'
        }
    elif abs(core_surprise) < 0.05:
        return {
            'signal': 'IN_LINE',
            'confidence': 'LOW',
            'rationale': 'as_expected_minimal_impact'
        }

    # Moderate surprise
    return {
        'signal': 'MODERATE_' + ('HOT' if core_surprise > 0 else 'COOL'),
        'rates': 'SLIGHTLY_' + ('HIGHER' if core_surprise > 0 else 'LOWER'),
        'confidence': 'MEDIUM'
    }
```

---

## GDP and Growth Data

### GDP Release Trading

```python
def trade_gdp_release(
    actual_growth: float,   # Annualized quarterly growth
    consensus: float,
    prior_quarter: float,
    gdp_components: dict    # Consumer, investment, govt, net exports
) -> dict:
    """
    Generate signal from GDP release.
    """
    surprise = actual_growth - consensus

    # Check composition
    consumer_driven = gdp_components.get('consumer', 0) > actual_growth * 0.6
    investment_driven = gdp_components.get('investment', 0) > actual_growth * 0.3

    # Quality of growth matters
    if actual_growth > 3.0 and consumer_driven:
        quality = 'SUSTAINABLE'
    elif actual_growth > 3.0 and not consumer_driven:
        quality = 'INVENTORY_DRIVEN'  # Less sustainable
    else:
        quality = 'MODERATE'

    # Signal generation
    if surprise > 0.5:  # Beat by 0.5%+
        return {
            'signal': 'STRONG_GROWTH',
            'equities': 'POSITIVE',
            'quality': quality,
            'fed_implication': 'no_urgency_to_ease'
        }
    elif surprise < -0.5:
        return {
            'signal': 'WEAK_GROWTH',
            'equities': 'NEGATIVE_BUT_FED_MAY_EASE',
            'quality': quality,
            'fed_implication': 'increases_easing_probability'
        }

    return {'signal': 'IN_LINE'}
```

---

## Risk Management

### Macro Event Position Sizing

```python
def macro_event_position_adjustment(
    event: str,
    hours_until: float,
    current_positions: dict,
    portfolio_beta: float
) -> dict:
    """
    Adjust positions before macro events.
    """
    event_impact = MACRO_CALENDAR.get(event, {}).get('impact', 'medium')

    if event_impact == 'very_high':  # FOMC
        if hours_until < 1:
            return {
                'action': 'REDUCE_ALL',
                'target_beta': 0.3,
                'reason': 'binary_event'
            }
        elif hours_until < 24:
            return {
                'action': 'REDUCE',
                'target_beta': 0.6,
                'reason': 'event_uncertainty'
            }

    elif event_impact == 'high':  # NFP, CPI
        if hours_until < 0.5:  # 30 minutes
            return {
                'action': 'REDUCE',
                'target_beta': 0.5,
                'reason': 'volatility_spike_expected'
            }

    return {'action': 'MAINTAIN'}

def calculate_event_var(
    position_value: float,
    event_type: str,
    historical_moves: list
) -> dict:
    """
    Calculate event-specific VaR.
    """
    # Historical event-day moves
    avg_move = np.mean(np.abs(historical_moves))
    max_move = np.max(np.abs(historical_moves))
    percentile_95 = np.percentile(np.abs(historical_moves), 95)

    return {
        'expected_move': avg_move,
        'var_95': position_value * percentile_95,
        'var_99': position_value * max_move * 0.8,
        'max_historical': max_move
    }
```

---

## Event Impact Table

| Event | SPX Avg Move | TLT Avg Move | Duration | Best Trade |
|-------|--------------|--------------|----------|------------|
| FOMC Decision | 1.0% | 0.8% | 2-3 days | Fade extremes |
| NFP | 0.5% | 0.4% | 4 hours | Trend continuation |
| CPI | 0.8% | 0.6% | 4 hours | Trend continuation |
| GDP | 0.3% | 0.2% | 2 hours | Fade if extreme |
| ISM | 0.3% | 0.2% | 1 hour | Sector rotation |

---

## Academic References

- Lucca & Moench (2015): "The Pre-FOMC Announcement Drift"
- Savor & Wilson (2013): "How Much Do Investors Care About Macroeconomic Risk?"
- Andersen et al. (2003): "Micro Effects of Macro Announcements"
- Bernanke & Kuttner (2005): "What Explains the Stock Market's Reaction to Federal Reserve Policy?"
