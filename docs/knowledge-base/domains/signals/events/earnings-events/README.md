# Earnings Events Trading

## Overview

Earnings announcements are among the most significant recurring events for individual stocks. They create predictable volatility windows and systematic trading opportunities through earnings surprise reactions and post-announcement drift.

---

## Earnings Calendar Management

### Event Timing

```python
class EarningsTiming:
    """
    Earnings release timing categories.
    """
    BMO = "before_market_open"   # 5:00-9:30 AM ET
    AMC = "after_market_close"   # 4:00-8:00 PM ET
    DURING = "during_market"     # 9:30 AM - 4:00 PM ET

# Blackout windows by timing
EARNINGS_BLACKOUT = {
    'BMO': {
        'start': 'prior_close',
        'end': 'today_open + 30min',
        'no_new_positions_hours': 16  # From prior close
    },
    'AMC': {
        'start': 'today_close - 30min',
        'end': 'next_open + 30min',
        'no_new_positions_hours': 4  # From market close
    }
}
```

### Earnings Surprise Metrics

```python
def calculate_surprise_metrics(
    actual_eps: float,
    consensus_eps: float,
    whisper_eps: float = None,
    actual_revenue: float = None,
    consensus_revenue: float = None
) -> dict:
    """
    Calculate comprehensive earnings surprise metrics.
    """
    # EPS surprise
    eps_surprise = (actual_eps - consensus_eps) / abs(consensus_eps) if consensus_eps else 0

    # Whisper surprise (more important for expectations)
    whisper_surprise = None
    if whisper_eps:
        whisper_surprise = (actual_eps - whisper_eps) / abs(whisper_eps)

    # Revenue surprise
    rev_surprise = None
    if actual_revenue and consensus_revenue:
        rev_surprise = (actual_revenue - consensus_revenue) / consensus_revenue

    # Quality of beat/miss
    quality = 'CLEAN'
    if eps_surprise > 0 and (rev_surprise is None or rev_surprise > 0):
        quality = 'QUALITY_BEAT'
    elif eps_surprise > 0 and rev_surprise < 0:
        quality = 'HOLLOW_BEAT'  # Beat EPS, missed revenue
    elif eps_surprise < 0 and rev_surprise > 0:
        quality = 'REVENUE_BEAT'  # Missed EPS, beat revenue

    return {
        'eps_surprise_pct': eps_surprise,
        'whisper_surprise_pct': whisper_surprise,
        'revenue_surprise_pct': rev_surprise,
        'quality': quality,
        'beat_magnitude': 'BIG' if abs(eps_surprise) > 0.10 else 'SMALL'
    }
```

---

## Post-Earnings Announcement Drift (PEAD)

### Academic Foundation

The PEAD anomaly shows that stocks continue drifting in the direction of earnings surprise for 60+ trading days after announcement.

```python
class PEADParameters:
    """
    PEAD strategy parameters based on academic research.
    """
    # Entry timing
    WAIT_AFTER_RELEASE_MINUTES = 15  # Let initial volatility settle
    MAX_ENTRY_DELAY_HOURS = 24       # Don't chase day-old news

    # Holding period
    TYPICAL_HOLDING_DAYS = 60        # Bernard & Thomas finding
    MIN_HOLDING_DAYS = 21            # At least one month
    MAX_HOLDING_DAYS = 90            # Diminishing returns after

    # Position sizing
    BASE_POSITION_PCT = 0.02         # 2% of portfolio base
    MAX_POSITION_PCT = 0.05          # 5% maximum
    SURPRISE_SCALING = True          # Scale by surprise magnitude

    # Entry thresholds
    MIN_EPS_SURPRISE = 0.05          # 5% surprise minimum
    MIN_VOLUME_RATIO = 1.5           # 150% of average volume
    REQUIRE_PRICE_CONFIRM = True     # Price must move with surprise
```

### PEAD Implementation

```python
class PEADStrategy:
    """
    Post-Earnings Announcement Drift strategy.
    """
    def __init__(self, params: PEADParameters = None):
        self.params = params or PEADParameters()
        self.positions = {}

    def evaluate_earnings(
        self,
        ticker: str,
        surprise_metrics: dict,
        price_reaction: float,
        volume_ratio: float,
        minutes_since_release: int
    ) -> dict:
        """
        Evaluate earnings for PEAD entry.
        """
        # Check timing
        if minutes_since_release < self.params.WAIT_AFTER_RELEASE_MINUTES:
            return {'action': 'WAIT', 'reason': 'too_early'}

        if minutes_since_release > self.params.MAX_ENTRY_DELAY_HOURS * 60:
            return {'action': 'SKIP', 'reason': 'too_late'}

        # Check surprise threshold
        eps_surprise = surprise_metrics['eps_surprise_pct']
        if abs(eps_surprise) < self.params.MIN_EPS_SURPRISE:
            return {'action': 'SKIP', 'reason': 'insufficient_surprise'}

        # Check volume
        if volume_ratio < self.params.MIN_VOLUME_RATIO:
            return {'action': 'SKIP', 'reason': 'weak_volume'}

        # Check price confirmation
        surprise_direction = 1 if eps_surprise > 0 else -1
        price_direction = 1 if price_reaction > 0 else -1

        if self.params.REQUIRE_PRICE_CONFIRM and surprise_direction != price_direction:
            return {
                'action': 'WATCH',
                'reason': 'price_divergence',
                'note': 'contrarian_opportunity_possible'
            }

        # Calculate position size
        position_size = self._calculate_size(eps_surprise, surprise_metrics['quality'])

        return {
            'action': 'ENTER',
            'direction': 'LONG' if eps_surprise > 0 else 'SHORT',
            'position_size': position_size,
            'holding_days': self.params.TYPICAL_HOLDING_DAYS,
            'surprise_pct': eps_surprise,
            'quality': surprise_metrics['quality']
        }

    def _calculate_size(self, surprise: float, quality: str) -> float:
        """
        Scale position by surprise magnitude and quality.
        """
        base = self.params.BASE_POSITION_PCT

        # Scale by surprise magnitude
        if self.params.SURPRISE_SCALING:
            magnitude_mult = min(abs(surprise) / 0.05, 2.0)  # Cap at 2x
        else:
            magnitude_mult = 1.0

        # Quality adjustment
        quality_mult = {
            'QUALITY_BEAT': 1.0,
            'HOLLOW_BEAT': 0.5,  # Lower confidence
            'REVENUE_BEAT': 0.7,
            'CLEAN': 0.8
        }.get(quality, 0.8)

        size = base * magnitude_mult * quality_mult
        return min(size, self.params.MAX_POSITION_PCT)
```

---

## Earnings Momentum

### Consecutive Surprise Tracking

```python
def track_earnings_momentum(
    ticker: str,
    surprise_history: list  # List of past N quarters
) -> dict:
    """
    Track pattern of consecutive beats/misses.
    """
    consecutive_beats = 0
    consecutive_misses = 0

    for surprise in reversed(surprise_history):
        if surprise > 0:
            if consecutive_misses > 0:
                break
            consecutive_beats += 1
        elif surprise < 0:
            if consecutive_beats > 0:
                break
            consecutive_misses += 1

    # Earnings momentum signal
    if consecutive_beats >= 4:
        momentum = 'STRONG_POSITIVE'
        next_surprise_probability = 0.65  # Tends to continue
    elif consecutive_beats >= 2:
        momentum = 'POSITIVE'
        next_surprise_probability = 0.58
    elif consecutive_misses >= 4:
        momentum = 'STRONG_NEGATIVE'
        next_surprise_probability = 0.35
    elif consecutive_misses >= 2:
        momentum = 'NEGATIVE'
        next_surprise_probability = 0.42
    else:
        momentum = 'NEUTRAL'
        next_surprise_probability = 0.50

    return {
        'momentum': momentum,
        'consecutive_beats': consecutive_beats,
        'consecutive_misses': consecutive_misses,
        'implied_next_beat_prob': next_surprise_probability
    }
```

---

## Guidance Analysis

### Forward Guidance Framework

```python
@dataclass
class GuidanceData:
    ticker: str
    quarter: str

    # Current guidance
    guidance_eps_low: float
    guidance_eps_high: float
    guidance_revenue_low: float
    guidance_revenue_high: float

    # Prior guidance (if exists)
    prior_guidance_eps_mid: Optional[float]
    prior_guidance_revenue_mid: Optional[float]

    # Consensus
    consensus_eps: float
    consensus_revenue: float

def analyze_guidance_change(data: GuidanceData) -> dict:
    """
    Analyze guidance relative to prior and consensus.
    """
    current_eps_mid = (data.guidance_eps_low + data.guidance_eps_high) / 2
    current_rev_mid = (data.guidance_revenue_low + data.guidance_revenue_high) / 2

    # Guidance vs consensus
    eps_vs_consensus = (current_eps_mid - data.consensus_eps) / abs(data.consensus_eps)
    rev_vs_consensus = (current_rev_mid - data.consensus_revenue) / data.consensus_revenue

    # Guidance change from prior
    eps_change = None
    if data.prior_guidance_eps_mid:
        eps_change = (current_eps_mid - data.prior_guidance_eps_mid) / abs(data.prior_guidance_eps_mid)

    # Interpret
    if eps_change:
        if eps_change > 0.02:
            guidance_trend = 'RAISED'
        elif eps_change < -0.02:
            guidance_trend = 'LOWERED'
        else:
            guidance_trend = 'MAINTAINED'
    else:
        guidance_trend = 'NEW'

    # Overall signal
    if guidance_trend == 'RAISED' and eps_vs_consensus > 0:
        signal = 'BULLISH'
    elif guidance_trend == 'LOWERED' and eps_vs_consensus < 0:
        signal = 'BEARISH'
    elif guidance_trend == 'LOWERED' and eps_vs_consensus > 0:
        signal = 'MIXED'  # Beat but lowered - watch carefully
    else:
        signal = 'NEUTRAL'

    return {
        'signal': signal,
        'guidance_trend': guidance_trend,
        'eps_vs_consensus': eps_vs_consensus,
        'revenue_vs_consensus': rev_vs_consensus,
        'eps_change_from_prior': eps_change
    }
```

---

## Earnings Volatility Trading

### Pre-Earnings Straddle

```python
def evaluate_earnings_straddle(
    ticker: str,
    days_to_earnings: int,
    implied_move: float,       # From ATM straddle price
    historical_move: float,    # Average absolute move past 8 quarters
    iv_percentile: float       # Current IV vs past year
) -> dict:
    """
    Evaluate pre-earnings volatility trade.
    """
    # Compare implied vs realized
    iv_premium = implied_move / historical_move

    if iv_premium > 1.3:
        # IV overpriced - consider selling
        recommendation = 'SELL_STRADDLE'
        confidence = min((iv_premium - 1) / 0.5, 1.0)
    elif iv_premium < 0.8:
        # IV underpriced - consider buying
        recommendation = 'BUY_STRADDLE'
        confidence = min((1 - iv_premium) / 0.3, 1.0)
    else:
        recommendation = 'NEUTRAL'
        confidence = 0.0

    # Entry timing
    optimal_entry_days = 5  # Enter 5 days before earnings typically
    if days_to_earnings > 10:
        timing = 'TOO_EARLY'
    elif days_to_earnings < 2:
        timing = 'TOO_LATE'
    else:
        timing = 'OPTIMAL'

    return {
        'recommendation': recommendation,
        'confidence': confidence,
        'iv_premium': iv_premium,
        'timing': timing,
        'implied_move': implied_move,
        'historical_move': historical_move
    }
```

---

## Performance Metrics

### Earnings Strategy Tracking

```python
def track_earnings_strategy_performance(trades: list) -> dict:
    """
    Track earnings-based strategy performance.
    """
    results = {
        'total_trades': len(trades),
        'wins': sum(1 for t in trades if t['pnl'] > 0),
        'losses': sum(1 for t in trades if t['pnl'] < 0),
        'total_pnl': sum(t['pnl'] for t in trades),
        'avg_win': 0,
        'avg_loss': 0
    }

    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] < 0]

    results['win_rate'] = results['wins'] / results['total_trades'] if results['total_trades'] > 0 else 0
    results['avg_win'] = sum(wins) / len(wins) if wins else 0
    results['avg_loss'] = sum(losses) / len(losses) if losses else 0

    # By surprise type
    big_beats = [t for t in trades if t.get('surprise_type') == 'BIG_BEAT']
    big_misses = [t for t in trades if t.get('surprise_type') == 'BIG_MISS']

    results['big_beat_win_rate'] = sum(1 for t in big_beats if t['pnl'] > 0) / len(big_beats) if big_beats else 0
    results['big_miss_win_rate'] = sum(1 for t in big_misses if t['pnl'] > 0) / len(big_misses) if big_misses else 0

    return results
```

---

## Risk Management

### Earnings-Specific Stops

```python
def calculate_earnings_stop(
    entry_price: float,
    surprise_direction: str,  # 'positive' or 'negative'
    historical_volatility: float
) -> dict:
    """
    Calculate stop loss for earnings trade.
    """
    # Use wider stops initially (earnings are volatile)
    initial_stop_pct = historical_volatility * 2

    # But don't let it run away
    max_stop_pct = 0.15  # 15% max

    stop_pct = min(initial_stop_pct, max_stop_pct)

    if surprise_direction == 'positive':
        stop_price = entry_price * (1 - stop_pct)
    else:  # short
        stop_price = entry_price * (1 + stop_pct)

    return {
        'stop_price': stop_price,
        'stop_pct': stop_pct,
        'method': 'volatility_based',
        'note': 'Consider trailing after 3-5 days'
    }
```

---

## Academic References

- Ball & Brown (1968): Original PEAD documentation
- Bernard & Thomas (1989): "Post-Earnings-Announcement Drift"
- Livnat & Mendenhall (2006): "Comparing the Post-Earnings Announcement Drift for Surprises"
- Chordia & Shivakumar (2006): "Earnings and Price Momentum"
