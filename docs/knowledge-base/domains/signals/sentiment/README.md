# Sentiment Analysis

## Overview

Sentiment analysis extracts trading signals from text data—news articles, SEC filings, social media, and analyst reports. Unlike price-based signals, sentiment provides forward-looking insight into market expectations and positioning.

---

## Directory Structure

```
14_sentiment_analysis/
├── README.md                    # This file
├── news_sentiment/
│   ├── README.md               # News-based sentiment
│   ├── loughran_mcdonald.md    # Financial lexicon approach
│   └── finbert.md              # Transformer-based
├── social_media/
│   ├── README.md               # Social sentiment
│   ├── twitter_sentiment.md    # Twitter/X analysis
│   └── reddit_wsb.md           # Reddit monitoring
└── alternative_data/
    ├── README.md               # Alt data sentiment
    ├── sec_filings.md          # 10-K/10-Q analysis
    └── earnings_calls.md       # Call transcript analysis
```

---

## Sentiment Scoring Framework

### Standard Sentiment Scale

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class SentimentLevel(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

@dataclass
class SentimentScore:
    """
    Standardized sentiment output.
    """
    # Core scores (-1.0 to 1.0)
    sentiment: float
    confidence: float
    magnitude: float  # Impact size (0.0 to 1.0)

    # Source metadata
    source: str
    source_reliability: float
    timestamp: datetime

    # Text metadata
    word_count: int
    entities_mentioned: list

    def to_level(self) -> SentimentLevel:
        if self.sentiment < -0.6:
            return SentimentLevel.VERY_NEGATIVE
        elif self.sentiment < -0.2:
            return SentimentLevel.NEGATIVE
        elif self.sentiment <= 0.2:
            return SentimentLevel.NEUTRAL
        elif self.sentiment <= 0.6:
            return SentimentLevel.POSITIVE
        else:
            return SentimentLevel.VERY_POSITIVE

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'HIGH': 0.80,
    'MODERATE': 0.50,
    'LOW': 0.30,
    'UNUSABLE': 0.0
}
```

---

## Source Reliability

### Source Tier System

```python
SOURCE_RELIABILITY = {
    # Tier 1: Official sources (reliability 1.0)
    'sec_edgar': 1.0,
    'company_ir': 1.0,      # Investor relations
    'fed_gov': 1.0,
    'exchange_notice': 1.0,

    # Tier 2: Major news wires (reliability 0.90-0.95)
    'reuters': 0.95,
    'bloomberg': 0.95,
    'dow_jones': 0.95,
    'ap': 0.90,

    # Tier 3: Major financial media (reliability 0.80-0.90)
    'wsj': 0.90,
    'ft': 0.90,
    'nyt_business': 0.85,
    'cnbc': 0.80,

    # Tier 4: Financial websites (reliability 0.60-0.75)
    'marketwatch': 0.75,
    'yahoo_finance': 0.70,
    'benzinga': 0.65,
    'seeking_alpha': 0.60,

    # Tier 5: Social/community (reliability 0.20-0.50)
    'twitter_verified': 0.50,
    'reddit': 0.40,
    'stocktwits': 0.35,
    'twitter_unverified': 0.25,

    # Excluded
    'unknown': 0.0,
    'promotional': 0.0
}

# Action thresholds
MIN_RELIABILITY_FOR_TRADE = 0.70
MIN_RELIABILITY_FOR_EXIT = 0.50  # Lower bar for risk management
MIN_RELIABILITY_FOR_ALERT = 0.40
```

---

## News Sentiment Analysis

### Loughran-McDonald Financial Lexicon

```python
class LoughranMcDonaldSentiment:
    """
    Financial-specific lexicon-based sentiment.
    Standard academic approach for financial text.
    """

    # Word categories from L-M dictionary
    CATEGORIES = {
        'negative': ['loss', 'decline', 'adverse', 'litigation', 'impairment',
                    'default', 'restated', 'investigation', 'fraud', 'violation'],
        'positive': ['profit', 'growth', 'improvement', 'exceeded', 'strong',
                    'successful', 'favorable', 'achievement', 'gain', 'increase'],
        'uncertainty': ['may', 'could', 'possible', 'uncertain', 'risk',
                       'approximately', 'believe', 'depend', 'fluctuate'],
        'litigious': ['lawsuit', 'plaintiff', 'defendant', 'court', 'jury',
                     'legal', 'claim', 'arbitration', 'settlement'],
        'constraining': ['require', 'obligation', 'restrict', 'prohibit',
                        'limit', 'comply', 'must', 'necessary']
    }

    def calculate_sentiment(self, text: str) -> SentimentScore:
        """
        Calculate sentiment using word count approach.
        """
        words = self._tokenize(text)
        total_words = len(words)

        if total_words == 0:
            return SentimentScore(sentiment=0, confidence=0, magnitude=0,
                                 source='lexicon', source_reliability=0.7,
                                 timestamp=datetime.now(), word_count=0,
                                 entities_mentioned=[])

        # Count category words
        counts = {}
        for category, word_list in self.CATEGORIES.items():
            counts[category] = sum(1 for w in words if w.lower() in word_list)

        # Calculate sentiment
        pos_pct = counts['positive'] / total_words
        neg_pct = counts['negative'] / total_words

        sentiment = (pos_pct - neg_pct) * 10  # Scale to -1 to 1 range
        sentiment = max(-1, min(1, sentiment))

        # Confidence based on word coverage
        coverage = sum(counts.values()) / total_words
        confidence = min(coverage * 5, 1.0)

        # Magnitude from uncertainty/litigious
        uncertainty = counts['uncertainty'] / total_words
        magnitude = 1 - uncertainty  # Higher uncertainty = lower magnitude

        return SentimentScore(
            sentiment=sentiment,
            confidence=confidence,
            magnitude=magnitude,
            source='loughran_mcdonald',
            source_reliability=0.75,
            timestamp=datetime.now(),
            word_count=total_words,
            entities_mentioned=self._extract_entities(text)
        )

    def _tokenize(self, text: str) -> list:
        """Simple word tokenization."""
        import re
        return re.findall(r'\b\w+\b', text.lower())

    def _extract_entities(self, text: str) -> list:
        """Extract ticker symbols and company names."""
        # Simplified - would use NER in production
        import re
        tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
        return list(set(tickers))
```

### FinBERT Transformer Approach

```python
class FinBERTSentiment:
    """
    Transformer-based financial sentiment using FinBERT.
    More accurate than lexicon for complex text.
    """

    def __init__(self):
        # Would load actual model in production
        self.model = None
        self.tokenizer = None

    def calculate_sentiment(self, text: str) -> SentimentScore:
        """
        Calculate sentiment using FinBERT model.
        """
        # Truncate to model max length
        max_length = 512
        if len(text) > max_length * 4:  # Rough char estimate
            text = text[:max_length * 4]

        # Model prediction (placeholder)
        # In production: outputs = self.model(self.tokenizer(text))
        # probabilities = softmax(outputs)

        # Simulated output structure
        probabilities = {
            'negative': 0.1,
            'neutral': 0.3,
            'positive': 0.6
        }

        # Convert to continuous score
        sentiment = (
            probabilities['positive'] * 1 +
            probabilities['neutral'] * 0 +
            probabilities['negative'] * -1
        )

        # Confidence from probability concentration
        confidence = max(probabilities.values())

        return SentimentScore(
            sentiment=sentiment,
            confidence=confidence,
            magnitude=abs(sentiment),
            source='finbert',
            source_reliability=0.85,
            timestamp=datetime.now(),
            word_count=len(text.split()),
            entities_mentioned=[]
        )
```

---

## News Item Processing

### News Schema

```python
@dataclass
class NewsItem:
    """
    Schema for processed news items.
    """
    # Identification
    id: str
    timestamp: datetime
    source: str
    source_reliability: float

    # Content
    headline: str
    summary: str
    full_text: Optional[str]

    # Classification
    tickers: list           # Affected symbols
    event_type: str         # 'earnings', 'ma', 'regulatory', etc.
    category: str           # Sector/industry

    # Analysis
    sentiment_score: float
    magnitude: float
    confidence: float

    # Metadata
    is_breaking: bool
    is_scheduled: bool      # Known event vs surprise
    related_news: list      # Previous coverage IDs
```

### News Processing Pipeline

```python
class NewsProcessor:
    """
    Process news items for trading signals.
    """

    def __init__(self, sentiment_model: str = 'loughran_mcdonald'):
        if sentiment_model == 'finbert':
            self.sentiment = FinBERTSentiment()
        else:
            self.sentiment = LoughranMcDonaldSentiment()

    def process(self, item: NewsItem) -> Optional[dict]:
        """
        Full processing pipeline for news item.
        """
        # Validate source
        if item.source_reliability < MIN_RELIABILITY_FOR_ALERT:
            return None

        # Check for duplicates
        if self._is_duplicate(item):
            return None

        # Calculate sentiment
        text = item.headline + " " + (item.summary or "")
        score = self.sentiment.calculate_sentiment(text)

        # Generate signal if significant
        if score.magnitude > 0.3 and score.confidence > 0.5:
            signal = self._generate_signal(item, score)
            return signal

        return None

    def _is_duplicate(self, item: NewsItem) -> bool:
        """Check if this is duplicate/stale news."""
        # Would check against recent news database
        return False

    def _generate_signal(self, item: NewsItem, score: SentimentScore) -> dict:
        """
        Generate trading signal from news + sentiment.
        """
        return {
            'tickers': item.tickers,
            'direction': 'bullish' if score.sentiment > 0 else 'bearish',
            'strength': abs(score.sentiment),
            'confidence': score.confidence,
            'source': item.source,
            'requires_confirmation': True,
            'timestamp': item.timestamp,
            'event_type': item.event_type
        }
```

---

## Sentiment Aggregation

### Multi-Source Aggregation

```python
def aggregate_sentiment(
    items: list,  # List of SentimentScore
    decay_hours: float = 24
) -> dict:
    """
    Aggregate sentiment across multiple sources.
    Weight by reliability and recency.
    """
    if not items:
        return {'sentiment': 0, 'confidence': 0, 'source_count': 0}

    now = datetime.now()
    weighted_sum = 0.0
    weight_total = 0.0

    for item in items:
        # Recency weight (exponential decay)
        hours_old = (now - item.timestamp).total_seconds() / 3600
        recency_weight = np.exp(-hours_old / decay_hours)

        # Combined weight
        weight = item.source_reliability * recency_weight * item.confidence

        weighted_sum += item.sentiment * weight
        weight_total += weight

    aggregated = weighted_sum / weight_total if weight_total > 0 else 0

    # Overall confidence
    avg_confidence = np.mean([i.confidence for i in items])

    return {
        'sentiment': aggregated,
        'confidence': avg_confidence,
        'source_count': len(items),
        'decay_hours': decay_hours
    }
```

### Sentiment Momentum

```python
def calculate_sentiment_momentum(
    historical_sentiment: pd.Series,  # Time-indexed sentiment scores
    short_window: int = 3,
    long_window: int = 10
) -> dict:
    """
    Calculate sentiment momentum (change in sentiment).
    """
    short_ma = historical_sentiment.rolling(short_window).mean()
    long_ma = historical_sentiment.rolling(long_window).mean()

    # Current momentum
    current_momentum = short_ma.iloc[-1] - long_ma.iloc[-1]

    # Trend
    if short_ma.iloc[-1] > short_ma.iloc[-2]:
        trend = 'IMPROVING'
    elif short_ma.iloc[-1] < short_ma.iloc[-2]:
        trend = 'DETERIORATING'
    else:
        trend = 'STABLE'

    # Reversal detection
    if historical_sentiment.iloc[-3:].mean() * historical_sentiment.iloc[-1] < 0:
        reversal = True
    else:
        reversal = False

    return {
        'current_sentiment': historical_sentiment.iloc[-1],
        'short_ma': short_ma.iloc[-1],
        'long_ma': long_ma.iloc[-1],
        'momentum': current_momentum,
        'trend': trend,
        'reversal_detected': reversal
    }
```

---

## Social Sentiment

### Social Media Analysis

```python
class SocialSentimentAnalyzer:
    """
    Analyze social media sentiment with appropriate filters.
    """

    # Social sentiment is SUPPLEMENTARY only
    SOCIAL_WEIGHT = 0.2  # vs 0.8 for traditional news

    QUALITY_FILTERS = {
        'min_account_age_days': 90,
        'min_followers': 1000,
        'verified_preferred': True,
        'exclude_bots': True,
        'exclude_promotional': True
    }

    def filter_quality_posts(self, posts: list) -> list:
        """
        Filter social posts for quality.
        """
        filtered = []
        for post in posts:
            if post.get('account_age_days', 0) < self.QUALITY_FILTERS['min_account_age_days']:
                continue
            if post.get('followers', 0) < self.QUALITY_FILTERS['min_followers']:
                continue
            if post.get('is_bot', False) and self.QUALITY_FILTERS['exclude_bots']:
                continue
            if post.get('is_promotional', False) and self.QUALITY_FILTERS['exclude_promotional']:
                continue
            filtered.append(post)
        return filtered

    def detect_unusual_activity(
        self,
        ticker: str,
        mention_count: int,
        avg_mentions: float
    ) -> dict:
        """
        Detect unusual social activity.
        """
        ratio = mention_count / avg_mentions if avg_mentions > 0 else 0

        if ratio > 5:
            alert = 'EXTREME_ACTIVITY'
            action = 'INVESTIGATE'
        elif ratio > 3:
            alert = 'HIGH_ACTIVITY'
            action = 'MONITOR'
        elif ratio > 2:
            alert = 'ELEVATED'
            action = 'NOTE'
        else:
            alert = 'NORMAL'
            action = 'NONE'

        return {
            'ticker': ticker,
            'mention_ratio': ratio,
            'alert_level': alert,
            'recommended_action': action,
            'warning': 'Social spikes without fundamental news may indicate retail speculation'
        }
```

---

## SEC Filing Sentiment

### 10-K/10-Q Analysis

```python
class SECFilingSentiment:
    """
    Analyze sentiment in SEC filings.
    """

    RISK_FACTOR_KEYWORDS = [
        'material adverse', 'significant risk', 'could harm',
        'uncertainty', 'cannot assure', 'no guarantee',
        'competitive pressures', 'regulatory', 'litigation'
    ]

    def analyze_filing(self, filing_text: str, filing_type: str) -> dict:
        """
        Analyze SEC filing for sentiment signals.
        """
        # Risk factors section
        risk_section = self._extract_section(filing_text, 'risk_factors')
        risk_score = self._score_risk_factors(risk_section)

        # MD&A section
        mda_section = self._extract_section(filing_text, 'mda')
        mda_sentiment = self._analyze_mda(mda_section)

        # Compare to prior filing
        word_count = len(filing_text.split())

        return {
            'risk_score': risk_score,
            'mda_sentiment': mda_sentiment,
            'word_count': word_count,
            'filing_type': filing_type,
            'overall_tone': self._calculate_overall(risk_score, mda_sentiment)
        }

    def _score_risk_factors(self, text: str) -> float:
        """
        Score risk factors section (more keywords = more risk).
        """
        if not text:
            return 0.5

        count = sum(1 for kw in self.RISK_FACTOR_KEYWORDS if kw.lower() in text.lower())
        normalized = count / len(self.RISK_FACTOR_KEYWORDS)

        return normalized

    def _analyze_mda(self, text: str) -> float:
        """
        Analyze Management Discussion & Analysis.
        """
        sentiment = LoughranMcDonaldSentiment()
        score = sentiment.calculate_sentiment(text)
        return score.sentiment

    def _extract_section(self, text: str, section: str) -> str:
        """Extract specific section from filing."""
        # Simplified - would use regex patterns for actual sections
        return text

    def _calculate_overall(self, risk: float, mda: float) -> str:
        """Calculate overall filing tone."""
        combined = (mda - risk) / 2
        if combined > 0.2:
            return 'POSITIVE'
        elif combined < -0.2:
            return 'NEGATIVE'
        return 'NEUTRAL'
```

---

## Trading Rules

### Sentiment-Based Entry Rules

```python
def sentiment_entry_rules(
    sentiment_signal: dict,
    price_action: dict
) -> dict:
    """
    Rules for entering positions based on sentiment.
    CRITICAL: Never trade on sentiment alone.
    """

    # Require price confirmation
    sentiment_dir = 1 if sentiment_signal['sentiment'] > 0 else -1
    price_dir = 1 if price_action['change'] > 0 else -1

    confirmed = sentiment_dir == price_dir

    if not confirmed:
        return {
            'action': 'WAIT',
            'reason': 'sentiment_price_divergence',
            'note': 'Wait for alignment before entry'
        }

    # Require volume confirmation
    if price_action.get('volume_ratio', 0) < 1.5:
        return {
            'action': 'WAIT',
            'reason': 'insufficient_volume'
        }

    # Check source reliability
    if sentiment_signal.get('source_reliability', 0) < MIN_RELIABILITY_FOR_TRADE:
        return {
            'action': 'PASS',
            'reason': 'unreliable_source'
        }

    # Confirmed entry
    return {
        'action': 'ENTER',
        'direction': 'LONG' if sentiment_dir > 0 else 'SHORT',
        'confidence': sentiment_signal['confidence'],
        'confirmed_by': ['price', 'volume']
    }
```

### Sentiment-Based Exit Rules

```python
def sentiment_exit_rules(
    position: dict,
    sentiment_update: dict
) -> dict:
    """
    Rules for exiting positions based on sentiment change.
    """
    position_dir = 1 if position['side'] == 'long' else -1
    sentiment_dir = 1 if sentiment_update['sentiment'] > 0 else -1

    # Adverse sentiment
    if position_dir != sentiment_dir:
        if sentiment_update['confidence'] > 0.7 and sentiment_update['magnitude'] > 0.7:
            return {
                'action': 'EXIT_IMMEDIATELY',
                'reason': 'high_confidence_adverse_sentiment'
            }
        else:
            return {
                'action': 'TIGHTEN_STOP',
                'adjustment': 0.5,  # 50% tighter
                'reason': 'adverse_sentiment_developing'
            }

    return {'action': 'HOLD'}
```

---

## Performance Characteristics

| Approach | Accuracy | Latency | Best Use |
|----------|----------|---------|----------|
| Loughran-McDonald | 65-70% | <10ms | SEC filings, formal text |
| FinBERT | 75-80% | 50-100ms | News headlines, articles |
| LLM-based | 80-85% | 500ms+ | Complex analysis |
| Social aggregate | 55-60% | Variable | Supplementary signal |

---

## Best Practices

1. **Never trade on sentiment alone**: Always require price/volume confirmation
2. **Source reliability matters**: Weight trusted sources significantly higher
3. **Time decay**: News impact diminishes rapidly (half-life ~24 hours)
4. **Aggregate multiple sources**: Single source can be misleading
5. **Different models for different text**: Lexicon for formal, FinBERT for informal
6. **Social is supplementary**: Weight at most 20% vs traditional sources
7. **Exit faster than entry**: Lower reliability bar for risk management

---

## Academic References

- Loughran & McDonald (2011): "When Is a Liability Not a Liability?"
- Tetlock (2007): "Giving Content to Investor Sentiment"
- Garcia (2013): "Sentiment during Recessions"
- Ke, Kelly & Xiu (2019): "Predicting Returns with Text Data"
- Boudoukh et al. (2019): "Information, Trading, and Volatility"

# Event-Driven Analysis

## Overview

Event-driven strategies capitalize on price movements caused by corporate events, macroeconomic announcements, and other catalysts. Unlike continuous trading strategies, event-driven approaches focus on specific time windows around known or expected events.

---

## Directory Structure

```
13_event_driven_analysis/
├── README.md                    # This file
├── earnings_events/
│   ├── README.md               # Earnings trading overview
│   ├── earnings_surprise.md    # EPS surprise strategies
│   └── guidance_trading.md     # Forward guidance plays
├── corporate_actions/
│   ├── README.md               # Corporate action overview
│   ├── merger_arbitrage.md     # M&A spread trading
│   └── spinoffs.md             # Spinoff strategies
└── macro_events/
    ├── README.md               # Macro event overview
    ├── fed_decisions.md        # FOMC trading
    └── economic_releases.md    # NFP, CPI, GDP
```

---

## Event Types Taxonomy

| Category | Event Types | Typical Impact | Volatility | Predictability |
|----------|-------------|----------------|------------|----------------|
| **Earnings** | Quarterly reports, guidance | High | High | Scheduled |
| **Corporate** | M&A, spinoffs, buybacks, dividends | Medium-High | Medium | Variable |
| **Management** | CEO changes, insider transactions | Medium | Low-Medium | Surprise |
| **Regulatory** | FDA decisions, SEC actions, lawsuits | High | High | Variable |
| **Macro** | Fed decisions, economic data, geopolitical | Market-wide | Variable | Scheduled |
| **Analyst** | Upgrades, downgrades, price targets | Low-Medium | Low | Surprise |

---

## Event Classification Schema

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

class EventType(Enum):
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "m_and_a"
    SPINOFF = "spinoff"
    DIVIDEND = "dividend"
    BUYBACK = "buyback"
    FDA_DECISION = "fda"
    REGULATORY = "regulatory"
    MANAGEMENT = "management"
    FOMC = "fomc"
    ECONOMIC_DATA = "economic"
    ANALYST_ACTION = "analyst"

class EventTiming(Enum):
    BMO = "before_market_open"
    AMC = "after_market_close"
    DURING = "during_market"
    SCHEDULED = "scheduled"
    SURPRISE = "surprise"

@dataclass
class CorporateEvent:
    """Schema for corporate events."""
    ticker: str
    event_type: EventType
    event_date: datetime
    timing: EventTiming

    # Expectations (if applicable)
    consensus_value: Optional[float] = None
    whisper_number: Optional[float] = None

    # Impact assessment
    expected_impact: str = "medium"  # low, medium, high
    historical_volatility: Optional[float] = None

    # Related events
    related_events: List[str] = None

    def is_scheduled(self) -> bool:
        return self.timing != EventTiming.SURPRISE
```

---

## Event Calendar Integration

### Scheduled Events Database

```python
ECONOMIC_CALENDAR = {
    'FOMC_DECISION': {
        'frequency': '8x_per_year',
        'impact': 'high',
        'affects': 'all_markets',
        'typical_time': '14:00_ET',
        'volatility_window': '2_days'
    },
    'NFP_REPORT': {
        'frequency': 'monthly_first_friday',
        'impact': 'high',
        'affects': 'all_markets',
        'typical_time': '08:30_ET',
        'volatility_window': '1_day'
    },
    'CPI_RELEASE': {
        'frequency': 'monthly',
        'impact': 'high',
        'affects': 'all_markets',
        'typical_time': '08:30_ET',
        'volatility_window': '1_day'
    },
    'GDP_RELEASE': {
        'frequency': 'quarterly',
        'impact': 'medium',
        'affects': 'all_markets',
        'typical_time': '08:30_ET',
        'volatility_window': '1_day'
    },
    'EARNINGS': {
        'frequency': 'quarterly_per_stock',
        'impact': 'high',
        'affects': 'specific_ticker',
        'timing': 'BMO_or_AMC',
        'volatility_window': '3_days'
    }
}
```

### Pre-Event Risk Management

```python
def pre_event_position_adjustment(
    event: CorporateEvent,
    current_position: float,
    hours_until_event: float
) -> dict:
    """
    Adjust position sizing before known events.
    """
    if event.expected_impact == 'high':
        if hours_until_event < 24:
            return {
                'action': 'reduce_exposure',
                'target_size': current_position * 0.5,
                'reason': 'high_impact_event_imminent'
            }
        if hours_until_event < 1:
            return {
                'action': 'close_or_hedge',
                'target_size': 0,
                'reason': 'binary_event_risk'
            }

    if event.expected_impact == 'medium':
        if hours_until_event < 4:
            return {
                'action': 'tighten_stops',
                'stop_adjustment': 0.5,  # 50% tighter
                'reason': 'elevated_event_risk'
            }

    return {'action': 'maintain', 'reason': 'normal_risk_parameters'}

# Event blackout rules
EVENT_BLACKOUT_RULES = {
    'EARNINGS': {
        'no_new_positions': 3,  # days before
        'reduce_size_start': 5,  # days before
        'size_reduction': 0.5
    },
    'FOMC': {
        'no_new_positions': 1,
        'reduce_size_start': 2,
        'size_reduction': 0.3
    },
    'FDA_DECISION': {
        'no_new_positions': 5,
        'reduce_size_start': 10,
        'size_reduction': 0.7
    }
}
```

---

## Earnings Event Trading

### Earnings Surprise Framework

```python
@dataclass
class EarningsReport:
    ticker: str
    report_date: datetime

    # EPS
    actual_eps: float
    consensus_eps: float
    whisper_eps: Optional[float]

    # Revenue
    actual_revenue: float
    consensus_revenue: float

    # Guidance
    guidance_eps: Optional[float]
    prior_guidance_eps: Optional[float]

    # Qualitative
    conference_call_sentiment: Optional[float]

def calculate_earnings_surprise(report: EarningsReport) -> dict:
    """
    Calculate earnings surprise metrics.
    """
    # EPS surprise
    eps_surprise_pct = (
        (report.actual_eps - report.consensus_eps) /
        abs(report.consensus_eps)
    ) if report.consensus_eps != 0 else 0

    # Revenue surprise
    rev_surprise_pct = (
        (report.actual_revenue - report.consensus_revenue) /
        report.consensus_revenue
    )

    # Guidance change
    guidance_change = None
    if report.guidance_eps and report.prior_guidance_eps:
        guidance_change = (
            (report.guidance_eps - report.prior_guidance_eps) /
            abs(report.prior_guidance_eps)
        )

    # Classification
    if eps_surprise_pct > 0.10:
        classification = 'BIG_BEAT'
    elif eps_surprise_pct > 0.02:
        classification = 'BEAT'
    elif eps_surprise_pct < -0.10:
        classification = 'BIG_MISS'
    elif eps_surprise_pct < -0.02:
        classification = 'MISS'
    else:
        classification = 'INLINE'

    return {
        'eps_surprise': eps_surprise_pct,
        'revenue_surprise': rev_surprise_pct,
        'guidance_change': guidance_change,
        'classification': classification
    }
```

### Post-Earnings Announcement Drift (PEAD)

```python
class PEADStrategy:
    """
    Post-Earnings Announcement Drift trading.
    Stocks continue drifting in surprise direction for 60+ days.
    """
    def __init__(
        self,
        surprise_threshold: float = 0.05,
        holding_period: int = 60,
        wait_period_minutes: int = 15
    ):
        self.surprise_threshold = surprise_threshold
        self.holding_period = holding_period
        self.wait_period = wait_period_minutes

    def generate_signal(
        self,
        report: EarningsReport,
        price_reaction: float,
        volume_ratio: float
    ) -> dict:
        """
        Generate PEAD signal after earnings release.
        """
        surprise = calculate_earnings_surprise(report)

        # Require minimum surprise
        if abs(surprise['eps_surprise']) < self.surprise_threshold:
            return {'signal': 'NEUTRAL', 'reason': 'insufficient_surprise'}

        # Require price confirmation
        surprise_direction = 1 if surprise['eps_surprise'] > 0 else -1
        price_direction = 1 if price_reaction > 0 else -1

        if surprise_direction != price_direction:
            return {
                'signal': 'CONTRARIAN_WATCH',
                'reason': 'price_diverges_from_surprise'
            }

        # Require volume confirmation
        if volume_ratio < 1.5:
            return {'signal': 'WEAK', 'reason': 'insufficient_volume'}

        # Strong signal
        if surprise['classification'] in ['BIG_BEAT', 'BIG_MISS']:
            strength = 'STRONG'
        else:
            strength = 'MODERATE'

        return {
            'signal': 'LONG' if surprise_direction > 0 else 'SHORT',
            'strength': strength,
            'surprise_pct': surprise['eps_surprise'],
            'holding_days': self.holding_period
        }

    def position_size_multiplier(self, signal: dict) -> float:
        """
        Scale position by signal strength.
        """
        if signal['strength'] == 'STRONG':
            return 1.0
        elif signal['strength'] == 'MODERATE':
            return 0.6
        return 0.0
```

### Earnings Guidance Trading

```python
def analyze_guidance(report: EarningsReport) -> dict:
    """
    Analyze forward guidance vs consensus.
    """
    if not report.guidance_eps:
        return {'signal': 'NO_GUIDANCE'}

    # Guidance vs consensus (for next quarter)
    guidance_vs_consensus = (
        (report.guidance_eps - report.consensus_eps) /
        abs(report.consensus_eps)
    )

    # Guidance vs prior guidance
    guidance_change = 'MAINTAINED'
    if report.prior_guidance_eps:
        change = (
            (report.guidance_eps - report.prior_guidance_eps) /
            abs(report.prior_guidance_eps)
        )
        if change > 0.02:
            guidance_change = 'RAISED'
        elif change < -0.02:
            guidance_change = 'LOWERED'

    # Signal generation
    if guidance_change == 'RAISED' and guidance_vs_consensus > 0:
        signal = 'BULLISH'
    elif guidance_change == 'LOWERED' and guidance_vs_consensus < 0:
        signal = 'BEARISH'
    elif guidance_change == 'LOWERED' and guidance_vs_consensus > 0:
        signal = 'MIXED_BEARISH'  # Beat but lowered
    else:
        signal = 'NEUTRAL'

    return {
        'signal': signal,
        'guidance_change': guidance_change,
        'guidance_vs_consensus': guidance_vs_consensus
    }
```

---

## Merger Arbitrage

### Deal Spread Calculation

```python
@dataclass
class MergerDeal:
    target_ticker: str
    acquirer_ticker: str
    announcement_date: datetime
    expected_close_date: datetime

    # Deal terms
    deal_type: str  # 'cash', 'stock', 'mixed'
    offer_price: float  # For cash deals
    exchange_ratio: Optional[float]  # For stock deals
    cash_component: Optional[float]

    # Conditions
    regulatory_approval_needed: bool
    shareholder_vote_needed: bool
    financing_contingent: bool

    # Market reaction
    target_price_pre: float
    target_price_post: float

def calculate_merger_spread(
    deal: MergerDeal,
    current_target_price: float,
    current_acquirer_price: float = None
) -> dict:
    """
    Calculate merger arbitrage spread.
    """
    if deal.deal_type == 'cash':
        # Simple cash deal spread
        spread = (deal.offer_price - current_target_price) / current_target_price
        implied_value = deal.offer_price

    elif deal.deal_type == 'stock':
        # Stock deal spread
        implied_value = deal.exchange_ratio * current_acquirer_price
        spread = (implied_value - current_target_price) / current_target_price

    else:  # mixed
        # Cash + stock
        implied_value = deal.cash_component + (deal.exchange_ratio * current_acquirer_price)
        spread = (implied_value - current_target_price) / current_target_price

    # Annualize spread
    days_to_close = (deal.expected_close_date - datetime.now()).days
    annual_spread = spread * (365 / max(days_to_close, 1))

    return {
        'current_spread': spread,
        'annualized_spread': annual_spread,
        'implied_value': implied_value,
        'days_to_close': days_to_close
    }

class MergerArbStrategy:
    """
    Merger arbitrage spread trading.
    """
    def __init__(
        self,
        min_spread: float = 0.03,
        min_annualized: float = 0.10,
        max_days_to_close: int = 365
    ):
        self.min_spread = min_spread
        self.min_annualized = min_annualized
        self.max_days = max_days_to_close

    def evaluate_deal(
        self,
        deal: MergerDeal,
        spread_info: dict
    ) -> dict:
        """
        Evaluate merger arb opportunity.
        """
        # Check spread minimums
        if spread_info['current_spread'] < self.min_spread:
            return {'trade': False, 'reason': 'spread_too_tight'}

        if spread_info['annualized_spread'] < self.min_annualized:
            return {'trade': False, 'reason': 'annualized_return_too_low'}

        if spread_info['days_to_close'] > self.max_days:
            return {'trade': False, 'reason': 'too_long_duration'}

        # Risk assessment
        risk_score = self._assess_deal_risk(deal)

        # Size based on risk
        if risk_score < 0.3:
            size_mult = 1.0
        elif risk_score < 0.6:
            size_mult = 0.6
        else:
            size_mult = 0.3

        return {
            'trade': True,
            'direction': 'long_target',
            'hedge': 'short_acquirer' if deal.deal_type == 'stock' else None,
            'size_multiplier': size_mult,
            'risk_score': risk_score
        }

    def _assess_deal_risk(self, deal: MergerDeal) -> float:
        """
        Assess probability of deal failure.
        Higher score = more risk.
        """
        risk = 0.0

        if deal.regulatory_approval_needed:
            risk += 0.2
        if deal.financing_contingent:
            risk += 0.3
        if deal.shareholder_vote_needed:
            risk += 0.1

        # Premium assessment
        premium = (deal.offer_price - deal.target_price_pre) / deal.target_price_pre
        if premium > 0.50:  # High premium deals more likely to fail
            risk += 0.2

        return min(risk, 1.0)
```

---

## Macro Event Trading

### FOMC Decision Framework

```python
class FOMCStrategy:
    """
    Trading around Federal Reserve decisions.
    """
    def __init__(self):
        self.pre_event_hours = 24
        self.post_event_hours = 4

    def pre_fomc_positioning(
        self,
        hours_until: float,
        rate_expectation: str,  # 'hike', 'cut', 'hold'
        market_pricing: float  # Probability priced in
    ) -> dict:
        """
        Position before FOMC.
        """
        if hours_until < 1:
            return {
                'action': 'no_new_positions',
                'reduce_existing': True,
                'reason': 'imminent_binary_event'
            }

        # If market is mispricing
        if market_pricing < 0.3 and rate_expectation == 'hike':
            return {
                'action': 'cautious_short_duration',
                'size': 'small',
                'reason': 'market_underpricing_hawkish'
            }

        return {'action': 'wait', 'reason': 'event_uncertainty'}

    def post_fomc_signal(
        self,
        decision: str,
        expected: str,
        statement_tone: str,  # 'hawkish', 'dovish', 'neutral'
        initial_reaction: float
    ) -> dict:
        """
        Generate signal after FOMC decision.
        """
        surprise = decision != expected

        if surprise:
            # Surprised moves tend to continue
            if initial_reaction > 0:
                return {
                    'signal': 'bullish_continuation',
                    'strength': 'strong',
                    'hold_hours': 4
                }
            else:
                return {
                    'signal': 'bearish_continuation',
                    'strength': 'strong',
                    'hold_hours': 4
                }

        # Statement tone matters when decision is expected
        if statement_tone == 'hawkish' and initial_reaction > 0:
            return {
                'signal': 'fade_rally',
                'reason': 'hawkish_tone_not_priced'
            }

        return {'signal': 'neutral', 'reason': 'as_expected'}
```

### Economic Data Releases

```python
ECONOMIC_SURPRISE_IMPACT = {
    'NFP': {
        'beat_by_50k': {'spx': -0.005, 'tlt': -0.01, 'dxy': 0.005},
        'miss_by_50k': {'spx': 0.005, 'tlt': 0.01, 'dxy': -0.005}
    },
    'CPI': {
        'beat_by_0.1': {'spx': -0.01, 'tlt': -0.02, 'dxy': 0.01},
        'miss_by_0.1': {'spx': 0.01, 'tlt': 0.02, 'dxy': -0.01}
    }
}

def trade_economic_release(
    release_type: str,
    actual: float,
    consensus: float,
    prior: float
) -> dict:
    """
    Generate signal from economic data release.
    """
    surprise = actual - consensus

    # NFP specific
    if release_type == 'NFP':
        if surprise > 100000:  # Big beat
            return {
                'rates': 'higher',
                'equities': 'lower_short_term',
                'dollar': 'stronger',
                'confidence': 'high'
            }
        elif surprise < -100000:  # Big miss
            return {
                'rates': 'lower',
                'equities': 'higher_short_term',
                'dollar': 'weaker',
                'confidence': 'high'
            }

    # CPI specific
    if release_type == 'CPI':
        if actual > consensus + 0.2:  # Hot inflation
            return {
                'rates': 'significantly_higher',
                'equities': 'sell',
                'dollar': 'stronger',
                'confidence': 'high'
            }

    return {'signal': 'neutral', 'reason': 'within_expectations'}
```

---

## Event Risk Management

### Position Sizing Around Events

```python
def event_adjusted_position_size(
    base_size: float,
    event_type: str,
    hours_until_event: float
) -> float:
    """
    Reduce position size near binary events.
    """
    if event_type in ['EARNINGS', 'FDA_DECISION']:
        # High impact binary events
        if hours_until_event < 4:
            return 0  # No position
        elif hours_until_event < 24:
            return base_size * 0.25
        elif hours_until_event < 72:
            return base_size * 0.5

    elif event_type in ['FOMC', 'CPI', 'NFP']:
        # Macro events
        if hours_until_event < 2:
            return base_size * 0.3
        elif hours_until_event < 24:
            return base_size * 0.6

    return base_size
```

### Event-Driven Stop Loss

```python
def event_stop_adjustment(
    normal_stop: float,
    event_type: str,
    position_side: str,
    surprise_direction: str
) -> float:
    """
    Adjust stops based on event outcome.
    """
    # If event confirms position, trail stop tighter
    if (position_side == 'long' and surprise_direction == 'positive') or \
       (position_side == 'short' and surprise_direction == 'negative'):
        return normal_stop * 0.7  # Tighter stop to protect gains

    # If event opposes position, use wider stop initially
    # (Allow for volatility before deciding)
    else:
        return normal_stop * 1.5  # Wider stop, but evaluate quickly
```

---

## Performance Characteristics

| Strategy | Annual Return | Volatility | Sharpe | Max DD | Win Rate |
|----------|---------------|------------|--------|--------|----------|
| PEAD Long | 8-12% | 15-20% | 0.5-0.7 | 25% | 55-60% |
| Merger Arb | 4-8% | 5-8% | 0.6-1.0 | 10% | 85-90% |
| FOMC Trading | Variable | High | Variable | 15% | 50-55% |

---

## Best Practices

1. **Event calendar discipline**: Always check calendar before entering positions
2. **Wait for confirmation**: Don't chase the initial move
3. **Size appropriately**: Binary events require smaller positions
4. **Hedge when possible**: Use options around events
5. **Track surprise patterns**: Some stocks consistently surprise
6. **Monitor related events**: Sector earnings can predict individual results
7. **Exit discipline**: Pre-defined exit rules, don't let winners become losers

---

## Academic References

- Ball & Brown (1968): "An Empirical Evaluation of Accounting Income Numbers"
- Bernard & Thomas (1989): Post-Earnings Announcement Drift
- Mitchell & Pulvino (2001): "Characteristics of Risk and Return in Risk Arbitrage"
- Lucca & Moench (2015): "The Pre-FOMC Announcement Drift"
- Savor & Wilson (2013): "How Much Do Investors Care About Macroeconomic Risk?"
