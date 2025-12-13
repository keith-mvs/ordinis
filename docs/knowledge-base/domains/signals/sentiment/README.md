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
