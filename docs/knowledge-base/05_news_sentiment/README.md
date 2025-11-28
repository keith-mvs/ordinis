# News, Headlines & Sentiment Integration - Knowledge Base

## Purpose

News and sentiment provide **event-driven signals**, **volatility triggers**, and **risk overlays** for the automated trading system. This section documents how to systematically integrate news while avoiding naive headline-chasing.

---

## 1. Event Classification

### 1.1 Event Types Taxonomy

| Category | Event Types | Typical Impact | Volatility |
|----------|-------------|----------------|------------|
| **Earnings** | Quarterly reports, guidance | High | High |
| **Corporate** | M&A, spinoffs, buybacks, dividends | Medium-High | Medium |
| **Management** | CEO changes, insider transactions | Medium | Low-Medium |
| **Regulatory** | FDA decisions, SEC actions, lawsuits | High | High |
| **Macro** | Fed decisions, economic data, geopolitical | Market-wide | Variable |
| **Analyst** | Upgrades, downgrades, price targets | Low-Medium | Low |
| **News** | Product launches, contracts, partnerships | Variable | Variable |

---

### 1.2 News Item Schema

```python
@dataclass
class NewsItem:
    # Identification
    id: str
    timestamp: datetime
    source: str
    source_reliability: float  # 0.0-1.0

    # Content
    headline: str
    summary: str
    full_text: Optional[str]

    # Classification
    tickers: List[str]  # Affected symbols
    event_type: str
    category: str

    # Analysis
    sentiment_score: float  # -1.0 to 1.0
    magnitude: float  # 0.0 to 1.0 (impact size)
    confidence: float  # 0.0 to 1.0

    # Metadata
    is_breaking: bool
    is_scheduled: bool  # Known event vs surprise
    related_news: List[str]  # Previous coverage
```

---

### 1.3 Source Reliability Tiers

**Tier 1 (Highest Reliability)**:
- SEC filings (EDGAR)
- Company press releases (official)
- Federal Reserve announcements
- Exchange/regulator notices

**Tier 2 (High Reliability)**:
- Major news wires: Reuters, AP, Bloomberg, Dow Jones
- Major financial outlets: WSJ, FT, NYT Business

**Tier 3 (Moderate Reliability)**:
- Industry publications
- Analyst reports (from reputable firms)
- Business news: CNBC, MarketWatch, Yahoo Finance

**Tier 4 (Use with Caution)**:
- Aggregators (require source verification)
- Smaller financial blogs
- Social media (verification required)

**Excluded**:
- Anonymous sources without corroboration
- Promotional content
- Unverified social media posts

```python
# Source reliability scores
SOURCE_RELIABILITY = {
    'sec_edgar': 1.0,
    'company_ir': 1.0,
    'fed_gov': 1.0,
    'reuters': 0.95,
    'bloomberg': 0.95,
    'wsj': 0.90,
    'cnbc': 0.75,
    'marketwatch': 0.70,
    'seeking_alpha': 0.50,
    'twitter': 0.30,  # Requires verification
}

# Minimum reliability for action
MIN_RELIABILITY_FOR_TRADE = 0.70
MIN_RELIABILITY_FOR_EXIT = 0.50  # Lower bar for risk management
```

---

## 2. Sentiment Analysis

### 2.1 Sentiment Scoring

**Methods**:
- **Lexicon-based**: Word lists (Loughran-McDonald financial dictionary)
- **ML-based**: Trained classifiers (FinBERT, custom models)
- **LLM-based**: GPT/Claude analysis with structured prompts

**Scoring Scale**:
```python
# Sentiment range
VERY_NEGATIVE = sentiment < -0.6
NEGATIVE = -0.6 <= sentiment < -0.2
NEUTRAL = -0.2 <= sentiment <= 0.2
POSITIVE = 0.2 < sentiment <= 0.6
VERY_POSITIVE = sentiment > 0.6

# Confidence thresholds
HIGH_CONFIDENCE = confidence > 0.8
MODERATE_CONFIDENCE = 0.5 < confidence <= 0.8
LOW_CONFIDENCE = confidence <= 0.5
```

---

### 2.2 Loughran-McDonald Approach

**Financial-Specific Lexicon**:
```python
# Sample word categories (Loughran-McDonald)
NEGATIVE_WORDS = ['loss', 'decline', 'adverse', 'litigation', 'impairment', ...]
POSITIVE_WORDS = ['profit', 'growth', 'improvement', 'exceeded', 'strong', ...]
UNCERTAINTY_WORDS = ['may', 'could', 'possible', 'uncertain', 'risk', ...]
LITIGIOUS_WORDS = ['lawsuit', 'plaintiff', 'defendant', 'court', ...]

# Scoring
def calculate_sentiment(text):
    words = tokenize(text)
    positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
    negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    total_words = len(words)

    sentiment = (positive_count - negative_count) / total_words
    return sentiment
```

---

### 2.3 Event-Specific Sentiment

**Earnings Events**:
```python
# Earnings sentiment signals
BEAT = actual_eps > consensus_eps
MISS = actual_eps < consensus_eps
INLINE = abs(actual_eps - consensus_eps) / consensus_eps < 0.02

# Magnitude
EARNINGS_SURPRISE_PCT = (actual_eps - consensus_eps) / abs(consensus_eps)
BIG_BEAT = EARNINGS_SURPRISE_PCT > 0.10  # >10% beat
BIG_MISS = EARNINGS_SURPRISE_PCT < -0.10  # >10% miss

# Guidance sentiment
GUIDANCE_RAISED = guidance > prior_guidance OR guidance > consensus
GUIDANCE_LOWERED = guidance < prior_guidance OR guidance < consensus
GUIDANCE_MAINTAINED = guidance unchanged
```

**M&A Events**:
```python
# M&A sentiment (for target)
ACQUISITION_PREMIUM = (offer_price - prior_close) / prior_close
POSITIVE_FOR_TARGET = ACQUISITION_PREMIUM > 0.10
HOSTILE_BID = not_recommended_by_board

# For acquirer
ACQUIRER_SENTIMENT = generally_neutral_to_negative  # Market skepticism
```

---

## 3. News-Based Trading Rules

### 3.1 Pre-Event Rules

```python
# Earnings blackout
EARNINGS_BLACKOUT = days_to_earnings <= 3
IF EARNINGS_BLACKOUT:
    action = "no_new_positions"
    reason = "earnings_uncertainty"

# Known event avoidance
MAJOR_EVENT_PENDING = (
    earnings_within(days=3) OR
    fda_decision_pending OR
    major_legal_ruling_pending
)
IF MAJOR_EVENT_PENDING:
    action = "reduce_position_size_50pct"
    reason = "binary_event_risk"

# Economic calendar
FED_MEETING = fomc_announcement_within(hours=24)
IF FED_MEETING:
    action = "no_new_rate_sensitive_positions"
```

---

### 3.2 Post-Event Rules

```python
# Earnings reaction
IF EARNINGS_RELEASED:
    wait_period = 15 minutes  # Let price stabilize
    IF BIG_BEAT AND price_up AND volume_high:
        signal = "bullish_continuation"
    IF BIG_MISS AND price_down AND volume_high:
        signal = "bearish_continuation"
    IF BIG_MISS AND price_up:
        signal = "potential_reversal"  # Negative priced in

# News-driven exit
IF NEGATIVE_NEWS AND POSITION_OPEN:
    IF price_breaks_support:
        action = "exit_position"
    ELSE:
        action = "tighten_stop"

# Surprise events
IF UNSCHEDULED_MATERIAL_NEWS:
    IF sentiment < -0.5 AND confidence > 0.7:
        action = "immediate_review"  # May need to exit
```

---

### 3.3 News Confirmation Requirements

**CRITICAL: Never trade on news alone**

```python
# News + Price confirmation
CONFIRMED_BULLISH_NEWS = (
    positive_news AND
    price > price_at_news_release AND
    volume > average_volume * 1.5
)

CONFIRMED_BEARISH_NEWS = (
    negative_news AND
    price < price_at_news_release AND
    volume > average_volume * 1.5
)

# Unconfirmed news = no action
IF news_sentiment != price_direction:
    action = "wait_for_confirmation"
    reason = "divergent_signals"

# Fade false moves
IF initial_reaction_fading:
    IF positive_news AND price_reversing_down:
        signal = "potential_short"  # Market disagrees
```

---

## 4. Sentiment Aggregation

### 4.1 Multi-Source Aggregation

```python
def aggregate_sentiment(news_items: List[NewsItem]) -> float:
    """
    Aggregate sentiment across multiple news sources.
    Weight by reliability and recency.
    """
    if not news_items:
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0

    for item in news_items:
        # Recency weight (exponential decay)
        hours_old = (now - item.timestamp).hours
        recency_weight = exp(-hours_old / 24)  # Half-life of 24 hours

        # Combined weight
        weight = item.source_reliability * recency_weight * item.confidence

        weighted_sum += item.sentiment_score * weight
        weight_total += weight

    return weighted_sum / weight_total if weight_total > 0 else 0.0
```

---

### 4.2 Sentiment Change Detection

```python
# Sentiment momentum
SENTIMENT_IMPROVING = current_sentiment > avg_sentiment_past_week
SENTIMENT_DETERIORATING = current_sentiment < avg_sentiment_past_week

# Sentiment shift
SENTIMENT_REVERSAL = (
    (prior_sentiment < -0.3 AND current_sentiment > 0.3) OR
    (prior_sentiment > 0.3 AND current_sentiment < -0.3)
)

# News velocity
NEWS_VELOCITY = count(news_items, past_24_hours)
HIGH_NEWS_ACTIVITY = NEWS_VELOCITY > avg_news_velocity * 2
```

---

## 5. Event Calendar Integration

### 5.1 Scheduled Events

```python
# Economic calendar
ECONOMIC_EVENTS = [
    {'name': 'FOMC_DECISION', 'impact': 'high', 'affects': 'all'},
    {'name': 'NFP_REPORT', 'impact': 'high', 'affects': 'all'},
    {'name': 'CPI_RELEASE', 'impact': 'high', 'affects': 'all'},
    {'name': 'GDP_RELEASE', 'impact': 'medium', 'affects': 'all'},
    {'name': 'EARNINGS', 'impact': 'high', 'affects': 'ticker'},
    {'name': 'EX_DIVIDEND', 'impact': 'low', 'affects': 'ticker'},
]

# Pre-event actions
def pre_event_check(event):
    if event.impact == 'high':
        if hours_until(event) < 24:
            return 'reduce_exposure'
        if hours_until(event) < 1:
            return 'no_new_positions'
    return 'normal'
```

---

### 5.2 Earnings Calendar

```python
# Earnings-specific handling
@dataclass
class EarningsEvent:
    ticker: str
    date: date
    time: str  # 'BMO', 'AMC', 'DURING'
    consensus_eps: float
    consensus_revenue: float
    whisper_number: Optional[float]

# Earnings rules
IF earnings_time == 'BMO':  # Before Market Open
    blackout_start = prior_close
    blackout_end = today_open + 30_minutes

IF earnings_time == 'AMC':  # After Market Close
    blackout_start = today_close - 30_minutes
    blackout_end = next_open + 30_minutes
```

---

## 6. Social Sentiment (With Caution)

### 6.1 Social Media Filters

```python
# Social sentiment is SUPPLEMENTARY only
SOCIAL_REQUIREMENTS = {
    'min_account_age': 90_days,
    'min_followers': 1000,
    'verified_only': preferred,
    'exclude_bots': True,
    'exclude_promotional': True,
}

# Weight social sentiment lower
SOCIAL_WEIGHT = 0.2  # vs 0.8 for traditional news
```

---

### 6.2 Unusual Social Activity

```python
# Detect unusual retail interest
SOCIAL_SPIKE = social_mentions > avg_mentions * 3
REDDIT_SURGE = wsb_mentions > threshold

# Warning: Elevated retail interest
IF SOCIAL_SPIKE AND NOT FUNDAMENTAL_NEWS:
    risk_flag = "retail_speculation"
    action = "avoid_or_reduce_size"

# Contrarian signal (after extreme sentiment)
IF extreme_bullish_social AND price_extended:
    signal = "potential_reversal"
```

---

## 7. News Processing Pipeline

### 7.1 Real-Time Pipeline

```
[News Sources] → [Ingestion] → [Parsing] → [Classification] → [Sentiment] → [Action]
      ↓              ↓            ↓              ↓               ↓            ↓
   Feeds API     Timestamp    Extract:      Event type       Score      Generate
   Webhooks      Dedup        - Ticker      Sector           Magnitude   Signal
   Scrapers      Store        - Entities    Impact level     Confidence  Alert
```

---

### 7.2 Processing Rules

```python
def process_news_item(item: NewsItem) -> Optional[Signal]:
    # Validate source
    if item.source_reliability < MIN_RELIABILITY:
        return None

    # Check for duplicates
    if is_duplicate(item):
        return None

    # Classify and score
    item.event_type = classify_event(item)
    item.sentiment_score = calculate_sentiment(item)

    # Generate signal if significant
    if item.magnitude > MIN_MAGNITUDE and item.confidence > MIN_CONFIDENCE:
        signal = generate_signal(item)
        signal.requires_price_confirmation = True
        return signal

    return None
```

---

## 8. Implementation Examples

### Example 1: Earnings Event Handler

```python
def handle_earnings_event(ticker: str, report: EarningsReport):
    """
    Process earnings release and generate appropriate signals.
    """
    # Calculate surprise
    eps_surprise = (report.actual_eps - report.consensus_eps) / abs(report.consensus_eps)
    rev_surprise = (report.actual_revenue - report.consensus_revenue) / report.consensus_revenue

    # Wait for price reaction
    wait(minutes=15)

    # Get price reaction
    price_change = current_price / price_at_release - 1
    volume_ratio = current_volume / average_volume

    # Generate signal
    if eps_surprise > 0.10 and price_change > 0.02 and volume_ratio > 2.0:
        return Signal(
            type='bullish',
            strength='strong',
            reason='earnings_beat_confirmed',
            confidence=0.8
        )
    elif eps_surprise < -0.10 and price_change < -0.02 and volume_ratio > 2.0:
        return Signal(
            type='bearish',
            strength='strong',
            reason='earnings_miss_confirmed',
            confidence=0.8
        )
    else:
        return Signal(type='neutral', reason='mixed_reaction')
```

### Example 2: News Filter

```python
def should_trade_on_news(news: NewsItem, position: Optional[Position]) -> Action:
    """
    Determine trading action based on news.
    """
    # Check source reliability
    if news.source_reliability < 0.7:
        return Action.IGNORE

    # Check if we have position
    if position and news.sentiment_score < -0.5:
        if news.confidence > 0.7 and news.magnitude > 0.7:
            return Action.REVIEW_IMMEDIATELY
        else:
            return Action.TIGHTEN_STOP

    # For new entries, require confirmation
    if not position and abs(news.sentiment_score) > 0.5:
        return Action.WAIT_FOR_PRICE_CONFIRMATION

    return Action.MONITOR
```

---

## Academic References

1. **Loughran, T. & McDonald, B. (2011)**: "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks" - Financial sentiment lexicon
2. **Tetlock, P. (2007)**: "Giving Content to Investor Sentiment: The Role of Media in the Stock Market"
3. **Garcia, D. (2013)**: "Sentiment during Recessions"
4. **Boudoukh, J. et al. (2019)**: "Information, Trading, and Volatility: Evidence from Firm-Specific News"
5. **Ke, Z., Kelly, B., & Xiu, D. (2019)**: "Predicting Returns with Text Data" - ML for news

---

## Key Takeaways

1. **Never trade on news alone**: Always require price/volume confirmation
2. **Source reliability matters**: Weight trusted sources higher
3. **Time decay**: News impact diminishes rapidly
4. **Scheduled vs. surprise**: Different handling required
5. **Sentiment ≠ Action**: Aggregate and confirm before acting
6. **Social caution**: Retail sentiment is supplementary, not primary
7. **Exit > Entry**: Lower reliability bar for risk management decisions
