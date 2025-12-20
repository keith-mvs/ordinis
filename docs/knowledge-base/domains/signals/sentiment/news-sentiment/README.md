# News Sentiment Analysis

## Overview

News sentiment analysis extracts trading signals from financial news articles, press releases, and wire services. The goal is to quantify market-moving information before it's fully reflected in prices.

---

## News Processing Pipeline

```
[Sources] → [Ingestion] → [Parsing] → [Classification] → [Sentiment] → [Signal]
    ↓           ↓            ↓              ↓               ↓            ↓
 Feeds      Timestamp     Extract:      Event type       Score       Generate
 Webhooks   Dedup         - Ticker      Sector          Magnitude    Alert
 Scrapers   Store         - Entities    Impact          Confidence   Trade
```

---

## Source Categories

### Primary Sources (Tier 1-2)

```python
PRIMARY_SOURCES = {
    # Official (Tier 1)
    'sec_edgar': {
        'reliability': 1.0,
        'latency': 'real-time',
        'content': ['8-K', '10-K', '10-Q', 'form_4'],
        'format': 'structured'
    },
    'company_ir': {
        'reliability': 1.0,
        'latency': 'real-time',
        'content': ['press_releases', 'earnings'],
        'format': 'mixed'
    },

    # News wires (Tier 2)
    'reuters': {
        'reliability': 0.95,
        'latency': 'seconds',
        'content': ['breaking', 'analysis'],
        'format': 'text'
    },
    'bloomberg': {
        'reliability': 0.95,
        'latency': 'seconds',
        'content': ['breaking', 'analysis', 'data'],
        'format': 'mixed'
    },
    'dow_jones': {
        'reliability': 0.95,
        'latency': 'seconds',
        'content': ['breaking', 'djns'],
        'format': 'text'
    }
}
```

### Secondary Sources (Tier 3-4)

```python
SECONDARY_SOURCES = {
    # Major media (Tier 3)
    'wsj': {'reliability': 0.90, 'use': 'analysis'},
    'ft': {'reliability': 0.90, 'use': 'analysis'},
    'cnbc': {'reliability': 0.80, 'use': 'breaking'},

    # Financial sites (Tier 4)
    'marketwatch': {'reliability': 0.75, 'use': 'aggregation'},
    'yahoo_finance': {'reliability': 0.70, 'use': 'aggregation'},
    'seeking_alpha': {'reliability': 0.60, 'use': 'opinion'}
}
```

---

## Lexicon-Based Sentiment

### Loughran-McDonald Dictionary

```python
class LMDictionary:
    """
    Loughran-McDonald financial sentiment lexicon.
    Specifically designed for financial text (unlike VADER/general lexicons).
    """

    # Full dictionaries have ~2,700 negative, ~350 positive words
    NEGATIVE_SAMPLE = [
        'loss', 'losses', 'decline', 'declined', 'declining',
        'adverse', 'adversely', 'against', 'litigation', 'liabilities',
        'impairment', 'impaired', 'default', 'defaults', 'restated',
        'investigation', 'investigations', 'fraud', 'violation', 'terminated',
        'writeoff', 'writedown', 'unfavorable', 'unsuccessful', 'weakness'
    ]

    POSITIVE_SAMPLE = [
        'achieve', 'achieved', 'achievement', 'achievements',
        'benefit', 'beneficial', 'best', 'better', 'breakthrough',
        'efficiency', 'efficient', 'enhance', 'enhanced', 'excellent',
        'exceptional', 'exceed', 'exceeded', 'gain', 'gained', 'gains',
        'growth', 'improve', 'improved', 'improvement', 'increase',
        'increased', 'innovation', 'opportunities', 'opportunity',
        'outperform', 'positive', 'profit', 'profitable', 'progress',
        'strength', 'strong', 'success', 'successful', 'surpass'
    ]

    UNCERTAINTY = [
        'almost', 'anticipate', 'apparent', 'appear', 'approximately',
        'assume', 'believe', 'could', 'depend', 'doubt', 'expose',
        'fluctuate', 'indicate', 'maybe', 'might', 'nearly', 'possible',
        'predict', 'presume', 'probable', 'risk', 'seem', 'suggest',
        'uncertain', 'unclear', 'unknown', 'variable', 'vary'
    ]

    LITIGIOUS = [
        'arbitration', 'attorney', 'claim', 'claims', 'claimant',
        'court', 'defendant', 'depose', 'lawsuit', 'lawsuits',
        'legal', 'litigate', 'litigation', 'plaintiff', 'settle',
        'settlement', 'tribunal', 'verdict'
    ]

    def score_text(self, text: str) -> dict:
        """
        Score text using word proportions.
        """
        words = text.lower().split()
        total = len(words)

        if total == 0:
            return {'sentiment': 0, 'uncertainty': 0, 'litigious': 0}

        neg_count = sum(1 for w in words if w in self.NEGATIVE_SAMPLE)
        pos_count = sum(1 for w in words if w in self.POSITIVE_SAMPLE)
        unc_count = sum(1 for w in words if w in self.UNCERTAINTY)
        lit_count = sum(1 for w in words if w in self.LITIGIOUS)

        # Proportional sentiment
        sentiment = (pos_count - neg_count) / total

        return {
            'sentiment': sentiment,
            'positive_pct': pos_count / total,
            'negative_pct': neg_count / total,
            'uncertainty_pct': unc_count / total,
            'litigious_pct': lit_count / total,
            'word_count': total
        }
```

---

## Transformer-Based Sentiment

### FinBERT Implementation

```python
class FinBERTAnalyzer:
    """
    FinBERT: Pre-trained on financial text for sentiment analysis.
    More accurate than lexicon for nuanced financial language.
    """

    MODEL_NAME = "ProsusAI/finbert"
    MAX_LENGTH = 512

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for text.
        """
        import torch

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.MAX_LENGTH,
            padding=True
        )

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        # Labels: negative, neutral, positive
        labels = ['negative', 'neutral', 'positive']
        scores = {label: float(probs[i]) for i, label in enumerate(labels)}

        # Continuous sentiment score
        sentiment = scores['positive'] - scores['negative']

        return {
            'sentiment': sentiment,
            'confidence': max(scores.values()),
            'probabilities': scores,
            'predicted_label': max(scores, key=scores.get)
        }

    def batch_predict(self, texts: list) -> list:
        """
        Batch prediction for efficiency.
        """
        return [self.predict(text) for text in texts]
```

### Comparison: Lexicon vs Transformer

| Aspect | Loughran-McDonald | FinBERT |
|--------|-------------------|---------|
| Speed | Very fast (<1ms) | Slower (50-100ms) |
| Accuracy | 65-70% | 75-80% |
| Context | No | Yes |
| Negation handling | Poor | Good |
| Best for | SEC filings, long text | Headlines, short text |
| Dependencies | None | PyTorch, transformers |

---

## Event Classification

### News Event Types

```python
class NewsEventClassifier:
    """
    Classify news by event type for appropriate handling.
    """

    EVENT_KEYWORDS = {
        'earnings': ['earnings', 'eps', 'revenue', 'quarterly', 'guidance',
                    'beat', 'miss', 'outlook', 'forecast'],
        'ma': ['acquisition', 'merger', 'acquire', 'bid', 'offer', 'deal',
               'buyout', 'takeover', 'combine'],
        'regulatory': ['fda', 'sec', 'ftc', 'doj', 'approval', 'investigation',
                      'subpoena', 'fine', 'settlement', 'ruling'],
        'management': ['ceo', 'cfo', 'executive', 'resign', 'appoint', 'hire',
                      'retire', 'depart', 'succession'],
        'analyst': ['upgrade', 'downgrade', 'price target', 'rating',
                   'outperform', 'underperform', 'buy', 'sell', 'hold'],
        'product': ['launch', 'release', 'announce', 'unveil', 'introduce',
                   'product', 'service', 'feature'],
        'legal': ['lawsuit', 'sue', 'court', 'litigation', 'settlement',
                 'verdict', 'appeal', 'class action']
    }

    def classify(self, text: str) -> dict:
        """
        Classify news event type.
        """
        text_lower = text.lower()
        scores = {}

        for event_type, keywords in self.EVENT_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            scores[event_type] = count

        if not any(scores.values()):
            return {'event_type': 'general', 'confidence': 0.5}

        best_type = max(scores, key=scores.get)
        total_matches = sum(scores.values())
        confidence = scores[best_type] / total_matches if total_matches > 0 else 0

        return {
            'event_type': best_type,
            'confidence': confidence,
            'all_scores': scores
        }
```

---

## News Processing

### Full Processing Pipeline

```python
class NewsProcessor:
    """
    Complete news processing pipeline.
    """

    def __init__(self, use_finbert: bool = False):
        self.classifier = NewsEventClassifier()
        self.lm_dict = LMDictionary()
        self.finbert = FinBERTAnalyzer() if use_finbert else None

    def process(self, headline: str, body: str = None, source: str = 'unknown') -> dict:
        """
        Process news item through full pipeline.
        """
        # Get source reliability
        reliability = SOURCE_RELIABILITY.get(source, 0.5)

        # Classify event
        event_info = self.classifier.classify(headline + (body or ''))

        # Calculate sentiment
        if self.finbert and len(headline) < 200:
            # Use FinBERT for headlines
            sentiment_info = self.finbert.predict(headline)
        else:
            # Use lexicon for longer text
            text = headline + ' ' + (body or '')
            sentiment_info = self.lm_dict.score_text(text)
            sentiment_info['confidence'] = 0.7  # Default lexicon confidence

        # Extract entities
        tickers = self._extract_tickers(headline + (body or ''))

        return {
            'tickers': tickers,
            'event_type': event_info['event_type'],
            'event_confidence': event_info['confidence'],
            'sentiment': sentiment_info.get('sentiment', 0),
            'sentiment_confidence': sentiment_info.get('confidence', 0.5),
            'source_reliability': reliability,
            'requires_confirmation': True
        }

    def _extract_tickers(self, text: str) -> list:
        """
        Extract ticker symbols from text.
        """
        import re
        # Match $TICKER or standalone uppercase 1-5 letters
        dollar_tickers = re.findall(r'\$([A-Z]{1,5})', text)
        # Filter common words
        excluded = {'A', 'I', 'AM', 'PM', 'CEO', 'CFO', 'SEC', 'FDA', 'US', 'UK', 'EU'}
        bare_tickers = [t for t in re.findall(r'\b([A-Z]{1,5})\b', text)
                       if t not in excluded]
        return list(set(dollar_tickers + bare_tickers))
```

---

## Signal Generation

### News to Trading Signal

```python
def news_to_signal(
    processed_news: dict,
    price_change: float,
    volume_ratio: float
) -> dict:
    """
    Convert processed news to trading signal.
    CRITICAL: Requires price and volume confirmation.
    """
    sentiment = processed_news['sentiment']
    reliability = processed_news['source_reliability']
    confidence = processed_news['sentiment_confidence']

    # Check minimum thresholds
    if reliability < MIN_RELIABILITY_FOR_TRADE:
        return {'signal': 'PASS', 'reason': 'source_reliability_too_low'}

    if abs(sentiment) < 0.3:
        return {'signal': 'NEUTRAL', 'reason': 'weak_sentiment'}

    # Check price confirmation
    sentiment_dir = 1 if sentiment > 0 else -1
    price_dir = 1 if price_change > 0 else -1

    if sentiment_dir != price_dir:
        return {
            'signal': 'DIVERGENCE',
            'reason': 'price_opposes_sentiment',
            'action': 'WAIT_FOR_CONFIRMATION'
        }

    # Check volume confirmation
    if volume_ratio < 1.5:
        return {
            'signal': 'WEAK',
            'reason': 'insufficient_volume',
            'action': 'MONITOR'
        }

    # Generate confirmed signal
    strength = 'STRONG' if abs(sentiment) > 0.6 and confidence > 0.7 else 'MODERATE'

    return {
        'signal': 'LONG' if sentiment > 0 else 'SHORT',
        'strength': strength,
        'sentiment': sentiment,
        'confirmed': True,
        'confirmations': ['price', 'volume'],
        'source_reliability': reliability
    }
```

---

## Academic References

- Loughran & McDonald (2011): "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks"
- Araci (2019): "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"
- Tetlock, Saar-Tsechansky & Macskassy (2008): "More Than Words: Quantifying Language to Measure Firms' Fundamentals"
