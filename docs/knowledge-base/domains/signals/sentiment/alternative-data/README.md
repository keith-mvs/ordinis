# Alternative Data Sentiment

## Overview

Alternative data sentiment analysis extracts signals from non-traditional sources: SEC filings, earnings call transcripts, patent filings, and other documents that provide insight into company fundamentals and management sentiment.

---

## SEC Filing Analysis

### 10-K and 10-Q Sentiment

```python
class SECFilingAnalyzer:
    """
    Analyze sentiment and changes in SEC filings.
    """

    SECTIONS = {
        '10-K': {
            'item1': 'business_description',
            'item1a': 'risk_factors',
            'item7': 'mda',  # Management Discussion & Analysis
            'item7a': 'market_risk',
            'item8': 'financial_statements'
        },
        '10-Q': {
            'part1_item2': 'mda',
            'part1_item3': 'market_risk',
            'part2_item1': 'legal_proceedings',
            'part2_item1a': 'risk_factors'
        }
    }

    def analyze_filing(self, filing_text: str, filing_type: str) -> dict:
        """
        Comprehensive filing analysis.
        """
        sections = self.SECTIONS.get(filing_type, {})
        analysis = {}

        for section_id, section_name in sections.items():
            section_text = self._extract_section(filing_text, section_id)
            if section_text:
                analysis[section_name] = self._analyze_section(section_text, section_name)

        # Overall assessment
        overall_sentiment = self._calculate_overall(analysis)

        return {
            'sections': analysis,
            'overall_sentiment': overall_sentiment,
            'filing_type': filing_type,
            'word_count': len(filing_text.split())
        }

    def _extract_section(self, text: str, section_id: str) -> str:
        """Extract specific section from filing."""
        # Simplified - would use regex patterns in production
        return text

    def _analyze_section(self, text: str, section_type: str) -> dict:
        """Analyze individual section."""
        lm = LoughranMcDonaldSentiment()
        scores = lm.score_text(text)

        # Section-specific interpretation
        if section_type == 'risk_factors':
            # More negative words in risk factors is normal
            adjusted_sentiment = scores['sentiment'] + 0.2  # Adjust baseline
        elif section_type == 'mda':
            adjusted_sentiment = scores['sentiment']
        else:
            adjusted_sentiment = scores['sentiment']

        return {
            'sentiment': adjusted_sentiment,
            'uncertainty': scores.get('uncertainty_pct', 0),
            'litigious': scores.get('litigious_pct', 0),
            'word_count': scores.get('word_count', 0)
        }

    def _calculate_overall(self, analysis: dict) -> float:
        """Calculate overall filing sentiment."""
        weights = {
            'mda': 0.4,          # Most important
            'risk_factors': 0.3,
            'business_description': 0.2,
            'market_risk': 0.1
        }

        weighted_sum = 0
        total_weight = 0

        for section, data in analysis.items():
            if section in weights:
                weighted_sum += data['sentiment'] * weights[section]
                total_weight += weights[section]

        return weighted_sum / total_weight if total_weight > 0 else 0
```

### Filing Change Detection

```python
class FilingChangeDetector:
    """
    Detect meaningful changes between filings.
    Changes in language often precede performance changes.
    """

    def compare_filings(
        self,
        current_filing: str,
        prior_filing: str,
        filing_type: str
    ) -> dict:
        """
        Compare current filing to prior period.
        """
        # Word count change
        current_words = len(current_filing.split())
        prior_words = len(prior_filing.split())
        length_change = (current_words - prior_words) / prior_words if prior_words > 0 else 0

        # Risk factor changes
        current_risks = self._extract_risk_factors(current_filing)
        prior_risks = self._extract_risk_factors(prior_filing)
        new_risks = self._find_new_content(current_risks, prior_risks)
        removed_risks = self._find_new_content(prior_risks, current_risks)

        # Sentiment change
        current_sentiment = self._calculate_sentiment(current_filing)
        prior_sentiment = self._calculate_sentiment(prior_filing)
        sentiment_change = current_sentiment - prior_sentiment

        # Key phrase changes
        key_changes = self._detect_key_phrase_changes(current_filing, prior_filing)

        return {
            'length_change_pct': length_change,
            'sentiment_change': sentiment_change,
            'new_risk_factors': new_risks,
            'removed_risk_factors': removed_risks,
            'key_phrase_changes': key_changes,
            'signal': self._interpret_changes(sentiment_change, new_risks, length_change)
        }

    def _extract_risk_factors(self, text: str) -> list:
        """Extract individual risk factor items."""
        # Would parse structured risk factors
        return []

    def _find_new_content(self, current: list, prior: list) -> list:
        """Find content in current not in prior."""
        # Would use similarity matching
        return []

    def _calculate_sentiment(self, text: str) -> float:
        """Calculate filing sentiment."""
        lm = LoughranMcDonaldSentiment()
        return lm.score_text(text)['sentiment']

    def _detect_key_phrase_changes(self, current: str, prior: str) -> list:
        """Detect changes in key phrases."""
        key_phrases = [
            'going concern',
            'material weakness',
            'significant deficiency',
            'covenant violation',
            'liquidity concerns',
            'restructuring',
            'impairment'
        ]

        changes = []
        for phrase in key_phrases:
            in_current = phrase in current.lower()
            in_prior = phrase in prior.lower()

            if in_current and not in_prior:
                changes.append({'phrase': phrase, 'change': 'ADDED', 'signal': 'NEGATIVE'})
            elif not in_current and in_prior:
                changes.append({'phrase': phrase, 'change': 'REMOVED', 'signal': 'POSITIVE'})

        return changes

    def _interpret_changes(
        self,
        sentiment_change: float,
        new_risks: list,
        length_change: float
    ) -> dict:
        """Interpret filing changes."""
        signals = []

        if sentiment_change < -0.1:
            signals.append('DETERIORATING_TONE')
        elif sentiment_change > 0.1:
            signals.append('IMPROVING_TONE')

        if len(new_risks) > 3:
            signals.append('INCREASED_RISK_DISCLOSURE')

        if length_change > 0.20:
            signals.append('SIGNIFICANTLY_LONGER')  # May indicate problems

        if not signals:
            return {'signal': 'NEUTRAL', 'confidence': 0.3}

        return {
            'signals': signals,
            'overall': 'NEGATIVE' if any('DETERIORATING' in s or 'RISK' in s for s in signals) else 'POSITIVE',
            'confidence': 0.6
        }
```

---

## Earnings Call Transcript Analysis

### Call Structure Analysis

```python
class EarningsCallAnalyzer:
    """
    Analyze earnings call transcripts.
    """

    SECTIONS = ['prepared_remarks', 'qa_session']

    def analyze_call(self, transcript: dict) -> dict:
        """
        Analyze full earnings call transcript.
        """
        # Prepared remarks (management-controlled)
        prepared = self._analyze_prepared_remarks(transcript.get('prepared_remarks', ''))

        # Q&A session (more revealing)
        qa = self._analyze_qa_session(transcript.get('qa', []))

        # Management tone analysis
        management_tone = self._analyze_management_tone(transcript)

        # Analyst questions sentiment
        analyst_sentiment = self._analyze_analyst_questions(transcript.get('qa', []))

        return {
            'prepared_remarks_sentiment': prepared['sentiment'],
            'qa_sentiment': qa['sentiment'],
            'management_tone': management_tone,
            'analyst_sentiment': analyst_sentiment,
            'overall_sentiment': (prepared['sentiment'] * 0.4 + qa['sentiment'] * 0.6),
            'key_topics': self._extract_topics(transcript),
            'uncertainty_level': (prepared['uncertainty'] + qa['uncertainty']) / 2
        }

    def _analyze_prepared_remarks(self, text: str) -> dict:
        """Analyze prepared remarks section."""
        lm = LoughranMcDonaldSentiment()
        scores = lm.score_text(text)

        return {
            'sentiment': scores['sentiment'],
            'uncertainty': scores.get('uncertainty_pct', 0),
            'word_count': scores.get('word_count', 0)
        }

    def _analyze_qa_session(self, qa_items: list) -> dict:
        """Analyze Q&A session."""
        if not qa_items:
            return {'sentiment': 0, 'uncertainty': 0}

        sentiments = []
        uncertainties = []
        lm = LoughranMcDonaldSentiment()

        for item in qa_items:
            answer = item.get('answer', '')
            if answer:
                scores = lm.score_text(answer)
                sentiments.append(scores['sentiment'])
                uncertainties.append(scores.get('uncertainty_pct', 0))

        return {
            'sentiment': np.mean(sentiments) if sentiments else 0,
            'uncertainty': np.mean(uncertainties) if uncertainties else 0
        }

    def _analyze_management_tone(self, transcript: dict) -> dict:
        """Analyze management tone indicators."""
        full_text = transcript.get('prepared_remarks', '') + ' '.join(
            q.get('answer', '') for q in transcript.get('qa', [])
        )

        # Confidence indicators
        confident_phrases = ['we are confident', 'we expect', 'we will', 'strong performance']
        hedging_phrases = ['we believe', 'we hope', 'we think', 'may', 'might', 'could']

        confident_count = sum(1 for p in confident_phrases if p in full_text.lower())
        hedging_count = sum(1 for p in hedging_phrases if p in full_text.lower())

        if confident_count + hedging_count == 0:
            confidence_ratio = 0.5
        else:
            confidence_ratio = confident_count / (confident_count + hedging_count)

        return {
            'confidence_level': 'HIGH' if confidence_ratio > 0.6 else 'LOW' if confidence_ratio < 0.4 else 'MODERATE',
            'confidence_ratio': confidence_ratio,
            'hedging_detected': hedging_count > confident_count
        }

    def _analyze_analyst_questions(self, qa_items: list) -> dict:
        """Analyze sentiment of analyst questions."""
        if not qa_items:
            return {'sentiment': 0, 'concern_level': 'UNKNOWN'}

        # Concerned questions often contain these
        concern_indicators = ['worried', 'concern', 'risk', 'decline', 'pressure', 'challenge']

        concerns = 0
        for item in qa_items:
            question = item.get('question', '').lower()
            if any(ind in question for ind in concern_indicators):
                concerns += 1

        concern_ratio = concerns / len(qa_items)

        return {
            'concern_ratio': concern_ratio,
            'concern_level': 'HIGH' if concern_ratio > 0.5 else 'MODERATE' if concern_ratio > 0.25 else 'LOW'
        }

    def _extract_topics(self, transcript: dict) -> list:
        """Extract key topics discussed."""
        # Simplified - would use topic modeling
        return []
```

### Call Comparison

```python
def compare_earnings_calls(
    current_call: dict,
    prior_call: dict
) -> dict:
    """
    Compare current call to prior quarter.
    """
    current_analysis = EarningsCallAnalyzer().analyze_call(current_call)
    prior_analysis = EarningsCallAnalyzer().analyze_call(prior_call)

    sentiment_change = current_analysis['overall_sentiment'] - prior_analysis['overall_sentiment']
    uncertainty_change = current_analysis['uncertainty_level'] - prior_analysis['uncertainty_level']

    # Tone shift detection
    if sentiment_change > 0.2:
        tone_shift = 'SIGNIFICANTLY_MORE_POSITIVE'
    elif sentiment_change > 0.1:
        tone_shift = 'MORE_POSITIVE'
    elif sentiment_change < -0.2:
        tone_shift = 'SIGNIFICANTLY_MORE_NEGATIVE'
    elif sentiment_change < -0.1:
        tone_shift = 'MORE_NEGATIVE'
    else:
        tone_shift = 'STABLE'

    return {
        'sentiment_change': sentiment_change,
        'uncertainty_change': uncertainty_change,
        'tone_shift': tone_shift,
        'management_confidence_change': (
            current_analysis['management_tone']['confidence_ratio'] -
            prior_analysis['management_tone']['confidence_ratio']
        ),
        'signal': 'BULLISH' if sentiment_change > 0.15 else 'BEARISH' if sentiment_change < -0.15 else 'NEUTRAL'
    }
```

---

## Form 4 Insider Trading

### Insider Sentiment

```python
class InsiderSentimentAnalyzer:
    """
    Analyze Form 4 insider trading filings.
    """

    def analyze_insider_activity(
        self,
        transactions: list,
        lookback_days: int = 90
    ) -> dict:
        """
        Analyze recent insider transactions.
        """
        if not transactions:
            return {'sentiment': 0, 'confidence': 0}

        buys = [t for t in transactions if t['type'] == 'buy']
        sells = [t for t in transactions if t['type'] == 'sell']

        # Value-weighted
        buy_value = sum(t['value'] for t in buys)
        sell_value = sum(t['value'] for t in sells)
        total_value = buy_value + sell_value

        if total_value == 0:
            return {'sentiment': 0, 'confidence': 0.3}

        # Sentiment score
        sentiment = (buy_value - sell_value) / total_value

        # Insider type weighting (CEO/CFO more meaningful)
        ceo_cfo_buys = sum(1 for t in buys if t['insider_type'] in ['CEO', 'CFO'])
        ceo_cfo_sells = sum(1 for t in sells if t['insider_type'] in ['CEO', 'CFO'])

        # Cluster detection (multiple insiders = stronger signal)
        unique_buyers = len(set(t['insider_name'] for t in buys))
        unique_sellers = len(set(t['insider_name'] for t in sells))

        cluster_signal = 'CLUSTERED_BUYING' if unique_buyers >= 3 else \
                        'CLUSTERED_SELLING' if unique_sellers >= 3 else 'SCATTERED'

        return {
            'sentiment': sentiment,
            'buy_value': buy_value,
            'sell_value': sell_value,
            'ceo_cfo_activity': {
                'buys': ceo_cfo_buys,
                'sells': ceo_cfo_sells
            },
            'cluster_signal': cluster_signal,
            'unique_insiders': {
                'buyers': unique_buyers,
                'sellers': unique_sellers
            },
            'confidence': 0.7 if cluster_signal.startswith('CLUSTERED') else 0.5
        }
```

---

## Signal Integration

### Combining Alternative Data

```python
def integrate_alternative_data_signals(
    filing_analysis: dict,
    call_analysis: dict,
    insider_analysis: dict
) -> dict:
    """
    Combine alternative data signals.
    """
    signals = []
    weights = {
        'filing': 0.25,
        'call': 0.50,  # Most forward-looking
        'insider': 0.25
    }

    weighted_sentiment = 0

    if filing_analysis:
        weighted_sentiment += filing_analysis.get('overall_sentiment', 0) * weights['filing']
        if filing_analysis.get('overall_sentiment', 0) < -0.2:
            signals.append('NEGATIVE_FILING_TONE')

    if call_analysis:
        weighted_sentiment += call_analysis.get('overall_sentiment', 0) * weights['call']
        if call_analysis.get('overall_sentiment', 0) < -0.2:
            signals.append('NEGATIVE_CALL_TONE')
        if call_analysis.get('management_tone', {}).get('hedging_detected'):
            signals.append('MANAGEMENT_HEDGING')

    if insider_analysis:
        weighted_sentiment += insider_analysis.get('sentiment', 0) * weights['insider']
        if insider_analysis.get('cluster_signal') == 'CLUSTERED_SELLING':
            signals.append('INSIDER_SELLING_CLUSTER')
        elif insider_analysis.get('cluster_signal') == 'CLUSTERED_BUYING':
            signals.append('INSIDER_BUYING_CLUSTER')

    overall = 'BULLISH' if weighted_sentiment > 0.15 else 'BEARISH' if weighted_sentiment < -0.15 else 'NEUTRAL'

    return {
        'weighted_sentiment': weighted_sentiment,
        'overall_signal': overall,
        'warning_signals': [s for s in signals if 'NEGATIVE' in s or 'SELLING' in s],
        'positive_signals': [s for s in signals if 'BUYING' in s],
        'confidence': 0.7 if len(signals) >= 2 else 0.5
    }
```

---

## Academic References

- Cohen, Malloy & Nguyen (2020): "Lazy Prices" - 10-K text changes predict returns
- Li (2010): "The Information Content of Forward-Looking Statements"
- Loughran & McDonald (2016): "Textual Analysis in Accounting and Finance"
- Segal & Segal (2016): "Are Managers Strategic in Reporting Non-Earnings News?"
