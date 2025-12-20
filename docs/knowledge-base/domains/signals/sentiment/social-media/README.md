# Social Media Sentiment

## Overview

Social media provides real-time retail sentiment signals but requires careful filtering due to low signal-to-noise ratio. Social sentiment is supplementary to traditional news—never primary.

---

## Usage Guidelines

```python
SOCIAL_SENTIMENT_RULES = {
    # Weighting
    'max_portfolio_weight': 0.20,  # Social signals max 20% of total sentiment
    'traditional_weight': 0.80,    # Traditional news 80%

    # Filtering requirements
    'require_verification': True,
    'require_fundamental_context': True,
    'avoid_meme_stocks': True,

    # Action thresholds
    'min_source_count': 10,        # Minimum posts for signal
    'min_engagement': 100,         # Minimum total engagement
    'max_bot_ratio': 0.20          # Maximum suspected bot content
}
```

---

## Quality Filtering

### Account Quality Filters

```python
class SocialAccountFilter:
    """
    Filter social media accounts for quality signals.
    """

    QUALITY_THRESHOLDS = {
        'twitter': {
            'min_account_age_days': 90,
            'min_followers': 1000,
            'min_tweets': 100,
            'verified_bonus': 1.5,  # Weight multiplier
            'finance_bio_bonus': 1.3
        },
        'reddit': {
            'min_account_age_days': 30,
            'min_karma': 500,
            'min_posts': 10
        },
        'stocktwits': {
            'min_account_age_days': 60,
            'min_followers': 100
        }
    }

    def score_account(self, account: dict, platform: str) -> float:
        """
        Score account quality (0-1).
        """
        thresholds = self.QUALITY_THRESHOLDS.get(platform, {})
        score = 0.0
        checks = 0

        # Age check
        if account.get('age_days', 0) >= thresholds.get('min_account_age_days', 0):
            score += 1
        checks += 1

        # Followers/karma check
        if platform == 'reddit':
            if account.get('karma', 0) >= thresholds.get('min_karma', 0):
                score += 1
        else:
            if account.get('followers', 0) >= thresholds.get('min_followers', 0):
                score += 1
        checks += 1

        # Activity check
        if account.get('post_count', 0) >= thresholds.get('min_posts', thresholds.get('min_tweets', 0)):
            score += 1
        checks += 1

        base_score = score / checks if checks > 0 else 0

        # Apply bonuses
        if account.get('verified', False):
            base_score *= thresholds.get('verified_bonus', 1.0)
        if account.get('finance_bio', False):
            base_score *= thresholds.get('finance_bio_bonus', 1.0)

        return min(base_score, 1.0)
```

### Bot Detection

```python
class BotDetector:
    """
    Detect likely bot accounts to filter from analysis.
    """

    BOT_INDICATORS = {
        'posting_frequency': 50,     # Posts per day threshold
        'identical_content_pct': 0.3,  # % duplicate content
        'follower_following_ratio': 0.01,  # Very low ratio
        'account_age_vs_posts': 100,   # Posts per day since creation
        'cashtag_density': 0.5         # High $ symbols per post
    }

    def is_likely_bot(self, account: dict, posts: list) -> dict:
        """
        Assess bot probability.
        """
        indicators = []
        bot_score = 0.0

        # Posting frequency
        posts_per_day = account.get('posts_per_day', 0)
        if posts_per_day > self.BOT_INDICATORS['posting_frequency']:
            indicators.append('high_posting_frequency')
            bot_score += 0.3

        # Duplicate content
        if posts:
            unique_posts = len(set(p.get('text', '') for p in posts))
            duplicate_pct = 1 - (unique_posts / len(posts))
            if duplicate_pct > self.BOT_INDICATORS['identical_content_pct']:
                indicators.append('duplicate_content')
                bot_score += 0.4

        # Follower ratio
        followers = account.get('followers', 1)
        following = account.get('following', 1)
        ratio = followers / following if following > 0 else 0
        if ratio < self.BOT_INDICATORS['follower_following_ratio']:
            indicators.append('suspicious_follow_ratio')
            bot_score += 0.2

        # Cashtag density
        if posts:
            avg_cashtags = sum(p.get('text', '').count('$') for p in posts) / len(posts)
            if avg_cashtags > 3:
                indicators.append('excessive_cashtags')
                bot_score += 0.1

        return {
            'is_bot': bot_score > 0.5,
            'bot_score': bot_score,
            'indicators': indicators
        }
```

---

## Platform-Specific Analysis

### Twitter/X Sentiment

```python
class TwitterSentimentAnalyzer:
    """
    Twitter-specific sentiment analysis.
    """

    def analyze_ticker_mentions(
        self,
        ticker: str,
        tweets: list,
        time_window_hours: int = 24
    ) -> dict:
        """
        Analyze sentiment for ticker mentions.
        """
        if not tweets:
            return {'sentiment': 0, 'confidence': 0, 'volume': 0}

        # Filter quality tweets
        quality_tweets = [t for t in tweets if self._is_quality_tweet(t)]

        if len(quality_tweets) < 10:
            return {'sentiment': 0, 'confidence': 0.2, 'volume': len(quality_tweets)}

        # Aggregate sentiment
        sentiments = []
        for tweet in quality_tweets:
            score = self._score_tweet(tweet)
            weight = self._calculate_weight(tweet)
            sentiments.append(score * weight)

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        # Calculate confidence from volume and consistency
        volume_conf = min(len(quality_tweets) / 100, 1.0)
        consistency = 1 - (np.std(sentiments) if len(sentiments) > 1 else 0.5)
        confidence = (volume_conf + consistency) / 2

        return {
            'sentiment': avg_sentiment,
            'confidence': confidence,
            'volume': len(quality_tweets),
            'total_engagement': sum(t.get('engagement', 0) for t in quality_tweets)
        }

    def _is_quality_tweet(self, tweet: dict) -> bool:
        """Check if tweet meets quality standards."""
        # Not a retweet
        if tweet.get('is_retweet', False):
            return False
        # Has some engagement
        if tweet.get('engagement', 0) < 5:
            return False
        # Minimum length
        if len(tweet.get('text', '')) < 20:
            return False
        return True

    def _score_tweet(self, tweet: dict) -> float:
        """Score individual tweet sentiment (-1 to 1)."""
        # Would use sentiment model in production
        text = tweet.get('text', '').lower()

        bullish_words = ['bullish', 'moon', 'buy', 'long', 'calls', 'breakout', 'undervalued']
        bearish_words = ['bearish', 'short', 'puts', 'sell', 'crash', 'overvalued', 'dump']

        bull_count = sum(1 for w in bullish_words if w in text)
        bear_count = sum(1 for w in bearish_words if w in text)

        if bull_count + bear_count == 0:
            return 0

        return (bull_count - bear_count) / (bull_count + bear_count)

    def _calculate_weight(self, tweet: dict) -> float:
        """Weight tweet by engagement and account quality."""
        engagement = tweet.get('engagement', 0)
        followers = tweet.get('author_followers', 0)
        verified = tweet.get('author_verified', False)

        base_weight = 1.0
        base_weight += min(engagement / 1000, 1.0)  # Engagement bonus
        base_weight += min(followers / 100000, 1.0)  # Follower bonus
        if verified:
            base_weight *= 1.5

        return base_weight
```

### Reddit Analysis

```python
class RedditSentimentAnalyzer:
    """
    Reddit-specific sentiment analysis, focused on investment subreddits.
    """

    SUBREDDITS = {
        'wallstreetbets': {'weight': 0.5, 'sentiment_type': 'retail_speculative'},
        'stocks': {'weight': 0.8, 'sentiment_type': 'retail_general'},
        'investing': {'weight': 0.9, 'sentiment_type': 'retail_conservative'},
        'options': {'weight': 0.7, 'sentiment_type': 'retail_derivatives'}
    }

    def analyze_subreddit_sentiment(
        self,
        ticker: str,
        posts: list,
        subreddit: str
    ) -> dict:
        """
        Analyze sentiment from subreddit posts and comments.
        """
        sub_config = self.SUBREDDITS.get(subreddit, {'weight': 0.5})

        if not posts:
            return {'sentiment': 0, 'confidence': 0}

        sentiments = []
        for post in posts:
            # Score post
            post_score = self._score_post(post)
            upvote_weight = np.log1p(post.get('score', 0)) / 10

            sentiments.append(post_score * upvote_weight)

            # Score comments
            for comment in post.get('comments', []):
                comment_score = self._score_post(comment)
                comment_weight = np.log1p(comment.get('score', 0)) / 20
                sentiments.append(comment_score * comment_weight)

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        # Apply subreddit weight
        weighted_sentiment = avg_sentiment * sub_config['weight']

        return {
            'sentiment': weighted_sentiment,
            'raw_sentiment': avg_sentiment,
            'confidence': min(len(posts) / 20, 1.0),
            'subreddit': subreddit,
            'sentiment_type': sub_config['sentiment_type']
        }

    def _score_post(self, post: dict) -> float:
        """Score post/comment sentiment."""
        text = post.get('text', post.get('body', '')).lower()

        # Simplified scoring
        bull_signals = ['buy', 'calls', 'bullish', 'moon', 'rocket', 'undervalued', 'long']
        bear_signals = ['sell', 'puts', 'bearish', 'crash', 'overvalued', 'short', 'dump']

        bull = sum(1 for s in bull_signals if s in text)
        bear = sum(1 for s in bear_signals if s in text)

        if bull + bear == 0:
            return 0

        return (bull - bear) / (bull + bear)

    def detect_unusual_activity(
        self,
        ticker: str,
        current_mentions: int,
        avg_mentions: float
    ) -> dict:
        """
        Detect unusual mention activity.
        """
        ratio = current_mentions / avg_mentions if avg_mentions > 0 else 0

        if ratio > 10:
            return {
                'alert': 'EXTREME',
                'ratio': ratio,
                'warning': 'Possible coordinated activity - trade with extreme caution',
                'action': 'AVOID_OR_CONTRARIAN'
            }
        elif ratio > 5:
            return {
                'alert': 'HIGH',
                'ratio': ratio,
                'warning': 'Unusual retail interest',
                'action': 'REDUCED_SIZE'
            }
        elif ratio > 3:
            return {
                'alert': 'ELEVATED',
                'ratio': ratio,
                'action': 'MONITOR'
            }

        return {'alert': 'NORMAL', 'ratio': ratio}
```

---

## Aggregation

### Cross-Platform Aggregation

```python
def aggregate_social_sentiment(
    twitter_sentiment: dict,
    reddit_sentiment: dict,
    stocktwits_sentiment: dict = None
) -> dict:
    """
    Aggregate sentiment across social platforms.
    """
    sources = []

    if twitter_sentiment and twitter_sentiment.get('confidence', 0) > 0.3:
        sources.append({
            'sentiment': twitter_sentiment['sentiment'],
            'weight': 0.4,
            'confidence': twitter_sentiment['confidence']
        })

    if reddit_sentiment and reddit_sentiment.get('confidence', 0) > 0.3:
        sources.append({
            'sentiment': reddit_sentiment['sentiment'],
            'weight': 0.4,
            'confidence': reddit_sentiment['confidence']
        })

    if stocktwits_sentiment and stocktwits_sentiment.get('confidence', 0) > 0.3:
        sources.append({
            'sentiment': stocktwits_sentiment['sentiment'],
            'weight': 0.2,
            'confidence': stocktwits_sentiment['confidence']
        })

    if not sources:
        return {'sentiment': 0, 'confidence': 0, 'source_count': 0}

    # Weighted average
    total_weight = sum(s['weight'] * s['confidence'] for s in sources)
    if total_weight == 0:
        return {'sentiment': 0, 'confidence': 0, 'source_count': len(sources)}

    weighted_sentiment = sum(
        s['sentiment'] * s['weight'] * s['confidence']
        for s in sources
    ) / total_weight

    avg_confidence = np.mean([s['confidence'] for s in sources])

    return {
        'sentiment': weighted_sentiment,
        'confidence': avg_confidence * 0.7,  # Reduce confidence for social
        'source_count': len(sources),
        'platform_breakdown': {
            'twitter': twitter_sentiment.get('sentiment') if twitter_sentiment else None,
            'reddit': reddit_sentiment.get('sentiment') if reddit_sentiment else None,
            'stocktwits': stocktwits_sentiment.get('sentiment') if stocktwits_sentiment else None
        }
    }
```

---

## Risk Warnings

### Social Sentiment Risks

```python
SOCIAL_SENTIMENT_RISKS = {
    'pump_and_dump': {
        'indicators': ['sudden_volume_spike', 'coordinated_posts', 'new_accounts'],
        'action': 'AVOID'
    },
    'meme_stock': {
        'indicators': ['extreme_mentions', 'emoji_heavy', 'yolo_posts'],
        'action': 'EXTREME_CAUTION'
    },
    'bot_activity': {
        'indicators': ['duplicate_content', 'suspicious_timing', 'low_account_quality'],
        'action': 'FILTER_OUT'
    },
    'echo_chamber': {
        'indicators': ['one_sided_sentiment', 'no_dissent', 'groupthink'],
        'action': 'CONTRARIAN_SIGNAL'
    }
}

def assess_social_risk(ticker: str, social_data: dict) -> dict:
    """
    Assess risks in social sentiment data.
    """
    risks = []

    # Volume spike
    if social_data.get('mention_ratio', 0) > 5:
        risks.append('pump_and_dump_risk')

    # One-sided sentiment
    sentiment_abs = abs(social_data.get('sentiment', 0))
    if sentiment_abs > 0.8:
        risks.append('echo_chamber_risk')

    # Bot ratio
    if social_data.get('bot_ratio', 0) > 0.2:
        risks.append('bot_activity_risk')

    # New account ratio
    if social_data.get('new_account_ratio', 0) > 0.3:
        risks.append('coordinated_activity_risk')

    risk_level = 'HIGH' if len(risks) >= 2 else 'MEDIUM' if risks else 'LOW'

    return {
        'risk_level': risk_level,
        'risks': risks,
        'recommendation': 'AVOID' if risk_level == 'HIGH' else 'CAUTION' if risk_level == 'MEDIUM' else 'PROCEED'
    }
```

---

## Best Practices

1. **Social is supplementary**: Never exceed 20% weight in total sentiment
2. **Filter aggressively**: Most social content is noise
3. **Verify with fundamentals**: Social spikes without news are suspicious
4. **Watch for manipulation**: Coordinated activity is common
5. **Fade extremes**: Extreme social sentiment often marks turning points
6. **Track unusual activity**: Spikes indicate either opportunity or trap
7. **Consider contrarian**: When "everyone" is bullish, be cautious

---

## Academic References

- Bollen, Mao & Zeng (2011): "Twitter mood predicts the stock market"
- Chen et al. (2014): "Wisdom of Crowds: The Value of Stock Opinions Transmitted Through Social Media"
- Cookson & Niessner (2020): "Why Don't We Agree? Evidence from a Social Network of Investors"
