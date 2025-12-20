# Liquidity Analysis Signals

## Overview

Liquidity signals measure the ease of executing trades without significant price impact. These signals provide **trade eligibility filters**, **execution cost estimates**, and **capacity constraints** for systematic strategies.

---

## 1. Bid-Ask Spread Analysis

### 1.1 Spread Metrics

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class LiquidityConfig:
    """Configuration for liquidity analysis."""

    # Spread thresholds (percentage)
    tight_spread_pct: float = 0.05      # <5 bps
    normal_spread_pct: float = 0.10     # <10 bps
    wide_spread_pct: float = 0.20       # <20 bps
    very_wide_pct: float = 0.50         # >50 bps = avoid

    # ADV thresholds
    high_liquidity_adv: int = 5_000_000
    liquid_adv: int = 1_000_000
    moderate_adv: int = 500_000
    illiquid_adv: int = 100_000

    # Position size limits (% of ADV)
    max_adv_pct: float = 0.01           # 1% of ADV
    warning_adv_pct: float = 0.005      # 0.5% of ADV


class SpreadAnalysis:
    """Analyze bid-ask spreads for liquidity assessment."""

    def __init__(self, config: LiquidityConfig = None):
        self.config = config or LiquidityConfig()

    def calculate_spread_metrics(
        self,
        bid: pd.Series,
        ask: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate bid-ask spread metrics.
        """
        signals = pd.DataFrame(index=bid.index)

        # Absolute spread
        spread = ask - bid
        signals['spread_abs'] = spread

        # Mid price
        mid = (bid + ask) / 2
        signals['mid_price'] = mid

        # Percentage spread
        spread_pct = spread / mid * 100
        signals['spread_pct'] = spread_pct

        # Spread in basis points
        spread_bps = spread_pct * 100
        signals['spread_bps'] = spread_bps

        # Spread classification
        signals['tight_spread'] = spread_pct < self.config.tight_spread_pct
        signals['normal_spread'] = (
            (spread_pct >= self.config.tight_spread_pct) &
            (spread_pct < self.config.normal_spread_pct)
        )
        signals['wide_spread'] = (
            (spread_pct >= self.config.normal_spread_pct) &
            (spread_pct < self.config.wide_spread_pct)
        )
        signals['very_wide_spread'] = spread_pct >= self.config.very_wide_pct

        # Spread dynamics
        spread_ma = spread_pct.rolling(20).mean()
        signals['spread_vs_avg'] = spread_pct / spread_ma.replace(0, np.nan)
        signals['spread_widening'] = spread_pct > spread_ma * 1.5
        signals['spread_narrowing'] = spread_pct < spread_ma * 0.5

        return signals

    def effective_spread_estimate(
        self,
        trade_price: pd.Series,
        mid_price: pd.Series,
        trade_direction: pd.Series  # 1 = buy, -1 = sell
    ) -> pd.DataFrame:
        """
        Estimate effective spread from actual trades.

        Effective spread = 2 * |Trade Price - Mid| / Mid
        """
        signals = pd.DataFrame(index=trade_price.index)

        # Price deviation from mid
        deviation = trade_price - mid_price

        # Effective half-spread
        effective_half_spread = deviation.abs() / mid_price

        # Full effective spread
        signals['effective_spread_pct'] = effective_half_spread * 2 * 100

        # Compare to quoted spread
        # If trade_direction available, check if got inside spread
        signals['price_improvement'] = np.where(
            trade_direction > 0,
            deviation < 0,  # Bought below mid
            deviation > 0   # Sold above mid
        )

        return signals
```

---

## 2. Average Daily Volume (ADV)

### 2.1 ADV Analysis

**Signal Logic**:
```python
class ADVAnalysis:
    """Average Daily Volume analysis for liquidity assessment."""

    def __init__(self, config: LiquidityConfig = None):
        self.config = config or LiquidityConfig()

    def calculate_adv_metrics(
        self,
        volume: pd.Series,
        price: pd.Series = None
    ) -> pd.DataFrame:
        """
        Calculate ADV-based liquidity metrics.
        """
        signals = pd.DataFrame(index=volume.index)

        # ADV calculations
        adv_5 = volume.rolling(5).mean()
        adv_20 = volume.rolling(20).mean()
        adv_50 = volume.rolling(50).mean()

        signals['adv_5'] = adv_5
        signals['adv_20'] = adv_20
        signals['adv_50'] = adv_50

        # Dollar volume (if price available)
        if price is not None:
            signals['dollar_volume'] = volume * price
            signals['adv_dollar_20'] = signals['dollar_volume'].rolling(20).mean()

        # Liquidity tier classification
        signals['highly_liquid'] = adv_20 > self.config.high_liquidity_adv
        signals['liquid'] = (
            (adv_20 > self.config.liquid_adv) &
            (adv_20 <= self.config.high_liquidity_adv)
        )
        signals['moderately_liquid'] = (
            (adv_20 > self.config.moderate_adv) &
            (adv_20 <= self.config.liquid_adv)
        )
        signals['illiquid'] = adv_20 < self.config.illiquid_adv

        # Volume trend
        adv_trend = adv_5 / adv_50.replace(0, np.nan)
        signals['adv_trending_up'] = adv_trend > 1.2
        signals['adv_trending_down'] = adv_trend < 0.8
        signals['adv_stable'] = (adv_trend >= 0.8) & (adv_trend <= 1.2)

        return signals

    def position_size_limits(
        self,
        adv: pd.Series,
        price: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate position size limits based on ADV.
        """
        signals = pd.DataFrame(index=adv.index)

        # Maximum shares (% of ADV)
        max_shares = adv * self.config.max_adv_pct
        signals['max_position_shares'] = max_shares

        # Maximum dollar amount
        signals['max_position_dollars'] = max_shares * price

        # Warning threshold
        warning_shares = adv * self.config.warning_adv_pct
        signals['warning_threshold_shares'] = warning_shares

        return signals

    def check_position_feasibility(
        self,
        desired_shares: int,
        adv: float,
        price: float
    ) -> Dict:
        """
        Check if desired position size is feasible.
        """
        adv_pct = desired_shares / adv if adv > 0 else float('inf')
        dollar_value = desired_shares * price

        max_shares = adv * self.config.max_adv_pct
        warning_shares = adv * self.config.warning_adv_pct

        return {
            'desired_shares': desired_shares,
            'adv_percentage': adv_pct * 100,
            'dollar_value': dollar_value,
            'feasible': desired_shares <= max_shares,
            'warning': desired_shares > warning_shares,
            'recommended_shares': min(desired_shares, max_shares),
            'days_to_accumulate': np.ceil(desired_shares / max_shares) if max_shares > 0 else np.inf
        }
```

---

## 3. Market Impact Estimation

### 3.1 Impact Models

**Signal Logic**:
```python
class MarketImpactEstimator:
    """Estimate market impact of trades."""

    def __init__(
        self,
        impact_coefficient: float = 0.1,
        permanent_ratio: float = 0.5
    ):
        self.impact_coefficient = impact_coefficient
        self.permanent_ratio = permanent_ratio

    def linear_impact_model(
        self,
        order_shares: int,
        adv: float,
        volatility: float,
        spread_pct: float
    ) -> Dict:
        """
        Simple linear market impact model.

        Impact = coefficient * (Order/ADV) * Volatility
        """
        participation_rate = order_shares / adv if adv > 0 else 0

        # Temporary impact (reverts)
        temp_impact_pct = (
            self.impact_coefficient *
            participation_rate *
            volatility * 100
        )

        # Permanent impact (persists)
        perm_impact_pct = temp_impact_pct * self.permanent_ratio

        # Half-spread cost
        spread_cost_pct = spread_pct / 2

        # Total execution cost
        total_cost_pct = temp_impact_pct + spread_cost_pct

        return {
            'participation_rate': participation_rate * 100,
            'temporary_impact_pct': temp_impact_pct,
            'permanent_impact_pct': perm_impact_pct,
            'spread_cost_pct': spread_cost_pct,
            'total_cost_pct': total_cost_pct,
            'cost_per_share': total_cost_pct / 100  # As decimal
        }

    def square_root_impact_model(
        self,
        order_shares: int,
        adv: float,
        volatility: float,
        price: float
    ) -> Dict:
        """
        Square root market impact model (Almgren-Chriss style).

        Impact = eta * sigma * sqrt(Order / ADV)
        """
        participation_rate = order_shares / adv if adv > 0 else 0

        # Impact scales with square root of order size
        impact_factor = np.sqrt(participation_rate)
        temp_impact_pct = self.impact_coefficient * volatility * impact_factor * 100

        perm_impact_pct = temp_impact_pct * self.permanent_ratio

        return {
            'participation_rate': participation_rate * 100,
            'temporary_impact_pct': temp_impact_pct,
            'permanent_impact_pct': perm_impact_pct,
            'impact_dollars': temp_impact_pct / 100 * price * order_shares
        }

    def impact_adjusted_size(
        self,
        target_impact_pct: float,
        adv: float,
        volatility: float
    ) -> int:
        """
        Calculate maximum order size for target impact.
        """
        # Solve: target = coefficient * (size/adv) * volatility
        # size = target * adv / (coefficient * volatility)

        if volatility == 0:
            return int(adv * self.impact_coefficient)

        max_participation = target_impact_pct / (self.impact_coefficient * volatility * 100)
        max_shares = int(max_participation * adv)

        return max_shares
```

---

## 4. Composite Liquidity Score

### 4.1 Multi-Factor Liquidity Assessment

**Signal Logic**:
```python
class LiquidityScorer:
    """Calculate composite liquidity scores."""

    def __init__(self, config: LiquidityConfig = None):
        self.config = config or LiquidityConfig()

    def calculate_liquidity_score(
        self,
        adv: float,
        spread_pct: float,
        price: float,
        volatility: float
    ) -> Dict:
        """
        Calculate composite liquidity score (0-100).

        Higher score = more liquid.
        """
        scores = {}

        # ADV component (0-40 points)
        if adv >= self.config.high_liquidity_adv:
            scores['adv_score'] = 40
        elif adv >= self.config.liquid_adv:
            scores['adv_score'] = 30
        elif adv >= self.config.moderate_adv:
            scores['adv_score'] = 20
        elif adv >= self.config.illiquid_adv:
            scores['adv_score'] = 10
        else:
            scores['adv_score'] = 0

        # Spread component (0-30 points)
        if spread_pct < self.config.tight_spread_pct:
            scores['spread_score'] = 30
        elif spread_pct < self.config.normal_spread_pct:
            scores['spread_score'] = 25
        elif spread_pct < self.config.wide_spread_pct:
            scores['spread_score'] = 15
        elif spread_pct < self.config.very_wide_pct:
            scores['spread_score'] = 5
        else:
            scores['spread_score'] = 0

        # Price level component (0-15 points)
        # Higher price generally means more institutional interest
        if price >= 50:
            scores['price_score'] = 15
        elif price >= 20:
            scores['price_score'] = 12
        elif price >= 10:
            scores['price_score'] = 8
        elif price >= 5:
            scores['price_score'] = 4
        else:
            scores['price_score'] = 0

        # Volatility adjustment (0-15 points)
        # Lower volatility = more predictable execution
        if volatility < 0.15:
            scores['volatility_score'] = 15
        elif volatility < 0.25:
            scores['volatility_score'] = 10
        elif volatility < 0.40:
            scores['volatility_score'] = 5
        else:
            scores['volatility_score'] = 0

        # Composite score
        scores['composite'] = sum(scores.values())

        # Liquidity tier
        if scores['composite'] >= 80:
            scores['tier'] = 'highly_liquid'
        elif scores['composite'] >= 60:
            scores['tier'] = 'liquid'
        elif scores['composite'] >= 40:
            scores['tier'] = 'moderate'
        elif scores['composite'] >= 20:
            scores['tier'] = 'illiquid'
        else:
            scores['tier'] = 'very_illiquid'

        return scores


class LiquiditySignalEngine:
    """
    Production liquidity signal engine.
    """

    def __init__(self, config: LiquidityConfig = None):
        self.config = config or LiquidityConfig()
        self.spread_analysis = SpreadAnalysis(config)
        self.adv_analysis = ADVAnalysis(config)
        self.impact_estimator = MarketImpactEstimator()
        self.scorer = LiquidityScorer(config)

    def generate_liquidity_signals(
        self,
        ohlcv: pd.DataFrame,
        bid: pd.Series = None,
        ask: pd.Series = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive liquidity signals.
        """
        signals = pd.DataFrame(index=ohlcv.index)

        # ADV analysis
        adv_signals = self.adv_analysis.calculate_adv_metrics(
            ohlcv['volume'],
            ohlcv['close']
        )
        signals = pd.concat([signals, adv_signals], axis=1)

        # Spread analysis (if bid/ask available)
        if bid is not None and ask is not None:
            spread_signals = self.spread_analysis.calculate_spread_metrics(bid, ask)
            signals = pd.concat([signals, spread_signals], axis=1)

        # Volatility for impact estimation
        returns = ohlcv['close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        signals['volatility_annual'] = volatility

        # Position size limits
        if 'adv_20' in signals.columns:
            size_limits = self.adv_analysis.position_size_limits(
                signals['adv_20'],
                ohlcv['close']
            )
            signals = pd.concat([signals, size_limits], axis=1)

        return signals

    def check_trade_feasibility(
        self,
        symbol: str,
        shares: int,
        signals: pd.DataFrame
    ) -> Dict:
        """
        Check if proposed trade is feasible.
        """
        if signals.empty:
            return {'feasible': False, 'reason': 'No liquidity data'}

        latest = signals.iloc[-1]

        adv = latest.get('adv_20', 0)
        spread_pct = latest.get('spread_pct', 0.1)
        price = latest.get('mid_price', 0)
        volatility = latest.get('volatility_annual', 0.3)

        # Liquidity score
        score = self.scorer.calculate_liquidity_score(
            adv, spread_pct, price, volatility
        )

        # Impact estimate
        impact = self.impact_estimator.linear_impact_model(
            shares, adv, volatility, spread_pct
        )

        # Feasibility check
        feasibility = self.adv_analysis.check_position_feasibility(
            shares, adv, price
        )

        return {
            'symbol': symbol,
            'liquidity_score': score['composite'],
            'liquidity_tier': score['tier'],
            'estimated_impact_pct': impact['total_cost_pct'],
            'feasibility': feasibility,
            'tradeable': (
                score['composite'] >= 40 and
                feasibility['feasible'] and
                impact['total_cost_pct'] < 1.0
            ),
            'warnings': self._generate_warnings(score, impact, feasibility)
        }

    def _generate_warnings(
        self,
        score: Dict,
        impact: Dict,
        feasibility: Dict
    ) -> List[str]:
        """
        Generate warning messages for trade.
        """
        warnings = []

        if score['composite'] < 40:
            warnings.append("Low liquidity score - execution may be difficult")

        if impact['total_cost_pct'] > 0.5:
            warnings.append(f"High impact cost: {impact['total_cost_pct']:.2f}%")

        if feasibility['warning']:
            warnings.append("Position exceeds recommended ADV threshold")

        if not feasibility['feasible']:
            warnings.append("Position exceeds maximum ADV limit - consider splitting")

        if feasibility['days_to_accumulate'] > 1:
            warnings.append(f"May take {feasibility['days_to_accumulate']:.0f} days to accumulate")

        return warnings
```

---

## 5. Liquidity Filters

### 5.1 Trading Eligibility

```python
class LiquidityFilter:
    """Filter securities for liquidity requirements."""

    def __init__(self, config: LiquidityConfig = None):
        self.config = config or LiquidityConfig()

    def apply_liquidity_filter(
        self,
        universe: pd.DataFrame,
        min_adv: int = None,
        max_spread_pct: float = None,
        min_price: float = 5.0
    ) -> pd.Index:
        """
        Filter universe for minimum liquidity requirements.
        """
        min_adv = min_adv or self.config.moderate_adv
        max_spread_pct = max_spread_pct or self.config.wide_spread_pct

        # ADV filter
        adv_pass = universe['adv_20'] >= min_adv

        # Spread filter
        if 'spread_pct' in universe.columns:
            spread_pass = universe['spread_pct'] <= max_spread_pct
        else:
            spread_pass = True

        # Price filter
        price_pass = universe['close'] >= min_price

        # Combined filter
        passes_all = adv_pass & spread_pass & price_pass

        return universe[passes_all].index

    def get_liquidity_tier(
        self,
        signals: pd.DataFrame
    ) -> pd.Series:
        """
        Assign liquidity tier to each security.
        """
        tiers = pd.Series('unknown', index=signals.index)

        if 'adv_20' in signals.columns:
            tiers = np.where(
                signals['adv_20'] >= self.config.high_liquidity_adv,
                'tier_1',
                np.where(
                    signals['adv_20'] >= self.config.liquid_adv,
                    'tier_2',
                    np.where(
                        signals['adv_20'] >= self.config.moderate_adv,
                        'tier_3',
                        'tier_4'
                    )
                )
            )

        return pd.Series(tiers, index=signals.index)
```

---

## Signal Usage Guidelines

### Liquidity Tier Requirements

| Tier | ADV | Spread | Max Position |
|------|-----|--------|--------------|
| Tier 1 | >5M | <5 bps | 1% ADV |
| Tier 2 | 1-5M | <10 bps | 0.5% ADV |
| Tier 3 | 500K-1M | <20 bps | 0.25% ADV |
| Tier 4 | <500K | >20 bps | Avoid |

### Integration with Ordinis

```python
# Liquidity filtering in universe selection
liq_engine = LiquiditySignalEngine()
signals = liq_engine.generate_liquidity_signals(ohlcv_data, bid, ask)

# Filter universe
eligible = LiquidityFilter().apply_liquidity_filter(
    signals, min_adv=500_000, max_spread_pct=0.15
)

# Check trade before execution
feasibility = liq_engine.check_trade_feasibility(
    symbol='AAPL', shares=10000, signals=signals
)

if feasibility['tradeable']:
    execute_trade()
else:
    for warning in feasibility['warnings']:
        log_warning(warning)
```

---

## Academic References

1. **Kyle (1985)**: "Continuous Auctions and Insider Trading"
2. **Glosten & Harris (1988)**: "Estimating the Components of Bid-Ask Spread"
3. **Almgren & Chriss (2001)**: "Optimal Execution of Portfolio Transactions"
4. **Hasbrouck (2007)**: "Empirical Market Microstructure"
5. **Kissell (2013)**: "The Science of Algorithmic Trading and Portfolio Management"
