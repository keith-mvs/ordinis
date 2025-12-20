# Market Regime Detection

## Overview

Regime detection identifies the current market state (trending, ranging, volatile, etc.) to select appropriate trading strategies. Different regimes require different approaches - trend-following fails in ranges, mean-reversion fails in trends.

---

## Regime Types

### Primary Regimes
| Regime | Characteristics | Strategy |
|--------|----------------|----------|
| Bull | Rising prices, higher highs/lows | Trend following, buy dips |
| Bear | Falling prices, lower highs/lows | Trend following short, sell rallies |
| Sideways | Range-bound, no clear trend | Mean reversion, range trading |
| Volatile | High ATR, erratic moves | Reduce size, widen stops |
| Low Volatility | Compressed ranges | Breakout anticipation |

### Sub-Regimes
- **Recovery**: Transition from bear to bull
- **Correction**: Pullback within bull market
- **Crash**: Rapid bear with high volatility
- **Accumulation**: Low volatility bottom building
- **Distribution**: Low volatility top building

---

## Detection Methods

### 1. ADX-Based Regime

```python
def adx_regime(data, period=14):
    adx = ADX(data, period)
    plus_di = PLUS_DI(data, period)
    minus_di = MINUS_DI(data, period)

    if adx < 20:
        return "RANGING"
    elif adx >= 25:
        if plus_di > minus_di:
            return "TRENDING_UP"
        else:
            return "TRENDING_DOWN"
    else:
        return "TRANSITIONING"
```

### 2. Moving Average Regime

```python
def ma_regime(data):
    ma_short = SMA(data['close'], 20)
    ma_medium = SMA(data['close'], 50)
    ma_long = SMA(data['close'], 200)
    close = data['close']

    # Strong uptrend: All aligned bullish
    if close > ma_short > ma_medium > ma_long:
        return "STRONG_BULL"

    # Strong downtrend: All aligned bearish
    elif close < ma_short < ma_medium < ma_long:
        return "STRONG_BEAR"

    # Recovering: Price above short, below long
    elif close > ma_short and close < ma_long:
        return "RECOVERY" if ma_short > ma_short.shift(5) else "CORRECTION"

    else:
        return "MIXED"
```

### 3. Volatility Regime

```python
def volatility_regime(data, period=20):
    atr = ATR(data, period)
    atr_percentile = percentile_rank(atr, 252)

    if atr_percentile > 80:
        return "HIGH_VOLATILITY"
    elif atr_percentile < 20:
        return "LOW_VOLATILITY"
    else:
        return "NORMAL_VOLATILITY"
```

### 4. Composite Regime

```python
def composite_regime(data):
    """
    Combine multiple regime indicators for robust classification.
    """
    # Trend strength
    adx = ADX(data, 14)
    trending = adx > 25

    # Trend direction
    ma_50 = SMA(data['close'], 50)
    ma_200 = SMA(data['close'], 200)
    bullish = ma_50 > ma_200

    # Volatility
    atr_pct = percentile_rank(ATR(data, 14), 252)
    high_vol = atr_pct > 75

    # Classify
    if trending:
        if bullish:
            regime = "BULL_TRENDING"
        else:
            regime = "BEAR_TRENDING"
    else:
        regime = "SIDEWAYS"

    # Add volatility modifier
    if high_vol:
        regime += "_VOLATILE"

    return regime
```

### 5. Hidden Markov Model Regime

```python
from hmmlearn import hmm

def hmm_regime(returns, n_states=3):
    """
    Use Hidden Markov Model for regime detection.
    States: 0=Low Vol, 1=High Vol, 2=Crash
    """
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full")
    model.fit(returns.values.reshape(-1, 1))

    # Predict current regime
    current_regime = model.predict(returns.values.reshape(-1, 1))[-1]

    # Get regime probabilities
    regime_probs = model.predict_proba(returns.values.reshape(-1, 1))[-1]

    return {
        'regime': current_regime,
        'probabilities': regime_probs,
        'means': model.means_.flatten(),
        'variances': model.covars_.flatten()
    }
```

---

## Strategy Selection by Regime

```python
def select_strategy(regime):
    strategies = {
        "STRONG_BULL": {
            "primary": "trend_following_long",
            "secondary": "buy_the_dip",
            "avoid": "mean_reversion_short",
            "position_size": 1.0
        },
        "STRONG_BEAR": {
            "primary": "trend_following_short",
            "secondary": "sell_the_rally",
            "avoid": "mean_reversion_long",
            "position_size": 0.8
        },
        "SIDEWAYS": {
            "primary": "mean_reversion",
            "secondary": "range_trading",
            "avoid": "trend_following",
            "position_size": 0.7
        },
        "HIGH_VOLATILITY": {
            "primary": "volatility_selling",
            "secondary": "wide_stops",
            "avoid": "tight_stops",
            "position_size": 0.5
        },
        "LOW_VOLATILITY": {
            "primary": "breakout_anticipation",
            "secondary": "range_trading",
            "avoid": "momentum",
            "position_size": 0.8
        }
    }

    return strategies.get(regime, strategies["SIDEWAYS"])
```

---

## Regime Transitions

### Detection
```python
def detect_regime_change(regime_history, lookback=5):
    """
    Identify when regime is transitioning.
    """
    recent = regime_history[-lookback:]
    unique_regimes = set(recent)

    if len(unique_regimes) > 1:
        return {
            'transitioning': True,
            'from': regime_history[-lookback],
            'to': regime_history[-1],
            'stability': recent.count(recent[-1]) / lookback
        }

    return {'transitioning': False, 'stability': 1.0}
```

### Transition Rules
```python
# Typical regime sequence
# Accumulation → Bull → Distribution → Bear → Accumulation

REGIME_TRANSITIONS = {
    "SIDEWAYS_LOW_VOL": ["BULL_TRENDING", "BEAR_TRENDING"],
    "BULL_TRENDING": ["SIDEWAYS", "BEAR_TRENDING"],
    "BEAR_TRENDING": ["SIDEWAYS", "BULL_TRENDING"],
    "HIGH_VOLATILITY": ["SIDEWAYS", "BEAR_TRENDING"]
}

def predict_next_regime(current_regime):
    return REGIME_TRANSITIONS.get(current_regime, ["SIDEWAYS"])
```

---

## Position Sizing by Regime

```python
def regime_position_size(base_size, regime, conviction):
    """
    Adjust position size based on regime.
    """
    regime_multipliers = {
        "STRONG_BULL": 1.2,
        "STRONG_BEAR": 1.0,
        "SIDEWAYS": 0.7,
        "HIGH_VOLATILITY": 0.5,
        "LOW_VOLATILITY": 0.8,
        "TRANSITIONING": 0.5
    }

    multiplier = regime_multipliers.get(regime, 0.7)
    adjusted_size = base_size * multiplier * conviction

    return min(adjusted_size, base_size * 1.5)  # Cap at 1.5x base
```

---

## Regime-Specific Parameters

```python
def get_regime_parameters(regime):
    """
    Adjust indicator parameters for regime.
    """
    parameters = {
        "TRENDING": {
            "rsi_oversold": 40,      # Less extreme for trend
            "rsi_overbought": 60,
            "atr_multiplier": 2.5,   # Wider stops
            "ma_period": 20          # Standard
        },
        "RANGING": {
            "rsi_oversold": 30,      # Standard levels
            "rsi_overbought": 70,
            "atr_multiplier": 1.5,   # Tighter stops
            "ma_period": 10          # Faster
        },
        "VOLATILE": {
            "rsi_oversold": 20,      # More extreme
            "rsi_overbought": 80,
            "atr_multiplier": 3.0,   # Much wider stops
            "ma_period": 50          # Slower, less noise
        }
    }

    return parameters.get(regime, parameters["RANGING"])
```

---

## Real-Time Regime Monitoring

```python
class RegimeMonitor:
    def __init__(self, data, lookback=50):
        self.data = data
        self.lookback = lookback
        self.regime_history = []

    def update(self, new_bar):
        self.data.append(new_bar)

        # Recalculate regime
        current_regime = self.detect_regime()
        self.regime_history.append(current_regime)

        # Check for regime change
        if len(self.regime_history) > 1:
            if current_regime != self.regime_history[-2]:
                self.on_regime_change(
                    self.regime_history[-2],
                    current_regime
                )

        return current_regime

    def on_regime_change(self, old_regime, new_regime):
        """
        Callback when regime changes.
        """
        print(f"Regime change: {old_regime} → {new_regime}")
        # Trigger strategy adjustments
        self.adjust_strategies(new_regime)
```

---

## Ordinis Regime Framework

The Ordinis system uses a 6-regime classification:

```python
ORDINIS_REGIMES = {
    "BULL": {
        "trend": True,
        "direction": "up",
        "volatility": "normal",
        "strategy_weights": {"momentum": 0.7, "mean_reversion": 0.3}
    },
    "BEAR": {
        "trend": True,
        "direction": "down",
        "volatility": "normal",
        "strategy_weights": {"momentum": 0.7, "mean_reversion": 0.3}
    },
    "SIDEWAYS": {
        "trend": False,
        "direction": "neutral",
        "volatility": "low",
        "strategy_weights": {"momentum": 0.3, "mean_reversion": 0.7}
    },
    "VOLATILE": {
        "trend": False,
        "direction": "uncertain",
        "volatility": "high",
        "strategy_weights": {"momentum": 0.2, "mean_reversion": 0.2, "cash": 0.6}
    },
    "RECOVERY": {
        "trend": True,
        "direction": "up",
        "volatility": "elevated",
        "strategy_weights": {"momentum": 0.5, "mean_reversion": 0.5}
    },
    "CORRECTION": {
        "trend": False,
        "direction": "down",
        "volatility": "normal",
        "strategy_weights": {"momentum": 0.3, "mean_reversion": 0.4, "cash": 0.3}
    }
}
```

---

## Common Pitfalls

1. **Regime Lag**: Detection is backward-looking; transitions are identified late
2. **Overfitting**: Training regime classifiers on limited data
3. **Binary Thinking**: Regimes have fuzzy boundaries
4. **Ignoring Transitions**: The shift between regimes is often the riskiest time

---

## Best Practices

1. **Multiple Indicators**: Don't rely on single regime detector
2. **Probability Framework**: Express regime as probabilities, not binary
3. **Slow Adaptation**: Don't switch strategies on every regime signal
4. **Backtest Each Regime**: Verify strategy works in target regime
5. **Include Transition Rules**: Define behavior during regime changes

---

## Implementation

```python
from src.strategies.regime_adaptive import RegimeDetector

detector = RegimeDetector(
    trend_indicator='adx',
    volatility_indicator='atr',
    lookback=50
)

# Detect current regime
regime = detector.detect(data)

# Get regime-adjusted parameters
params = detector.get_parameters(regime)

# Select strategy
strategy = detector.select_strategy(regime)

# Get position sizing
size = detector.position_size(regime, base_size=1000)
```

---

## Academic Notes

- **Hamilton (1989)**: Markov Switching Models for regime detection
- **Ang & Bekaert (2002)**: Regime switches in stock markets
- **Key Insight**: Matching strategy to regime improves risk-adjusted returns

**Best Practice**: Focus on correctly identifying the current regime rather than predicting the next one. React to regime changes rather than anticipate them.
