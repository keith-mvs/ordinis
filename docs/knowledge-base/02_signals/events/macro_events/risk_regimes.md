# Risk Regimes: Risk-On/Risk-Off Detection

## Overview

Markets alternate between risk-on (risk appetite) and risk-off (risk aversion) regimes. Detecting regime shifts enables systematic allocation adjustments, hedging decisions, and strategy selection. This document covers quantitative methods for regime detection and trading.

---

## Regime Framework

### Risk Regime Classification

```python
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List, Dict, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats


class RiskRegime(Enum):
    """Market risk regime states."""
    STRONG_RISK_ON = "strong_risk_on"
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"
    STRONG_RISK_OFF = "strong_risk_off"
    CRISIS = "crisis"


@dataclass
class RegimeState:
    """Current regime state with confidence."""
    regime: RiskRegime
    confidence: float
    start_date: date
    duration_days: int

    # Contributing factors
    vol_signal: str
    credit_signal: str
    safe_haven_signal: str
    momentum_signal: str

    # Regime stability
    transition_probability: float


class RegimeIndicators:
    """
    Indicators for regime detection.
    """

    # Risk-on indicators (higher = more risk appetite)
    RISK_ON_INDICATORS = {
        'vix_level': {'threshold_low': 15, 'threshold_high': 25, 'direction': 'inverse'},
        'credit_spreads': {'threshold_low': 300, 'threshold_high': 500, 'direction': 'inverse'},
        'junk_vs_treasury': {'threshold_low': 0, 'threshold_high': -0.02, 'direction': 'direct'},
        'em_vs_dm': {'threshold_low': -0.02, 'threshold_high': 0.02, 'direction': 'direct'},
        'small_vs_large': {'threshold_low': -0.01, 'threshold_high': 0.01, 'direction': 'direct'},
        'gold_vs_copper': {'threshold_low': 0.02, 'threshold_high': -0.02, 'direction': 'inverse'}
    }

    # Crisis indicators
    CRISIS_THRESHOLDS = {
        'vix': 40,
        'credit_spread_move_1d': 50,  # bps
        'spx_drawdown': -0.10,
        'correlation_breakdown': 0.90
    }


class RegimeDetector:
    """
    Multi-factor risk regime detection.
    """

    def __init__(self, lookback_days: int = 20):
        self.lookback = lookback_days
        self.regime_history = []

    def detect_regime(
        self,
        vix: float,
        credit_spread: float,
        junk_return: float,
        treasury_return: float,
        em_return: float,
        dm_return: float,
        gold_return: float,
        copper_return: float,
        spx_return: float
    ) -> RegimeState:
        """
        Detect current risk regime from market indicators.
        """
        signals = {}

        # 1. Volatility signal
        signals['vol'] = self._vol_signal(vix)

        # 2. Credit signal
        signals['credit'] = self._credit_signal(credit_spread, junk_return, treasury_return)

        # 3. Safe haven signal
        signals['safe_haven'] = self._safe_haven_signal(gold_return, treasury_return)

        # 4. Risk appetite signal
        signals['risk_appetite'] = self._risk_appetite_signal(
            em_return, dm_return, copper_return, gold_return
        )

        # 5. Momentum signal
        signals['momentum'] = self._momentum_signal(spx_return)

        # Aggregate signals
        regime, confidence = self._aggregate_signals(signals)

        # Check for crisis
        if self._check_crisis_conditions(vix, credit_spread, spx_return):
            regime = RiskRegime.CRISIS
            confidence = 0.95

        # Build regime state
        duration = self._calculate_duration(regime)

        return RegimeState(
            regime=regime,
            confidence=confidence,
            start_date=date.today() - pd.Timedelta(days=duration),
            duration_days=duration,
            vol_signal=signals['vol'],
            credit_signal=signals['credit'],
            safe_haven_signal=signals['safe_haven'],
            momentum_signal=signals['momentum'],
            transition_probability=self._estimate_transition_prob(regime, duration)
        )

    def _vol_signal(self, vix: float) -> str:
        """Generate volatility signal."""
        if vix < 15:
            return 'STRONG_RISK_ON'
        elif vix < 20:
            return 'RISK_ON'
        elif vix < 25:
            return 'NEUTRAL'
        elif vix < 35:
            return 'RISK_OFF'
        else:
            return 'STRONG_RISK_OFF'

    def _credit_signal(
        self,
        spread: float,
        junk_return: float,
        treasury_return: float
    ) -> str:
        """Generate credit market signal."""
        # Spread level
        if spread < 300:
            level_signal = 'RISK_ON'
        elif spread < 450:
            level_signal = 'NEUTRAL'
        else:
            level_signal = 'RISK_OFF'

        # Relative performance (junk vs treasuries)
        junk_vs_treasury = junk_return - treasury_return

        if junk_vs_treasury > 0.005:  # Junk outperforming
            flow_signal = 'RISK_ON'
        elif junk_vs_treasury < -0.005:  # Treasuries outperforming
            flow_signal = 'RISK_OFF'
        else:
            flow_signal = 'NEUTRAL'

        # Combine
        if level_signal == flow_signal:
            return level_signal
        else:
            return 'NEUTRAL'

    def _safe_haven_signal(
        self,
        gold_return: float,
        treasury_return: float
    ) -> str:
        """Generate safe haven demand signal."""
        # Both gold and treasuries rallying = risk-off
        if gold_return > 0.005 and treasury_return > 0.003:
            return 'RISK_OFF'
        # Both selling off = risk-on
        elif gold_return < -0.005 and treasury_return < -0.003:
            return 'RISK_ON'
        else:
            return 'NEUTRAL'

    def _risk_appetite_signal(
        self,
        em_return: float,
        dm_return: float,
        copper_return: float,
        gold_return: float
    ) -> str:
        """Generate risk appetite signal from cross-asset."""
        # EM vs DM
        em_vs_dm = em_return - dm_return

        # Copper vs Gold (Dr. Copper indicator)
        copper_vs_gold = copper_return - gold_return

        score = 0

        if em_vs_dm > 0.005:
            score += 1
        elif em_vs_dm < -0.005:
            score -= 1

        if copper_vs_gold > 0.005:
            score += 1
        elif copper_vs_gold < -0.005:
            score -= 1

        if score >= 2:
            return 'STRONG_RISK_ON'
        elif score >= 1:
            return 'RISK_ON'
        elif score <= -2:
            return 'STRONG_RISK_OFF'
        elif score <= -1:
            return 'RISK_OFF'
        else:
            return 'NEUTRAL'

    def _momentum_signal(self, spx_return: float) -> str:
        """Generate momentum signal."""
        if spx_return > 0.02:
            return 'STRONG_RISK_ON'
        elif spx_return > 0.005:
            return 'RISK_ON'
        elif spx_return < -0.02:
            return 'STRONG_RISK_OFF'
        elif spx_return < -0.005:
            return 'RISK_OFF'
        else:
            return 'NEUTRAL'

    def _aggregate_signals(self, signals: Dict[str, str]) -> Tuple[RiskRegime, float]:
        """Aggregate individual signals into regime."""
        # Convert to scores
        score_map = {
            'STRONG_RISK_ON': 2,
            'RISK_ON': 1,
            'NEUTRAL': 0,
            'RISK_OFF': -1,
            'STRONG_RISK_OFF': -2
        }

        scores = [score_map.get(s, 0) for s in signals.values()]
        avg_score = np.mean(scores)
        agreement = 1 - (np.std(scores) / 2)  # Higher agreement = higher confidence

        # Map score to regime
        if avg_score > 1.5:
            regime = RiskRegime.STRONG_RISK_ON
        elif avg_score > 0.5:
            regime = RiskRegime.RISK_ON
        elif avg_score > -0.5:
            regime = RiskRegime.NEUTRAL
        elif avg_score > -1.5:
            regime = RiskRegime.RISK_OFF
        else:
            regime = RiskRegime.STRONG_RISK_OFF

        confidence = min(0.95, max(0.3, agreement))

        return regime, confidence

    def _check_crisis_conditions(
        self,
        vix: float,
        credit_spread: float,
        spx_return: float
    ) -> bool:
        """Check if crisis conditions are met."""
        crisis_indicators = 0

        if vix > 40:
            crisis_indicators += 1
        if credit_spread > 600:
            crisis_indicators += 1
        if spx_return < -0.05:
            crisis_indicators += 1

        return crisis_indicators >= 2

    def _estimate_transition_prob(
        self,
        current_regime: RiskRegime,
        duration: int
    ) -> float:
        """
        Estimate probability of regime transition.

        Longer duration = lower transition probability (persistence)
        """
        # Base transition rates (daily)
        base_rates = {
            RiskRegime.STRONG_RISK_ON: 0.03,
            RiskRegime.RISK_ON: 0.05,
            RiskRegime.NEUTRAL: 0.10,
            RiskRegime.RISK_OFF: 0.07,
            RiskRegime.STRONG_RISK_OFF: 0.05,
            RiskRegime.CRISIS: 0.15
        }

        base = base_rates.get(current_regime, 0.05)

        # Duration adjustment (longer regimes more persistent)
        duration_factor = max(0.3, 1 - (duration / 60))

        return base * duration_factor
```

---

## Hidden Markov Model Regime Detection

### HMM Implementation

```python
from scipy.special import logsumexp


class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection.
    """

    def __init__(self, n_states: int = 3):
        """
        Args:
            n_states: Number of hidden states (regimes)
                     Typically 3: risk-on, neutral, risk-off
        """
        self.n_states = n_states
        self.transition_matrix = None
        self.emission_means = None
        self.emission_stds = None
        self.initial_probs = None

    def fit(self, returns: np.ndarray, max_iter: int = 100):
        """
        Fit HMM using Baum-Welch algorithm.
        """
        n = len(returns)

        # Initialize parameters
        self._initialize_params(returns)

        for _ in range(max_iter):
            # E-step: Forward-backward
            alpha, beta, gamma, xi = self._forward_backward(returns)

            # M-step: Update parameters
            self._update_params(returns, gamma, xi)

        return self

    def _initialize_params(self, returns: np.ndarray):
        """Initialize HMM parameters."""
        # Sort returns into quantiles for initial state assignment
        sorted_returns = np.sort(returns)
        n = len(returns)

        # Initial means based on quantiles
        self.emission_means = np.array([
            np.mean(sorted_returns[:n//3]),      # Risk-off (low returns)
            np.mean(sorted_returns[n//3:2*n//3]), # Neutral
            np.mean(sorted_returns[2*n//3:])     # Risk-on (high returns)
        ])

        self.emission_stds = np.array([
            np.std(sorted_returns[:n//3]) + 0.001,
            np.std(sorted_returns[n//3:2*n//3]) + 0.001,
            np.std(sorted_returns[2*n//3:]) + 0.001
        ])

        # Transition matrix (high persistence)
        self.transition_matrix = np.array([
            [0.90, 0.08, 0.02],  # Risk-off tends to stay
            [0.05, 0.90, 0.05],  # Neutral
            [0.02, 0.08, 0.90]   # Risk-on tends to stay
        ])

        # Initial probabilities
        self.initial_probs = np.array([0.2, 0.6, 0.2])

    def _forward_backward(self, returns: np.ndarray) -> Tuple:
        """Forward-backward algorithm."""
        n = len(returns)

        # Emission probabilities
        emission_probs = self._compute_emissions(returns)

        # Forward pass
        alpha = np.zeros((n, self.n_states))
        alpha[0] = self.initial_probs * emission_probs[0]
        alpha[0] /= alpha[0].sum()

        for t in range(1, n):
            alpha[t] = emission_probs[t] * (alpha[t-1] @ self.transition_matrix)
            alpha[t] /= alpha[t].sum() + 1e-10

        # Backward pass
        beta = np.zeros((n, self.n_states))
        beta[-1] = 1

        for t in range(n-2, -1, -1):
            beta[t] = self.transition_matrix @ (emission_probs[t+1] * beta[t+1])
            beta[t] /= beta[t].sum() + 1e-10

        # Gamma (state probabilities)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10

        # Xi (transition probabilities)
        xi = np.zeros((n-1, self.n_states, self.n_states))
        for t in range(n-1):
            xi[t] = (alpha[t][:, None] * self.transition_matrix *
                    emission_probs[t+1] * beta[t+1])
            xi[t] /= xi[t].sum() + 1e-10

        return alpha, beta, gamma, xi

    def _compute_emissions(self, returns: np.ndarray) -> np.ndarray:
        """Compute emission probabilities."""
        n = len(returns)
        probs = np.zeros((n, self.n_states))

        for s in range(self.n_states):
            probs[:, s] = stats.norm.pdf(
                returns,
                self.emission_means[s],
                self.emission_stds[s]
            )

        return probs + 1e-10

    def _update_params(
        self,
        returns: np.ndarray,
        gamma: np.ndarray,
        xi: np.ndarray
    ):
        """Update HMM parameters (M-step)."""
        # Update transition matrix
        self.transition_matrix = xi.sum(axis=0) / xi.sum(axis=(0, 2))[:, None]
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)

        # Update emission parameters
        for s in range(self.n_states):
            weight = gamma[:, s]
            self.emission_means[s] = np.average(returns, weights=weight)
            self.emission_stds[s] = np.sqrt(
                np.average((returns - self.emission_means[s])**2, weights=weight)
            ) + 0.001

    def predict_regime(self, returns: np.ndarray) -> np.ndarray:
        """Predict most likely regime sequence."""
        # Viterbi algorithm
        n = len(returns)
        emission_probs = self._compute_emissions(returns)

        # Initialize
        viterbi = np.zeros((n, self.n_states))
        backpointer = np.zeros((n, self.n_states), dtype=int)

        viterbi[0] = np.log(self.initial_probs + 1e-10) + np.log(emission_probs[0] + 1e-10)

        # Forward pass
        log_trans = np.log(self.transition_matrix + 1e-10)

        for t in range(1, n):
            for s in range(self.n_states):
                probs = viterbi[t-1] + log_trans[:, s]
                backpointer[t, s] = np.argmax(probs)
                viterbi[t, s] = probs[backpointer[t, s]] + np.log(emission_probs[t, s] + 1e-10)

        # Backtrack
        path = np.zeros(n, dtype=int)
        path[-1] = np.argmax(viterbi[-1])

        for t in range(n-2, -1, -1):
            path[t] = backpointer[t+1, path[t+1]]

        return path

    def get_regime_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """Get smoothed regime probabilities."""
        _, _, gamma, _ = self._forward_backward(returns)
        return gamma
```

---

## Regime-Based Trading

### Strategy Allocation

```python
class RegimeBasedAllocator:
    """
    Adjust strategy allocation based on detected regime.
    """

    # Regime-specific allocations
    REGIME_ALLOCATIONS = {
        RiskRegime.STRONG_RISK_ON: {
            'equity_beta': 1.2,
            'high_yield': 0.15,
            'emerging_markets': 0.15,
            'safe_havens': 0.0,
            'cash': 0.05,
            'vol_position': 'short'
        },
        RiskRegime.RISK_ON: {
            'equity_beta': 1.0,
            'high_yield': 0.10,
            'emerging_markets': 0.10,
            'safe_havens': 0.05,
            'cash': 0.10,
            'vol_position': 'neutral'
        },
        RiskRegime.NEUTRAL: {
            'equity_beta': 0.7,
            'high_yield': 0.05,
            'emerging_markets': 0.05,
            'safe_havens': 0.10,
            'cash': 0.15,
            'vol_position': 'neutral'
        },
        RiskRegime.RISK_OFF: {
            'equity_beta': 0.4,
            'high_yield': 0.0,
            'emerging_markets': 0.0,
            'safe_havens': 0.20,
            'cash': 0.25,
            'vol_position': 'long'
        },
        RiskRegime.STRONG_RISK_OFF: {
            'equity_beta': 0.2,
            'high_yield': 0.0,
            'emerging_markets': 0.0,
            'safe_havens': 0.30,
            'cash': 0.35,
            'vol_position': 'long'
        },
        RiskRegime.CRISIS: {
            'equity_beta': 0.0,
            'high_yield': 0.0,
            'emerging_markets': 0.0,
            'safe_havens': 0.40,
            'cash': 0.50,
            'vol_position': 'long'
        }
    }

    def __init__(self, transition_speed: float = 0.3):
        """
        Args:
            transition_speed: How quickly to transition (0-1)
                            Lower = smoother transitions
        """
        self.transition_speed = transition_speed
        self.current_allocation = None

    def get_target_allocation(
        self,
        regime_state: RegimeState,
        current_allocation: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Get target allocation for current regime.
        """
        target = self.REGIME_ALLOCATIONS.get(
            regime_state.regime,
            self.REGIME_ALLOCATIONS[RiskRegime.NEUTRAL]
        ).copy()

        # Adjust by confidence
        if regime_state.confidence < 0.7:
            # Blend with neutral allocation
            neutral = self.REGIME_ALLOCATIONS[RiskRegime.NEUTRAL]
            blend_factor = regime_state.confidence

            for key in target:
                if key != 'vol_position':
                    target[key] = (target[key] * blend_factor +
                                  neutral[key] * (1 - blend_factor))

        # Smooth transition if we have current allocation
        if current_allocation and self.current_allocation:
            for key in target:
                if key != 'vol_position':
                    target[key] = (target[key] * self.transition_speed +
                                  current_allocation.get(key, target[key]) * (1 - self.transition_speed))

        self.current_allocation = target
        return target

    def get_rebalance_trades(
        self,
        current_positions: Dict[str, float],
        target_allocation: Dict[str, float],
        portfolio_value: float,
        min_trade_size: float = 1000
    ) -> List[Dict]:
        """
        Generate trades to achieve target allocation.
        """
        trades = []

        # Map allocation categories to actual holdings
        category_mapping = {
            'equity_beta': ['SPY', 'QQQ'],
            'high_yield': ['HYG', 'JNK'],
            'emerging_markets': ['EEM', 'VWO'],
            'safe_havens': ['GLD', 'TLT'],
            'cash': ['SHY', 'BIL']
        }

        for category, target_pct in target_allocation.items():
            if category == 'vol_position':
                continue

            tickers = category_mapping.get(category, [])
            if not tickers:
                continue

            target_value = portfolio_value * target_pct / len(tickers)

            for ticker in tickers:
                current_value = current_positions.get(ticker, 0)
                diff = target_value - current_value

                if abs(diff) > min_trade_size:
                    trades.append({
                        'ticker': ticker,
                        'action': 'BUY' if diff > 0 else 'SELL',
                        'value': abs(diff),
                        'category': category
                    })

        return trades


class RegimeStrategySelector:
    """
    Select trading strategies based on regime.
    """

    REGIME_STRATEGIES = {
        RiskRegime.STRONG_RISK_ON: {
            'preferred': ['momentum', 'growth', 'small_cap', 'emerging'],
            'avoid': ['defensive', 'low_vol', 'value'],
            'leverage': 1.2
        },
        RiskRegime.RISK_ON: {
            'preferred': ['momentum', 'quality_growth'],
            'avoid': ['defensive'],
            'leverage': 1.0
        },
        RiskRegime.NEUTRAL: {
            'preferred': ['diversified', 'factor_balanced'],
            'avoid': [],
            'leverage': 0.8
        },
        RiskRegime.RISK_OFF: {
            'preferred': ['defensive', 'low_vol', 'quality'],
            'avoid': ['momentum', 'small_cap', 'emerging'],
            'leverage': 0.5
        },
        RiskRegime.STRONG_RISK_OFF: {
            'preferred': ['defensive', 'low_vol', 'market_neutral'],
            'avoid': ['momentum', 'leveraged', 'small_cap'],
            'leverage': 0.3
        },
        RiskRegime.CRISIS: {
            'preferred': ['cash', 'short_vol_sellers', 'gold'],
            'avoid': ['all_long_equity'],
            'leverage': 0.0
        }
    }

    def select_strategies(self, regime_state: RegimeState) -> Dict:
        """
        Select appropriate strategies for regime.
        """
        config = self.REGIME_STRATEGIES.get(
            regime_state.regime,
            self.REGIME_STRATEGIES[RiskRegime.NEUTRAL]
        )

        return {
            'active_strategies': config['preferred'],
            'disabled_strategies': config['avoid'],
            'leverage_cap': config['leverage'],
            'confidence': regime_state.confidence,
            'regime': regime_state.regime.value
        }
```

---

## Regime Transition Detection

### Early Warning Signals

```python
class RegimeTransitionDetector:
    """
    Detect early signs of regime transition.
    """

    def __init__(self, sensitivity: float = 0.7):
        """
        Args:
            sensitivity: 0-1, higher = more sensitive to transitions
        """
        self.sensitivity = sensitivity
        self.indicator_history = []

    def check_transition_signals(
        self,
        current_regime: RiskRegime,
        indicators: Dict[str, float]
    ) -> Dict:
        """
        Check for early warning signs of regime transition.
        """
        warnings = []

        # Store for trend analysis
        self.indicator_history.append({
            'timestamp': datetime.now(),
            **indicators
        })

        # Keep last 20 observations
        if len(self.indicator_history) > 20:
            self.indicator_history = self.indicator_history[-20:]

        # Check divergences
        if current_regime in [RiskRegime.RISK_ON, RiskRegime.STRONG_RISK_ON]:
            # Check for risk-off warnings
            warnings.extend(self._check_risk_off_warnings(indicators))

        elif current_regime in [RiskRegime.RISK_OFF, RiskRegime.STRONG_RISK_OFF]:
            # Check for risk-on warnings
            warnings.extend(self._check_risk_on_warnings(indicators))

        # Check for acceleration
        trend_warning = self._check_trend_acceleration()
        if trend_warning:
            warnings.append(trend_warning)

        # Aggregate
        if len(warnings) >= 3:
            transition_signal = 'HIGH'
        elif len(warnings) >= 2:
            transition_signal = 'MEDIUM'
        elif len(warnings) >= 1:
            transition_signal = 'LOW'
        else:
            transition_signal = 'NONE'

        return {
            'transition_signal': transition_signal,
            'warnings': warnings,
            'current_regime': current_regime.value,
            'recommended_action': self._get_recommended_action(transition_signal, current_regime)
        }

    def _check_risk_off_warnings(self, indicators: Dict) -> List[str]:
        """Check for warnings in a risk-on environment."""
        warnings = []

        # VIX rising
        if indicators.get('vix_change_5d', 0) > 0.20:
            warnings.append('VIX rising significantly')

        # Credit spreads widening
        if indicators.get('credit_spread_change', 0) > 20:
            warnings.append('Credit spreads widening')

        # Safe havens rallying
        if indicators.get('gold_return_5d', 0) > 0.02:
            warnings.append('Gold rallying - safe haven demand')

        # EM underperforming
        if indicators.get('em_vs_dm_5d', 0) < -0.02:
            warnings.append('EM underperforming - risk appetite fading')

        return warnings

    def _check_risk_on_warnings(self, indicators: Dict) -> List[str]:
        """Check for warnings in a risk-off environment."""
        warnings = []

        # VIX falling
        if indicators.get('vix_change_5d', 0) < -0.15:
            warnings.append('VIX declining - fear subsiding')

        # Credit spreads tightening
        if indicators.get('credit_spread_change', 0) < -15:
            warnings.append('Credit spreads tightening')

        # Risk assets outperforming
        if indicators.get('junk_vs_treasury_5d', 0) > 0.02:
            warnings.append('High yield outperforming - risk appetite returning')

        return warnings

    def _get_recommended_action(
        self,
        signal: str,
        current_regime: RiskRegime
    ) -> str:
        """Get recommended action based on transition signal."""
        if signal == 'HIGH':
            return 'Immediately adjust allocation toward new regime'
        elif signal == 'MEDIUM':
            return 'Begin gradual allocation shift'
        elif signal == 'LOW':
            return 'Monitor closely, prepare for shift'
        else:
            return 'Maintain current allocation'
```

---

## Performance Analytics

### Regime-Based Performance

```python
class RegimePerformanceAnalyzer:
    """
    Analyze strategy performance across regimes.
    """

    def __init__(self, returns: pd.DataFrame, regimes: pd.Series):
        """
        Args:
            returns: DataFrame of strategy returns
            regimes: Series of regime classifications
        """
        self.returns = returns
        self.regimes = regimes

    def performance_by_regime(self) -> Dict:
        """
        Calculate performance metrics by regime.
        """
        results = {}

        for regime in RiskRegime:
            mask = self.regimes == regime.value
            if mask.sum() == 0:
                continue

            regime_returns = self.returns[mask]

            results[regime.value] = {
                'count_days': mask.sum(),
                'mean_return': regime_returns.mean().to_dict(),
                'volatility': regime_returns.std().to_dict(),
                'sharpe': (regime_returns.mean() / regime_returns.std()).to_dict(),
                'max_drawdown': self._calculate_max_dd(regime_returns).to_dict()
            }

        return results

    def regime_timing_value(self) -> Dict:
        """
        Calculate value added from regime timing.
        """
        # Compare regime-aware vs buy-and-hold
        regime_allocations = RegimeBasedAllocator.REGIME_ALLOCATIONS

        # Calculate regime-aware returns
        regime_aware_returns = []

        for idx, regime in self.regimes.items():
            if regime in regime_allocations:
                beta = regime_allocations[regime]['equity_beta']
            else:
                beta = 0.7

            regime_aware_returns.append(self.returns.loc[idx].mean() * beta)

        regime_aware = pd.Series(regime_aware_returns, index=self.regimes.index)

        # Buy and hold
        buy_hold = self.returns.mean(axis=1)

        return {
            'regime_aware_total_return': (1 + regime_aware).prod() - 1,
            'buy_hold_total_return': (1 + buy_hold).prod() - 1,
            'value_added': ((1 + regime_aware).prod() - 1) - ((1 + buy_hold).prod() - 1),
            'regime_aware_sharpe': regime_aware.mean() / regime_aware.std() * np.sqrt(252),
            'buy_hold_sharpe': buy_hold.mean() / buy_hold.std() * np.sqrt(252)
        }
```

---

## Academic References

1. **Hamilton, J. D. (1989)**. "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*.

2. **Ang, A., & Bekaert, G. (2002)**. "International Asset Allocation with Regime Shifts." *Review of Financial Studies*.

3. **Kritzman, M., Page, S., & Turkington, D. (2012)**. "Regime Shifts: Implications for Dynamic Strategies." *Financial Analysts Journal*.

4. **Guidolin, M., & Timmermann, A. (2007)**. "Asset Allocation Under Multivariate Regime Switching." *Journal of Economic Dynamics and Control*.

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
tags: ["risk-regimes", "hmm", "risk-on-risk-off", "allocation", "regime-detection"]
code_lines: 750
```

---

**END OF DOCUMENT**
