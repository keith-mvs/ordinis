# Advanced Risk Management Methods

## Overview

This document extends the core risk management framework with quantitative methods for sophisticated risk measurement, dynamic adjustment, and NVIDIA model integration.

---

## 1. Correlation Risk Management

### 1.1 Correlation Matrix Calculation

```python
import numpy as np
import pandas as pd
from scipy import stats

class CorrelationRiskEngine:
    """
    Real-time correlation tracking and risk management.
    """

    def __init__(
        self,
        lookback_days: int = 60,
        update_frequency: str = 'daily',
        correlation_threshold: float = 0.70
    ):
        self.lookback = lookback_days
        self.update_freq = update_frequency
        self.threshold = correlation_threshold
        self.correlation_matrix: pd.DataFrame = None
        self.last_update: datetime = None

    def calculate_correlation_matrix(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate rolling correlation matrix from returns.

        Args:
            returns: DataFrame with symbol columns, date index

        Returns:
            Correlation matrix (n_symbols x n_symbols)
        """
        # Use recent data only
        recent_returns = returns.tail(self.lookback)

        # Pearson correlation
        corr_matrix = recent_returns.corr(method='pearson')

        self.correlation_matrix = corr_matrix
        self.last_update = datetime.now()

        return corr_matrix

    def get_correlated_pairs(
        self,
        threshold: float = None
    ) -> List[Tuple[str, str, float]]:
        """
        Find all pairs with correlation above threshold.
        """
        if threshold is None:
            threshold = self.threshold

        pairs = []
        symbols = self.correlation_matrix.columns.tolist()

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                corr = self.correlation_matrix.loc[sym1, sym2]
                if abs(corr) >= threshold:
                    pairs.append((sym1, sym2, corr))

        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    def calculate_portfolio_correlation_risk(
        self,
        positions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level correlation exposure.

        Returns:
            - avg_correlation: Average pairwise correlation
            - correlation_concentration: Measure of correlated exposure
            - diversification_ratio: DR = weighted vol / portfolio vol
        """
        symbols = list(positions.keys())
        weights = np.array([positions[s] for s in symbols])
        weights = weights / weights.sum()  # Normalize

        # Extract relevant correlations
        corr_sub = self.correlation_matrix.loc[symbols, symbols].values

        # Average pairwise correlation (excluding diagonal)
        n = len(symbols)
        mask = ~np.eye(n, dtype=bool)
        avg_corr = corr_sub[mask].mean()

        # Correlation concentration (weighted)
        weighted_corr = 0
        total_weight = 0
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i != j:
                    pair_weight = weights[i] * weights[j]
                    weighted_corr += pair_weight * abs(corr_sub[i, j])
                    total_weight += pair_weight

        corr_concentration = weighted_corr / total_weight if total_weight > 0 else 0

        return {
            'avg_pairwise_correlation': avg_corr,
            'correlation_concentration': corr_concentration,
            'highly_correlated_pairs': len(self.get_correlated_pairs())
        }


### 1.2 Correlation-Adjusted Position Sizing

```python
def correlation_adjusted_size(
    base_size: float,
    new_symbol: str,
    existing_positions: Dict[str, float],
    correlation_matrix: pd.DataFrame,
    max_correlated_exposure: float = 0.40
) -> float:
    """
    Reduce position size based on correlation with existing holdings.

    Logic: If new position is highly correlated with existing,
    reduce size to limit correlated exposure.
    """
    if not existing_positions:
        return base_size

    # Calculate correlation with existing positions
    max_corr = 0
    total_correlated_weight = 0

    for symbol, weight in existing_positions.items():
        if symbol in correlation_matrix.columns and new_symbol in correlation_matrix.columns:
            corr = abs(correlation_matrix.loc[new_symbol, symbol])
            max_corr = max(max_corr, corr)
            if corr > 0.5:  # Moderate correlation threshold
                total_correlated_weight += weight * corr

    # Adjustment factor
    if total_correlated_weight > max_correlated_exposure:
        # Already have significant correlated exposure
        adjustment = max(0.25, 1 - (total_correlated_weight - max_correlated_exposure))
    elif max_corr > 0.8:
        # Very high correlation with a single position
        adjustment = 0.5
    elif max_corr > 0.7:
        adjustment = 0.75
    else:
        adjustment = 1.0

    return base_size * adjustment


### 1.3 Dynamic Correlation Monitoring

```python
class CorrelationRegimeDetector:
    """
    Detect changes in correlation structure.
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 60,
        change_threshold: float = 0.15
    ):
        self.short = short_window
        self.long = long_window
        self.threshold = change_threshold

    def detect_correlation_shift(
        self,
        returns: pd.DataFrame,
        symbol1: str,
        symbol2: str
    ) -> Dict:
        """
        Detect if correlation regime is shifting.
        """
        short_corr = returns[[symbol1, symbol2]].tail(self.short).corr().iloc[0, 1]
        long_corr = returns[[symbol1, symbol2]].tail(self.long).corr().iloc[0, 1]

        change = short_corr - long_corr

        return {
            'short_term_correlation': short_corr,
            'long_term_correlation': long_corr,
            'correlation_change': change,
            'regime_shift': abs(change) > self.threshold,
            'direction': 'INCREASING' if change > 0 else 'DECREASING'
        }

    def portfolio_correlation_shift(
        self,
        returns: pd.DataFrame,
        symbols: List[str]
    ) -> Dict:
        """
        Detect portfolio-wide correlation shifts.
        """
        short_corr = returns[symbols].tail(self.short).corr()
        long_corr = returns[symbols].tail(self.long).corr()

        # Average correlation change (excluding diagonal)
        n = len(symbols)
        mask = ~np.eye(n, dtype=bool)

        short_avg = short_corr.values[mask].mean()
        long_avg = long_corr.values[mask].mean()

        return {
            'short_term_avg_correlation': short_avg,
            'long_term_avg_correlation': long_avg,
            'portfolio_correlation_shift': short_avg - long_avg,
            'alert': abs(short_avg - long_avg) > self.threshold
        }
```

---

## 2. Tail Risk & Extreme Events

### 2.1 Value at Risk (VaR) Methods

```python
from scipy import stats
from typing import Literal

class VaRCalculator:
    """
    Multiple VaR calculation methods.
    """

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def historical_var(
        self,
        returns: pd.Series,
        horizon_days: int = 1
    ) -> float:
        """
        Historical simulation VaR.
        Non-parametric - uses actual return distribution.
        """
        if horizon_days > 1:
            # Multi-day VaR using rolling returns
            rolling_returns = returns.rolling(horizon_days).sum().dropna()
            return rolling_returns.quantile(1 - self.confidence)

        return returns.quantile(1 - self.confidence)

    def parametric_var(
        self,
        returns: pd.Series,
        horizon_days: int = 1
    ) -> float:
        """
        Parametric (Gaussian) VaR.
        Assumes normal distribution.
        """
        mu = returns.mean()
        sigma = returns.std()

        z_score = stats.norm.ppf(1 - self.confidence)

        # Scale for horizon
        mu_horizon = mu * horizon_days
        sigma_horizon = sigma * np.sqrt(horizon_days)

        return mu_horizon + z_score * sigma_horizon

    def cornish_fisher_var(
        self,
        returns: pd.Series,
        horizon_days: int = 1
    ) -> float:
        """
        Cornish-Fisher VaR.
        Adjusts for skewness and kurtosis.
        """
        mu = returns.mean()
        sigma = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()  # Excess kurtosis

        z = stats.norm.ppf(1 - self.confidence)

        # Cornish-Fisher expansion
        z_cf = (z +
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)

        # Scale for horizon
        mu_horizon = mu * horizon_days
        sigma_horizon = sigma * np.sqrt(horizon_days)

        return mu_horizon + z_cf * sigma_horizon

    def calculate_all_var(
        self,
        returns: pd.Series,
        horizon_days: int = 1
    ) -> Dict[str, float]:
        """
        Calculate VaR using all methods for comparison.
        """
        return {
            'historical_var': self.historical_var(returns, horizon_days),
            'parametric_var': self.parametric_var(returns, horizon_days),
            'cornish_fisher_var': self.cornish_fisher_var(returns, horizon_days),
            'confidence': self.confidence,
            'horizon_days': horizon_days
        }


### 2.2 Expected Shortfall (CVaR)

```python
class ExpectedShortfall:
    """
    Conditional Value at Risk - average loss beyond VaR.
    More coherent risk measure than VaR.
    """

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def historical_es(
        self,
        returns: pd.Series
    ) -> float:
        """
        Historical Expected Shortfall.
        """
        var = returns.quantile(1 - self.confidence)
        tail_losses = returns[returns <= var]
        return tail_losses.mean()

    def parametric_es(
        self,
        returns: pd.Series
    ) -> float:
        """
        Parametric ES assuming normal distribution.
        """
        mu = returns.mean()
        sigma = returns.std()

        z = stats.norm.ppf(1 - self.confidence)

        # ES for normal distribution
        es_standard = -stats.norm.pdf(z) / (1 - self.confidence)

        return mu + sigma * es_standard

    def marginal_es(
        self,
        portfolio_returns: pd.Series,
        asset_returns: pd.Series,
        asset_weight: float
    ) -> float:
        """
        Marginal contribution of asset to portfolio ES.
        """
        portfolio_var = portfolio_returns.quantile(1 - self.confidence)

        # Find dates where portfolio is in tail
        tail_dates = portfolio_returns[portfolio_returns <= portfolio_var].index

        # Asset's contribution on those dates
        asset_tail_contribution = asset_returns.loc[tail_dates].mean()

        return asset_tail_contribution * asset_weight


### 2.3 Stress Testing Framework

```python
class StressTestEngine:
    """
    Scenario-based stress testing.
    """

    HISTORICAL_SCENARIOS = {
        'black_monday_1987': {'equity': -0.226, 'vol_spike': 3.0},
        'asian_crisis_1997': {'equity': -0.07, 'emerging': -0.30},
        'dot_com_crash_2000': {'tech': -0.40, 'equity': -0.15},
        'gfc_2008': {'equity': -0.50, 'credit': -0.30, 'vol_spike': 4.0},
        'flash_crash_2010': {'equity': -0.09, 'duration': '15min'},
        'covid_crash_2020': {'equity': -0.34, 'vol_spike': 5.0, 'duration': '23days'},
        'rate_shock_2022': {'equity': -0.25, 'bonds': -0.15},
    }

    def __init__(self, portfolio: Dict[str, float]):
        self.portfolio = portfolio

    def historical_scenario_test(
        self,
        scenario_name: str,
        sector_exposures: Dict[str, float]
    ) -> Dict:
        """
        Apply historical scenario to portfolio.
        """
        scenario = self.HISTORICAL_SCENARIOS.get(scenario_name)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        total_impact = 0
        impacts = {}

        # Apply scenario shocks
        if 'equity' in scenario:
            equity_impact = scenario['equity'] * sum(sector_exposures.values())
            total_impact += equity_impact
            impacts['equity'] = equity_impact

        if 'tech' in scenario and 'technology' in sector_exposures:
            tech_impact = scenario['tech'] * sector_exposures.get('technology', 0)
            total_impact += tech_impact
            impacts['tech'] = tech_impact

        return {
            'scenario': scenario_name,
            'total_impact': total_impact,
            'component_impacts': impacts,
            'survives': total_impact > -0.50  # Example survival threshold
        }

    def hypothetical_scenario_test(
        self,
        shocks: Dict[str, float],
        correlations_spike: bool = True
    ) -> Dict:
        """
        Custom hypothetical scenario.

        Args:
            shocks: Dict of factor shocks (e.g., {'SPY': -0.15, 'TLT': 0.05})
            correlations_spike: Assume correlations increase in stress
        """
        if correlations_spike:
            # In stress, correlations typically go to 1
            correlation_adjustment = 1.3  # 30% worse due to correlation
        else:
            correlation_adjustment = 1.0

        total_impact = 0
        for symbol, shock in shocks.items():
            position_weight = self.portfolio.get(symbol, 0)
            impact = shock * position_weight * correlation_adjustment
            total_impact += impact

        return {
            'custom_scenario': shocks,
            'correlation_adjusted': correlations_spike,
            'total_impact': total_impact,
            'portfolio_survives': total_impact > -0.50
        }

    def reverse_stress_test(
        self,
        target_loss: float = -0.25
    ) -> Dict:
        """
        Find scenarios that would cause target_loss.
        """
        # What uniform shock causes target loss?
        total_exposure = sum(abs(w) for w in self.portfolio.values())
        required_shock = target_loss / total_exposure

        # What if only top position is hit?
        top_position = max(self.portfolio.items(), key=lambda x: abs(x[1]))
        single_stock_shock = target_loss / abs(top_position[1])

        return {
            'target_loss': target_loss,
            'uniform_shock_required': required_shock,
            'single_stock_shock': {
                'symbol': top_position[0],
                'required_move': single_stock_shock
            }
        }
```

---

## 3. Dynamic Risk Adjustment

### 3.1 Regime-Based Risk Limits

```python
class DynamicRiskLimits:
    """
    Adjust risk limits based on market regime.
    """

    REGIME_LIMITS = {
        'low_volatility': {
            'risk_per_trade': 0.015,      # Can take more risk
            'max_positions': 12,
            'max_portfolio_heat': 0.08,
            'stop_multiplier': 1.5        # Tighter stops
        },
        'normal': {
            'risk_per_trade': 0.01,
            'max_positions': 10,
            'max_portfolio_heat': 0.06,
            'stop_multiplier': 2.0
        },
        'high_volatility': {
            'risk_per_trade': 0.0075,     # Reduce risk
            'max_positions': 6,
            'max_portfolio_heat': 0.04,
            'stop_multiplier': 2.5        # Wider stops
        },
        'crisis': {
            'risk_per_trade': 0.005,
            'max_positions': 3,
            'max_portfolio_heat': 0.02,
            'stop_multiplier': 3.0
        }
    }

    def __init__(self, base_limits: Dict):
        self.base = base_limits

    def get_adjusted_limits(
        self,
        current_regime: str,
        volatility_percentile: float,
        drawdown: float
    ) -> Dict:
        """
        Get risk limits adjusted for current conditions.
        """
        # Start with regime-based limits
        limits = self.REGIME_LIMITS.get(current_regime, self.REGIME_LIMITS['normal']).copy()

        # Further adjust for drawdown
        if drawdown < -0.10:
            limits['risk_per_trade'] *= 0.5
            limits['max_positions'] = max(3, limits['max_positions'] // 2)
        elif drawdown < -0.05:
            limits['risk_per_trade'] *= 0.75

        # Adjust for volatility
        if volatility_percentile > 0.90:
            limits['risk_per_trade'] *= 0.75
            limits['stop_multiplier'] *= 1.25

        return limits


### 3.2 Adaptive Position Sizing

```python
class AdaptivePositionSizer:
    """
    Position sizing that adapts to market conditions.
    """

    def __init__(
        self,
        base_risk_pct: float = 0.01,
        max_adjustment: float = 2.0,
        min_adjustment: float = 0.25
    ):
        self.base_risk = base_risk_pct
        self.max_adj = max_adjustment
        self.min_adj = min_adjustment

    def calculate_adaptive_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        conditions: Dict
    ) -> Dict:
        """
        Calculate position size with adaptive adjustments.
        """
        # Base size
        risk_amount = equity * self.base_risk
        risk_per_share = abs(entry_price - stop_price)
        base_shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

        # Calculate adjustment multiplier
        multiplier = 1.0
        adjustments = []

        # Volatility adjustment (inverse)
        vol_percentile = conditions.get('volatility_percentile', 0.5)
        if vol_percentile > 0.8:
            vol_adj = 0.6
            adjustments.append(f"High vol ({vol_percentile:.0%}): 0.6x")
        elif vol_percentile > 0.6:
            vol_adj = 0.8
            adjustments.append(f"Elevated vol ({vol_percentile:.0%}): 0.8x")
        elif vol_percentile < 0.2:
            vol_adj = 1.25
            adjustments.append(f"Low vol ({vol_percentile:.0%}): 1.25x")
        else:
            vol_adj = 1.0
        multiplier *= vol_adj

        # Drawdown adjustment
        drawdown = conditions.get('drawdown', 0)
        if drawdown < -0.15:
            dd_adj = 0.5
            adjustments.append(f"Drawdown ({drawdown:.1%}): 0.5x")
        elif drawdown < -0.10:
            dd_adj = 0.75
            adjustments.append(f"Drawdown ({drawdown:.1%}): 0.75x")
        else:
            dd_adj = 1.0
        multiplier *= dd_adj

        # Winning streak boost (cautious)
        win_streak = conditions.get('winning_streak', 0)
        if win_streak >= 5:
            streak_adj = 1.15
            adjustments.append(f"Win streak ({win_streak}): 1.15x")
        else:
            streak_adj = 1.0
        multiplier *= streak_adj

        # Losing streak reduction
        lose_streak = conditions.get('losing_streak', 0)
        if lose_streak >= 3:
            streak_adj = 0.75
            adjustments.append(f"Lose streak ({lose_streak}): 0.75x")
            multiplier *= streak_adj

        # Signal strength adjustment
        signal_strength = conditions.get('signal_strength', 0.5)
        if signal_strength > 0.8:
            sig_adj = 1.2
            adjustments.append(f"Strong signal ({signal_strength:.1%}): 1.2x")
        elif signal_strength < 0.5:
            sig_adj = 0.75
            adjustments.append(f"Weak signal ({signal_strength:.1%}): 0.75x")
        else:
            sig_adj = 1.0
        multiplier *= sig_adj

        # Clamp multiplier
        multiplier = max(self.min_adj, min(self.max_adj, multiplier))

        adjusted_shares = int(base_shares * multiplier)

        return {
            'base_shares': base_shares,
            'multiplier': multiplier,
            'adjusted_shares': adjusted_shares,
            'adjustments': adjustments,
            'effective_risk': (adjusted_shares * risk_per_share) / equity
        }
```

---

## 4. Multi-Timeframe Risk Management

### 4.1 Timeframe Risk Hierarchy

```python
class MultiTimeframeRisk:
    """
    Risk management across multiple timeframes.
    """

    def __init__(self):
        self.timeframes = {
            'intraday': {
                'max_loss': 0.01,          # 1% max intraday loss
                'position_limit': 5,
                'reset': 'daily'
            },
            'weekly': {
                'max_loss': 0.03,          # 3% max weekly loss
                'position_limit': 10,
                'reset': 'weekly'
            },
            'monthly': {
                'max_loss': 0.08,          # 8% max monthly loss
                'position_limit': 15,
                'reset': 'monthly'
            },
            'annual': {
                'max_loss': 0.20,          # 20% max annual drawdown
                'position_limit': 20,
                'reset': 'yearly'
            }
        }

    def check_all_timeframes(
        self,
        pnl_history: pd.Series,
        current_equity: float,
        peak_equity: float
    ) -> Dict:
        """
        Check risk limits across all timeframes.
        """
        results = {}

        for tf_name, limits in self.timeframes.items():
            tf_pnl = self._get_timeframe_pnl(pnl_history, tf_name)
            tf_pnl_pct = tf_pnl / peak_equity

            results[tf_name] = {
                'pnl': tf_pnl,
                'pnl_pct': tf_pnl_pct,
                'limit': limits['max_loss'],
                'breached': tf_pnl_pct <= -limits['max_loss'],
                'utilization': abs(tf_pnl_pct) / limits['max_loss']
            }

        return results

    def _get_timeframe_pnl(
        self,
        pnl_history: pd.Series,
        timeframe: str
    ) -> float:
        """
        Calculate PnL for specific timeframe.
        """
        now = datetime.now()

        if timeframe == 'intraday':
            start = now.replace(hour=9, minute=30, second=0)
        elif timeframe == 'weekly':
            start = now - timedelta(days=now.weekday())
        elif timeframe == 'monthly':
            start = now.replace(day=1)
        elif timeframe == 'annual':
            start = now.replace(month=1, day=1)

        return pnl_history[pnl_history.index >= start].sum()
```

---

## 5. Liquidity Risk Management

### 5.1 Liquidity Scoring

```python
class LiquidityRiskManager:
    """
    Assess and manage liquidity risk.
    """

    def __init__(
        self,
        min_adv_multiple: float = 0.01,  # Position < 1% of ADV
        max_spread_bps: float = 20       # Max 20 bps spread
    ):
        self.min_adv = min_adv_multiple
        self.max_spread = max_spread_bps

    def calculate_liquidity_score(
        self,
        symbol: str,
        market_data: Dict
    ) -> float:
        """
        Score liquidity 0-1 (1 = highly liquid).
        """
        adv = market_data.get('avg_daily_volume', 0)
        spread = market_data.get('bid_ask_spread_pct', 0.01)
        price = market_data.get('price', 0)

        # Volume score
        adv_dollar = adv * price
        if adv_dollar > 100_000_000:
            vol_score = 1.0
        elif adv_dollar > 10_000_000:
            vol_score = 0.8
        elif adv_dollar > 1_000_000:
            vol_score = 0.6
        else:
            vol_score = 0.3

        # Spread score
        spread_bps = spread * 10000
        if spread_bps < 5:
            spread_score = 1.0
        elif spread_bps < 10:
            spread_score = 0.8
        elif spread_bps < 20:
            spread_score = 0.6
        elif spread_bps < 50:
            spread_score = 0.3
        else:
            spread_score = 0.1

        return (vol_score * 0.6 + spread_score * 0.4)

    def max_position_by_liquidity(
        self,
        symbol: str,
        price: float,
        adv: int,
        participation_rate: float = 0.05
    ) -> Dict:
        """
        Calculate maximum position size based on liquidity.
        """
        # Max shares based on ADV participation
        max_shares_adv = int(adv * participation_rate)

        # Max position value
        max_value = max_shares_adv * price

        # Estimated market impact
        impact_pct = self._estimate_market_impact(max_shares_adv, adv)

        return {
            'max_shares': max_shares_adv,
            'max_value': max_value,
            'participation_rate': participation_rate,
            'estimated_impact_pct': impact_pct
        }

    def _estimate_market_impact(
        self,
        order_shares: int,
        adv: int
    ) -> float:
        """
        Estimate market impact using square-root model.
        Impact ~ 0.1 * sqrt(order_size / ADV)
        """
        participation = order_shares / adv if adv > 0 else 1
        impact = 0.10 * np.sqrt(participation)
        return impact


### 5.2 Liquidity-Adjusted Position Sizing

```python
def liquidity_adjusted_position(
    base_shares: int,
    liquidity_score: float,
    urgency: str = 'normal'
) -> Dict:
    """
    Adjust position size for liquidity constraints.
    """
    # Urgency affects how much we're willing to pay for liquidity
    urgency_multiplier = {
        'low': 0.7,      # Patient, wait for liquidity
        'normal': 1.0,
        'high': 1.3      # Accept more impact for speed
    }.get(urgency, 1.0)

    # Liquidity adjustment
    if liquidity_score > 0.8:
        liq_adj = 1.0    # Highly liquid, no adjustment
    elif liquidity_score > 0.6:
        liq_adj = 0.8
    elif liquidity_score > 0.4:
        liq_adj = 0.5
    else:
        liq_adj = 0.25   # Illiquid, significant reduction

    adjusted_shares = int(base_shares * liq_adj * urgency_multiplier)

    return {
        'base_shares': base_shares,
        'liquidity_score': liquidity_score,
        'liquidity_adjustment': liq_adj,
        'urgency_adjustment': urgency_multiplier,
        'adjusted_shares': adjusted_shares
    }
```

---

## 6. Risk Metrics Dashboard

### 6.1 Comprehensive Risk Report

```python
@dataclass
class RiskDashboard:
    """
    Comprehensive risk metrics for monitoring.
    """
    # Position-level
    total_positions: int
    largest_position_pct: float
    avg_position_size_pct: float

    # Exposure
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float

    # Correlation
    avg_pairwise_correlation: float
    max_pairwise_correlation: float
    correlation_concentration: float

    # Tail risk
    var_95_1day: float
    es_95_1day: float
    var_99_1day: float

    # Drawdown
    current_drawdown: float
    max_drawdown: float
    drawdown_duration_days: int

    # Portfolio heat
    portfolio_heat: float
    available_risk_budget: float

    # Liquidity
    avg_liquidity_score: float
    min_liquidity_score: float
    days_to_liquidate: float

    # Regime
    volatility_regime: str
    market_regime: str

    def risk_score(self) -> float:
        """
        Overall risk score 0-100 (higher = riskier).
        """
        score = 0

        # Drawdown component (0-30 points)
        score += min(30, abs(self.current_drawdown) * 150)

        # VaR component (0-20 points)
        score += min(20, abs(self.var_95_1day) * 400)

        # Correlation component (0-20 points)
        score += self.avg_pairwise_correlation * 20

        # Concentration component (0-15 points)
        score += self.largest_position_pct * 75

        # Liquidity component (0-15 points)
        score += (1 - self.avg_liquidity_score) * 15

        return min(100, score)
```

---

## Academic References

1. **Jorion, P. (2006)**: "Value at Risk" - Comprehensive VaR methodology
2. **McNeil, A., Frey, R., Embrechts, P. (2015)**: "Quantitative Risk Management" - Advanced risk measures
3. **Artzner et al. (1999)**: "Coherent Measures of Risk" - ES vs VaR
4. **Cont, R. (2001)**: "Empirical Properties of Asset Returns" - Fat tails
5. **Ang, A. & Chen, J. (2002)**: "Asymmetric Correlations of Equity Portfolios" - Correlation dynamics
6. **Kyle, A. (1985)**: "Continuous Auctions and Insider Trading" - Market impact
7. **Almgren, R. & Chriss, N. (2001)**: "Optimal Execution of Portfolio Transactions" - Execution risk

---

## Integration with RiskGuard Engine

These methods integrate with the existing RiskGuard engine at `src/engines/riskguard/`:

```python
# Example integration point
from engines.riskguard.core.engine import RiskGuardEngine

class EnhancedRiskGuard(RiskGuardEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correlation_engine = CorrelationRiskEngine()
        self.var_calculator = VaRCalculator()
        self.stress_tester = StressTestEngine({})
        self.adaptive_sizer = AdaptivePositionSizer()

    def evaluate_signal_enhanced(
        self,
        signal: Signal,
        proposed_trade: ProposedTrade,
        portfolio_state: PortfolioState,
        market_conditions: Dict
    ) -> Tuple[bool, List[RiskCheckResult], ProposedTrade]:
        """
        Enhanced evaluation with advanced risk methods.
        """
        # Standard rule checks
        passed, results, adjusted = super().evaluate_signal(
            signal, proposed_trade, portfolio_state
        )

        if not passed:
            return passed, results, adjusted

        # Additional correlation check
        corr_risk = self.correlation_engine.calculate_portfolio_correlation_risk(
            portfolio_state.positions
        )
        if corr_risk['correlation_concentration'] > 0.50:
            results.append(RiskCheckResult(
                rule_name='correlation_concentration',
                passed=False,
                current_value=corr_risk['correlation_concentration'],
                threshold=0.50
            ))
            passed = False

        # VaR check
        var_95 = self.var_calculator.historical_var(portfolio_state.returns)
        if abs(var_95) > 0.03:  # 3% VaR limit
            results.append(RiskCheckResult(
                rule_name='var_95_limit',
                passed=False,
                current_value=var_95,
                threshold=-0.03
            ))
            passed = False

        return passed, results, adjusted
```
