# Strategy Formulation Framework

## Overview

This document provides a comprehensive framework for systematic strategy development across all supported asset classes. The framework integrates with the Cortex engine and NVIDIA AI models for enhanced hypothesis generation and validation.

---

## Supported Asset Classes

| Asset Class | Symbol Format | Data Requirements | Risk Considerations |
|-------------|---------------|-------------------|---------------------|
| **Equities (Stocks)** | AAPL, MSFT | OHLCV, fundamentals | Market risk, sector concentration |
| **Options** | AAPL240119C00150000 | Greeks, IV, chain | Gamma, vega, time decay |
| **Bonds/Fixed Income** | TLT, BND, individual | Yield, duration, credit | Interest rate, credit risk |
| **Crypto** | BTC-USD, ETH-USD | 24/7 OHLCV | Extreme volatility, liquidity |

---

## 1. Strategy Development Lifecycle

### 1.1 Lifecycle Phases

```
Phase 1: IDEATION
├── Market observation
├── Academic research review
├── Edge hypothesis formation
└── NVIDIA Cortex hypothesis generation

Phase 2: SPECIFICATION
├── Entry rules (programmatic)
├── Exit rules (stop, target, time)
├── Position sizing rules
├── Filter conditions
└── Asset class adaptation

Phase 3: VALIDATION
├── Historical backtest
├── Walk-forward analysis
├── Out-of-sample testing
├── Monte Carlo simulation
└── Regime stress testing

Phase 4: OPTIMIZATION
├── Parameter sensitivity
├── Robustness verification
├── Transaction cost analysis
└── Capacity estimation

Phase 5: DEPLOYMENT
├── Paper trading validation
├── Gradual capital allocation
├── Live monitoring setup
└── Kill switch configuration

Phase 6: MONITORING
├── Performance tracking
├── Regime adaptation
├── Decay detection
└── Retirement criteria
```

### 1.2 Strategy Specification Template

```python
@dataclass
class StrategySpecification:
    """
    Complete strategy specification for all asset classes.
    """
    # Identification
    name: str
    version: str
    asset_class: Literal['equities', 'options', 'bonds', 'crypto']
    author: str
    created_date: date

    # Edge Definition
    edge_description: str
    market_inefficiency: str
    theoretical_basis: str
    academic_references: List[str]

    # Universe
    universe_definition: str
    liquidity_requirements: Dict
    sector_constraints: Optional[Dict]

    # Entry Logic
    entry_conditions: List[EntryCondition]
    entry_order_type: str
    entry_timing: str

    # Exit Logic
    stop_loss_method: StopLossSpec
    profit_target_method: Optional[ProfitTargetSpec]
    time_stop: Optional[TimeStopSpec]
    trailing_stop: Optional[TrailingStopSpec]

    # Position Sizing
    sizing_method: SizingMethod
    max_position_pct: float
    correlation_adjustment: bool

    # Filters
    regime_filters: List[RegimeFilter]
    volatility_filters: List[VolatilityFilter]
    calendar_filters: List[CalendarFilter]

    # Risk Limits
    max_drawdown_halt: float
    daily_loss_limit: float
    max_correlated_positions: int

    # Expected Performance
    expected_win_rate: float
    expected_profit_factor: float
    expected_sharpe: float
    expected_max_drawdown: float

    # Validation Requirements
    min_backtest_years: int
    min_trades_required: int
    required_out_of_sample: bool
    required_walk_forward: bool
```

---

## 2. Asset Class Specific Strategies

### 2.1 Equity Strategies

```python
class EquityStrategyFramework:
    """
    Framework for equity (stock) strategies.
    """

    STRATEGY_TYPES = {
        'momentum': {
            'edge': 'Price persistence over intermediate timeframes',
            'typical_holding': '1-3 months',
            'indicators': ['RSI', 'ROC', 'relative_strength'],
            'risks': ['Momentum crashes', 'regime shifts']
        },
        'mean_reversion': {
            'edge': 'Short-term overreaction correction',
            'typical_holding': '1-5 days',
            'indicators': ['RSI_oversold', 'BB_deviation', 'volume_spike'],
            'risks': ['Trending markets', 'fundamental changes']
        },
        'trend_following': {
            'edge': 'Extended price movements in trends',
            'typical_holding': 'Weeks to months',
            'indicators': ['SMA_crossover', 'ADX', 'breakout'],
            'risks': ['Choppy markets', 'whipsaws']
        },
        'value': {
            'edge': 'Mispricing relative to fundamentals',
            'typical_holding': 'Months to years',
            'indicators': ['PE_ratio', 'PB_ratio', 'FCF_yield'],
            'risks': ['Value traps', 'sector rotation']
        },
        'quality': {
            'edge': 'Superior business fundamentals persist',
            'typical_holding': 'Months to years',
            'indicators': ['ROE', 'margins', 'earnings_stability'],
            'risks': ['Valuation compression', 'competition']
        }
    }

    def generate_equity_strategy(
        self,
        strategy_type: str,
        regime: str,
        constraints: Dict
    ) -> StrategySpecification:
        """
        Generate equity strategy specification.
        """
        base = self.STRATEGY_TYPES.get(strategy_type)

        spec = StrategySpecification(
            name=f"{strategy_type}_{regime}",
            asset_class='equities',
            edge_description=base['edge'],
            # ... fill other fields
        )

        return spec


### 2.2 Options Strategies

```python
class OptionsStrategyFramework:
    """
    Framework for options strategies.
    """

    STRATEGY_ARCHETYPES = {
        # Directional
        'long_call': {
            'bias': 'bullish',
            'max_loss': 'premium_paid',
            'ideal_iv': 'low',
            'theta': 'negative'
        },
        'long_put': {
            'bias': 'bearish',
            'max_loss': 'premium_paid',
            'ideal_iv': 'low',
            'theta': 'negative'
        },

        # Premium Selling
        'covered_call': {
            'bias': 'neutral_to_bullish',
            'max_loss': 'stock_downside - premium',
            'ideal_iv': 'high',
            'theta': 'positive'
        },
        'cash_secured_put': {
            'bias': 'neutral_to_bullish',
            'max_loss': 'strike - premium',
            'ideal_iv': 'high',
            'theta': 'positive'
        },
        'iron_condor': {
            'bias': 'neutral',
            'max_loss': 'width - premium',
            'ideal_iv': 'high',
            'theta': 'positive'
        },

        # Volatility
        'long_straddle': {
            'bias': 'neutral (expect move)',
            'max_loss': 'premium_paid',
            'ideal_iv': 'low (before vol expansion)',
            'theta': 'negative'
        },
        'short_straddle': {
            'bias': 'neutral (expect stability)',
            'max_loss': 'unlimited',
            'ideal_iv': 'high (before vol contraction)',
            'theta': 'positive'
        },

        # Spreads
        'bull_call_spread': {
            'bias': 'moderately_bullish',
            'max_loss': 'debit_paid',
            'ideal_iv': 'low_to_moderate',
            'theta': 'depends_on_position'
        },
        'bear_put_spread': {
            'bias': 'moderately_bearish',
            'max_loss': 'debit_paid',
            'ideal_iv': 'low_to_moderate',
            'theta': 'depends_on_position'
        }
    }

    def select_options_strategy(
        self,
        directional_bias: str,
        iv_rank: float,
        expected_move_pct: float,
        risk_tolerance: str
    ) -> str:
        """
        Select appropriate options strategy.
        """
        if iv_rank > 50:  # High IV
            if directional_bias == 'neutral':
                return 'iron_condor'
            elif directional_bias == 'bullish':
                return 'bull_put_spread'  # Credit spread
            else:
                return 'bear_call_spread'
        else:  # Low IV
            if expected_move_pct > 5:
                return 'long_straddle'
            elif directional_bias == 'bullish':
                return 'bull_call_spread'  # Debit spread
            else:
                return 'bear_put_spread'

    def calculate_options_position_size(
        self,
        account_equity: float,
        max_risk_pct: float,
        strategy: str,
        greeks: Dict
    ) -> Dict:
        """
        Position sizing for options based on risk.
        """
        max_risk_dollars = account_equity * max_risk_pct

        strategy_info = self.STRATEGY_ARCHETYPES.get(strategy)

        if 'premium_paid' in strategy_info['max_loss']:
            # Defined risk: max loss = premium
            max_contracts = int(max_risk_dollars / (greeks['premium'] * 100))
        elif 'width' in strategy_info['max_loss']:
            # Spread: max loss = width - premium
            width = greeks.get('spread_width', 5) * 100
            max_loss_per = width - (greeks['premium'] * 100)
            max_contracts = int(max_risk_dollars / max_loss_per)
        else:
            # Undefined risk: use maintenance margin
            margin_required = greeks.get('margin_requirement', 5000)
            max_contracts = int((account_equity * 0.25) / margin_required)

        return {
            'max_contracts': max_contracts,
            'max_risk_dollars': max_risk_dollars,
            'risk_per_contract': max_risk_dollars / max_contracts if max_contracts > 0 else 0
        }


### 2.3 Fixed Income Strategies

```python
class BondStrategyFramework:
    """
    Framework for bond/fixed income strategies.
    Status: PLANNED
    """

    STRATEGY_TYPES = {
        'duration_timing': {
            'edge': 'Interest rate direction prediction',
            'instruments': ['TLT', 'IEF', 'SHY', 'individual_bonds'],
            'key_metrics': ['duration', 'yield', 'fed_funds_rate'],
            'risks': ['Rate shock', 'credit spread widening']
        },
        'credit_rotation': {
            'edge': 'Credit spread mean reversion',
            'instruments': ['LQD', 'HYG', 'JNK', 'corporate_bonds'],
            'key_metrics': ['credit_spread', 'default_rate', 'economic_cycle'],
            'risks': ['Credit events', 'liquidity crisis']
        },
        'yield_curve': {
            'edge': 'Yield curve shape prediction',
            'instruments': ['2Y/10Y spread', 'butterfly trades'],
            'key_metrics': ['curve_slope', 'term_premium'],
            'risks': ['Fed policy surprise', 'curve inversion']
        }
    }

    def calculate_bond_duration_risk(
        self,
        position_value: float,
        modified_duration: float,
        yield_change_bps: float
    ) -> float:
        """
        Calculate price impact of yield change.
        """
        # Price change ~ -Duration * Yield change
        price_change_pct = -modified_duration * (yield_change_bps / 100)
        return position_value * (price_change_pct / 100)


### 2.4 Crypto Strategies (Placeholder)

```python
class CryptoStrategyFramework:
    """
    Framework for cryptocurrency strategies.
    Status: PLACEHOLDER - API integration pending
    """

    STRATEGY_TYPES = {
        'momentum': {
            'edge': 'Strong trend persistence in crypto',
            'typical_holding': 'Days to weeks',
            'considerations': ['24/7 markets', 'extreme volatility', 'correlation spikes']
        },
        'mean_reversion': {
            'edge': 'Overreaction correction',
            'typical_holding': 'Hours to days',
            'considerations': ['Flash crashes', 'liquidity gaps']
        },
        'arbitrage': {
            'edge': 'Cross-exchange price differences',
            'typical_holding': 'Minutes',
            'considerations': ['Transfer times', 'exchange risk']
        }
    }

    RISK_MULTIPLIERS = {
        # Crypto needs different risk parameters
        'position_size_multiplier': 0.25,  # 25% of equity strategy size
        'stop_loss_multiplier': 2.0,       # Wider stops needed
        'max_portfolio_allocation': 0.10   # Max 10% in crypto
    }

    def adapt_equity_strategy_to_crypto(
        self,
        equity_strategy: StrategySpecification
    ) -> StrategySpecification:
        """
        Adapt equity strategy for crypto with appropriate adjustments.
        """
        crypto_strategy = copy.deepcopy(equity_strategy)

        # Adjust for crypto volatility
        crypto_strategy.asset_class = 'crypto'
        crypto_strategy.max_position_pct *= self.RISK_MULTIPLIERS['position_size_multiplier']
        crypto_strategy.stop_loss_method.multiplier *= self.RISK_MULTIPLIERS['stop_loss_multiplier']

        # Add crypto-specific filters
        crypto_strategy.filters.append(
            VolatilityFilter(
                max_24h_volatility=0.15,  # Skip if >15% daily vol
                action='skip_trade'
            )
        )

        return crypto_strategy
```

---

## 3. Signal Generation Framework

### 3.1 Multi-Factor Signal Aggregation

```python
class MultiFactorSignalGenerator:
    """
    Combine multiple factors into trading signals.
    """

    def __init__(self, factors: List[Factor], weights: Dict[str, float] = None):
        self.factors = factors
        self.weights = weights or {f.name: 1.0/len(factors) for f in factors}

    def generate_composite_signal(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> CompositeSignal:
        """
        Generate weighted composite signal.
        """
        factor_scores = {}
        factor_signals = {}

        for factor in self.factors:
            score = factor.calculate(data, symbol)
            signal = factor.generate_signal(score)

            factor_scores[factor.name] = score
            factor_signals[factor.name] = signal

        # Weighted average
        composite_score = sum(
            score * self.weights[name]
            for name, score in factor_scores.items()
        )

        # Direction consensus
        directions = [s.direction for s in factor_signals.values()]
        direction_consensus = max(set(directions), key=directions.count)
        agreement_rate = directions.count(direction_consensus) / len(directions)

        return CompositeSignal(
            symbol=symbol,
            score=composite_score,
            direction=direction_consensus,
            agreement_rate=agreement_rate,
            factor_breakdown=factor_scores,
            confidence=self._calculate_confidence(agreement_rate, composite_score)
        )

    def _calculate_confidence(
        self,
        agreement: float,
        score: float
    ) -> float:
        """
        Calculate signal confidence.
        """
        # High agreement + strong score = high confidence
        base_confidence = agreement * abs(score)

        # Penalize extreme scores (potential overfit)
        if abs(score) > 0.9:
            base_confidence *= 0.8

        return min(base_confidence, 1.0)


### 3.2 Regime-Adaptive Signals

```python
class RegimeAdaptiveSignalGenerator:
    """
    Adjust signal generation based on market regime.
    """

    REGIME_SIGNAL_WEIGHTS = {
        'bull_trending': {
            'momentum': 0.40,
            'trend': 0.35,
            'mean_reversion': 0.10,
            'volatility': 0.15
        },
        'bear_trending': {
            'momentum': 0.25,
            'trend': 0.30,
            'mean_reversion': 0.20,
            'volatility': 0.25
        },
        'sideways': {
            'momentum': 0.15,
            'trend': 0.10,
            'mean_reversion': 0.50,
            'volatility': 0.25
        },
        'high_volatility': {
            'momentum': 0.20,
            'trend': 0.15,
            'mean_reversion': 0.25,
            'volatility': 0.40
        }
    }

    def generate_regime_adaptive_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        current_regime: str
    ) -> Signal:
        """
        Generate signal with regime-appropriate weighting.
        """
        weights = self.REGIME_SIGNAL_WEIGHTS.get(
            current_regime,
            self.REGIME_SIGNAL_WEIGHTS['sideways']
        )

        # Calculate each factor
        momentum_score = self.calculate_momentum(data)
        trend_score = self.calculate_trend(data)
        mean_rev_score = self.calculate_mean_reversion(data)
        vol_score = self.calculate_volatility_signal(data)

        # Weight by regime
        composite = (
            momentum_score * weights['momentum'] +
            trend_score * weights['trend'] +
            mean_rev_score * weights['mean_reversion'] +
            vol_score * weights['volatility']
        )

        return Signal(
            symbol=symbol,
            score=composite,
            direction='LONG' if composite > 0.2 else 'SHORT' if composite < -0.2 else 'NEUTRAL',
            regime=current_regime,
            weights_used=weights
        )
```

---

## 4. Validation Framework

### 4.1 Comprehensive Validation Pipeline

```python
class StrategyValidationPipeline:
    """
    Complete validation pipeline for strategies.
    """

    def __init__(self, data: pd.DataFrame, strategy: Strategy):
        self.data = data
        self.strategy = strategy
        self.results = {}

    def run_full_validation(self) -> ValidationReport:
        """
        Run all validation steps.
        """
        # Step 1: Basic backtest
        backtest_results = self.run_backtest()
        self.results['backtest'] = backtest_results

        # Step 2: Walk-forward
        wf_results = self.run_walk_forward()
        self.results['walk_forward'] = wf_results

        # Step 3: Out-of-sample
        oos_results = self.run_out_of_sample()
        self.results['out_of_sample'] = oos_results

        # Step 4: Monte Carlo
        mc_results = self.run_monte_carlo()
        self.results['monte_carlo'] = mc_results

        # Step 5: Regime analysis
        regime_results = self.run_regime_analysis()
        self.results['regime'] = regime_results

        # Step 6: Parameter sensitivity
        sensitivity_results = self.run_sensitivity_analysis()
        self.results['sensitivity'] = sensitivity_results

        # Compile report
        return self.compile_validation_report()

    def compile_validation_report(self) -> ValidationReport:
        """
        Compile all results into final report.
        """
        # Check minimum criteria
        criteria_passed = self.check_minimum_criteria()

        # Calculate overall score
        overall_score = self.calculate_validation_score()

        return ValidationReport(
            strategy_name=self.strategy.name,
            validation_date=datetime.now(),
            results=self.results,
            criteria_passed=criteria_passed,
            overall_score=overall_score,
            recommendation=self.generate_recommendation(overall_score, criteria_passed),
            ready_for_production=criteria_passed['all_passed'] and overall_score > 70
        )

    def check_minimum_criteria(self) -> Dict[str, bool]:
        """
        Check against minimum validation criteria.
        """
        bt = self.results['backtest']
        oos = self.results['out_of_sample']
        wf = self.results['walk_forward']

        return {
            'min_trades': bt.total_trades >= 100,
            'min_sharpe': bt.sharpe >= 1.0,
            'min_profit_factor': bt.profit_factor >= 1.5,
            'max_drawdown': abs(bt.max_drawdown) <= 0.20,
            'oos_profitable': oos.total_return > 0,
            'wf_stable': wf.degradation < 0.30,
            'all_passed': None  # Set below
        }

    def calculate_validation_score(self) -> float:
        """
        Calculate overall validation score (0-100).
        """
        bt = self.results['backtest']
        oos = self.results['out_of_sample']
        mc = self.results['monte_carlo']

        score = 0

        # Sharpe contribution (0-30)
        score += min(30, bt.sharpe * 15)

        # Profit factor contribution (0-20)
        score += min(20, (bt.profit_factor - 1) * 20)

        # OOS performance (0-20)
        if oos.sharpe > 0:
            score += min(20, oos.sharpe * 10)

        # Monte Carlo robustness (0-15)
        if mc.p5_equity > mc.initial_equity:
            score += 15
        elif mc.p5_equity > mc.initial_equity * 0.9:
            score += 10

        # Drawdown (0-15)
        dd_penalty = abs(bt.max_drawdown) * 75
        score += max(0, 15 - dd_penalty)

        return min(100, score)
```

---

## 5. Production Deployment

### 5.1 Deployment Checklist

```python
DEPLOYMENT_CHECKLIST = {
    'validation': [
        'All minimum criteria passed',
        'Walk-forward analysis completed',
        'Out-of-sample test passed',
        'Monte Carlo shows robustness',
        'Regime analysis reviewed'
    ],
    'risk_management': [
        'Stop loss method defined',
        'Position sizing configured',
        'Max drawdown halt set',
        'Daily loss limit set',
        'Kill switch configured'
    ],
    'infrastructure': [
        'Data feed connected',
        'Broker API authenticated',
        'Order routing tested',
        'Monitoring dashboard active',
        'Alert notifications configured'
    ],
    'documentation': [
        'Strategy specification complete',
        'Trade log initialized',
        'Performance tracking setup',
        'Decay criteria defined'
    ]
}


### 5.2 Capital Allocation Ramp

```python
class CapitalAllocationRamp:
    """
    Gradual capital allocation for new strategies.
    """

    RAMP_SCHEDULE = {
        'week_1': 0.10,    # 10% of target allocation
        'week_2': 0.25,    # 25%
        'month_1': 0.50,   # 50%
        'month_2': 0.75,   # 75%
        'month_3': 1.00    # Full allocation
    }

    def __init__(
        self,
        target_allocation: float,
        start_date: date,
        performance_gates: Dict = None
    ):
        self.target = target_allocation
        self.start = start_date
        self.gates = performance_gates or {
            'min_trades': 10,
            'min_win_rate': 0.35,
            'max_drawdown': 0.10
        }

    def get_current_allocation(
        self,
        current_date: date,
        live_performance: Dict
    ) -> float:
        """
        Get allowed allocation based on time and performance.
        """
        # Time-based ramp
        days_live = (current_date - self.start).days
        if days_live < 7:
            time_allocation = self.RAMP_SCHEDULE['week_1']
        elif days_live < 14:
            time_allocation = self.RAMP_SCHEDULE['week_2']
        elif days_live < 30:
            time_allocation = self.RAMP_SCHEDULE['month_1']
        elif days_live < 60:
            time_allocation = self.RAMP_SCHEDULE['month_2']
        else:
            time_allocation = self.RAMP_SCHEDULE['month_3']

        # Performance gates can reduce allocation
        performance_multiplier = 1.0

        if live_performance.get('drawdown', 0) < -self.gates['max_drawdown']:
            performance_multiplier = 0.5  # Reduce by half

        if live_performance.get('trades', 0) < self.gates['min_trades']:
            # Not enough data, stay conservative
            performance_multiplier = min(performance_multiplier, 0.75)

        return self.target * time_allocation * performance_multiplier
```

---

## 6. Strategy Decay & Retirement

### 6.1 Decay Detection

```python
class StrategyDecayDetector:
    """
    Detect strategy performance decay.
    """

    DECAY_THRESHOLDS = {
        'sharpe_decline': 0.50,      # Sharpe dropped by 50%
        'win_rate_decline': 0.20,    # Win rate dropped 20 percentage points
        'drawdown_exceeded': 0.25,   # Exceeded expected max DD by 25%
        'consecutive_losses': 10,    # 10 consecutive losses
        'underwater_days': 90        # 90 days in drawdown
    }

    def check_decay(
        self,
        backtest_metrics: Dict,
        live_metrics: Dict,
        rolling_window: int = 60  # days
    ) -> DecayAssessment:
        """
        Compare live performance to backtest expectations.
        """
        decay_signals = []

        # Sharpe decay
        bt_sharpe = backtest_metrics.get('sharpe', 1.5)
        live_sharpe = live_metrics.get('rolling_sharpe', 1.5)
        if live_sharpe < bt_sharpe * (1 - self.DECAY_THRESHOLDS['sharpe_decline']):
            decay_signals.append(DecaySignal(
                type='sharpe_decay',
                severity='HIGH',
                bt_value=bt_sharpe,
                live_value=live_sharpe
            ))

        # Win rate decay
        bt_wr = backtest_metrics.get('win_rate', 0.5)
        live_wr = live_metrics.get('rolling_win_rate', 0.5)
        if live_wr < bt_wr - self.DECAY_THRESHOLDS['win_rate_decline']:
            decay_signals.append(DecaySignal(
                type='win_rate_decay',
                severity='MEDIUM',
                bt_value=bt_wr,
                live_value=live_wr
            ))

        # Drawdown exceeded
        bt_dd = abs(backtest_metrics.get('max_drawdown', 0.15))
        live_dd = abs(live_metrics.get('current_drawdown', 0))
        if live_dd > bt_dd * (1 + self.DECAY_THRESHOLDS['drawdown_exceeded']):
            decay_signals.append(DecaySignal(
                type='drawdown_exceeded',
                severity='CRITICAL',
                bt_value=bt_dd,
                live_value=live_dd
            ))

        return DecayAssessment(
            signals=decay_signals,
            decay_detected=len(decay_signals) > 0,
            severity=max([s.severity for s in decay_signals], default='NONE'),
            recommendation=self._generate_recommendation(decay_signals)
        )

    def _generate_recommendation(self, signals: List[DecaySignal]) -> str:
        """
        Generate action recommendation based on decay signals.
        """
        if not signals:
            return 'CONTINUE'

        severities = [s.severity for s in signals]

        if 'CRITICAL' in severities:
            return 'HALT_AND_REVIEW'
        elif severities.count('HIGH') >= 2:
            return 'REDUCE_ALLOCATION_50PCT'
        elif 'HIGH' in severities:
            return 'REDUCE_ALLOCATION_25PCT'
        else:
            return 'MONITOR_CLOSELY'
```

---

## Academic References

1. **Lopez de Prado, M. (2018)**: "Advances in Financial Machine Learning"
2. **Chan, E. (2009)**: "Quantitative Trading"
3. **Aronson, D. (2006)**: "Evidence-Based Technical Analysis"
4. **Bailey & Lopez de Prado (2014)**: "The Deflated Sharpe Ratio"
5. **Harvey et al. (2016)**: "...and the Cross-Section of Expected Returns"
