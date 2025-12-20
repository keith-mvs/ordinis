# Earnings Volatility Trading

## Overview

Earnings announcements create predictable volatility spikes, offering systematic options trading opportunities. This document covers volatility-based strategies including pre-earnings straddles, IV crush exploitation, and earnings volatility forecasting.

---

## Implied Volatility Dynamics

### Earnings IV Lifecycle

```python
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class EarningsIVProfile:
    """Implied volatility profile around earnings."""
    symbol: str
    earnings_date: date

    # IV levels
    iv_30d_prior: float        # Normal IV level
    iv_7d_prior: float         # IV building
    iv_1d_prior: float         # Peak IV
    iv_post_earnings: float    # IV after crush

    # ATM straddle pricing
    straddle_price_7d: float
    straddle_price_1d: float

    # Historical moves
    historical_avg_move: float  # Average absolute earnings move
    historical_moves: List[float]  # Last 8 quarters


class EarningsIVAnalyzer:
    """
    Analyze implied volatility patterns around earnings.
    """

    def __init__(self, options_data: pd.DataFrame, price_data: pd.DataFrame):
        """
        Args:
            options_data: Options chain data with IV
            price_data: Historical price data
        """
        self.options = options_data
        self.prices = price_data

    def get_implied_move(
        self,
        symbol: str,
        earnings_date: date,
        current_price: float
    ) -> Dict:
        """
        Calculate implied move from ATM straddle pricing.

        The market-implied move = ATM straddle price / stock price
        """
        # Get ATM options expiring just after earnings
        chain = self._get_earnings_chain(symbol, earnings_date)

        if chain.empty:
            return {'error': 'No options data'}

        # Find ATM strike
        atm_strike = self._find_atm_strike(chain, current_price)

        # Get ATM call and put
        atm_call = chain[
            (chain['strike'] == atm_strike) &
            (chain['type'] == 'call')
        ].iloc[0] if len(chain) > 0 else None

        atm_put = chain[
            (chain['strike'] == atm_strike) &
            (chain['type'] == 'put')
        ].iloc[0] if len(chain) > 0 else None

        if atm_call is None or atm_put is None:
            return {'error': 'Cannot find ATM options'}

        # Straddle price
        straddle_price = atm_call['mid'] + atm_put['mid']

        # Implied move percentage
        implied_move = straddle_price / current_price

        return {
            'implied_move_pct': implied_move,
            'straddle_price': straddle_price,
            'atm_strike': atm_strike,
            'call_iv': atm_call['iv'],
            'put_iv': atm_put['iv'],
            'avg_iv': (atm_call['iv'] + atm_put['iv']) / 2,
            'interpretation': f"Market expects +/- {implied_move*100:.1f}% move"
        }

    def get_historical_moves(
        self,
        symbol: str,
        num_quarters: int = 8
    ) -> Dict:
        """
        Calculate historical earnings moves.
        """
        # Get earnings dates (would come from calendar)
        earnings_dates = self._get_historical_earnings_dates(symbol, num_quarters)

        moves = []
        for earn_date in earnings_dates:
            # Get close before and after earnings
            pre_close = self._get_price(symbol, earn_date - timedelta(days=1))
            post_close = self._get_price(symbol, earn_date + timedelta(days=1))

            if pre_close and post_close:
                move = (post_close - pre_close) / pre_close
                moves.append({
                    'date': earn_date,
                    'move': move,
                    'abs_move': abs(move)
                })

        if not moves:
            return {'error': 'No historical data'}

        abs_moves = [m['abs_move'] for m in moves]

        return {
            'moves': moves,
            'avg_move': np.mean(abs_moves),
            'median_move': np.median(abs_moves),
            'max_move': np.max(abs_moves),
            'min_move': np.min(abs_moves),
            'std_move': np.std(abs_moves),
            'positive_pct': sum(1 for m in moves if m['move'] > 0) / len(moves)
        }

    def compare_implied_vs_realized(
        self,
        symbol: str,
        current_implied: float,
        num_quarters: int = 8
    ) -> Dict:
        """
        Compare current implied move to historical realized moves.
        """
        historical = self.get_historical_moves(symbol, num_quarters)

        if 'error' in historical:
            return historical

        # Premium/discount to historical
        premium = (current_implied - historical['avg_move']) / historical['avg_move']

        # Percentile rank
        pct_rank = stats.percentileofscore(
            [m['abs_move'] for m in historical['moves']],
            current_implied
        )

        # Trading signal
        if premium > 0.30:
            signal = 'IV_OVERPRICED'
            strategy = 'SELL_STRADDLE'
            confidence = min((premium - 0.30) / 0.30, 1.0)
        elif premium < -0.20:
            signal = 'IV_UNDERPRICED'
            strategy = 'BUY_STRADDLE'
            confidence = min((abs(premium) - 0.20) / 0.30, 1.0)
        else:
            signal = 'FAIR_VALUE'
            strategy = 'NEUTRAL'
            confidence = 0.0

        return {
            'implied_move': current_implied,
            'historical_avg': historical['avg_move'],
            'premium_to_historical': premium,
            'percentile_rank': pct_rank,
            'signal': signal,
            'strategy': strategy,
            'confidence': confidence,
            'historical_stats': historical
        }


def calculate_iv_crush_expected(
    pre_earnings_iv: float,
    post_earnings_iv_typical: float,
    days_to_expiry: int,
    vega: float
) -> Dict:
    """
    Estimate expected P&L from IV crush.
    """
    iv_drop = pre_earnings_iv - post_earnings_iv_typical

    # Vega P&L (per 1 vol point)
    vega_pnl = vega * iv_drop * 100  # Convert to percentage points

    # Time value also decays
    # Theta accelerates into expiry

    return {
        'expected_iv_drop': iv_drop,
        'vega_pnl_estimate': vega_pnl,
        'note': 'Does not account for delta/gamma from price move'
    }
```

---

## Pre-Earnings Strategies

### Long Straddle Strategy

```python
@dataclass
class StraddlePosition:
    """Straddle position details."""
    symbol: str
    strike: float
    expiry: date

    call_price: float
    put_price: float
    total_cost: float

    # Greeks
    delta: float
    gamma: float
    vega: float
    theta: float

    # Breakevens
    upper_breakeven: float
    lower_breakeven: float


class EarningsStraddleStrategy:
    """
    Pre-earnings straddle trading strategy.
    """

    def __init__(self, iv_analyzer: EarningsIVAnalyzer):
        self.iv_analyzer = iv_analyzer

    def evaluate_straddle_trade(
        self,
        symbol: str,
        earnings_date: date,
        current_price: float,
        days_to_earnings: int
    ) -> Dict:
        """
        Evaluate straddle trade opportunity.
        """
        # Get implied vs historical
        implied_info = self.iv_analyzer.get_implied_move(
            symbol, earnings_date, current_price
        )

        if 'error' in implied_info:
            return implied_info

        comparison = self.iv_analyzer.compare_implied_vs_realized(
            symbol, implied_info['implied_move_pct']
        )

        # Entry timing analysis
        timing = self._analyze_entry_timing(days_to_earnings)

        # Build recommendation
        if comparison['signal'] == 'IV_UNDERPRICED':
            recommendation = {
                'action': 'BUY_STRADDLE',
                'confidence': comparison['confidence'],
                'rationale': f"IV implies {implied_info['implied_move_pct']*100:.1f}% "
                           f"vs {comparison['historical_avg']*100:.1f}% historical avg",
                'timing': timing
            }
        elif comparison['signal'] == 'IV_OVERPRICED':
            recommendation = {
                'action': 'SELL_STRADDLE',
                'confidence': comparison['confidence'],
                'rationale': f"IV premium of {comparison['premium_to_historical']*100:.0f}%",
                'timing': timing,
                'warning': 'Unlimited risk on short straddle'
            }
        else:
            recommendation = {
                'action': 'NO_TRADE',
                'rationale': 'IV fairly priced'
            }

        return {
            'recommendation': recommendation,
            'implied_move': implied_info,
            'historical_comparison': comparison
        }

    def _analyze_entry_timing(self, days_to_earnings: int) -> Dict:
        """
        Analyze optimal entry timing.
        """
        # IV typically builds into earnings
        # Optimal entry: 5-7 days before for long straddle
        # Later entry for short straddle (capture more IV crush)

        if days_to_earnings > 14:
            timing = 'TOO_EARLY'
            recommendation = 'Wait for IV to build'
        elif 5 <= days_to_earnings <= 10:
            timing = 'OPTIMAL_FOR_LONG'
            recommendation = 'Good entry for long straddle'
        elif 2 <= days_to_earnings < 5:
            timing = 'LATE_LONG_OPTIMAL_SHORT'
            recommendation = 'Better for short straddle if selling'
        elif days_to_earnings < 2:
            timing = 'ELEVATED_RISK'
            recommendation = 'High gamma risk, smaller size'
        else:
            timing = 'UNKNOWN'
            recommendation = None

        return {
            'timing': timing,
            'days_to_earnings': days_to_earnings,
            'recommendation': recommendation
        }

    def calculate_position_size(
        self,
        portfolio_value: float,
        straddle_cost: float,
        max_risk_pct: float = 0.02,
        confidence: float = 0.5
    ) -> Dict:
        """
        Calculate position size for straddle.
        """
        # Max loss on long straddle = premium paid
        max_risk_dollars = portfolio_value * max_risk_pct

        # Base contracts
        base_contracts = int(max_risk_dollars / (straddle_cost * 100))

        # Adjust by confidence
        adjusted_contracts = int(base_contracts * confidence)

        # Minimum 1 contract
        contracts = max(1, adjusted_contracts)

        return {
            'contracts': contracts,
            'total_cost': contracts * straddle_cost * 100,
            'max_loss': contracts * straddle_cost * 100,
            'risk_pct': (contracts * straddle_cost * 100) / portfolio_value
        }


class StraddleRiskManager:
    """
    Risk management for straddle positions.
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.50,
        take_profit_pct: float = 0.50,
        time_stop_days: int = 1
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.time_stop = time_stop_days

    def check_exit_conditions(
        self,
        position: StraddlePosition,
        current_value: float,
        days_since_entry: int,
        earnings_released: bool
    ) -> Dict:
        """
        Check if position should be exited.
        """
        pnl_pct = (current_value - position.total_cost) / position.total_cost

        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return {
                'exit': True,
                'reason': 'STOP_LOSS',
                'pnl_pct': pnl_pct
            }

        # Take profit
        if pnl_pct >= self.take_profit_pct:
            return {
                'exit': True,
                'reason': 'TAKE_PROFIT',
                'pnl_pct': pnl_pct
            }

        # Time stop (for pre-earnings trades)
        if earnings_released and days_since_entry >= self.time_stop:
            return {
                'exit': True,
                'reason': 'TIME_STOP_POST_EARNINGS',
                'pnl_pct': pnl_pct
            }

        return {'exit': False, 'pnl_pct': pnl_pct}
```

---

## Post-Earnings IV Crush

### Selling Volatility Strategy

```python
class IVCrushStrategy:
    """
    Exploit IV crush after earnings release.
    """

    def __init__(
        self,
        min_iv_premium: float = 0.20,
        max_days_to_expiry: int = 14
    ):
        self.min_iv_premium = min_iv_premium
        self.max_days = max_days_to_expiry

    def evaluate_iv_crush_trade(
        self,
        symbol: str,
        pre_earnings_iv: float,
        post_earnings_iv_estimate: float,
        straddle_price: float,
        current_price: float,
        days_to_expiry: int
    ) -> Dict:
        """
        Evaluate selling premium to capture IV crush.
        """
        # Expected IV drop
        iv_drop = pre_earnings_iv - post_earnings_iv_estimate
        iv_drop_pct = iv_drop / pre_earnings_iv

        if iv_drop_pct < self.min_iv_premium:
            return {
                'trade': False,
                'reason': f'IV drop ({iv_drop_pct:.0%}) below minimum ({self.min_iv_premium:.0%})'
            }

        if days_to_expiry > self.max_days:
            return {
                'trade': False,
                'reason': 'Expiry too far out - IV crush less impactful'
            }

        # Expected P&L from IV crush alone
        implied_move = straddle_price / current_price
        expected_iv_pnl = straddle_price * (iv_drop_pct * 0.7)  # Rough estimate

        return {
            'trade': True,
            'strategy': 'SELL_STRADDLE' if implied_move > 0.08 else 'SELL_STRANGLE',
            'expected_iv_drop': iv_drop,
            'expected_iv_pnl': expected_iv_pnl,
            'max_loss': 'UNLIMITED' if True else straddle_price,
            'probability_profit': 0.65,  # Historical average
            'risk_warning': 'Large move can exceed IV crush benefit'
        }

    def select_strikes(
        self,
        current_price: float,
        implied_move: float,
        strategy: str
    ) -> Dict:
        """
        Select optimal strikes for short vol trade.
        """
        if strategy == 'SELL_STRADDLE':
            # ATM strikes
            call_strike = round(current_price / 5) * 5  # Round to nearest $5
            put_strike = call_strike

        elif strategy == 'SELL_STRANGLE':
            # OTM strikes beyond expected move
            buffer = 1.2  # 20% beyond implied move

            call_strike = current_price * (1 + implied_move * buffer)
            call_strike = round(call_strike / 5) * 5

            put_strike = current_price * (1 - implied_move * buffer)
            put_strike = round(put_strike / 5) * 5

        elif strategy == 'IRON_CONDOR':
            # Sell inside expected move, buy protection outside
            inner_buffer = 0.9
            outer_buffer = 1.5

            sell_call = current_price * (1 + implied_move * inner_buffer)
            buy_call = current_price * (1 + implied_move * outer_buffer)
            sell_put = current_price * (1 - implied_move * inner_buffer)
            buy_put = current_price * (1 - implied_move * outer_buffer)

            return {
                'sell_call': round(sell_call / 5) * 5,
                'buy_call': round(buy_call / 5) * 5,
                'sell_put': round(sell_put / 5) * 5,
                'buy_put': round(buy_put / 5) * 5
            }

        return {
            'call_strike': call_strike,
            'put_strike': put_strike
        }
```

---

## Volatility Forecasting

### Earnings Move Prediction

```python
class EarningsMovePredictor:
    """
    Forecast earnings move magnitude.
    """

    def __init__(self, historical_data: pd.DataFrame):
        """
        Args:
            historical_data: DataFrame with columns:
                symbol, earnings_date, pre_iv, post_iv, actual_move,
                surprise_pct, guidance_change, volume_ratio
        """
        self.data = historical_data

    def predict_move_distribution(
        self,
        symbol: str,
        current_implied: float,
        surprise_expectation: str = 'unknown'
    ) -> Dict:
        """
        Predict distribution of possible earnings moves.
        """
        # Get historical moves
        symbol_data = self.data[self.data['symbol'] == symbol]

        if len(symbol_data) < 4:
            return self._use_sector_proxy(symbol, current_implied)

        moves = symbol_data['actual_move'].values

        # Fit distribution
        mean_move = np.mean(moves)
        std_move = np.std(moves)

        # Generate percentiles
        percentiles = {
            'p10': np.percentile(moves, 10),
            'p25': np.percentile(moves, 25),
            'p50': np.percentile(moves, 50),
            'p75': np.percentile(moves, 75),
            'p90': np.percentile(moves, 90)
        }

        # Compare to current implied
        implied_percentile = stats.percentileofscore(np.abs(moves), current_implied)

        return {
            'mean_move': mean_move,
            'std_move': std_move,
            'percentiles': percentiles,
            'implied_vs_historical_percentile': implied_percentile,
            'sample_size': len(moves)
        }

    def predict_move_given_surprise(
        self,
        symbol: str,
        surprise_magnitude: str  # 'big_beat', 'beat', 'miss', 'big_miss'
    ) -> Dict:
        """
        Predict move conditional on surprise type.
        """
        symbol_data = self.data[self.data['symbol'] == symbol]

        # Classify historical surprises
        symbol_data['surprise_category'] = symbol_data['surprise_pct'].apply(
            lambda x: 'big_beat' if x > 0.10 else
                     'beat' if x > 0.02 else
                     'big_miss' if x < -0.10 else
                     'miss' if x < -0.02 else 'inline'
        )

        # Filter to matching surprise type
        matching = symbol_data[symbol_data['surprise_category'] == surprise_magnitude]

        if len(matching) < 2:
            return {'error': 'Insufficient data for conditional prediction'}

        moves = matching['actual_move'].values

        return {
            'conditional_mean': np.mean(moves),
            'conditional_std': np.std(moves),
            'sample_size': len(moves),
            'surprise_type': surprise_magnitude
        }

    def calculate_edge(
        self,
        implied_move: float,
        predicted_move: float,
        prediction_confidence: float
    ) -> Dict:
        """
        Calculate trading edge from move prediction.
        """
        edge = predicted_move - implied_move

        # Adjust for confidence
        confidence_adjusted_edge = edge * prediction_confidence

        if abs(confidence_adjusted_edge) < 0.02:
            signal = 'NO_EDGE'
        elif confidence_adjusted_edge > 0.02:
            signal = 'BUY_VOL'  # Move expected to be larger than implied
        else:
            signal = 'SELL_VOL'  # Move expected to be smaller than implied

        return {
            'raw_edge': edge,
            'confidence_adjusted_edge': confidence_adjusted_edge,
            'signal': signal,
            'implied_move': implied_move,
            'predicted_move': predicted_move
        }
```

---

## Iron Condor Earnings Strategy

### Defined Risk Short Vol

```python
class EarningsIronCondor:
    """
    Iron condor strategy for earnings with defined risk.
    """

    def __init__(
        self,
        wing_width: float = 0.10,  # 10% OTM for wings
        probability_profit_target: float = 0.60
    ):
        self.wing_width = wing_width
        self.pop_target = probability_profit_target

    def design_iron_condor(
        self,
        current_price: float,
        implied_move: float,
        options_chain: pd.DataFrame
    ) -> Dict:
        """
        Design iron condor around expected move.
        """
        # Short strikes just beyond expected move
        short_call = current_price * (1 + implied_move * 1.1)
        short_put = current_price * (1 - implied_move * 1.1)

        # Long strikes for protection
        long_call = short_call * (1 + self.wing_width)
        long_put = short_put * (1 - self.wing_width)

        # Round to available strikes
        strikes = self._round_to_strikes(
            options_chain,
            short_call, short_put, long_call, long_put
        )

        # Calculate premium and risk
        premium = self._calculate_premium(options_chain, strikes)
        max_loss = self._calculate_max_loss(strikes, premium)

        return {
            'strikes': strikes,
            'premium_received': premium,
            'max_loss': max_loss,
            'risk_reward': premium / max_loss if max_loss > 0 else 0,
            'breakeven_upper': strikes['short_call'] + premium,
            'breakeven_lower': strikes['short_put'] - premium,
            'probability_profit': self._estimate_pop(
                current_price, strikes, implied_move
            )
        }

    def _estimate_pop(
        self,
        current_price: float,
        strikes: Dict,
        implied_move: float
    ) -> float:
        """
        Estimate probability of profit.
        """
        # Simplified: probability stock stays between short strikes
        upper_distance = (strikes['short_call'] - current_price) / current_price
        lower_distance = (current_price - strikes['short_put']) / current_price

        # Using normal distribution assumption
        # Probability stock stays within range
        upper_prob = stats.norm.cdf(upper_distance / implied_move)
        lower_prob = stats.norm.cdf(lower_distance / implied_move)

        # Joint probability (simplified)
        return upper_prob * lower_prob * 2  # Rough approximation

    def manage_iron_condor(
        self,
        position: Dict,
        current_price: float,
        days_to_expiry: int,
        earnings_released: bool
    ) -> Dict:
        """
        Position management rules.
        """
        strikes = position['strikes']

        # Test breaches
        upper_breach = current_price > strikes['short_call']
        lower_breach = current_price < strikes['short_put']

        # Pre-earnings: wider stops
        if not earnings_released:
            if upper_breach or lower_breach:
                return {
                    'action': 'CLOSE',
                    'reason': 'Strike breach before earnings - elevated risk'
                }

        # Post-earnings: normal management
        if earnings_released:
            # Take profit at 50% of max
            current_value = self._get_current_value(position, current_price)
            pnl_pct = (position['premium_received'] - current_value) / position['premium_received']

            if pnl_pct > 0.50:
                return {
                    'action': 'CLOSE',
                    'reason': 'Take profit - 50% of max captured'
                }

            # Close if strike tested
            if upper_breach or lower_breach:
                return {
                    'action': 'CLOSE',
                    'reason': 'Short strike breached'
                }

        return {'action': 'HOLD'}
```

---

## Performance Analytics

### Volatility Strategy Tracking

```python
class EarningsVolPerformance:
    """
    Track earnings volatility strategy performance.
    """

    def __init__(self):
        self.trades = []

    def record_trade(
        self,
        symbol: str,
        strategy: str,
        entry_date: date,
        exit_date: date,
        pnl: float,
        implied_move: float,
        actual_move: float,
        iv_pre: float,
        iv_post: float
    ):
        """Record completed trade."""
        self.trades.append({
            'symbol': symbol,
            'strategy': strategy,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'pnl': pnl,
            'implied_move': implied_move,
            'actual_move': actual_move,
            'iv_pre': iv_pre,
            'iv_post': iv_post,
            'move_vs_implied': actual_move - implied_move,
            'iv_crush': iv_pre - iv_post
        })

    def get_performance_summary(self) -> Dict:
        """Get overall performance metrics."""
        if not self.trades:
            return {'error': 'No trades recorded'}

        df = pd.DataFrame(self.trades)

        summary = {
            'total_trades': len(df),
            'total_pnl': df['pnl'].sum(),
            'avg_pnl': df['pnl'].mean(),
            'win_rate': (df['pnl'] > 0).mean(),
            'profit_factor': abs(df[df['pnl'] > 0]['pnl'].sum()) /
                           abs(df[df['pnl'] < 0]['pnl'].sum()) if (df['pnl'] < 0).any() else np.inf,
            'avg_iv_crush': df['iv_crush'].mean(),
            'times_move_exceeded_implied': (df['actual_move'].abs() > df['implied_move']).mean()
        }

        # By strategy
        for strategy in df['strategy'].unique():
            strat_df = df[df['strategy'] == strategy]
            summary[f'{strategy}_win_rate'] = (strat_df['pnl'] > 0).mean()
            summary[f'{strategy}_avg_pnl'] = strat_df['pnl'].mean()

        return summary

    def analyze_edge_decay(self) -> Dict:
        """
        Analyze if edge has decayed over time.
        """
        df = pd.DataFrame(self.trades)
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df = df.sort_values('entry_date')

        # Rolling win rate
        df['rolling_win_rate'] = (df['pnl'] > 0).rolling(20).mean()

        # Rolling edge (implied vs actual)
        df['rolling_edge'] = (df['implied_move'] - df['actual_move'].abs()).rolling(20).mean()

        # Trend analysis
        recent = df.tail(20)
        earlier = df.head(20)

        return {
            'recent_win_rate': (recent['pnl'] > 0).mean(),
            'earlier_win_rate': (earlier['pnl'] > 0).mean(),
            'edge_trend': 'DECAYING' if recent['rolling_edge'].mean() < earlier['rolling_edge'].mean() else 'STABLE',
            'recommendation': 'Review strategy parameters' if recent['rolling_win_rate'].mean() < 0.45 else 'Continue'
        }
```

---

## Risk Management

### Volatility Position Limits

```python
class EarningsVolRiskManager:
    """
    Risk management for earnings volatility positions.
    """

    def __init__(
        self,
        max_portfolio_risk_pct: float = 0.05,
        max_single_position_pct: float = 0.02,
        max_concurrent_earnings: int = 5
    ):
        self.max_portfolio_risk = max_portfolio_risk_pct
        self.max_single_position = max_single_position_pct
        self.max_concurrent = max_concurrent_earnings

    def check_position_limits(
        self,
        proposed_trade: Dict,
        current_positions: List[Dict],
        portfolio_value: float
    ) -> Dict:
        """
        Check if proposed trade fits within limits.
        """
        # Count current earnings positions
        earnings_positions = [
            p for p in current_positions
            if p.get('strategy_type') == 'earnings_vol'
        ]

        if len(earnings_positions) >= self.max_concurrent:
            return {
                'approved': False,
                'reason': f'Max concurrent earnings positions ({self.max_concurrent}) reached'
            }

        # Check single position size
        position_risk = proposed_trade.get('max_loss', 0)
        position_risk_pct = position_risk / portfolio_value

        if position_risk_pct > self.max_single_position:
            return {
                'approved': False,
                'reason': f'Position risk {position_risk_pct:.1%} exceeds limit {self.max_single_position:.1%}'
            }

        # Check total portfolio risk
        total_risk = sum(p.get('max_loss', 0) for p in earnings_positions)
        total_risk += position_risk
        total_risk_pct = total_risk / portfolio_value

        if total_risk_pct > self.max_portfolio_risk:
            return {
                'approved': False,
                'reason': f'Total earnings risk {total_risk_pct:.1%} would exceed limit {self.max_portfolio_risk:.1%}'
            }

        return {
            'approved': True,
            'position_risk_pct': position_risk_pct,
            'total_risk_pct': total_risk_pct
        }
```

---

## Academic References

1. **Patell, J. M., & Wolfson, M. A. (1981)**. "The Ex Ante and Ex Post Price Effects of Quarterly Earnings Announcements." *Journal of Accounting Research*.

2. **Dubinsky, A., & Johannes, M. (2006)**. "Earnings Announcements and Equity Options." *Working Paper*.

3. **Ederington, L. H., & Lee, J. H. (1996)**. "The Creation and Resolution of Market Uncertainty: The Impact of Information Releases on Implied Volatility." *Journal of Financial and Quantitative Analysis*.

4. **Ni, S. X., Pan, J., & Poteshman, A. M. (2008)**. "Volatility Information Trading in the Option Market." *Journal of Finance*.

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
tags: ["earnings", "volatility", "options", "straddle", "iv-crush"]
code_lines: 700
```

---

**END OF DOCUMENT**
