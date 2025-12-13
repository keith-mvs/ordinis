# Dividend Trading Strategies

## Overview

Dividend events create systematic trading opportunities through capture strategies, ex-date dynamics, and yield-based signals. This document covers dividend trading mechanics, tax considerations, and quantitative approaches.

---

## Dividend Mechanics

### Key Dates

```python
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, List, Dict
import numpy as np
import pandas as pd


@dataclass
class DividendEvent:
    """Dividend event with key dates."""
    symbol: str
    dividend_amount: float
    dividend_type: str  # 'regular', 'special', 'stock'

    # Key dates
    declaration_date: date
    ex_dividend_date: date
    record_date: date
    payment_date: date

    # Additional info
    frequency: str  # 'quarterly', 'monthly', 'annual', 'special'
    prior_dividend: Optional[float] = None
    yield_pct: Optional[float] = None


class DividendCalendar:
    """
    Manage dividend calendar and calculations.
    """

    def __init__(self, dividend_history: pd.DataFrame):
        """
        Args:
            dividend_history: DataFrame with columns:
                symbol, ex_date, amount, type, frequency
        """
        self.history = dividend_history

    def get_upcoming_dividends(
        self,
        symbols: List[str],
        days_ahead: int = 30
    ) -> List[DividendEvent]:
        """Get upcoming dividend events."""
        cutoff = date.today() + timedelta(days=days_ahead)

        upcoming = self.history[
            (self.history['symbol'].isin(symbols)) &
            (self.history['ex_date'] >= date.today()) &
            (self.history['ex_date'] <= cutoff)
        ].sort_values('ex_date')

        events = []
        for _, row in upcoming.iterrows():
            # Get prior dividend
            prior = self._get_prior_dividend(row['symbol'], row['ex_date'])

            events.append(DividendEvent(
                symbol=row['symbol'],
                dividend_amount=row['amount'],
                dividend_type=row['type'],
                declaration_date=row.get('declaration_date', row['ex_date'] - timedelta(days=14)),
                ex_dividend_date=row['ex_date'],
                record_date=row.get('record_date', row['ex_date'] + timedelta(days=1)),
                payment_date=row.get('payment_date', row['ex_date'] + timedelta(days=30)),
                frequency=row['frequency'],
                prior_dividend=prior
            ))

        return events

    def _get_prior_dividend(
        self,
        symbol: str,
        current_ex_date: date
    ) -> Optional[float]:
        """Get prior dividend amount."""
        prior = self.history[
            (self.history['symbol'] == symbol) &
            (self.history['ex_date'] < current_ex_date)
        ].sort_values('ex_date', ascending=False)

        if len(prior) > 0:
            return prior.iloc[0]['amount']
        return None

    def calculate_yield(
        self,
        symbol: str,
        current_price: float,
        annualize: bool = True
    ) -> Dict:
        """Calculate dividend yield."""
        # Get last 4 quarterly dividends or 12 monthly
        recent = self.history[
            (self.history['symbol'] == symbol) &
            (self.history['ex_date'] >= date.today() - timedelta(days=365))
        ].sort_values('ex_date', ascending=False)

        if len(recent) == 0:
            return {'yield': 0, 'trailing_12m': 0}

        total_dividends = recent['amount'].sum()

        # Annualize if needed
        if annualize:
            frequency = recent.iloc[0]['frequency']
            if frequency == 'quarterly':
                annual_div = total_dividends if len(recent) >= 4 else recent['amount'].mean() * 4
            elif frequency == 'monthly':
                annual_div = total_dividends if len(recent) >= 12 else recent['amount'].mean() * 12
            else:
                annual_div = total_dividends
        else:
            annual_div = total_dividends

        return {
            'yield': annual_div / current_price,
            'trailing_12m': total_dividends,
            'num_payments': len(recent),
            'frequency': recent.iloc[0]['frequency']
        }
```

---

## Dividend Capture Strategy

### Core Implementation

```python
class DividendCaptureStrategy:
    """
    Systematic dividend capture trading.
    """

    def __init__(
        self,
        min_yield: float = 0.03,
        holding_period_days: int = 5,
        max_position_pct: float = 0.05
    ):
        """
        Args:
            min_yield: Minimum annualized yield to consider
            holding_period_days: Days to hold around ex-date
            max_position_pct: Maximum position as % of portfolio
        """
        self.min_yield = min_yield
        self.holding_period = holding_period_days
        self.max_position = max_position_pct

    def evaluate_capture_opportunity(
        self,
        event: DividendEvent,
        current_price: float,
        avg_daily_volume: int,
        bid_ask_spread: float,
        tax_rate: float = 0.20
    ) -> Dict:
        """
        Evaluate dividend capture opportunity.
        """
        # Calculate capture return
        gross_yield = event.dividend_amount / current_price

        # Transaction costs
        round_trip_cost = bid_ask_spread * 2  # Buy and sell

        # Tax (qualified vs ordinary)
        if event.dividend_type == 'qualified':
            tax_drag = event.dividend_amount * tax_rate * 0.5  # Lower rate
        else:
            tax_drag = event.dividend_amount * tax_rate

        # Net capture return
        net_capture = event.dividend_amount - (round_trip_cost * current_price) - tax_drag
        net_yield = net_capture / current_price

        # Annualized return
        annualized = net_yield * (365 / self.holding_period)

        # Liquidity check
        min_volume = 100000  # Minimum daily dollar volume
        daily_dollar_volume = avg_daily_volume * current_price

        if daily_dollar_volume < min_volume:
            liquidity_pass = False
            max_shares = 0
        else:
            liquidity_pass = True
            # Limit to 1% of daily volume
            max_shares = int(avg_daily_volume * 0.01)

        # Overall assessment
        if net_yield > 0 and liquidity_pass and annualized > self.min_yield:
            recommendation = 'CAPTURE'
        elif net_yield > 0 and not liquidity_pass:
            recommendation = 'SKIP_LIQUIDITY'
        elif net_yield <= 0:
            recommendation = 'SKIP_NEGATIVE'
        else:
            recommendation = 'SKIP'

        return {
            'recommendation': recommendation,
            'gross_yield': gross_yield,
            'net_yield': net_yield,
            'annualized_return': annualized,
            'transaction_cost': round_trip_cost,
            'tax_drag': tax_drag / current_price,
            'max_shares': max_shares,
            'liquidity_pass': liquidity_pass
        }

    def generate_capture_trades(
        self,
        opportunities: List[Dict],
        portfolio_value: float
    ) -> List[Dict]:
        """
        Generate trades for dividend capture.
        """
        trades = []

        # Sort by net yield
        opportunities.sort(key=lambda x: x.get('net_yield', 0), reverse=True)

        total_allocated = 0
        max_total = portfolio_value * 0.30  # Max 30% in capture strategy

        for opp in opportunities:
            if opp['recommendation'] != 'CAPTURE':
                continue

            if total_allocated >= max_total:
                break

            # Position size
            position_value = min(
                portfolio_value * self.max_position,
                max_total - total_allocated
            )

            trades.append({
                'symbol': opp['symbol'],
                'action': 'BUY',
                'value': position_value,
                'entry_date': opp['ex_date'] - timedelta(days=self.holding_period // 2),
                'exit_date': opp['ex_date'] + timedelta(days=self.holding_period // 2 + 1),
                'expected_yield': opp['net_yield']
            })

            total_allocated += position_value

        return trades


class DividendCaptureRiskManager:
    """
    Risk management for dividend capture.
    """

    def __init__(
        self,
        max_gap_risk: float = 0.05,
        stop_loss_pct: float = 0.03
    ):
        self.max_gap_risk = max_gap_risk
        self.stop_loss = stop_loss_pct

    def assess_ex_date_risk(
        self,
        event: DividendEvent,
        current_price: float,
        historical_ex_date_drops: List[float]
    ) -> Dict:
        """
        Assess risk around ex-dividend date.
        """
        expected_drop = event.dividend_amount / current_price

        # Historical analysis
        if historical_ex_date_drops:
            avg_drop = np.mean(historical_ex_date_drops)
            std_drop = np.std(historical_ex_date_drops)

            # Does stock typically drop more or less than dividend?
            typical_excess_drop = avg_drop - expected_drop
        else:
            avg_drop = expected_drop
            std_drop = 0.01
            typical_excess_drop = 0

        # Risk assessment
        if typical_excess_drop > 0.01:
            risk_level = 'HIGH'
            note = 'Stock typically drops more than dividend'
        elif typical_excess_drop < -0.01:
            risk_level = 'LOW'
            note = 'Stock typically drops less than dividend'
        else:
            risk_level = 'NORMAL'
            note = 'Stock drops approximately dividend amount'

        return {
            'expected_drop': expected_drop,
            'historical_avg_drop': avg_drop,
            'excess_drop': typical_excess_drop,
            'risk_level': risk_level,
            'note': note
        }

    def calculate_position_stop(
        self,
        entry_price: float,
        dividend_amount: float,
        side: str = 'long'
    ) -> Dict:
        """
        Calculate stop loss for dividend position.
        """
        # Account for expected dividend drop
        adjusted_entry = entry_price - dividend_amount  # Post-ex price expectation

        if side == 'long':
            stop_price = adjusted_entry * (1 - self.stop_loss)
        else:
            stop_price = adjusted_entry * (1 + self.stop_loss)

        return {
            'stop_price': stop_price,
            'stop_pct': self.stop_loss,
            'note': 'Adjusted for expected dividend drop'
        }
```

---

## Ex-Date Price Dynamics

### Ex-Date Analysis

```python
class ExDateAnalyzer:
    """
    Analyze price behavior around ex-dividend dates.
    """

    def __init__(self, price_data: pd.DataFrame, dividend_data: pd.DataFrame):
        """
        Args:
            price_data: OHLCV data
            dividend_data: Dividend events
        """
        self.prices = price_data
        self.dividends = dividend_data

    def analyze_ex_date_behavior(
        self,
        symbol: str,
        lookback_events: int = 20
    ) -> Dict:
        """
        Analyze historical ex-date price behavior.
        """
        # Get historical ex-dates
        events = self.dividends[
            self.dividends['symbol'] == symbol
        ].sort_values('ex_date', ascending=False).head(lookback_events)

        if len(events) == 0:
            return {'error': 'No dividend history'}

        drops = []
        recoveries = []

        for _, event in events.iterrows():
            ex_date = event['ex_date']
            dividend = event['amount']

            # Get prices
            pre_close = self._get_price(symbol, ex_date - timedelta(days=1), 'close')
            ex_open = self._get_price(symbol, ex_date, 'open')
            ex_close = self._get_price(symbol, ex_date, 'close')
            post_5d_close = self._get_price(symbol, ex_date + timedelta(days=5), 'close')

            if pre_close and ex_open:
                # Overnight drop
                overnight_drop = (pre_close - ex_open) / pre_close
                expected_drop = dividend / pre_close
                excess_drop = overnight_drop - expected_drop

                drops.append({
                    'ex_date': ex_date,
                    'overnight_drop': overnight_drop,
                    'expected_drop': expected_drop,
                    'excess_drop': excess_drop
                })

            if ex_close and post_5d_close:
                # Recovery (from ex-date close to 5 days later)
                recovery = (post_5d_close - ex_close) / ex_close
                recoveries.append(recovery)

        # Summary statistics
        if drops:
            avg_overnight = np.mean([d['overnight_drop'] for d in drops])
            avg_excess = np.mean([d['excess_drop'] for d in drops])
        else:
            avg_overnight = 0
            avg_excess = 0

        avg_recovery = np.mean(recoveries) if recoveries else 0

        return {
            'symbol': symbol,
            'num_events': len(events),
            'avg_overnight_drop': avg_overnight,
            'avg_excess_drop': avg_excess,
            'avg_5d_recovery': avg_recovery,
            'drops': drops,
            'interpretation': self._interpret_behavior(avg_excess, avg_recovery)
        }

    def _interpret_behavior(
        self,
        avg_excess: float,
        avg_recovery: float
    ) -> str:
        """Interpret ex-date behavior."""
        if avg_excess < -0.005 and avg_recovery > 0.005:
            return 'FAVORABLE - Under-drops and recovers, capture likely profitable'
        elif avg_excess > 0.005:
            return 'UNFAVORABLE - Over-drops, capture may not recover'
        else:
            return 'NEUTRAL - Drops roughly dividend amount'

    def find_under_dropping_stocks(
        self,
        min_events: int = 8,
        max_excess_drop: float = -0.003
    ) -> List[Dict]:
        """
        Find stocks that consistently under-drop on ex-date.
        """
        candidates = []

        for symbol in self.dividends['symbol'].unique():
            analysis = self.analyze_ex_date_behavior(symbol, min_events)

            if 'error' in analysis:
                continue

            if (analysis['num_events'] >= min_events and
                analysis['avg_excess_drop'] < max_excess_drop):
                candidates.append({
                    'symbol': symbol,
                    'avg_excess_drop': analysis['avg_excess_drop'],
                    'avg_recovery': analysis['avg_5d_recovery'],
                    'num_events': analysis['num_events']
                })

        # Sort by excess drop (most negative = best)
        candidates.sort(key=lambda x: x['avg_excess_drop'])

        return candidates
```

---

## Dividend Growth Investing

### Dividend Growth Signals

```python
class DividendGrowthAnalyzer:
    """
    Analyze dividend growth for investment signals.
    """

    def __init__(self, dividend_history: pd.DataFrame):
        self.history = dividend_history

    def calculate_growth_metrics(
        self,
        symbol: str,
        years: int = 5
    ) -> Dict:
        """
        Calculate dividend growth metrics.
        """
        cutoff = date.today() - timedelta(days=years * 365)

        history = self.history[
            (self.history['symbol'] == symbol) &
            (self.history['ex_date'] >= cutoff) &
            (self.history['type'] == 'regular')
        ].sort_values('ex_date')

        if len(history) < 4:  # Need at least 1 year
            return {'error': 'Insufficient history'}

        # Annual dividends
        history['year'] = pd.to_datetime(history['ex_date']).dt.year
        annual = history.groupby('year')['amount'].sum()

        # Growth rate (CAGR)
        if len(annual) >= 2:
            start_div = annual.iloc[0]
            end_div = annual.iloc[-1]
            n_years = len(annual) - 1

            if start_div > 0:
                cagr = (end_div / start_div) ** (1 / n_years) - 1
            else:
                cagr = 0
        else:
            cagr = 0

        # Consecutive growth years
        consecutive_growth = 0
        for i in range(len(annual) - 1, 0, -1):
            if annual.iloc[i] > annual.iloc[i-1]:
                consecutive_growth += 1
            else:
                break

        # Payout stability
        yoy_changes = annual.pct_change().dropna()
        cut_count = (yoy_changes < 0).sum()

        return {
            'cagr': cagr,
            'consecutive_growth_years': consecutive_growth,
            'dividend_cuts': cut_count,
            'avg_annual_growth': yoy_changes.mean() if len(yoy_changes) > 0 else 0,
            'growth_volatility': yoy_changes.std() if len(yoy_changes) > 1 else 0,
            'annual_dividends': annual.to_dict(),
            'years_analyzed': len(annual)
        }

    def screen_dividend_growers(
        self,
        min_cagr: float = 0.05,
        min_consecutive_years: int = 5,
        max_cuts: int = 0
    ) -> List[Dict]:
        """
        Screen for dividend growth stocks.
        """
        results = []

        for symbol in self.history['symbol'].unique():
            metrics = self.calculate_growth_metrics(symbol, years=10)

            if 'error' in metrics:
                continue

            if (metrics['cagr'] >= min_cagr and
                metrics['consecutive_growth_years'] >= min_consecutive_years and
                metrics['dividend_cuts'] <= max_cuts):

                results.append({
                    'symbol': symbol,
                    'cagr': metrics['cagr'],
                    'consecutive_years': metrics['consecutive_growth_years'],
                    'cuts': metrics['dividend_cuts']
                })

        # Sort by CAGR
        results.sort(key=lambda x: x['cagr'], reverse=True)

        return results

    def dividend_cut_warning(
        self,
        symbol: str,
        payout_ratio: float,
        earnings_trend: str,
        debt_ratio: float
    ) -> Dict:
        """
        Assess risk of dividend cut.
        """
        risk_score = 0
        warnings = []

        # Payout ratio risk
        if payout_ratio > 1.0:
            risk_score += 3
            warnings.append('Payout ratio exceeds earnings')
        elif payout_ratio > 0.80:
            risk_score += 2
            warnings.append('High payout ratio')
        elif payout_ratio > 0.60:
            risk_score += 1

        # Earnings trend
        if earnings_trend == 'declining':
            risk_score += 2
            warnings.append('Declining earnings')
        elif earnings_trend == 'volatile':
            risk_score += 1

        # Debt
        if debt_ratio > 0.60:
            risk_score += 2
            warnings.append('High debt levels')
        elif debt_ratio > 0.40:
            risk_score += 1

        # Overall assessment
        if risk_score >= 5:
            cut_risk = 'HIGH'
        elif risk_score >= 3:
            cut_risk = 'MODERATE'
        else:
            cut_risk = 'LOW'

        return {
            'cut_risk': cut_risk,
            'risk_score': risk_score,
            'warnings': warnings,
            'recommendation': 'Monitor closely' if cut_risk == 'HIGH' else 'Normal monitoring'
        }
```

---

## Special Dividend Trading

### Special Dividend Opportunities

```python
class SpecialDividendStrategy:
    """
    Trading strategies for special dividends.
    """

    def __init__(self, min_yield: float = 0.03):
        self.min_yield = min_yield

    def evaluate_special_dividend(
        self,
        event: DividendEvent,
        current_price: float,
        reason: str  # 'asset_sale', 'cash_balance', 'restructuring'
    ) -> Dict:
        """
        Evaluate special dividend opportunity.
        """
        yield_pct = event.dividend_amount / current_price

        # Special dividend signals
        signals = {}

        if yield_pct > 0.10:
            signals['size'] = 'LARGE'
            signals['price_impact'] = 'SIGNIFICANT'
        elif yield_pct > 0.05:
            signals['size'] = 'MODERATE'
            signals['price_impact'] = 'MODERATE'
        else:
            signals['size'] = 'SMALL'
            signals['price_impact'] = 'MINOR'

        # Reason-based assessment
        if reason == 'asset_sale':
            signals['sustainability'] = 'ONE_TIME'
            signals['forward_outlook'] = 'Watch for earnings impact'
        elif reason == 'cash_balance':
            signals['sustainability'] = 'BALANCE_SHEET_DRIVEN'
            signals['forward_outlook'] = 'May repeat if cash builds'
        elif reason == 'restructuring':
            signals['sustainability'] = 'ONE_TIME'
            signals['forward_outlook'] = 'Company in transition'

        # Trading approach
        if yield_pct > 0.05:
            # Large special dividends often see full drop
            trading_approach = {
                'pre_announcement': 'Already priced in announcement reaction',
                'pre_ex_date': 'Stock may drift up as yield attracts buyers',
                'ex_date': 'Expect full drop',
                'post_ex_date': 'Recovery depends on fundamentals'
            }
        else:
            trading_approach = {
                'capture': 'May be capture opportunity if under-drops'
            }

        return {
            'yield': yield_pct,
            'signals': signals,
            'trading_approach': trading_approach
        }

    def find_special_dividend_candidates(
        self,
        stocks_with_excess_cash: List[Dict],
        min_cash_ratio: float = 0.20
    ) -> List[Dict]:
        """
        Find stocks likely to announce special dividends.
        """
        candidates = []

        for stock in stocks_with_excess_cash:
            if stock.get('cash_to_market_cap', 0) > min_cash_ratio:
                # High cash balance relative to market cap
                candidates.append({
                    'symbol': stock['symbol'],
                    'cash_ratio': stock['cash_to_market_cap'],
                    'catalyst': 'excess_cash',
                    'probability': 'MODERATE'
                })

            if stock.get('recent_asset_sale', False):
                candidates.append({
                    'symbol': stock['symbol'],
                    'catalyst': 'asset_sale_proceeds',
                    'probability': 'HIGH'
                })

        return candidates
```

---

## Tax-Efficient Dividend Strategies

### Tax Optimization

```python
class DividendTaxOptimizer:
    """
    Optimize dividend strategies for tax efficiency.
    """

    def __init__(
        self,
        qualified_rate: float = 0.20,
        ordinary_rate: float = 0.37,
        state_rate: float = 0.05
    ):
        self.qualified_rate = qualified_rate
        self.ordinary_rate = ordinary_rate
        self.state_rate = state_rate

    def calculate_after_tax_yield(
        self,
        gross_yield: float,
        dividend_type: str,
        holding_period_days: int
    ) -> Dict:
        """
        Calculate after-tax yield.
        """
        # Qualified dividend requires 61+ day holding
        is_qualified = (dividend_type == 'qualified' and holding_period_days >= 61)

        if is_qualified:
            tax_rate = self.qualified_rate
        else:
            tax_rate = self.ordinary_rate

        total_rate = tax_rate + self.state_rate

        after_tax_yield = gross_yield * (1 - total_rate)

        return {
            'gross_yield': gross_yield,
            'tax_rate': total_rate,
            'after_tax_yield': after_tax_yield,
            'qualified': is_qualified,
            'holding_requirement': 61 if dividend_type == 'qualified' else 0
        }

    def optimize_holding_period(
        self,
        capture_yield: float,
        transaction_costs: float,
        short_holding_days: int = 5,
        long_holding_days: int = 65
    ) -> Dict:
        """
        Compare short vs long holding periods.
        """
        # Short hold (ordinary tax rate)
        short_after_tax = capture_yield * (1 - self.ordinary_rate - self.state_rate)
        short_net = short_after_tax - transaction_costs
        short_annualized = short_net * (365 / short_holding_days)

        # Long hold (qualified rate)
        long_after_tax = capture_yield * (1 - self.qualified_rate - self.state_rate)
        long_net = long_after_tax - transaction_costs
        long_annualized = long_net * (365 / long_holding_days)

        if short_annualized > long_annualized:
            recommendation = 'SHORT_HOLD'
            reason = 'Higher annualized return despite higher tax'
        else:
            recommendation = 'LONG_HOLD'
            reason = 'Tax savings outweigh time cost'

        return {
            'short_hold': {
                'days': short_holding_days,
                'net_yield': short_net,
                'annualized': short_annualized
            },
            'long_hold': {
                'days': long_holding_days,
                'net_yield': long_net,
                'annualized': long_annualized
            },
            'recommendation': recommendation,
            'reason': reason
        }

    def wash_sale_check(
        self,
        symbol: str,
        trade_date: date,
        recent_trades: List[Dict]
    ) -> Dict:
        """
        Check for wash sale rule violations.
        """
        wash_sale_window = timedelta(days=30)

        # Find losses within window
        for trade in recent_trades:
            if trade['symbol'] != symbol:
                continue

            trade_date_dt = trade['date']
            days_diff = abs((trade_date - trade_date_dt).days)

            if days_diff <= 30 and trade.get('pnl', 0) < 0:
                return {
                    'wash_sale_risk': True,
                    'conflicting_trade': trade,
                    'warning': 'Loss may be disallowed under wash sale rule'
                }

        return {'wash_sale_risk': False}
```

---

## Performance Analytics

### Dividend Strategy Tracking

```python
class DividendPerformanceTracker:
    """
    Track dividend strategy performance.
    """

    def __init__(self):
        self.captures = []
        self.growth_holdings = []

    def record_capture(
        self,
        symbol: str,
        entry_date: date,
        exit_date: date,
        dividend_received: float,
        price_change: float,
        costs: float
    ):
        """Record dividend capture trade."""
        self.captures.append({
            'symbol': symbol,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'dividend': dividend_received,
            'price_change': price_change,
            'costs': costs,
            'net_return': dividend_received + price_change - costs
        })

    def capture_performance_summary(self) -> Dict:
        """Summarize capture strategy performance."""
        if not self.captures:
            return {'error': 'No captures recorded'}

        df = pd.DataFrame(self.captures)

        return {
            'total_captures': len(df),
            'total_dividends': df['dividend'].sum(),
            'total_price_change': df['price_change'].sum(),
            'total_costs': df['costs'].sum(),
            'total_net_return': df['net_return'].sum(),
            'win_rate': (df['net_return'] > 0).mean(),
            'avg_return': df['net_return'].mean(),
            'best_capture': df.loc[df['net_return'].idxmax()].to_dict(),
            'worst_capture': df.loc[df['net_return'].idxmin()].to_dict()
        }
```

---

## Academic References

1. **Elton, E. J., & Gruber, M. J. (1970)**. "Marginal Stockholder Tax Rates and the Clientele Effect." *Review of Economics and Statistics*.

2. **Kalay, A. (1982)**. "The Ex-Dividend Day Behavior of Stock Prices: A Re-Examination of the Clientele Effect." *Journal of Finance*.

3. **Graham, J. R., Michaely, R., & Roberts, M. R. (2003)**. "Do Price Discreteness and Transactions Costs Affect Stock Returns? Comparing Ex-Dividend Pricing Before and After Decimalization." *Journal of Finance*.

4. **Hartzmark, S. M., & Solomon, D. H. (2019)**. "The Dividend Disconnect." *Journal of Finance*.

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
tags: ["dividends", "capture-strategy", "ex-date", "yield", "tax-optimization"]
code_lines: 750
```

---

**END OF DOCUMENT**
