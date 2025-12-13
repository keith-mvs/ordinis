# Algorithmic Trading Strategies

## Overview

This document provides a comprehensive guide to algorithmic trading strategies, covering passive exploitation, arbitrage, market making, mean reversion, and execution algorithms. Content is organized for both conceptual understanding and practical implementation.

**Last Updated**: December 8, 2025

---

## 1. Index Fund Rebalancing Strategies

### 1.1 Trading Ahead of Index Rebalancing

Index funds, which manage retirement savings including pension funds, 401(k), and IRAs, must periodically "rebalance" their portfolios to match changes in their benchmark indices. This creates predictable trading patterns that algorithmic traders exploit.

**The Index Rebalancing Effect:**
- When stocks are added to or removed from indices (S&P 500, Russell 2000), index funds must buy/sell to maintain tracking
- This creates predictable demand/supply imbalances
- Active traders front-run these flows by positioning before rebalancing dates

**Estimated Impact on Passive Investors:**

| Index | Annual Cost to Passive Investors |
|-------|--------------------------------|
| S&P 500 | 21-28 basis points |
| Russell 2000 | 38-77 basis points |

```python
class IndexRebalanceStrategy:
    """
    Strategy to exploit index fund rebalancing flows.

    Key Events:
    - S&P 500: Quarterly rebalancing, ad-hoc additions/deletions
    - Russell: Annual reconstitution (June)
    - MSCI: Quarterly reviews
    """

    def __init__(self):
        self.rebalance_calendar = self._load_rebalance_calendar()

    def identify_opportunities(
        self,
        announced_additions: list,
        announced_deletions: list,
        rebalance_date: datetime
    ) -> list:
        """
        Identify trading opportunities from index changes.

        Additions: Expected buying pressure -> Go long before
        Deletions: Expected selling pressure -> Go short before
        """
        opportunities = []

        for symbol in announced_additions:
            # Estimate index fund demand
            demand = self._estimate_index_demand(symbol)

            opportunities.append({
                "symbol": symbol,
                "direction": "long",
                "entry_date": rebalance_date - timedelta(days=5),
                "exit_date": rebalance_date + timedelta(days=1),
                "expected_demand": demand,
                "confidence": self._calculate_confidence(symbol)
            })

        for symbol in announced_deletions:
            supply = self._estimate_index_supply(symbol)

            opportunities.append({
                "symbol": symbol,
                "direction": "short",
                "entry_date": rebalance_date - timedelta(days=5),
                "exit_date": rebalance_date + timedelta(days=1),
                "expected_supply": supply,
                "confidence": self._calculate_confidence(symbol)
            })

        return opportunities

    def _estimate_index_demand(self, symbol: str) -> float:
        """
        Estimate buying demand from index funds.

        Factors:
        - Total AUM tracking the index
        - Stock's weight in the index
        - Average daily volume (ADV)
        """
        # Implementation
        pass
```

**Key Considerations:**
- Announcement timing varies by index provider
- Competition from other algo traders has reduced alpha
- Transaction costs must be carefully managed
- Regulatory scrutiny on front-running practices

---

## 2. Pairs Trading

### 2.1 Concept

Pairs trading is a market-neutral strategy that profits from relative value discrepancies between correlated securities. Unlike pure arbitrage, pairs trading does not guarantee convergence.

**Key Characteristics:**
- Long-short position in correlated assets
- Market-neutral (beta-hedged)
- Profits from mean reversion of spread
- Belongs to statistical arbitrage family

### 2.2 Implementation

```python
from statsmodels.tsa.stattools import coint
import numpy as np
import pandas as pd


class PairsTrader:
    """
    Pairs trading strategy based on cointegration.
    """

    def __init__(
        self,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        lookback: int = 60,
        stop_loss_zscore: float = 4.0
    ):
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.lookback = lookback
        self.stop_loss_zscore = stop_loss_zscore

    def find_cointegrated_pairs(
        self,
        price_data: pd.DataFrame,
        significance: float = 0.05
    ) -> list:
        """
        Find cointegrated pairs using Engle-Granger test.

        Returns:
            List of (stock1, stock2, p_value, hedge_ratio) tuples
        """
        symbols = price_data.columns.tolist()
        pairs = []

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                # Test for cointegration
                score, pvalue, _ = coint(
                    price_data[sym1],
                    price_data[sym2]
                )

                if pvalue < significance:
                    # Calculate hedge ratio via OLS
                    hedge_ratio = self._calculate_hedge_ratio(
                        price_data[sym1],
                        price_data[sym2]
                    )

                    pairs.append({
                        "stock1": sym1,
                        "stock2": sym2,
                        "pvalue": pvalue,
                        "hedge_ratio": hedge_ratio
                    })

        return sorted(pairs, key=lambda x: x["pvalue"])

    def calculate_spread(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        hedge_ratio: float
    ) -> pd.Series:
        """
        Calculate the spread between two cointegrated series.

        Spread = Price1 - hedge_ratio * Price2
        """
        return prices1 - hedge_ratio * prices2

    def calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """
        Calculate z-score of spread for signal generation.
        """
        mean = spread.rolling(self.lookback).mean()
        std = spread.rolling(self.lookback).std()
        return (spread - mean) / std

    def generate_signal(self, zscore: float) -> str:
        """
        Generate trading signal based on z-score.

        Returns:
            'long_spread': Buy stock1, sell stock2
            'short_spread': Sell stock1, buy stock2
            'close': Close position
            'stop_loss': Stop loss triggered
            'hold': No action
        """
        if abs(zscore) > self.stop_loss_zscore:
            return "stop_loss"

        if zscore > self.entry_zscore:
            return "short_spread"  # Spread too high, expect reversion
        elif zscore < -self.entry_zscore:
            return "long_spread"  # Spread too low, expect reversion
        elif abs(zscore) < self.exit_zscore:
            return "close"

        return "hold"

    def _calculate_hedge_ratio(
        self,
        y: pd.Series,
        x: pd.Series
    ) -> float:
        """Calculate hedge ratio via OLS regression."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(x.values.reshape(-1, 1), y.values)
        return model.coef_[0]
```

### 2.3 Risks and Considerations

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Divergence risk** | Spread may not converge | Stop-loss limits |
| **Regime change** | Correlation breakdown | Regular revalidation |
| **Execution risk** | Legs executed at different prices | Simultaneous execution |
| **Liquidity risk** | Wide spreads in one leg | Liquidity filters |
| **Model risk** | Cointegration may be spurious | Out-of-sample testing |

---

## 3. Arbitrage Strategies

### 3.1 Arbitrage Fundamentals

Arbitrage exploits price discrepancies between related instruments for risk-free profit. In academic terms: a transaction with no negative cash flow in any state and positive cash flow in at least one state.

**Conditions for Arbitrage:**
1. Same asset trades at different prices across markets
2. Two assets with identical cash flows trade at different prices
3. Asset with known future price doesn't trade at discounted present value

### 3.2 Types of Arbitrage

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity."""
    strategy_type: str
    instruments: list
    expected_profit: float
    risk_level: str
    execution_window_ms: int
    capital_required: float


class ArbitrageStrategy(ABC):
    """Base class for arbitrage strategies."""

    @abstractmethod
    def scan_opportunities(self) -> list:
        """Scan for arbitrage opportunities."""
        pass

    @abstractmethod
    def execute(self, opportunity: ArbitrageOpportunity) -> dict:
        """Execute arbitrage trade."""
        pass


class IndexArbitrage(ArbitrageStrategy):
    """
    S&P 500 Index vs. Futures Arbitrage.

    When futures price diverges from fair value:
    - Futures premium too high: Buy basket, sell futures
    - Futures discount too deep: Sell basket, buy futures
    """

    def __init__(self, fair_value_threshold_bps: float = 10):
        self.threshold = fair_value_threshold_bps

    def calculate_fair_value(
        self,
        spot_index: float,
        risk_free_rate: float,
        dividend_yield: float,
        days_to_expiry: int
    ) -> float:
        """
        Calculate theoretical futures fair value.

        F = S * e^((r - d) * T)

        Where:
            S = spot index level
            r = risk-free rate
            d = dividend yield
            T = time to expiry (years)
        """
        T = days_to_expiry / 365
        return spot_index * np.exp((risk_free_rate - dividend_yield) * T)

    def scan_opportunities(
        self,
        spot_price: float,
        futures_price: float,
        fair_value: float
    ) -> ArbitrageOpportunity:
        """
        Scan for index arbitrage opportunities.
        """
        basis = (futures_price - fair_value) / fair_value * 10000  # bps

        if abs(basis) > self.threshold:
            direction = "buy_basket_sell_futures" if basis > 0 else "sell_basket_buy_futures"

            return ArbitrageOpportunity(
                strategy_type="index_arbitrage",
                instruments=["SPY", "ES"],  # ETF and futures
                expected_profit=abs(basis) * self._notional_size(),
                risk_level="low",
                execution_window_ms=100,
                capital_required=self._calculate_capital()
            )

        return None


class TriangularArbitrage(ArbitrageStrategy):
    """
    Currency triangular arbitrage.

    Exploits inconsistencies in FX cross rates.
    Example: USD -> EUR -> GBP -> USD
    """

    def scan_opportunities(
        self,
        usd_eur: float,
        eur_gbp: float,
        gbp_usd: float
    ) -> ArbitrageOpportunity:
        """
        Check for triangular arbitrage opportunity.

        If USD -> EUR -> GBP -> USD != 1, opportunity exists.
        """
        # Forward path: USD -> EUR -> GBP -> USD
        forward_rate = (1 / usd_eur) * (1 / eur_gbp) * gbp_usd

        # Reverse path: USD -> GBP -> EUR -> USD
        reverse_rate = (1 / gbp_usd) * eur_gbp * usd_eur

        # Check for profitable paths
        profit_forward = forward_rate - 1
        profit_reverse = reverse_rate - 1

        if profit_forward > 0.0001:  # 1 bps threshold
            return ArbitrageOpportunity(
                strategy_type="triangular_fx",
                instruments=["EURUSD", "EURGBP", "GBPUSD"],
                expected_profit=profit_forward,
                risk_level="low",
                execution_window_ms=50,
                capital_required=1_000_000
            )

        return None
```

### 3.3 Execution Risk

True arbitrage requires simultaneous execution. In practice:
- "Leg-in and leg-out" risk: One leg executes, other doesn't
- Market impact: Large orders move prices
- Technology requirements: Ultra-low latency needed
- Capital requirements: Despite being "risk-free", capital must be posted

---

## 4. Delta-Neutral Strategies

### 4.1 Concept

Delta-neutral portfolios are constructed so the portfolio value remains unchanged due to small changes in the underlying asset price. This is achieved by offsetting positive and negative delta components.

```python
class DeltaNeutralPortfolio:
    """
    Manage delta-neutral options portfolio.
    """

    def __init__(self, target_delta: float = 0.0):
        self.target_delta = target_delta
        self.positions = []

    def calculate_portfolio_delta(self) -> float:
        """Calculate total portfolio delta."""
        total_delta = 0

        for position in self.positions:
            if position["type"] == "stock":
                total_delta += position["shares"]
            elif position["type"] == "option":
                total_delta += position["contracts"] * 100 * position["delta"]

        return total_delta

    def hedge_to_neutral(self, stock_price: float) -> dict:
        """
        Calculate hedge needed to achieve delta neutrality.

        Returns:
            Dictionary with hedge instructions
        """
        current_delta = self.calculate_portfolio_delta()
        delta_to_hedge = current_delta - self.target_delta

        if abs(delta_to_hedge) < 1:  # Threshold
            return {"action": "none", "shares": 0}

        if delta_to_hedge > 0:
            return {
                "action": "sell",
                "shares": int(delta_to_hedge),
                "reason": "Portfolio too long delta"
            }
        else:
            return {
                "action": "buy",
                "shares": int(abs(delta_to_hedge)),
                "reason": "Portfolio too short delta"
            }

    def rebalance_frequency(self, gamma: float, volatility: float) -> str:
        """
        Determine rebalancing frequency based on gamma exposure.

        Higher gamma = more frequent rebalancing needed
        """
        gamma_dollars = abs(gamma) * 100  # Per 1% move

        if gamma_dollars > 10000:
            return "continuous"  # Every few minutes
        elif gamma_dollars > 1000:
            return "hourly"
        elif gamma_dollars > 100:
            return "daily"
        else:
            return "weekly"
```

### 4.2 Applications

| Strategy | Description | Risk Profile |
|----------|-------------|--------------|
| **Gamma scalping** | Profit from rebalancing gains | Long gamma, short theta |
| **Volatility trading** | Bet on vol changes | Vega exposure |
| **Market making** | Provide liquidity | Inventory risk |
| **Dispersion trading** | Index vol vs component vol | Correlation risk |

---

## 5. Mean Reversion

### 5.1 Concept

Mean reversion assumes that extreme prices are temporary and will revert to an average over time. The Ornstein-Uhlenbeck process is a common model:

dX(t) = θ(μ - X(t))dt + σdW(t)

Where:
- θ = speed of mean reversion
- μ = long-term mean
- σ = volatility
- W(t) = Wiener process

### 5.2 Implementation

```python
class MeanReversionStrategy:
    """
    Mean reversion trading strategy.
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_std: float = 2.0,
        exit_std: float = 0.5
    ):
        self.lookback = lookback
        self.entry_std = entry_std
        self.exit_std = exit_std

    def calculate_signals(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate mean reversion signals.

        Buy when price is entry_std below mean
        Sell when price is entry_std above mean
        Exit when price returns to exit_std of mean
        """
        df = pd.DataFrame(index=prices.index)

        # Calculate rolling statistics
        df["price"] = prices
        df["mean"] = prices.rolling(self.lookback).mean()
        df["std"] = prices.rolling(self.lookback).std()

        # Z-score
        df["zscore"] = (df["price"] - df["mean"]) / df["std"]

        # Signals
        df["signal"] = 0
        df.loc[df["zscore"] < -self.entry_std, "signal"] = 1   # Buy
        df.loc[df["zscore"] > self.entry_std, "signal"] = -1   # Sell

        return df

    def estimate_half_life(self, spread: pd.Series) -> float:
        """
        Estimate mean reversion half-life using OLS.

        Half-life = -ln(2) / ln(1 + β)

        Where β is from: ΔS(t) = α + β*S(t-1) + ε
        """
        spread_lag = spread.shift(1)
        spread_diff = spread.diff()

        # Remove NaN
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.loc[spread_lag.index]

        # OLS regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(spread_lag.values.reshape(-1, 1), spread_diff.values)

        beta = model.coef_[0]

        if beta >= 0:
            return float("inf")  # No mean reversion

        half_life = -np.log(2) / np.log(1 + beta)
        return half_life
```

### 5.3 When Mean Reversion Works

| Condition | Favorable | Unfavorable |
|-----------|-----------|-------------|
| **Market regime** | Range-bound, sideways | Strong trending |
| **Volatility** | Moderate | Extreme |
| **Liquidity** | High | Low |
| **News flow** | Normal | Event-driven |

---

## 6. Scalping and Market Making

### 6.1 Scalping

Scalping involves providing liquidity by capturing the bid-ask spread, with positions held for minutes or less.

```python
class ScalpingStrategy:
    """
    Scalping strategy for capturing bid-ask spread.
    """

    def __init__(
        self,
        min_spread_bps: float = 5,
        max_position_time_seconds: int = 300,
        max_loss_per_trade_bps: float = 10
    ):
        self.min_spread_bps = min_spread_bps
        self.max_position_time = max_position_time_seconds
        self.max_loss = max_loss_per_trade_bps

    def evaluate_opportunity(
        self,
        bid: float,
        ask: float,
        mid: float,
        volume: int
    ) -> dict:
        """
        Evaluate scalping opportunity.
        """
        spread_bps = (ask - bid) / mid * 10000

        if spread_bps < self.min_spread_bps:
            return {"action": "skip", "reason": "Spread too tight"}

        # Estimate fill probability
        fill_prob = self._estimate_fill_probability(volume, spread_bps)

        expected_profit = (spread_bps / 2) * fill_prob - self.max_loss * (1 - fill_prob)

        if expected_profit > 0:
            return {
                "action": "quote",
                "bid_price": bid + 0.01,
                "ask_price": ask - 0.01,
                "expected_profit_bps": expected_profit
            }

        return {"action": "skip", "reason": "Negative expected value"}
```

### 6.2 Market Making Obligations

Registered market makers have exchange-mandated obligations:

| Exchange | Requirement |
|----------|-------------|
| **NASDAQ** | Post at least one bid and one ask at some price level |
| **NYSE** | Maintain continuous two-sided quotes |
| **CME** | Meet volume and spread requirements |

---

## 7. Execution Algorithms

### 7.1 Algorithm Types

Large orders must be broken into smaller pieces to minimize market impact.

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **TWAP** | Time-weighted average price | Spread execution evenly over time |
| **VWAP** | Volume-weighted average price | Match market volume profile |
| **Implementation Shortfall** | Minimize slippage from decision price | Urgent execution |
| **POV** | Percentage of volume | Limit market impact |
| **Iceberg** | Hide order size | Large orders in thin markets |
| **Sniper** | Target hidden liquidity | Find dark pool orders |

### 7.2 TWAP Implementation

```python
class TWAPExecutor:
    """
    Time-Weighted Average Price execution algorithm.
    """

    def __init__(
        self,
        total_quantity: int,
        duration_minutes: int,
        interval_seconds: int = 60
    ):
        self.total_quantity = total_quantity
        self.duration = duration_minutes
        self.interval = interval_seconds

    def generate_schedule(self) -> list:
        """
        Generate execution schedule.

        Splits order evenly across time periods.
        """
        num_intervals = (self.duration * 60) // self.interval
        quantity_per_interval = self.total_quantity // num_intervals
        remainder = self.total_quantity % num_intervals

        schedule = []
        for i in range(num_intervals):
            qty = quantity_per_interval
            if i < remainder:
                qty += 1  # Distribute remainder

            schedule.append({
                "interval": i,
                "time_offset_seconds": i * self.interval,
                "quantity": qty
            })

        return schedule

    def execute_slice(self, slice_quantity: int, market_data: dict) -> dict:
        """
        Execute one slice of the order.

        Uses limit orders slightly better than market to avoid
        taking liquidity aggressively.
        """
        mid = (market_data["bid"] + market_data["ask"]) / 2
        spread = market_data["ask"] - market_data["bid"]

        # Place order at mid or slightly better
        limit_price = mid - (spread * 0.1)  # 10% inside mid

        return {
            "order_type": "limit",
            "quantity": slice_quantity,
            "limit_price": limit_price,
            "time_in_force": "IOC"  # Immediate or cancel
        }
```

### 7.3 VWAP Implementation

```python
class VWAPExecutor:
    """
    Volume-Weighted Average Price execution algorithm.
    """

    def __init__(
        self,
        total_quantity: int,
        historical_volume_profile: pd.Series
    ):
        self.total_quantity = total_quantity
        self.volume_profile = historical_volume_profile

    def generate_schedule(self) -> list:
        """
        Generate execution schedule based on volume profile.

        Allocates more shares to high-volume periods.
        """
        # Normalize volume profile
        normalized = self.volume_profile / self.volume_profile.sum()

        schedule = []
        cumulative = 0

        for time, vol_pct in normalized.items():
            quantity = int(self.total_quantity * vol_pct)
            cumulative += quantity

            schedule.append({
                "time": time,
                "target_quantity": quantity,
                "volume_weight": vol_pct,
                "cumulative_target": cumulative
            })

        return schedule

    def calculate_benchmark(
        self,
        executions: list,
        market_vwap: float
    ) -> dict:
        """
        Calculate execution quality vs VWAP benchmark.
        """
        total_shares = sum(e["quantity"] for e in executions)
        total_cost = sum(e["quantity"] * e["price"] for e in executions)
        execution_vwap = total_cost / total_shares

        slippage_bps = (execution_vwap - market_vwap) / market_vwap * 10000

        return {
            "execution_vwap": execution_vwap,
            "market_vwap": market_vwap,
            "slippage_bps": slippage_bps,
            "total_shares": total_shares
        }
```

---

## 8. Dark Pool Strategies

### 8.1 Dark Pool Characteristics

Dark pools are private trading venues that provide anonymity and reduced market impact:

- Orders are hidden ("iceberged")
- No public order book
- Typically better execution for large orders
- Reduced information leakage

### 8.2 Detection Algorithms

Algorithmic traders use "sniffing" or "pinging" to detect large hidden orders:

```python
class DarkPoolDetector:
    """
    Detect large hidden orders in dark pools.

    WARNING: Some detection practices may be considered
    predatory trading. Ensure compliance with regulations.
    """

    def __init__(self, ping_size: int = 100):
        self.ping_size = ping_size
        self.detection_threshold = 0.8  # 80% fill rate

    def ping_for_liquidity(
        self,
        symbol: str,
        side: str,
        price_levels: list
    ) -> dict:
        """
        Send small orders to detect hidden liquidity.

        If small orders consistently fill, large hidden
        order may be present.
        """
        results = []

        for price in price_levels:
            # Send small IOC order
            fill_result = self._send_ping_order(
                symbol=symbol,
                side=side,
                quantity=self.ping_size,
                price=price
            )

            results.append({
                "price": price,
                "filled": fill_result["filled"],
                "fill_rate": fill_result["fill_rate"]
            })

        # Analyze results
        high_fill_prices = [
            r["price"] for r in results
            if r["fill_rate"] > self.detection_threshold
        ]

        if len(high_fill_prices) > 2:
            return {
                "detected": True,
                "estimated_size": self._estimate_hidden_size(results),
                "price_range": (min(high_fill_prices), max(high_fill_prices))
            }

        return {"detected": False}
```

### 8.3 Ethical Considerations

Dark pool detection strategies are controversial:
- May constitute predatory trading
- Can harm institutional investors
- Subject to regulatory scrutiny
- "Arms race" dynamics reduce profitability

---

## 9. Market Timing and Backtesting

### 9.1 Market Timing Approaches

```python
class MarketTimingStrategy:
    """
    Market timing strategy using technical indicators
    and pattern recognition.
    """

    def __init__(self):
        self.states = ["bullish", "bearish", "neutral"]
        self.current_state = "neutral"

    def analyze_regime(self, data: pd.DataFrame) -> str:
        """
        Classify current market regime using multiple indicators.
        """
        signals = {}

        # Moving average signals
        signals["ma_cross"] = self._ma_crossover_signal(data)

        # Momentum
        signals["momentum"] = self._momentum_signal(data)

        # Volatility regime
        signals["volatility"] = self._volatility_signal(data)

        # Pattern recognition (FSM-based)
        signals["pattern"] = self._pattern_signal(data)

        # Aggregate signals
        return self._aggregate_signals(signals)

    def _ma_crossover_signal(self, data: pd.DataFrame) -> str:
        """Moving average crossover signal."""
        ma_short = data["close"].rolling(50).mean().iloc[-1]
        ma_long = data["close"].rolling(200).mean().iloc[-1]

        if ma_short > ma_long * 1.02:
            return "bullish"
        elif ma_short < ma_long * 0.98:
            return "bearish"
        return "neutral"
```

### 9.2 Backtesting Best Practices

| Stage | Purpose | Pitfalls to Avoid |
|-------|---------|-------------------|
| **Backtest** | Test on in-sample data | Overfitting, look-ahead bias |
| **Walk-forward** | Out-of-sample validation | Data snooping |
| **Paper trading** | Live market conditions | Execution assumptions |
| **Live trading** | Real capital | Position sizing |

**Optimization Guidelines:**
- Modify inputs +/- 10% to test robustness
- Run Monte Carlo simulations
- Account for slippage and commissions
- Use multiple performance metrics

---

## 10. Non-Ergodicity in Trading

### 10.1 Concept

Modern algorithmic trading recognizes that financial markets are **non-ergodic**: time averages do not equal ensemble averages. Returns are neither independent nor normally distributed.

### 10.2 Practical Implications

```python
from scipy.stats import binom


class StrategyValidator:
    """
    Validate strategy using non-ergodic assumptions.

    Uses Binomial Evolution Function to estimate if
    results could be achieved randomly.
    """

    def calculate_random_probability(
        self,
        trades: list
    ) -> float:
        """
        Calculate probability of achieving results randomly.

        Steps:
        1. Aggregate consecutive same-direction trades
        2. Convert to binary sequence (1=win, 0=loss)
        3. Calculate binomial probability

        Low probability = real predictive capacity
        High probability = results may be random
        """
        # Step 1: Aggregate consecutive trades
        aggregated = self._aggregate_consecutive_trades(trades)

        # Step 2: Convert to binary
        binary_sequence = [1 if t["pnl"] > 0 else 0 for t in aggregated]

        # Step 3: Calculate binomial probability
        n = len(binary_sequence)
        k = sum(binary_sequence)  # Number of wins
        p = 0.5  # Fair coin assumption

        # P(X >= k) where X ~ Binomial(n, 0.5)
        prob_random = 1 - binom.cdf(k - 1, n, p)

        return prob_random

    def _aggregate_consecutive_trades(self, trades: list) -> list:
        """
        Aggregate consecutive trades in same direction.
        """
        if not trades:
            return []

        aggregated = []
        current_group = {"direction": trades[0]["direction"], "pnl": 0}

        for trade in trades:
            if trade["direction"] == current_group["direction"]:
                current_group["pnl"] += trade["pnl"]
            else:
                aggregated.append(current_group)
                current_group = {
                    "direction": trade["direction"],
                    "pnl": trade["pnl"]
                }

        aggregated.append(current_group)
        return aggregated

    def assess_predictive_capacity(
        self,
        trades: list
    ) -> dict:
        """
        Assess if strategy has genuine predictive capacity.
        """
        prob = self.calculate_random_probability(trades)

        if prob < 0.01:
            assessment = "strong"
            confidence = "Strategy likely has real predictive capacity"
        elif prob < 0.05:
            assessment = "moderate"
            confidence = "Results unlikely to be purely random"
        elif prob < 0.10:
            assessment = "weak"
            confidence = "Some evidence of predictive capacity"
        else:
            assessment = "none"
            confidence = "Results consistent with random trading"

        return {
            "random_probability": prob,
            "assessment": assessment,
            "confidence": confidence,
            "total_trades": len(trades),
            "win_count": sum(1 for t in trades if t.get("pnl", 0) > 0)
        }
```

---

## 11. References

### Academic Sources
- De Prado, M. (2018): "Advances in Financial Machine Learning"
- Chan, E. (2013): "Algorithmic Trading: Winning Strategies"
- Kissell, R. (2013): "The Science of Algorithmic Trading"
- Johnson, B. (2010): "Algorithmic Trading & DMA"

### Regulatory References
- SEC Rule 15c3-5: Market Access Rule
- FINRA Rule 5310: Best Execution
- MiFID II: Algorithmic Trading Requirements

### Key Papers
- Petajisto, A. (2011): "The Index Premium and Hidden Costs of Index Investing"
- Montgomery, J.: "Front-Running Index Funds" (Bridgeway Capital)
- Gatev et al. (2006): "Pairs Trading: Performance of Relative-Value Arbitrage Rule"
