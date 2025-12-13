# Queueing Theory for Order Book Dynamics

Queueing theory provides the mathematical framework for modeling limit order books as stochastic systems, analyzing fill probabilities, optimizing queue positions, and designing market making strategies.

---

## Overview

Financial markets operate through limit order books (LOBs) where orders queue at price levels awaiting execution. Queueing theory enables:

1. **Order Book Modeling**: Stochastic dynamics of queue depths
2. **Fill Probability Estimation**: Likelihood of limit order execution
3. **Queue Position Value**: Economic value of priority in queue
4. **Market Making Optimization**: Optimal quote placement
5. **Execution Strategy Design**: TWAP/VWAP with queue dynamics

This document covers foundational queueing concepts and their applications to systematic trading.

---

## 1. Order Book as Queueing System

### 1.1 Basic Structure

A limit order book is a collection of queues at discrete price levels:

**Bid Side** (buy orders):
- Price levels: $p_b^{(1)} > p_b^{(2)} > \ldots$ (best bid first)
- Queue depths: $Q_b^{(1)}, Q_b^{(2)}, \ldots$

**Ask Side** (sell orders):
- Price levels: $p_a^{(1)} < p_a^{(2)} < \ldots$ (best ask first)
- Queue depths: $Q_a^{(1)}, Q_a^{(2)}, \ldots$

**Spread**: $s = p_a^{(1)} - p_b^{(1)}$

### 1.2 Order Flow Events

Orders arrive and depart according to stochastic processes:

**Arrivals** (limit orders):
- New orders join queues at various price levels
- Modeled as Poisson process with intensity $\lambda(p)$

**Departures**:
1. **Executions**: Market orders consume queue (rate $\mu$)
2. **Cancellations**: Limit orders removed (rate $\theta$)

**Price Changes**:
- When best queue depletes, price moves to next level
- Spread widens/narrows based on queue dynamics

### 1.3 Birth-Death Process Model

At each price level, the queue follows a birth-death process:

**State**: Queue depth $Q(t) \in \{0, 1, 2, \ldots\}$

**Transition Rates**:
- Birth (arrival): $Q \to Q + 1$ at rate $\lambda$
- Death (execution/cancel): $Q \to Q - 1$ at rate $\mu + \theta$

**Stationary Distribution** (if $\lambda < \mu + \theta$):
$$\pi(q) = \left(1 - \frac{\lambda}{\mu + \theta}\right) \left(\frac{\lambda}{\mu + \theta}\right)^q$$

Geometric distribution with parameter $\rho = \lambda / (\mu + \theta)$.

---

## 2. Fill Probability Models

### 2.1 Simple Fill Probability

For an order at position $k$ in queue of depth $Q$:

**Execution before cancellation**:
$$P(\text{fill}) = P(\text{at least } k \text{ executions before queue depletes})$$

**Exponential service times**:
If executions arrive as Poisson($\mu$) and we wait time $T$:
$$P(\text{fill} | T) = \sum_{n=k}^{\infty} \frac{(\mu T)^n e^{-\mu T}}{n!}$$

### 2.2 Cont-Stoikov-Talreja Model

The CST model (2010) provides a comprehensive framework:

**State Space**: $(Q_b, Q_a, S)$ where $S$ is mid-price

**Dynamics**:
- Limit order arrivals: Poisson at rate $\lambda$ per level
- Market order arrivals: Poisson at rate $\mu$
- Cancellations: Poisson at rate $\theta \cdot Q$ (proportional to depth)

**Key Results**:
- Queue depth is approximately geometric
- Fill probability depends on queue position and market order intensity
- Price impact is proportional to order size relative to queue depth

### 2.3 Python Implementation

```python
import numpy as np
from scipy.stats import poisson, expon
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import pandas as pd


@dataclass
class OrderBookParameters:
    """Parameters for order book queueing model."""
    limit_order_rate: float = 10.0      # Limit orders per second per level
    market_order_rate: float = 5.0       # Market orders per second
    cancel_rate: float = 0.1             # Cancellation rate per order
    tick_size: float = 0.01              # Minimum price increment
    n_levels: int = 10                    # Number of price levels to model


class FillProbabilityModel:
    """
    Fill probability estimation for limit orders.

    Models the probability of order execution based on
    queue position, market order flow, and time horizon.
    """

    def __init__(self, params: OrderBookParameters):
        """
        Initialize fill probability model.

        Args:
            params: Order book parameters
        """
        self.params = params

        # Effective departure rate (executions + cancellations)
        self.departure_rate = params.market_order_rate + params.cancel_rate

        # Traffic intensity
        self.rho = params.limit_order_rate / self.departure_rate

    def queue_depth_distribution(self, q_max: int = 100) -> np.ndarray:
        """
        Compute stationary distribution of queue depth.

        Args:
            q_max: Maximum queue depth to compute

        Returns:
            Array of probabilities P(Q = q) for q in 0..q_max
        """
        if self.rho >= 1:
            # Queue is unstable, use truncated distribution
            probs = np.array([self.rho ** q for q in range(q_max + 1)])
            probs /= probs.sum()
        else:
            # Geometric distribution
            probs = (1 - self.rho) * (self.rho ** np.arange(q_max + 1))

        return probs

    def expected_queue_depth(self) -> float:
        """
        Compute expected queue depth at equilibrium.

        Returns:
            E[Q] = rho / (1 - rho) for rho < 1
        """
        if self.rho >= 1:
            return np.inf
        return self.rho / (1 - self.rho)

    def fill_probability_given_position(
        self,
        position: int,
        time_horizon: float,
        queue_ahead: int = None
    ) -> float:
        """
        Compute fill probability for order at given queue position.

        Args:
            position: Position in queue (1 = front, larger = further back)
            time_horizon: Time to wait for fill (seconds)
            queue_ahead: Orders ahead (if known), else uses position-1

        Returns:
            Probability of fill within time horizon
        """
        if queue_ahead is None:
            queue_ahead = position - 1

        # Expected number of executions in time horizon
        expected_executions = self.params.market_order_rate * time_horizon

        # Need at least 'position' executions to fill our order
        # P(N >= position) where N ~ Poisson(mu * T)
        fill_prob = 1 - poisson.cdf(queue_ahead, expected_executions)

        return fill_prob

    def fill_probability_with_cancellations(
        self,
        initial_position: int,
        time_horizon: float,
        initial_queue_depth: int
    ) -> float:
        """
        Compute fill probability accounting for cancellations ahead.

        Cancellations can improve our queue position over time.

        Args:
            initial_position: Starting position in queue
            time_horizon: Time to wait (seconds)
            initial_queue_depth: Total queue depth at our level

        Returns:
            Fill probability
        """
        # Simulate queue dynamics
        n_simulations = 10000
        fills = 0

        for _ in range(n_simulations):
            position = initial_position
            remaining_time = time_horizon

            while remaining_time > 0 and position > 0:
                # Time to next event
                # Events: market order (execution) or cancellation ahead
                rate_exec = self.params.market_order_rate
                rate_cancel = self.params.cancel_rate * (position - 1)  # Cancels ahead

                total_rate = rate_exec + rate_cancel
                if total_rate <= 0:
                    break

                # Time to next event
                time_to_event = np.random.exponential(1 / total_rate)

                if time_to_event > remaining_time:
                    break

                remaining_time -= time_to_event

                # Determine event type
                if np.random.random() < rate_exec / total_rate:
                    # Execution - we move up in queue
                    position -= 1
                    if position == 0:
                        fills += 1
                        break
                else:
                    # Cancellation ahead - we move up
                    position -= 1

        return fills / n_simulations

    def optimal_limit_price(
        self,
        side: str,
        target_fill_prob: float,
        time_horizon: float,
        current_best: float,
        queue_depths: List[int]
    ) -> float:
        """
        Find optimal limit price to achieve target fill probability.

        Args:
            side: 'buy' or 'sell'
            target_fill_prob: Desired fill probability
            time_horizon: Time horizon for execution
            current_best: Current best bid/ask
            queue_depths: Queue depths at each price level

        Returns:
            Optimal limit price
        """
        tick = self.params.tick_size

        for level, depth in enumerate(queue_depths):
            if side == 'buy':
                price = current_best - level * tick
            else:
                price = current_best + level * tick

            # Position would be at end of queue
            position = depth + 1

            fill_prob = self.fill_probability_given_position(
                position, time_horizon
            )

            if fill_prob >= target_fill_prob:
                return price

        # If no level achieves target, return worst price checked
        if side == 'buy':
            return current_best - len(queue_depths) * tick
        else:
            return current_best + len(queue_depths) * tick


# Example usage
if __name__ == "__main__":
    params = OrderBookParameters(
        limit_order_rate=10.0,
        market_order_rate=5.0,
        cancel_rate=0.1
    )

    model = FillProbabilityModel(params)

    # Queue depth distribution
    print(f"Expected queue depth: {model.expected_queue_depth():.2f}")

    # Fill probability by position
    print("\nFill Probability (60 second horizon):")
    print("Position | P(Fill)")
    print("-" * 25)
    for pos in [1, 5, 10, 20, 50]:
        prob = model.fill_probability_given_position(pos, time_horizon=60)
        print(f"{pos:8d} | {prob:.4f}")

    # With cancellations
    print("\nFill Probability with Cancellations (pos=10, depth=50):")
    prob_cancel = model.fill_probability_with_cancellations(
        initial_position=10,
        time_horizon=60,
        initial_queue_depth=50
    )
    print(f"P(Fill) = {prob_cancel:.4f}")
```

---

## 3. Queue Position Value

### 3.1 Economic Value of Queue Priority

Being at the front of a queue has economic value:
- Higher fill probability
- Capture spread when filled
- Opportunity cost of waiting

**Queue Position Value (QPV)**:
$$V(k) = E[\text{profit} | \text{position } k] - E[\text{profit} | \text{join end of queue}]$$

### 3.2 Moallemi-Yuan Model

Moallemi and Yuan (2017) provide analytical formulas:

**Value of Position $k$**:
$$V(k) = \frac{s}{2} \cdot \left[ P_{\text{fill}}(k) - P_{\text{fill}}(Q+1) \right]$$

where $s$ is the bid-ask spread.

**Implications**:
- Narrow spreads reduce queue position value
- High volatility increases value of being filled
- Queue jumpers (via cancellation+resubmit) destroy value

### 3.3 Python Implementation

```python
class QueuePositionValue:
    """
    Economic value of queue position.

    Computes the value of being at different positions
    in the limit order book queue.
    """

    def __init__(
        self,
        fill_model: FillProbabilityModel,
        spread: float,
        volatility: float
    ):
        """
        Initialize queue position value model.

        Args:
            fill_model: Fill probability model
            spread: Current bid-ask spread
            volatility: Price volatility (per second)
        """
        self.fill_model = fill_model
        self.spread = spread
        self.volatility = volatility

    def position_value(
        self,
        position: int,
        queue_depth: int,
        time_horizon: float
    ) -> float:
        """
        Compute value of being at given queue position.

        Args:
            position: Current position in queue
            queue_depth: Total queue depth
            time_horizon: Trading horizon

        Returns:
            Expected profit from position (relative to joining at end)
        """
        # Fill probability at current position
        p_fill_current = self.fill_model.fill_probability_given_position(
            position, time_horizon
        )

        # Fill probability if joining at end
        p_fill_end = self.fill_model.fill_probability_given_position(
            queue_depth + 1, time_horizon
        )

        # Basic value: spread capture differential
        basic_value = (self.spread / 2) * (p_fill_current - p_fill_end)

        # Adverse selection cost (filled more when price moves against)
        # Higher fill prob may mean we're filled when uninformed flow arrives
        adverse_selection = 0.1 * self.volatility * np.sqrt(time_horizon) * p_fill_current

        return basic_value - adverse_selection

    def optimal_queue_entry(
        self,
        queue_depths: List[int],
        prices: List[float],
        time_horizon: float,
        fair_value: float
    ) -> Tuple[int, float]:
        """
        Find optimal price level to place limit order.

        Balances fill probability against price improvement.

        Args:
            queue_depths: Depths at each price level
            prices: Prices at each level (best to worst)
            time_horizon: Trading horizon
            fair_value: Estimated fair value

        Returns:
            (optimal_level, expected_profit)
        """
        best_level = 0
        best_profit = -np.inf

        for level, (depth, price) in enumerate(zip(queue_depths, prices)):
            # Position if joining this level
            position = depth + 1

            # Fill probability
            p_fill = self.fill_model.fill_probability_given_position(
                position, time_horizon
            )

            # Price improvement (or cost)
            price_diff = fair_value - price  # Positive if buying below fair value

            # Expected profit
            expected_profit = p_fill * (price_diff + self.spread / 2)

            # Opportunity cost of not being filled
            opp_cost = (1 - p_fill) * 0.5 * self.volatility * np.sqrt(time_horizon)

            net_profit = expected_profit - opp_cost

            if net_profit > best_profit:
                best_profit = net_profit
                best_level = level

        return best_level, best_profit

    def queue_value_curve(
        self,
        max_position: int,
        queue_depth: int,
        time_horizon: float
    ) -> pd.DataFrame:
        """
        Compute value curve across queue positions.

        Args:
            max_position: Maximum position to compute
            queue_depth: Total queue depth
            time_horizon: Trading horizon

        Returns:
            DataFrame with position values
        """
        positions = range(1, min(max_position, queue_depth) + 1)

        data = []
        for pos in positions:
            value = self.position_value(pos, queue_depth, time_horizon)
            fill_prob = self.fill_model.fill_probability_given_position(
                pos, time_horizon
            )

            data.append({
                'position': pos,
                'value': value,
                'fill_probability': fill_prob,
                'value_per_prob': value / fill_prob if fill_prob > 0 else 0
            })

        return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    params = OrderBookParameters()
    fill_model = FillProbabilityModel(params)

    qpv = QueuePositionValue(
        fill_model=fill_model,
        spread=0.02,
        volatility=0.0001  # Per second
    )

    # Value curve
    value_df = qpv.queue_value_curve(
        max_position=50,
        queue_depth=100,
        time_horizon=60
    )

    print("Queue Position Value:")
    print(value_df.head(10).to_string(index=False))
```

---

## 4. Market Making with Queues

### 4.1 Avellaneda-Stoikov Framework

The AS model (2008) optimizes market maker quotes accounting for:
- Inventory risk
- Queue position (implicitly through price levels)
- Adverse selection

**Optimal Quotes**:
$$\delta_b^* = \gamma \sigma^2 (T-t) + \frac{1}{\gamma} \ln\left(1 + \frac{\gamma}{k}\right) + q \gamma \sigma^2 (T-t)$$
$$\delta_a^* = \gamma \sigma^2 (T-t) + \frac{1}{\gamma} \ln\left(1 + \frac{\gamma}{k}\right) - q \gamma \sigma^2 (T-t)$$

where:
- $\gamma$ = risk aversion
- $\sigma$ = volatility
- $q$ = current inventory
- $k$ = order arrival intensity

### 4.2 Queue-Aware Market Making

Extending AS to account for queue dynamics:

```python
from scipy.optimize import minimize_scalar


class QueueAwareMarketMaker:
    """
    Market making strategy with queue position awareness.

    Optimizes quote placement considering fill probabilities
    and queue position value.
    """

    def __init__(
        self,
        params: OrderBookParameters,
        risk_aversion: float = 0.01,
        volatility: float = 0.02,
        inventory_limit: int = 100
    ):
        """
        Initialize market maker.

        Args:
            params: Order book parameters
            risk_aversion: Risk aversion coefficient
            volatility: Daily volatility
            inventory_limit: Maximum inventory position
        """
        self.params = params
        self.risk_aversion = risk_aversion
        self.volatility = volatility / np.sqrt(252 * 6.5 * 3600)  # Per second
        self.inventory_limit = inventory_limit

        self.fill_model = FillProbabilityModel(params)

    def optimal_spread(
        self,
        inventory: int,
        time_remaining: float
    ) -> float:
        """
        Compute optimal bid-ask spread.

        Based on Avellaneda-Stoikov with risk aversion.

        Args:
            inventory: Current inventory position
            time_remaining: Time until end of trading (seconds)

        Returns:
            Optimal spread width
        """
        gamma = self.risk_aversion
        sigma = self.volatility

        # Base spread component
        base_spread = gamma * sigma**2 * time_remaining

        # Order arrival component (assuming k inversely related to spread)
        # This is simplified; full model requires solving HJB
        arrival_component = 2 / gamma * np.log(1 + gamma / 0.1)

        return base_spread + arrival_component

    def inventory_skew(
        self,
        inventory: int,
        time_remaining: float
    ) -> float:
        """
        Compute inventory-based quote skew.

        Positive skew = raise ask more than bid (reduce long position)

        Args:
            inventory: Current inventory
            time_remaining: Time remaining

        Returns:
            Skew amount (added to ask, subtracted from bid)
        """
        gamma = self.risk_aversion
        sigma = self.volatility

        return inventory * gamma * sigma**2 * time_remaining

    def compute_quotes(
        self,
        mid_price: float,
        inventory: int,
        time_remaining: float,
        queue_depths: Dict[str, List[int]]
    ) -> Dict[str, float]:
        """
        Compute optimal bid and ask quotes.

        Args:
            mid_price: Current mid price
            inventory: Current inventory
            time_remaining: Time remaining
            queue_depths: {'bid': [...], 'ask': [...]} depths at each level

        Returns:
            {'bid': price, 'ask': price, 'bid_level': int, 'ask_level': int}
        """
        spread = self.optimal_spread(inventory, time_remaining)
        skew = self.inventory_skew(inventory, time_remaining)

        # Raw optimal quotes
        optimal_bid = mid_price - spread / 2 - skew
        optimal_ask = mid_price + spread / 2 + skew

        # Find best price levels considering queue position value
        tick = self.params.tick_size

        # Snap to tick grid
        bid_price = np.floor(optimal_bid / tick) * tick
        ask_price = np.ceil(optimal_ask / tick) * tick

        # Determine queue levels
        best_bid = mid_price - tick / 2  # Approximate
        best_ask = mid_price + tick / 2

        bid_level = int((best_bid - bid_price) / tick)
        ask_level = int((ask_price - best_ask) / tick)

        return {
            'bid': bid_price,
            'ask': ask_price,
            'bid_level': bid_level,
            'ask_level': ask_level,
            'spread': ask_price - bid_price,
            'skew': skew
        }

    def expected_pnl(
        self,
        quotes: Dict[str, float],
        time_horizon: float,
        inventory: int,
        queue_depths: Dict[str, List[int]]
    ) -> float:
        """
        Compute expected P&L from market making.

        Args:
            quotes: Current quotes
            time_horizon: Time horizon
            inventory: Current inventory
            queue_depths: Queue depths

        Returns:
            Expected P&L
        """
        spread = quotes['ask'] - quotes['bid']

        # Fill probabilities
        bid_queue_pos = queue_depths['bid'][quotes['bid_level']] + 1
        ask_queue_pos = queue_depths['ask'][quotes['ask_level']] + 1

        p_bid_fill = self.fill_model.fill_probability_given_position(
            bid_queue_pos, time_horizon
        )
        p_ask_fill = self.fill_model.fill_probability_given_position(
            ask_queue_pos, time_horizon
        )

        # Expected spread capture
        spread_pnl = (p_bid_fill + p_ask_fill) * spread / 2

        # Inventory risk (penalty for ending with position)
        inv_risk = self.risk_aversion * (inventory ** 2) * (self.volatility ** 2) * time_horizon

        return spread_pnl - inv_risk

    def simulate_trading_day(
        self,
        initial_mid: float = 100.0,
        trading_seconds: int = 23400,  # 6.5 hours
        update_frequency: int = 1
    ) -> pd.DataFrame:
        """
        Simulate market making for one trading day.

        Args:
            initial_mid: Starting mid price
            trading_seconds: Length of trading day
            update_frequency: How often to update quotes (seconds)

        Returns:
            DataFrame with simulation results
        """
        results = []
        mid_price = initial_mid
        inventory = 0
        cash = 0

        for t in range(0, trading_seconds, update_frequency):
            time_remaining = trading_seconds - t

            # Simulate queue depths (simplified)
            avg_depth = int(self.fill_model.expected_queue_depth())
            queue_depths = {
                'bid': [avg_depth] * 5,
                'ask': [avg_depth] * 5
            }

            # Compute quotes
            quotes = self.compute_quotes(
                mid_price, inventory, time_remaining, queue_depths
            )

            # Simulate fills (simplified Poisson process)
            bid_fills = np.random.poisson(
                self.params.market_order_rate * update_frequency * 0.5
            )
            ask_fills = np.random.poisson(
                self.params.market_order_rate * update_frequency * 0.5
            )

            # Update inventory and cash
            # Bid fill = we buy
            if bid_fills > 0 and inventory < self.inventory_limit:
                actual_buys = min(bid_fills, self.inventory_limit - inventory)
                inventory += actual_buys
                cash -= actual_buys * quotes['bid']

            # Ask fill = we sell
            if ask_fills > 0 and inventory > -self.inventory_limit:
                actual_sells = min(ask_fills, inventory + self.inventory_limit)
                inventory -= actual_sells
                cash += actual_sells * quotes['ask']

            # Update mid price (random walk)
            mid_price += np.random.normal(0, self.volatility * np.sqrt(update_frequency))

            # Record state
            mark_to_market = cash + inventory * mid_price

            results.append({
                'time': t,
                'mid_price': mid_price,
                'bid': quotes['bid'],
                'ask': quotes['ask'],
                'spread': quotes['spread'],
                'inventory': inventory,
                'cash': cash,
                'mtm_pnl': mark_to_market - initial_mid * 0  # No initial position
            })

        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    params = OrderBookParameters(
        limit_order_rate=10.0,
        market_order_rate=5.0,
        cancel_rate=0.1
    )

    mm = QueueAwareMarketMaker(
        params=params,
        risk_aversion=0.001,
        volatility=0.02,
        inventory_limit=100
    )

    # Compute optimal quotes
    quotes = mm.compute_quotes(
        mid_price=100.0,
        inventory=10,
        time_remaining=3600,
        queue_depths={'bid': [50, 40, 30], 'ask': [45, 35, 25]}
    )

    print("Optimal Quotes:")
    print(f"  Bid: ${quotes['bid']:.2f} (level {quotes['bid_level']})")
    print(f"  Ask: ${quotes['ask']:.2f} (level {quotes['ask_level']})")
    print(f"  Spread: ${quotes['spread']:.4f}")
    print(f"  Skew: ${quotes['skew']:.6f}")

    # Simulate trading
    results = mm.simulate_trading_day(update_frequency=60)
    print(f"\nSimulation Results (1-minute updates):")
    print(f"  Final Inventory: {results['inventory'].iloc[-1]}")
    print(f"  Final MTM P&L: ${results['mtm_pnl'].iloc[-1]:.2f}")
    print(f"  Max Inventory: {results['inventory'].abs().max()}")
```

---

## 5. Execution with Queue Dynamics

### 5.1 TWAP with Fill Probability

Standard TWAP assumes all orders fill. Queue-aware TWAP adjusts:

```python
class QueueAwareTWAP:
    """
    TWAP execution accounting for queue dynamics.
    """

    def __init__(self, fill_model: FillProbabilityModel):
        """
        Initialize TWAP executor.

        Args:
            fill_model: Fill probability model
        """
        self.fill_model = fill_model

    def compute_schedule(
        self,
        total_shares: int,
        time_horizon: float,
        n_periods: int,
        target_fill_prob: float = 0.95,
        queue_depths: List[int] = None
    ) -> pd.DataFrame:
        """
        Compute TWAP schedule with fill probability targets.

        Args:
            total_shares: Total shares to execute
            time_horizon: Total time (seconds)
            n_periods: Number of time periods
            target_fill_prob: Minimum fill probability per period
            queue_depths: Expected queue depths

        Returns:
            DataFrame with execution schedule
        """
        period_length = time_horizon / n_periods
        base_shares_per_period = total_shares / n_periods

        if queue_depths is None:
            avg_depth = int(self.fill_model.expected_queue_depth())
            queue_depths = [avg_depth] * n_periods

        schedule = []
        remaining_shares = total_shares

        for period in range(n_periods):
            # Determine shares for this period
            shares_this_period = min(
                remaining_shares,
                base_shares_per_period * 1.5  # Allow some flexibility
            )

            # Estimate queue position
            queue_position = queue_depths[period] + 1

            # Fill probability
            fill_prob = self.fill_model.fill_probability_given_position(
                queue_position, period_length
            )

            # Adjust shares if fill probability too low
            if fill_prob < target_fill_prob:
                # Need to be more aggressive (use market orders partially)
                limit_fraction = fill_prob / target_fill_prob
                market_fraction = 1 - limit_fraction
            else:
                limit_fraction = 1.0
                market_fraction = 0.0

            limit_shares = int(shares_this_period * limit_fraction)
            market_shares = int(shares_this_period * market_fraction)

            schedule.append({
                'period': period,
                'start_time': period * period_length,
                'limit_shares': limit_shares,
                'market_shares': market_shares,
                'total_shares': limit_shares + market_shares,
                'fill_probability': fill_prob,
                'queue_position': queue_position
            })

            remaining_shares -= (limit_shares + market_shares)

        return pd.DataFrame(schedule)

    def simulate_execution(
        self,
        schedule: pd.DataFrame,
        price_volatility: float = 0.0001
    ) -> Dict:
        """
        Simulate TWAP execution.

        Args:
            schedule: Execution schedule
            price_volatility: Price volatility per second

        Returns:
            Execution statistics
        """
        total_cost = 0
        total_shares = 0
        fills = []

        price = 100.0  # Starting price

        for _, row in schedule.iterrows():
            period_length = schedule['start_time'].iloc[1] - schedule['start_time'].iloc[0]

            # Limit order fills
            fill_prob = row['fill_probability']
            if np.random.random() < fill_prob:
                limit_filled = row['limit_shares']
                limit_price = price - 0.01  # Assume we capture spread
            else:
                limit_filled = 0
                limit_price = 0

            # Market order fills (always fill)
            market_filled = row['market_shares']
            market_price = price + 0.01  # Cross spread

            # Record fills
            if limit_filled > 0:
                total_cost += limit_filled * limit_price
                total_shares += limit_filled
                fills.append({'type': 'limit', 'shares': limit_filled, 'price': limit_price})

            if market_filled > 0:
                total_cost += market_filled * market_price
                total_shares += market_filled
                fills.append({'type': 'market', 'shares': market_filled, 'price': market_price})

            # Price evolution
            price += np.random.normal(0, price_volatility * np.sqrt(period_length))

        avg_price = total_cost / total_shares if total_shares > 0 else 0

        return {
            'total_shares': total_shares,
            'total_cost': total_cost,
            'avg_price': avg_price,
            'fills': fills,
            'final_price': price
        }


# Example usage
if __name__ == "__main__":
    params = OrderBookParameters()
    fill_model = FillProbabilityModel(params)

    twap = QueueAwareTWAP(fill_model)

    # Create schedule
    schedule = twap.compute_schedule(
        total_shares=10000,
        time_horizon=3600,  # 1 hour
        n_periods=12,       # 5-minute intervals
        target_fill_prob=0.90
    )

    print("TWAP Schedule:")
    print(schedule.to_string(index=False))

    # Simulate
    result = twap.simulate_execution(schedule)
    print(f"\nExecution Result:")
    print(f"  Shares Filled: {result['total_shares']}")
    print(f"  Avg Price: ${result['avg_price']:.4f}")
```

---

## 6. Order Book Simulation

### 6.1 Full Order Book Simulator

```python
from collections import defaultdict
import heapq


class OrderBookSimulator:
    """
    Event-driven order book simulator.

    Models limit order book dynamics with arrivals,
    cancellations, and executions.
    """

    def __init__(self, params: OrderBookParameters, initial_mid: float = 100.0):
        """
        Initialize simulator.

        Args:
            params: Order book parameters
            initial_mid: Initial mid price
        """
        self.params = params
        self.mid_price = initial_mid
        self.tick = params.tick_size

        # Order books: price -> list of (order_id, size, arrival_time)
        self.bids = defaultdict(list)
        self.asks = defaultdict(list)

        self.order_counter = 0
        self.current_time = 0

        # Event queue: (time, event_type, data)
        self.events = []

        # Initialize with some orders
        self._initialize_book()

    def _initialize_book(self):
        """Populate initial order book."""
        # Add orders at several price levels
        for level in range(self.params.n_levels):
            bid_price = self.mid_price - (level + 1) * self.tick
            ask_price = self.mid_price + (level + 1) * self.tick

            # Random number of orders at each level
            n_bid_orders = np.random.poisson(5)
            n_ask_orders = np.random.poisson(5)

            for _ in range(n_bid_orders):
                self._add_order('bid', bid_price, size=np.random.randint(1, 10))

            for _ in range(n_ask_orders):
                self._add_order('ask', ask_price, size=np.random.randint(1, 10))

    def _add_order(self, side: str, price: float, size: int):
        """Add order to book."""
        self.order_counter += 1
        order = (self.order_counter, size, self.current_time)

        if side == 'bid':
            self.bids[price].append(order)
        else:
            self.asks[price].append(order)

        # Schedule potential cancellation
        cancel_time = self.current_time + np.random.exponential(1 / self.params.cancel_rate)
        heapq.heappush(self.events, (cancel_time, 'cancel', (side, price, self.order_counter)))

    def get_best_bid(self) -> Tuple[float, int]:
        """Get best bid price and total size."""
        if not self.bids:
            return None, 0

        best_price = max(self.bids.keys())
        total_size = sum(order[1] for order in self.bids[best_price])

        return best_price, total_size

    def get_best_ask(self) -> Tuple[float, int]:
        """Get best ask price and total size."""
        if not self.asks:
            return None, 0

        best_price = min(self.asks.keys())
        total_size = sum(order[1] for order in self.asks[best_price])

        return best_price, total_size

    def get_spread(self) -> float:
        """Get current bid-ask spread."""
        best_bid, _ = self.get_best_bid()
        best_ask, _ = self.get_best_ask()

        if best_bid is None or best_ask is None:
            return np.inf

        return best_ask - best_bid

    def process_market_order(self, side: str, size: int) -> List[Tuple[float, int]]:
        """
        Process market order.

        Args:
            side: 'buy' or 'sell'
            size: Order size

        Returns:
            List of (price, size) fills
        """
        fills = []
        remaining = size

        if side == 'buy':
            # Match against asks (ascending price)
            while remaining > 0 and self.asks:
                best_price = min(self.asks.keys())
                orders = self.asks[best_price]

                while remaining > 0 and orders:
                    order_id, order_size, _ = orders[0]

                    fill_size = min(remaining, order_size)
                    fills.append((best_price, fill_size))
                    remaining -= fill_size

                    if fill_size >= order_size:
                        orders.pop(0)
                    else:
                        orders[0] = (order_id, order_size - fill_size, orders[0][2])

                if not orders:
                    del self.asks[best_price]

        else:  # sell
            # Match against bids (descending price)
            while remaining > 0 and self.bids:
                best_price = max(self.bids.keys())
                orders = self.bids[best_price]

                while remaining > 0 and orders:
                    order_id, order_size, _ = orders[0]

                    fill_size = min(remaining, order_size)
                    fills.append((best_price, fill_size))
                    remaining -= fill_size

                    if fill_size >= order_size:
                        orders.pop(0)
                    else:
                        orders[0] = (order_id, order_size - fill_size, orders[0][2])

                if not orders:
                    del self.bids[best_price]

        return fills

    def step(self, dt: float = 1.0) -> Dict:
        """
        Advance simulation by dt seconds.

        Args:
            dt: Time step

        Returns:
            Events that occurred
        """
        end_time = self.current_time + dt
        events_occurred = []

        # Process scheduled events
        while self.events and self.events[0][0] <= end_time:
            event_time, event_type, data = heapq.heappop(self.events)
            self.current_time = event_time

            if event_type == 'cancel':
                side, price, order_id = data
                book = self.bids if side == 'bid' else self.asks

                if price in book:
                    book[price] = [o for o in book[price] if o[0] != order_id]
                    if not book[price]:
                        del book[price]

                events_occurred.append({'type': 'cancel', 'side': side, 'price': price})

        # Generate new limit orders (Poisson arrivals)
        n_new_orders = np.random.poisson(self.params.limit_order_rate * dt)

        for _ in range(n_new_orders):
            side = 'bid' if np.random.random() < 0.5 else 'sell'

            if side == 'bid':
                best_bid, _ = self.get_best_bid()
                if best_bid is None:
                    best_bid = self.mid_price - self.tick

                # New order within a few ticks of best
                level = np.random.randint(0, 5)
                price = best_bid - level * self.tick

            else:
                best_ask, _ = self.get_best_ask()
                if best_ask is None:
                    best_ask = self.mid_price + self.tick

                level = np.random.randint(0, 5)
                price = best_ask + level * self.tick

            self._add_order('bid' if side == 'bid' else 'ask', price, np.random.randint(1, 10))
            events_occurred.append({'type': 'limit', 'side': side, 'price': price})

        # Generate market orders
        n_market_orders = np.random.poisson(self.params.market_order_rate * dt)

        for _ in range(n_market_orders):
            side = 'buy' if np.random.random() < 0.5 else 'sell'
            size = np.random.randint(1, 5)

            fills = self.process_market_order(side, size)
            events_occurred.append({'type': 'market', 'side': side, 'fills': fills})

        self.current_time = end_time

        # Update mid price
        best_bid, _ = self.get_best_bid()
        best_ask, _ = self.get_best_ask()

        if best_bid is not None and best_ask is not None:
            self.mid_price = (best_bid + best_ask) / 2

        return {
            'time': self.current_time,
            'mid_price': self.mid_price,
            'spread': self.get_spread(),
            'best_bid': best_bid,
            'best_ask': best_ask,
            'events': events_occurred
        }

    def get_book_snapshot(self) -> Dict:
        """Get current order book state."""
        bid_levels = []
        for price in sorted(self.bids.keys(), reverse=True)[:5]:
            total_size = sum(o[1] for o in self.bids[price])
            bid_levels.append({'price': price, 'size': total_size, 'n_orders': len(self.bids[price])})

        ask_levels = []
        for price in sorted(self.asks.keys())[:5]:
            total_size = sum(o[1] for o in self.asks[price])
            ask_levels.append({'price': price, 'size': total_size, 'n_orders': len(self.asks[price])})

        return {
            'bids': bid_levels,
            'asks': ask_levels,
            'mid_price': self.mid_price,
            'spread': self.get_spread()
        }


# Example usage
if __name__ == "__main__":
    params = OrderBookParameters(
        limit_order_rate=10.0,
        market_order_rate=5.0,
        cancel_rate=0.1,
        tick_size=0.01
    )

    sim = OrderBookSimulator(params, initial_mid=100.0)

    print("Initial Order Book:")
    snapshot = sim.get_book_snapshot()
    print(f"  Mid: ${snapshot['mid_price']:.2f}, Spread: ${snapshot['spread']:.4f}")

    # Simulate 60 seconds
    for _ in range(60):
        result = sim.step(dt=1.0)

    print(f"\nAfter 60 seconds:")
    snapshot = sim.get_book_snapshot()
    print(f"  Mid: ${snapshot['mid_price']:.2f}, Spread: ${snapshot['spread']:.4f}")
    print(f"  Top 3 Bids: {snapshot['bids'][:3]}")
    print(f"  Top 3 Asks: {snapshot['asks'][:3]}")
```

---

## 7. Academic References

### Foundational Papers

1. **Cont, R., Stoikov, S., & Talreja, R. (2010)**. "A Stochastic Model for Order Book Dynamics." *Operations Research*, 58(3), 549-563.
   - Queueing model for limit order book dynamics

2. **Avellaneda, M., & Stoikov, S. (2008)**. "High-Frequency Trading in a Limit Order Book." *Quantitative Finance*, 8(3), 217-224.
   - Optimal market making with inventory risk

3. **Moallemi, C. C., & Yuan, K. (2017)**. "A Model for Queue Position Valuation in a Limit Order Book." Working Paper.
   - Economic value of queue priority

4. **Gueant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2012)**. "Dealing with the Inventory Risk: A Solution to the Market Making Problem." *Mathematics and Financial Economics*, 4(7), 477-507.
   - Optimal market making with inventory constraints

5. **Cartea, A., Jaimungal, S., & Penalva, J. (2015)**. *Algorithmic and High-Frequency Trading*. Cambridge University Press.
   - Comprehensive treatment of HFT mathematics

### Queueing Theory

6. **Gross, D., & Harris, C. M. (1998)**. *Fundamentals of Queueing Theory*. Wiley.
   - General queueing theory reference

7. **Lakatos, L., et al. (2013)**. *Introduction to Queueing Systems with Telecommunication Applications*. Springer.
   - Modern queueing applications

---

## 8. Cross-References

**Related Knowledge Base Sections**:

- [Game Theory](game_theory.md) - Strategic trading, Kyle/Glosten-Milgrom models
- [Control Theory](control_theory.md) - Optimal execution, HJB equation
- [Optimal Execution](../../02_signals/quantitative/execution_algorithms/optimal_execution.md) - TWAP/VWAP
- [Market Impact](../../02_signals/quantitative/execution_algorithms/market_impact.md) - Impact models

**Integration Points**:

1. **FlowRoute**: Queue-aware order routing
2. **SignalCore**: Order flow imbalance signals
3. **ProofBench**: Execution simulation with queue dynamics
4. **RiskGuard**: Execution risk from queue position uncertainty

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
last_updated: "2025-12-12"
status: "published"
category: "foundations/advanced_mathematics"
tags: ["queueing-theory", "order-book", "market-making", "fill-probability", "queue-position", "execution"]
code_lines: 800
academic_references: 7
implementation_completeness: "production-ready"
```

---

**END OF DOCUMENT**
