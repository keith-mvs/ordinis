"""Execution simulation with realistic slippage and commission modeling."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import random
import uuid

from ordinis.domain.enums import OrderSide, OrderStatus, OrderType
from ordinis.domain.market_data import Bar
from ordinis.domain.orders import Fill, Order


class FillMode(Enum):
    """Fill modeling granularity."""

    BAR_OPEN = "BAR_OPEN"  # Fill at bar open (previous behavior)
    INTRA_BAR = "INTRA_BAR"  # Approximate intrabar extremes (high/low biased by side)
    REALISTIC = "REALISTIC"  # Blend open/close with range to mimic intrabar path


@dataclass
class ExecutionConfig:
    """Configuration for execution simulator.

    Attributes:
        estimated_spread: Estimated bid-ask spread (as fraction, e.g., 0.001 = 0.1%)
        impact_coefficient: Market impact coefficient for slippage
        volatility_factor: Volatility factor for slippage
        max_slippage: Maximum allowed slippage (as fraction)
        commission_per_share: Commission per share
        commission_per_trade: Flat commission per trade
        commission_pct: Commission as percentage of trade value
        min_commission: Minimum commission per trade
        partial_fills: Allow partial fills
        fill_probability: Probability of fill for limit orders
        fill_mode: How to model intrabar fills for market/stop orders
    """

    estimated_spread: float = 0.001  # 0.1%
    impact_coefficient: float = 0.1
    volatility_factor: float = 0.5
    max_slippage: float = 0.01  # 1%
    commission_per_share: float = 0.0
    commission_per_trade: float = 0.0
    commission_pct: float = 0.001  # 0.1%
    min_commission: float = 0.0
    partial_fills: bool = False
    fill_probability: float = 1.0
    fill_mode: FillMode = FillMode.BAR_OPEN

    def __post_init__(self):
        """Validate configuration."""
        if self.estimated_spread < 0:
            raise ValueError("Spread cannot be negative")
        if self.max_slippage < 0:
            raise ValueError("Max slippage cannot be negative")
        if not 0 <= self.fill_probability <= 1:
            raise ValueError("Fill probability must be between 0 and 1")


class ExecutionSimulator:
    """Simulates realistic order execution with slippage and commissions.

    Uses configurable slippage model with three components:
    1. Fixed component (bid-ask spread)
    2. Variable component (market impact based on volume)
    3. Volatility component (based on bar volatility)
    """

    def __init__(self, config: ExecutionConfig | None = None):
        """Initialize execution simulator.

        Args:
            config: Execution configuration (uses defaults if None)
        """
        self.config = config or ExecutionConfig()

    def simulate_fill(
        self, order: Order, bar: Bar, timestamp: datetime, multiplier: float = 1.0
    ) -> Fill | None:
        """Simulate order fill for given market conditions.

        Args:
            order: Order to fill
            bar: Current market bar
            timestamp: Fill timestamp
            multiplier: Contract multiplier (default 1.0)

        Returns:
            Fill object if order can be filled, None otherwise
        """
        if order.status == OrderStatus.FILLED:
            return None

        # Apply fill probability check for limit orders
        if order.order_type == OrderType.LIMIT and self.config.fill_probability < 1.0:
            if random.random() > self.config.fill_probability:
                return None  # Order not filled due to probability

        if order.order_type == OrderType.MARKET:
            return self._fill_market_order(order, bar, timestamp, multiplier)
        if order.order_type == OrderType.LIMIT:
            return self._fill_limit_order(order, bar, timestamp, multiplier)
        if order.order_type == OrderType.STOP:
            return self._fill_stop_order(order, bar, timestamp, multiplier)
        if order.order_type == OrderType.STOP_LIMIT:
            return self._fill_stop_limit_order(order, bar, timestamp, multiplier)

        return None

    def _fill_market_order(
        self, order: Order, bar: Bar, timestamp: datetime, multiplier: float = 1.0
    ) -> Fill:
        """Fill market order at current price with slippage.

        Args:
            order: Market order to fill
            bar: Current market bar
            timestamp: Fill timestamp
            multiplier: Contract multiplier

        Returns:
            Fill object
        """
        # Choose base price depending on fill mode
        if self.config.fill_mode == FillMode.BAR_OPEN:
            base_price = bar.open
        elif self.config.fill_mode == FillMode.INTRA_BAR:
            # Side-biased toward worst-case intrabar price
            base_price = (
                (bar.high + bar.open) / 2
                if order.side == OrderSide.BUY
                else (bar.low + bar.open) / 2
            )
        elif order.side == OrderSide.BUY:
            base_price = min(bar.high, (bar.open + bar.close + bar.high) / 3)
        else:
            base_price = max(bar.low, (bar.open + bar.close + bar.low) / 3)

        # Calculate slippage
        slippage = self._calculate_slippage(order, bar)

        # Apply slippage based on order side
        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + slippage)
        else:
            fill_price = base_price * (1 - slippage)

        # Calculate commission
        commission = self._calculate_commission(order.remaining_quantity, fill_price, multiplier)

        # Determine fill quantity (support partial fills)
        if self.config.partial_fills and order.remaining_quantity > 1:
            # Simulate partial fill: fill between 50% and 100% of remaining
            fill_ratio = 0.5 + random.random() * 0.5
            fill_quantity = max(1, int(order.remaining_quantity * fill_ratio))
        else:
            fill_quantity = order.remaining_quantity

        return Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage,
            timestamp=timestamp,
            multiplier=multiplier,
        )

    def _fill_limit_order(
        self, order: Order, bar: Bar, timestamp: datetime, multiplier: float = 1.0
    ) -> Fill | None:
        """Fill limit order if price conditions are met.

        Args:
            order: Limit order to fill
            bar: Current market bar
            timestamp: Fill timestamp
            multiplier: Contract multiplier

        Returns:
            Fill object if conditions met, None otherwise
        """
        if order.limit_price is None:
            return None

        # Check if limit price was reached
        if order.side == OrderSide.BUY:
            # For buy limit, fill if low <= limit_price
            if bar.low > order.limit_price:
                return None
            fill_price = min(order.limit_price, bar.open)
        else:
            # For sell limit, fill if high >= limit_price
            if bar.high < order.limit_price:
                return None
            fill_price = max(order.limit_price, bar.open)

        # Small slippage even for limit orders
        slippage = self.config.estimated_spread / 2
        commission = self._calculate_commission(order.remaining_quantity, fill_price, multiplier)

        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.remaining_quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage,
            timestamp=timestamp,
            multiplier=multiplier,
        )

    def _fill_stop_order(
        self, order: Order, bar: Bar, timestamp: datetime, multiplier: float = 1.0
    ) -> Fill | None:
        """Fill stop order if price conditions are met.

        Args:
            order: Stop order to fill
            bar: Current market bar
            timestamp: Fill timestamp
            multiplier: Contract multiplier

        Returns:
            Fill object if conditions met, None otherwise
        """
        if order.stop_price is None:
            return None

        # Check if stop price was hit
        if order.side == OrderSide.BUY:
            # For buy stop, trigger if high >= stop_price
            if bar.high < order.stop_price:
                return None
        elif bar.low > order.stop_price:
            return None

        # Once triggered, fill as market order
        return self._fill_market_order(order, bar, timestamp, multiplier)

    def _fill_stop_limit_order(
        self, order: Order, bar: Bar, timestamp: datetime, multiplier: float = 1.0
    ) -> Fill | None:
        """Fill stop-limit order if conditions are met.

        Args:
            order: Stop-limit order to fill
            bar: Current market bar
            timestamp: Fill timestamp
            multiplier: Contract multiplier

        Returns:
            Fill object if conditions met, None otherwise
        """
        # First check if stop is triggered (same logic as stop order)
        temp_stop_order = Order(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=OrderType.STOP,
            stop_price=order.stop_price,
        )

        if self._fill_stop_order(temp_stop_order, bar, timestamp, multiplier) is None:
            return None

        # If stop triggered, try to fill as limit order
        temp_limit_order = Order(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=OrderType.LIMIT,
            limit_price=order.limit_price,
        )

        return self._fill_limit_order(temp_limit_order, bar, timestamp, multiplier)

    def _calculate_slippage(self, order: Order, bar: Bar) -> float:
        """Calculate slippage for an order.

        Slippage has three components:
        1. Fixed: Half of bid-ask spread
        2. Variable: Market impact based on order size vs volume
        3. Volatility: Based on bar volatility

        Args:
            order: Order being filled
            bar: Current market bar

        Returns:
            Total slippage as a fraction
        """
        # Fixed component (half spread)
        spread_slippage = self.config.estimated_spread / 2

        # Variable component (market impact)
        if bar.volume > 0:
            volume_ratio = order.remaining_quantity / bar.volume
            impact_slippage = self.config.impact_coefficient * volume_ratio
        else:
            impact_slippage = 0.0

        # Volatility component
        if bar.close > 0:
            volatility = (bar.high - bar.low) / bar.close
            vol_slippage = volatility * self.config.volatility_factor
        else:
            vol_slippage = 0.0

        # Total slippage
        total_slippage = spread_slippage + impact_slippage + vol_slippage

        # Cap at maximum
        return min(total_slippage, self.config.max_slippage)

    def _calculate_commission(self, quantity: int, price: float, multiplier: float = 1.0) -> float:
        """Calculate commission for a trade.

        Commission can have multiple components:
        1. Per-share commission
        2. Flat per-trade commission
        3. Percentage of trade value

        Args:
            quantity: Number of shares
            price: Price per share
            multiplier: Contract multiplier

        Returns:
            Total commission
        """
        commission = 0.0

        # Per-share component
        commission += self.config.commission_per_share * quantity

        # Flat component
        commission += self.config.commission_per_trade

        # Percentage component
        trade_value = quantity * price * multiplier
        commission += trade_value * self.config.commission_pct

        # Apply minimum
        return max(commission, self.config.min_commission)
