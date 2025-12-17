"""Integration test for Futures trading cycle."""

from datetime import UTC, datetime

import pytest

from ordinis.domain.enums import OrderSide, OrderType
from ordinis.domain.market_data import Bar
from ordinis.domain.orders import Order
from ordinis.engines.portfolio.core.engine import PortfolioEngine
from ordinis.engines.proofbench.core.execution import ExecutionConfig, ExecutionSimulator


class TestFuturesTrading:
    """Test futures trading cycle including execution and portfolio updates."""

    @pytest.fixture
    def portfolio_engine(self):
        """Provide portfolio engine."""
        engine = PortfolioEngine()
        # engine.initialize() # Not async in this test context for simplicity, or mock
        return engine

    @pytest.fixture
    def execution_simulator(self):
        """Provide execution simulator."""
        config = ExecutionConfig(commission_pct=0.0)  # Simplify commission
        return ExecutionSimulator(config)

    @pytest.mark.asyncio
    async def test_futures_execution_and_portfolio_update(
        self, portfolio_engine, execution_simulator
    ):
        """Test full cycle for a futures contract."""

        # 1. Define Future Contract specs
        symbol = "ES"
        multiplier = 50.0
        initial_margin = 12000.0

        # 2. Create Order
        order = Order(symbol=symbol, side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET)

        # 3. Create Market Data (Bar)
        bar = Bar(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            open=4000.0,
            high=4010.0,
            low=3990.0,
            close=4005.0,
            volume=1000,
        )

        # 4. Simulate Execution
        fill = execution_simulator.simulate_fill(
            order=order, bar=bar, timestamp=datetime.now(UTC), multiplier=multiplier
        )

        assert fill is not None
        assert fill.multiplier == multiplier
        # Allow for slippage outside the bar range
        assert fill.price >= 3990.0 * 0.99 and fill.price <= 4010.0 * 1.01

        # 5. Update Portfolio
        # Manually inject initial margin into position since fill doesn't carry it
        # In real system, PortfolioEngine would look up instrument specs.
        # Here we rely on PortfolioEngine creating the position, then we update it.

        await portfolio_engine.update([fill])

        assert symbol in portfolio_engine.positions
        position = portfolio_engine.positions[symbol]
        assert position.quantity == 1
        assert position.multiplier == multiplier

        # Set margin manually for test (simulating instrument lookup)
        position.initial_margin = initial_margin

        # 6. Check Margin Calculation
        margin_req = portfolio_engine.calculate_margin()
        assert margin_req == initial_margin

        # 7. Check PnL Update
        # Price moves up 10 points
        new_price = 4015.0
        position.update_price(new_price, datetime.now(UTC))

        expected_pnl = (new_price - position.avg_entry_price) * 1 * multiplier
        assert position.unrealized_pnl == pytest.approx(expected_pnl, abs=0.01)

        # 8. Close Position
        close_order = Order(
            symbol=symbol, side=OrderSide.SELL, quantity=1, order_type=OrderType.MARKET
        )

        close_bar = Bar(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            open=4015.0,
            high=4020.0,
            low=4010.0,
            close=4018.0,
            volume=1000,
        )

        close_fill = execution_simulator.simulate_fill(
            close_order, close_bar, datetime.now(UTC), multiplier=multiplier
        )

        await portfolio_engine.update([close_fill])

        assert symbol not in portfolio_engine.positions
        # Cash should reflect PnL
        # Initial cash 100k.
        # Entry cost: ~4000 * 50 = 200k (but futures don't pay this, spot model does)
        # Exit proceeds: ~4015 * 50 = 200.75k
        # Net change: +750

        # Since PortfolioEngine uses Spot accounting (debit cost, credit proceeds),
        # the net cash change should be correct PnL.

        initial_cash = 100000.0
        expected_cash_change = (close_fill.price - fill.price) * multiplier
        # Note: fill.price includes slippage, so we use actual fill prices

        assert portfolio_engine.cash == pytest.approx(initial_cash + expected_cash_change, abs=1.0)
