"""
Tests for Position domain models.
Tests cover:
- Position model
- Trade model
"""

from datetime import UTC, datetime
import pytest

from ordinis.domain.enums import PositionSide
from ordinis.domain.positions import Position, Trade


class TestPosition:
    """Tests for Position model."""

    @pytest.mark.unit
    def test_position_creation_default(self):
        """Test creating a Position with defaults."""
        position = Position(symbol="AAPL")

        assert position.symbol == "AAPL"
        assert position.side == PositionSide.FLAT
        assert position.quantity == 0
        assert position.avg_entry_price == 0.0
        assert position.current_price == 0.0
        assert position.realized_pnl == 0.0
        assert position.unrealized_pnl == 0.0
        assert position.multiplier == 1.0

    @pytest.mark.unit
    def test_position_long(self):
        """Test creating a long position."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=100,
            avg_entry_price=150.00,
            current_price=155.00,
        )

        assert position.side == PositionSide.LONG
        assert position.quantity == 100
        assert position.avg_entry_price == 150.00

    @pytest.mark.unit
    def test_position_short(self):
        """Test creating a short position."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.SHORT,
            quantity=100,
            avg_entry_price=150.00,
            current_price=145.00,
        )

        assert position.side == PositionSide.SHORT
        assert position.quantity == 100

    @pytest.mark.unit
    def test_position_market_value(self):
        """Test market_value property."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            current_price=155.00,
        )

        # market_value = 100 * 155 * 1 = 15500
        assert position.market_value == 15500.00

    @pytest.mark.unit
    def test_position_market_value_with_multiplier(self):
        """Test market_value with multiplier."""
        position = Position(
            symbol="AAPL",
            quantity=10,
            current_price=5.00,
            multiplier=100.0,
        )

        # market_value = 10 * 5 * 100 = 5000
        assert position.market_value == 5000.00

    @pytest.mark.unit
    def test_position_cost_basis(self):
        """Test cost_basis property."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=150.00,
        )

        # cost_basis = 100 * 150 * 1 = 15000
        assert position.cost_basis == 15000.00

    @pytest.mark.unit
    def test_position_total_pnl(self):
        """Test total_pnl property."""
        position = Position(
            symbol="AAPL",
            realized_pnl=500.00,
            unrealized_pnl=200.00,
        )

        assert position.total_pnl == 700.00

    @pytest.mark.unit
    def test_position_pnl_pct(self):
        """Test pnl_pct property."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=100.00,
            realized_pnl=10.00,
            unrealized_pnl=10.00,
        )

        # cost_basis = 100 * 100 = 10000
        # total_pnl = 20
        # pnl_pct = (20 / 10000) * 100 = 0.2%
        assert position.pnl_pct == pytest.approx(0.2)

    @pytest.mark.unit
    def test_position_pnl_pct_zero_cost_basis(self):
        """Test pnl_pct returns 0 when cost basis is zero."""
        position = Position(symbol="AAPL")

        assert position.pnl_pct == 0.0

    @pytest.mark.unit
    def test_position_is_open_true(self):
        """Test is_open is True for open position."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=100,
        )

        assert position.is_open is True

    @pytest.mark.unit
    def test_position_is_open_false_flat(self):
        """Test is_open is False for flat position."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.FLAT,
            quantity=100,
        )

        assert position.is_open is False

    @pytest.mark.unit
    def test_position_is_open_false_zero_quantity(self):
        """Test is_open is False when quantity is zero."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=0,
        )

        assert position.is_open is False

    @pytest.mark.unit
    def test_position_is_flat(self):
        """Test is_flat method."""
        position = Position(symbol="AAPL")

        assert position.is_flat() is True

    @pytest.mark.unit
    def test_position_is_flat_false(self):
        """Test is_flat is False for open position."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=100,
        )

        assert position.is_flat() is False

    @pytest.mark.unit
    def test_position_update_price_long(self):
        """Test update_price for long position."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            quantity=100,
            avg_entry_price=150.00,
        )

        now = datetime.now(UTC)
        position.update_price(160.00, now)

        assert position.current_price == 160.00
        assert position.last_update_time == now
        # unrealized = (160 - 150) * 100 * 1 = 1000
        assert position.unrealized_pnl == 1000.00

    @pytest.mark.unit
    def test_position_update_price_short(self):
        """Test update_price for short position."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.SHORT,
            quantity=100,
            avg_entry_price=150.00,
        )

        now = datetime.now(UTC)
        position.update_price(140.00, now)

        assert position.current_price == 140.00
        # unrealized for short = -(140 - 150) * 100 = 1000 (profit)
        assert position.unrealized_pnl == 1000.00

    @pytest.mark.unit
    def test_position_update_price_short_loss(self):
        """Test update_price for short position with loss."""
        position = Position(
            symbol="AAPL",
            side=PositionSide.SHORT,
            quantity=100,
            avg_entry_price=150.00,
        )

        now = datetime.now(UTC)
        position.update_price(160.00, now)

        # unrealized for short = -(160 - 150) * 100 = -1000 (loss)
        assert position.unrealized_pnl == -1000.00

    @pytest.mark.unit
    def test_position_update_price_zero_quantity(self):
        """Test update_price with zero quantity."""
        position = Position(
            symbol="AAPL",
            quantity=0,
        )

        now = datetime.now(UTC)
        position.update_price(160.00, now)

        assert position.unrealized_pnl == 0.0

    @pytest.mark.unit
    def test_position_with_metadata(self):
        """Test position with optional metadata."""
        entry = datetime.now(UTC)
        position = Position(
            symbol="AAPL",
            sector="Technology",
            entry_time=entry,
            initial_margin=5000.00,
            maintenance_margin=2500.00,
        )

        assert position.sector == "Technology"
        assert position.entry_time == entry
        assert position.initial_margin == 5000.00
        assert position.maintenance_margin == 2500.00


class TestTrade:
    """Tests for Trade model."""

    @pytest.mark.unit
    def test_trade_creation(self):
        """Test creating a Trade."""
        entry = datetime.now(UTC)
        exit = datetime.now(UTC)
        trade = Trade(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_time=entry,
            exit_time=exit,
            entry_price=150.00,
            exit_price=155.00,
            quantity=100,
            pnl=500.00,
            pnl_pct=3.33,
            commission=2.00,
            duration=3600.0,
        )

        assert trade.symbol == "AAPL"
        assert trade.side == PositionSide.LONG
        assert trade.entry_price == 150.00
        assert trade.exit_price == 155.00
        assert trade.quantity == 100
        assert trade.pnl == 500.00
        assert trade.commission == 2.00
        assert trade.duration == 3600.0
        assert trade.trade_id is not None

    @pytest.mark.unit
    def test_trade_is_winner(self):
        """Test is_winner property for winning trade."""
        trade = Trade(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_time=datetime.now(UTC),
            exit_time=datetime.now(UTC),
            entry_price=150.00,
            exit_price=155.00,
            quantity=100,
            pnl=500.00,
            pnl_pct=3.33,
            commission=2.00,
            duration=3600.0,
        )

        assert trade.is_winner is True

    @pytest.mark.unit
    def test_trade_is_winner_false(self):
        """Test is_winner is False for losing trade."""
        trade = Trade(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_time=datetime.now(UTC),
            exit_time=datetime.now(UTC),
            entry_price=150.00,
            exit_price=145.00,
            quantity=100,
            pnl=-500.00,
            pnl_pct=-3.33,
            commission=2.00,
            duration=3600.0,
        )

        assert trade.is_winner is False

    @pytest.mark.unit
    def test_trade_is_loser(self):
        """Test is_loser property for losing trade."""
        trade = Trade(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_time=datetime.now(UTC),
            exit_time=datetime.now(UTC),
            entry_price=150.00,
            exit_price=145.00,
            quantity=100,
            pnl=-500.00,
            pnl_pct=-3.33,
            commission=2.00,
            duration=3600.0,
        )

        assert trade.is_loser is True

    @pytest.mark.unit
    def test_trade_is_loser_false(self):
        """Test is_loser is False for winning trade."""
        trade = Trade(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_time=datetime.now(UTC),
            exit_time=datetime.now(UTC),
            entry_price=150.00,
            exit_price=155.00,
            quantity=100,
            pnl=500.00,
            pnl_pct=3.33,
            commission=2.00,
            duration=3600.0,
        )

        assert trade.is_loser is False

    @pytest.mark.unit
    def test_trade_breakeven(self):
        """Test trade at breakeven."""
        trade = Trade(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_time=datetime.now(UTC),
            exit_time=datetime.now(UTC),
            entry_price=150.00,
            exit_price=150.00,
            quantity=100,
            pnl=0.00,
            pnl_pct=0.0,
            commission=2.00,
            duration=3600.0,
        )

        assert trade.is_winner is False
        assert trade.is_loser is False
