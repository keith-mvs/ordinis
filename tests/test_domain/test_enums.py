"""
Tests for domain enums.
Tests cover:
- OrderSide enum
- OrderType enum
- TimeInForce enum
- OrderStatus enum
- PositionSide enum
"""

import pytest

from ordinis.domain.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)


class TestOrderSide:
    """Tests for OrderSide enum."""

    @pytest.mark.unit
    def test_order_side_values(self):
        """Test OrderSide has expected values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    @pytest.mark.unit
    def test_order_side_is_str_enum(self):
        """Test OrderSide is a string enum."""
        assert isinstance(OrderSide.BUY, str)
        assert OrderSide.BUY.value == "BUY"

    @pytest.mark.unit
    def test_order_side_count(self):
        """Test OrderSide has 2 values."""
        assert len(OrderSide) == 2


class TestOrderType:
    """Tests for OrderType enum."""

    @pytest.mark.unit
    def test_order_type_values(self):
        """Test OrderType has expected values."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"
        assert OrderType.STOP_LIMIT.value == "STOP_LIMIT"
        assert OrderType.TRAILING_STOP.value == "TRAILING_STOP"

    @pytest.mark.unit
    def test_order_type_is_str_enum(self):
        """Test OrderType is a string enum."""
        assert isinstance(OrderType.MARKET, str)

    @pytest.mark.unit
    def test_order_type_count(self):
        """Test OrderType has 5 values."""
        assert len(OrderType) == 5


class TestTimeInForce:
    """Tests for TimeInForce enum."""

    @pytest.mark.unit
    def test_time_in_force_values(self):
        """Test TimeInForce has expected values."""
        assert TimeInForce.DAY.value == "DAY"
        assert TimeInForce.GTC.value == "GTC"
        assert TimeInForce.IOC.value == "IOC"
        assert TimeInForce.FOK.value == "FOK"

    @pytest.mark.unit
    def test_time_in_force_is_str_enum(self):
        """Test TimeInForce is a string enum."""
        assert isinstance(TimeInForce.DAY, str)

    @pytest.mark.unit
    def test_time_in_force_count(self):
        """Test TimeInForce has 4 values."""
        assert len(TimeInForce) == 4


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    @pytest.mark.unit
    def test_order_status_values(self):
        """Test OrderStatus has expected values."""
        assert OrderStatus.CREATED.value == "CREATED"
        assert OrderStatus.VALIDATED.value == "VALIDATED"
        assert OrderStatus.PENDING_SUBMIT.value == "PENDING_SUBMIT"
        assert OrderStatus.SUBMITTED.value == "SUBMITTED"
        assert OrderStatus.ACKNOWLEDGED.value == "ACKNOWLEDGED"
        assert OrderStatus.PARTIALLY_FILLED.value == "PARTIALLY_FILLED"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.REJECTED.value == "REJECTED"
        assert OrderStatus.EXPIRED.value == "EXPIRED"
        assert OrderStatus.ERROR.value == "ERROR"

    @pytest.mark.unit
    def test_order_status_is_str_enum(self):
        """Test OrderStatus is a string enum."""
        assert isinstance(OrderStatus.CREATED, str)

    @pytest.mark.unit
    def test_order_status_count(self):
        """Test OrderStatus has 11 values."""
        assert len(OrderStatus) == 11


class TestPositionSide:
    """Tests for PositionSide enum."""

    @pytest.mark.unit
    def test_position_side_values(self):
        """Test PositionSide has expected values."""
        assert PositionSide.LONG.value == "LONG"
        assert PositionSide.SHORT.value == "SHORT"
        assert PositionSide.FLAT.value == "FLAT"

    @pytest.mark.unit
    def test_position_side_is_str_enum(self):
        """Test PositionSide is a string enum."""
        assert isinstance(PositionSide.LONG, str)

    @pytest.mark.unit
    def test_position_side_count(self):
        """Test PositionSide has 3 values."""
        assert len(PositionSide) == 3
