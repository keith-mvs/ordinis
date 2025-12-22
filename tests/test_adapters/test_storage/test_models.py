"""
Tests for database row models.

Coverage focus:
- PositionRow: from_row, to_insert_tuple, to_update_tuple
- OrderRow: from_row, to_insert_tuple, JSON parsing methods
- FillRow: from_row, to_insert_tuple
- TradeRow: from_row, to_insert_tuple
- SystemStateRow: from_row, get_typed_value with all types
- PortfolioSnapshotRow: from_row, to_insert_tuple, get_positions
"""

import json

import pytest

from ordinis.adapters.storage.models import (
    FillRow,
    OrderRow,
    PortfolioSnapshotRow,
    PositionRow,
    SystemStateRow,
    TradeRow,
)

# ==================== PositionRow Tests ====================


@pytest.mark.unit
def test_position_row_defaults():
    """Test PositionRow with default values."""
    row = PositionRow(symbol="AAPL", last_update="2025-01-01T00:00:00")

    assert row.symbol == "AAPL"
    assert row.side == "FLAT"
    assert row.quantity == 0
    assert row.avg_cost == 0.0
    assert row.current_price == 0.0
    assert row.realized_pnl == 0.0
    assert row.unrealized_pnl == 0.0
    assert row.entry_time is None
    assert row.id is None
    assert row.created_at is None
    assert row.updated_at is None


@pytest.mark.unit
def test_position_row_from_row():
    """Test PositionRow.from_row with full data."""
    db_row = (
        1,  # id
        "AAPL",  # symbol
        "LONG",  # side
        100,  # quantity
        150.0,  # avg_cost
        155.0,  # current_price
        200.0,  # realized_pnl
        500.0,  # unrealized_pnl
        "2025-01-01T10:00:00",  # entry_time
        "2025-01-01T15:00:00",  # last_update
        "2025-01-01T09:00:00",  # created_at
        "2025-01-01T15:00:00",  # updated_at
    )

    row = PositionRow.from_row(db_row)

    assert row.id == 1
    assert row.symbol == "AAPL"
    assert row.side == "LONG"
    assert row.quantity == 100
    assert row.avg_cost == 150.0
    assert row.current_price == 155.0
    assert row.realized_pnl == 200.0
    assert row.unrealized_pnl == 500.0
    assert row.entry_time == "2025-01-01T10:00:00"
    assert row.last_update == "2025-01-01T15:00:00"
    assert row.created_at == "2025-01-01T09:00:00"
    assert row.updated_at == "2025-01-01T15:00:00"


@pytest.mark.unit
def test_position_row_from_row_with_nulls():
    """Test PositionRow.from_row with NULL values."""
    db_row = (
        None,  # id (can be NULL)
        "TSLA",  # symbol
        "FLAT",  # side
        0,  # quantity
        0.0,  # avg_cost
        0.0,  # current_price
        0.0,  # realized_pnl
        0.0,  # unrealized_pnl
        None,  # entry_time (NULL)
        "2025-01-01T15:00:00",  # last_update
        None,  # created_at (NULL)
        None,  # updated_at (NULL)
    )

    row = PositionRow.from_row(db_row)

    assert row.id is None
    assert row.symbol == "TSLA"
    assert row.entry_time is None
    assert row.created_at is None
    assert row.updated_at is None


@pytest.mark.unit
def test_position_row_to_insert_tuple():
    """Test PositionRow.to_insert_tuple excludes id and timestamps."""
    row = PositionRow(
        id=999,  # Should be excluded
        symbol="AAPL",
        side="LONG",
        quantity=100,
        avg_cost=150.0,
        current_price=155.0,
        realized_pnl=200.0,
        unrealized_pnl=500.0,
        entry_time="2025-01-01T10:00:00",
        last_update="2025-01-01T15:00:00",
        created_at="2025-01-01T09:00:00",  # Should be excluded
        updated_at="2025-01-01T15:00:00",  # Should be excluded
    )

    result = row.to_insert_tuple()

    # Should exclude id, created_at, updated_at
    assert len(result) == 9
    assert result == (
        "AAPL",
        "LONG",
        100,
        150.0,
        155.0,
        200.0,
        500.0,
        "2025-01-01T10:00:00",
        "2025-01-01T15:00:00",
    )
    assert 999 not in result


@pytest.mark.unit
def test_position_row_to_update_tuple():
    """Test PositionRow.to_update_tuple includes symbol for WHERE clause."""
    row = PositionRow(
        id=1,
        symbol="AAPL",
        side="LONG",
        quantity=100,
        avg_cost=150.0,
        current_price=155.0,
        realized_pnl=200.0,
        unrealized_pnl=500.0,
        entry_time="2025-01-01T10:00:00",
        last_update="2025-01-01T15:00:00",
    )

    result = row.to_update_tuple()

    # Should have symbol at the end for WHERE clause
    assert len(result) == 9
    assert result[-1] == "AAPL"
    assert result == (
        "LONG",
        100,
        150.0,
        155.0,
        200.0,
        500.0,
        "2025-01-01T10:00:00",
        "2025-01-01T15:00:00",
        "AAPL",
    )


# ==================== OrderRow Tests ====================


@pytest.mark.unit
def test_order_row_defaults():
    """Test OrderRow with default values."""
    row = OrderRow(
        order_id="ORD001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type="market",
        remaining_quantity=100,
        created_at="2025-01-01T10:00:00",
    )

    assert row.order_id == "ORD001"
    assert row.time_in_force == "day"
    assert row.status == "created"
    assert row.filled_quantity == 0
    assert row.avg_fill_price == 0.0
    assert row.retry_count == 0
    assert row.limit_price is None
    assert row.stop_price is None
    assert row.broker_response is None
    assert row.metadata is None


@pytest.mark.unit
def test_order_row_from_row():
    """Test OrderRow.from_row with full data."""
    db_row = (
        1,  # id
        "ORD001",  # order_id
        "AAPL",  # symbol
        "buy",  # side
        100,  # quantity
        "limit",  # order_type
        150.0,  # limit_price
        None,  # stop_price
        "gtc",  # time_in_force
        "filled",  # status
        100,  # filled_quantity
        0,  # remaining_quantity
        150.05,  # avg_fill_price
        "2025-01-01T10:00:00",  # created_at
        "2025-01-01T10:00:01",  # submitted_at
        "2025-01-01T10:00:05",  # filled_at
        "INTENT001",  # intent_id
        "SIG001",  # signal_id
        "STRAT001",  # strategy_id
        "SESSION001",  # session_id
        "BRK001",  # broker_order_id
        '{"status": "filled"}',  # broker_response
        None,  # error_message
        0,  # retry_count
        1,  # chroma_synced
        '{"notes": "test"}',  # metadata
        "2025-01-01T10:00:05",  # updated_at
    )

    row = OrderRow.from_row(db_row)

    assert row.id == 1
    assert row.order_id == "ORD001"
    assert row.symbol == "AAPL"
    assert row.side == "buy"
    assert row.quantity == 100
    assert row.order_type == "limit"
    assert row.limit_price == 150.0
    assert row.stop_price is None
    assert row.time_in_force == "gtc"
    assert row.status == "filled"
    assert row.filled_quantity == 100
    assert row.remaining_quantity == 0
    assert row.avg_fill_price == 150.05
    assert row.intent_id == "INTENT001"
    assert row.signal_id == "SIG001"
    assert row.strategy_id == "STRAT001"
    assert row.session_id == "SESSION001"
    assert row.broker_order_id == "BRK001"
    assert row.broker_response == '{"status": "filled"}'
    assert row.error_message is None
    assert row.retry_count == 0
    assert row.chroma_synced == 1
    assert row.metadata == '{"notes": "test"}'


@pytest.mark.unit
def test_order_row_to_insert_tuple():
    """Test OrderRow.to_insert_tuple excludes id and updated_at."""
    row = OrderRow(
        id=999,
        order_id="ORD001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type="limit",
        limit_price=150.0,
        stop_price=None,
        time_in_force="gtc",
        status="created",
        filled_quantity=0,
        remaining_quantity=100,
        avg_fill_price=0.0,
        created_at="2025-01-01T10:00:00",
        submitted_at=None,
        filled_at=None,
        intent_id="INTENT001",
        signal_id="SIG001",
        strategy_id="STRAT001",
        session_id=None,
        broker_order_id=None,
        broker_response=None,
        error_message=None,
        retry_count=0,
        chroma_synced=0,
        metadata='{"test": true}',
        updated_at="2025-01-01T10:00:00",
    )

    result = row.to_insert_tuple()

    # Should exclude id and updated_at, include session_id and chroma_synced
    assert len(result) == 25
    assert 999 not in result
    assert result[0] == "ORD001"
    assert result[-1] == '{"test": true}'


@pytest.mark.unit
def test_order_row_get_broker_response_dict_with_json():
    """Test get_broker_response_dict with valid JSON."""
    row = OrderRow(
        order_id="ORD001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type="market",
        remaining_quantity=100,
        created_at="2025-01-01T10:00:00",
        broker_response='{"status": "filled", "price": 150.0}',
    )

    result = row.get_broker_response_dict()

    assert isinstance(result, dict)
    assert result["status"] == "filled"
    assert result["price"] == 150.0


@pytest.mark.unit
def test_order_row_get_broker_response_dict_with_none():
    """Test get_broker_response_dict with None returns empty dict."""
    row = OrderRow(
        order_id="ORD001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type="market",
        remaining_quantity=100,
        created_at="2025-01-01T10:00:00",
        broker_response=None,
    )

    result = row.get_broker_response_dict()

    assert result == {}


@pytest.mark.unit
def test_order_row_get_metadata_dict_with_json():
    """Test get_metadata_dict with valid JSON."""
    row = OrderRow(
        order_id="ORD001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type="market",
        remaining_quantity=100,
        created_at="2025-01-01T10:00:00",
        metadata='{"notes": "test order", "priority": "high"}',
    )

    result = row.get_metadata_dict()

    assert isinstance(result, dict)
    assert result["notes"] == "test order"
    assert result["priority"] == "high"


@pytest.mark.unit
def test_order_row_get_metadata_dict_with_none():
    """Test get_metadata_dict with None returns empty dict."""
    row = OrderRow(
        order_id="ORD001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type="market",
        remaining_quantity=100,
        created_at="2025-01-01T10:00:00",
        metadata=None,
    )

    result = row.get_metadata_dict()

    assert result == {}


# ==================== FillRow Tests ====================


@pytest.mark.unit
def test_fill_row_defaults():
    """Test FillRow with default values."""
    row = FillRow(
        fill_id="FILL001",
        order_id="ORD001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        price=150.0,
        timestamp="2025-01-01T10:00:00",
    )

    assert row.fill_id == "FILL001"
    assert row.commission == 0.0
    assert row.latency_ms == 0.0
    assert row.slippage_bps == 0.0
    assert row.vs_arrival_bps == 0.0
    assert row.metadata is None
    assert row.created_at is None


@pytest.mark.unit
def test_fill_row_from_row():
    """Test FillRow.from_row with full data."""
    db_row = (
        1,  # id
        "FILL001",  # fill_id
        "ORD001",  # order_id
        "AAPL",  # symbol
        "buy",  # side
        100,  # quantity
        150.05,  # price
        1.50,  # commission
        "2025-01-01T10:00:05",  # timestamp
        125.5,  # latency_ms
        2.5,  # slippage_bps
        1.8,  # vs_arrival_bps
        '{"venue": "NASDAQ"}',  # metadata
        "2025-01-01T10:00:05",  # created_at
    )

    row = FillRow.from_row(db_row)

    assert row.id == 1
    assert row.fill_id == "FILL001"
    assert row.order_id == "ORD001"
    assert row.symbol == "AAPL"
    assert row.side == "buy"
    assert row.quantity == 100
    assert row.price == 150.05
    assert row.commission == 1.50
    assert row.timestamp == "2025-01-01T10:00:05"
    assert row.latency_ms == 125.5
    assert row.slippage_bps == 2.5
    assert row.vs_arrival_bps == 1.8
    assert row.metadata == '{"venue": "NASDAQ"}'
    assert row.created_at == "2025-01-01T10:00:05"


@pytest.mark.unit
def test_fill_row_to_insert_tuple():
    """Test FillRow.to_insert_tuple excludes id and created_at."""
    row = FillRow(
        id=999,
        fill_id="FILL001",
        order_id="ORD001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        price=150.05,
        commission=1.50,
        timestamp="2025-01-01T10:00:05",
        latency_ms=125.5,
        slippage_bps=2.5,
        vs_arrival_bps=1.8,
        metadata='{"venue": "NASDAQ"}',
        created_at="2025-01-01T10:00:05",
    )

    result = row.to_insert_tuple()

    # Should exclude id and created_at
    assert len(result) == 12
    assert 999 not in result
    assert result[0] == "FILL001"
    assert result[1] == "ORD001"
    assert result[-1] == '{"venue": "NASDAQ"}'


# ==================== TradeRow Tests ====================


@pytest.mark.unit
def test_trade_row_defaults():
    """Test TradeRow with default values."""
    row = TradeRow(
        trade_id="TRADE001",
        symbol="AAPL",
        side="LONG",
        entry_time="2025-01-01T10:00:00",
        exit_time="2025-01-01T11:00:00",
        entry_price=150.0,
        exit_price=155.0,
        quantity=100,
        pnl=500.0,
        pnl_pct=3.33,
        duration_seconds=3600.0,
    )

    assert row.trade_id == "TRADE001"
    assert row.commission == 0.0
    assert row.entry_order_id is None
    assert row.exit_order_id is None
    assert row.strategy_id is None
    assert row.metadata is None
    assert row.created_at is None


@pytest.mark.unit
def test_trade_row_from_row():
    """Test TradeRow.from_row with full data."""
    db_row = (
        1,  # id
        "TRADE001",  # trade_id
        "AAPL",  # symbol
        "LONG",  # side
        "2025-01-01T10:00:00",  # entry_time
        "2025-01-01T11:00:00",  # exit_time
        150.0,  # entry_price
        155.0,  # exit_price
        100,  # quantity
        500.0,  # pnl
        3.33,  # pnl_pct
        3.0,  # commission
        3600.0,  # duration_seconds
        "ORD001",  # entry_order_id
        "ORD002",  # exit_order_id
        "STRAT001",  # strategy_id
        "SESSION001",  # session_id
        1,  # chroma_synced
        "CHROMA001",  # chroma_id
        '{"reason": "target"}',  # metadata
        "2025-01-01T11:00:00",  # created_at
    )

    row = TradeRow.from_row(db_row)

    assert row.id == 1
    assert row.trade_id == "TRADE001"
    assert row.symbol == "AAPL"
    assert row.side == "LONG"
    assert row.entry_time == "2025-01-01T10:00:00"
    assert row.exit_time == "2025-01-01T11:00:00"
    assert row.entry_price == 150.0
    assert row.exit_price == 155.0
    assert row.quantity == 100
    assert row.pnl == 500.0
    assert row.pnl_pct == 3.33
    assert row.commission == 3.0
    assert row.duration_seconds == 3600.0
    assert row.entry_order_id == "ORD001"
    assert row.exit_order_id == "ORD002"
    assert row.strategy_id == "STRAT001"
    assert row.session_id == "SESSION001"
    assert row.chroma_synced == 1
    assert row.chroma_id == "CHROMA001"
    assert row.metadata == '{"reason": "target"}'
    assert row.created_at == "2025-01-01T11:00:00"


@pytest.mark.unit
def test_trade_row_to_insert_tuple():
    """Test TradeRow.to_insert_tuple excludes id and created_at."""
    row = TradeRow(
        id=999,
        trade_id="TRADE001",
        symbol="AAPL",
        side="LONG",
        entry_time="2025-01-01T10:00:00",
        exit_time="2025-01-01T11:00:00",
        entry_price=150.0,
        exit_price=155.0,
        quantity=100,
        pnl=500.0,
        pnl_pct=3.33,
        commission=3.0,
        duration_seconds=3600.0,
        entry_order_id="ORD001",
        exit_order_id="ORD002",
        strategy_id="STRAT001",
        session_id=None,
        chroma_synced=0,
        chroma_id=None,
        metadata='{"reason": "target"}',
        created_at="2025-01-01T11:00:00",
    )

    result = row.to_insert_tuple()

    # Should exclude id and created_at, include session_id, chroma_synced, chroma_id
    assert len(result) == 19
    assert 999 not in result
    assert result[0] == "TRADE001"
    assert result[-1] == '{"reason": "target"}'


# ==================== SystemStateRow Tests ====================


@pytest.mark.unit
def test_system_state_row_defaults():
    """Test SystemStateRow with default values."""
    row = SystemStateRow(key="test_key", value="test_value")

    assert row.key == "test_key"
    assert row.value == "test_value"
    assert row.value_type == "string"
    assert row.description is None
    assert row.id is None
    assert row.updated_at is None
    assert row.created_at is None


@pytest.mark.unit
def test_system_state_row_from_row():
    """Test SystemStateRow.from_row with full data."""
    db_row = (
        1,  # id
        "trading_enabled",  # key
        "true",  # value
        "bool",  # value_type
        "Whether trading is enabled",  # description
        "2025-01-01T10:00:00",  # updated_at
        "2025-01-01T09:00:00",  # created_at
    )

    row = SystemStateRow.from_row(db_row)

    assert row.id == 1
    assert row.key == "trading_enabled"
    assert row.value == "true"
    assert row.value_type == "bool"
    assert row.description == "Whether trading is enabled"
    assert row.updated_at == "2025-01-01T10:00:00"
    assert row.created_at == "2025-01-01T09:00:00"


@pytest.mark.unit
def test_system_state_get_typed_value_bool_true():
    """Test get_typed_value for bool type with true values."""
    test_cases = [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("Yes", True),
        ("YES", True),
    ]

    for value, expected in test_cases:
        row = SystemStateRow(key="test", value=value, value_type="bool")
        assert row.get_typed_value() is expected


@pytest.mark.unit
def test_system_state_get_typed_value_bool_false():
    """Test get_typed_value for bool type with false values."""
    test_cases = ["false", "False", "FALSE", "0", "no", "No", "NO", "anything"]

    for value in test_cases:
        row = SystemStateRow(key="test", value=value, value_type="bool")
        assert row.get_typed_value() is False


@pytest.mark.unit
def test_system_state_get_typed_value_int():
    """Test get_typed_value for int type."""
    row = SystemStateRow(key="max_positions", value="10", value_type="int")

    result = row.get_typed_value()

    assert result == 10
    assert isinstance(result, int)


@pytest.mark.unit
def test_system_state_get_typed_value_float():
    """Test get_typed_value for float type."""
    row = SystemStateRow(key="risk_factor", value="0.02", value_type="float")

    result = row.get_typed_value()

    assert result == 0.02
    assert isinstance(result, float)


@pytest.mark.unit
def test_system_state_get_typed_value_json():
    """Test get_typed_value for json type."""
    json_value = '{"setting1": "value1", "setting2": 42}'
    row = SystemStateRow(key="config", value=json_value, value_type="json")

    result = row.get_typed_value()

    assert isinstance(result, dict)
    assert result["setting1"] == "value1"
    assert result["setting2"] == 42


@pytest.mark.unit
def test_system_state_get_typed_value_string():
    """Test get_typed_value for string type (default)."""
    row = SystemStateRow(key="api_key", value="secret123", value_type="string")

    result = row.get_typed_value()

    assert result == "secret123"
    assert isinstance(result, str)


@pytest.mark.unit
def test_system_state_get_typed_value_unknown_type():
    """Test get_typed_value for unknown type defaults to string."""
    row = SystemStateRow(key="test", value="some_value", value_type="unknown")

    result = row.get_typed_value()

    assert result == "some_value"


# ==================== PortfolioSnapshotRow Tests ====================


@pytest.mark.unit
def test_portfolio_snapshot_row_defaults():
    """Test PortfolioSnapshotRow with default values."""
    positions = [{"symbol": "AAPL", "quantity": 100}]
    row = PortfolioSnapshotRow(
        snapshot_date="2025-01-01",
        cash=10000.0,
        total_equity=25000.0,
        total_position_value=15000.0,
        positions_json=json.dumps(positions),
    )

    assert row.snapshot_date == "2025-01-01"
    assert row.cash == 10000.0
    assert row.total_equity == 25000.0
    assert row.total_position_value == 15000.0
    assert row.id is None
    assert row.created_at is None


@pytest.mark.unit
def test_portfolio_snapshot_row_from_row():
    """Test PortfolioSnapshotRow.from_row with full data."""
    positions = [
        {"symbol": "AAPL", "quantity": 100, "value": 15000.0},
        {"symbol": "GOOGL", "quantity": 50, "value": 10000.0},
    ]

    db_row = (
        1,  # id
        "2025-01-01",  # snapshot_date
        10000.0,  # cash
        35000.0,  # total_equity
        25000.0,  # total_position_value
        json.dumps(positions),  # positions_json
        "2025-01-01T16:00:00",  # created_at
    )

    row = PortfolioSnapshotRow.from_row(db_row)

    assert row.id == 1
    assert row.snapshot_date == "2025-01-01"
    assert row.cash == 10000.0
    assert row.total_equity == 35000.0
    assert row.total_position_value == 25000.0
    assert row.created_at == "2025-01-01T16:00:00"


@pytest.mark.unit
def test_portfolio_snapshot_get_positions():
    """Test get_positions parses JSON correctly."""
    positions = [
        {"symbol": "AAPL", "quantity": 100, "value": 15000.0},
        {"symbol": "GOOGL", "quantity": 50, "value": 10000.0},
    ]

    row = PortfolioSnapshotRow(
        snapshot_date="2025-01-01",
        cash=10000.0,
        total_equity=35000.0,
        total_position_value=25000.0,
        positions_json=json.dumps(positions),
    )

    result = row.get_positions()

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["symbol"] == "AAPL"
    assert result[0]["quantity"] == 100
    assert result[0]["value"] == 15000.0
    assert result[1]["symbol"] == "GOOGL"


@pytest.mark.unit
def test_portfolio_snapshot_get_positions_empty():
    """Test get_positions with empty array."""
    row = PortfolioSnapshotRow(
        snapshot_date="2025-01-01",
        cash=10000.0,
        total_equity=10000.0,
        total_position_value=0.0,
        positions_json="[]",
    )

    result = row.get_positions()

    assert result == []


@pytest.mark.unit
def test_portfolio_snapshot_to_insert_tuple():
    """Test PortfolioSnapshotRow.to_insert_tuple excludes id and created_at."""
    positions = [{"symbol": "AAPL", "quantity": 100}]
    row = PortfolioSnapshotRow(
        id=999,
        snapshot_date="2025-01-01",
        cash=10000.0,
        total_equity=25000.0,
        total_position_value=15000.0,
        positions_json=json.dumps(positions),
        created_at="2025-01-01T16:00:00",
    )

    result = row.to_insert_tuple()

    # Should exclude id and created_at
    assert len(result) == 5
    assert 999 not in result
    assert result[0] == "2025-01-01"
    assert result[1] == 10000.0
    assert result[2] == 25000.0
    assert result[3] == 15000.0
    assert isinstance(json.loads(result[4]), list)


# ==================== Edge Cases and Integration ====================


@pytest.mark.unit
def test_order_row_with_all_optional_fields_none():
    """Test OrderRow with all optional fields as None."""
    row = OrderRow(
        order_id="ORD001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type="market",
        remaining_quantity=100,
        created_at="2025-01-01T10:00:00",
        limit_price=None,
        stop_price=None,
        submitted_at=None,
        filled_at=None,
        intent_id=None,
        signal_id=None,
        strategy_id=None,
        session_id=None,
        broker_order_id=None,
        broker_response=None,
        error_message=None,
        retry_count=0,
        chroma_synced=0,
        metadata=None,
    )

    assert row.get_broker_response_dict() == {}
    assert row.get_metadata_dict() == {}

    # Should still be able to create insert tuple
    # Includes: order_id, symbol, side, quantity, order_type, limit_price, stop_price,
    # time_in_force, status, filled_quantity, remaining_quantity, avg_fill_price,
    # created_at, submitted_at, filled_at, intent_id, signal_id, strategy_id,
    # session_id, broker_order_id, broker_response, error_message, retry_count, chroma_synced, metadata
    result = row.to_insert_tuple()
    assert len(result) == 25


@pytest.mark.unit
def test_position_row_round_trip():
    """Test PositionRow round-trip from_row -> to_insert_tuple."""
    original_data = (
        1,
        "AAPL",
        "LONG",
        100,
        150.0,
        155.0,
        200.0,
        500.0,
        "2025-01-01T10:00:00",
        "2025-01-01T15:00:00",
        "2025-01-01T09:00:00",
        "2025-01-01T15:00:00",
    )

    row = PositionRow.from_row(original_data)
    insert_tuple = row.to_insert_tuple()

    # Insert tuple should match original data (excluding id and timestamps)
    assert insert_tuple[0] == original_data[1]  # symbol
    assert insert_tuple[1] == original_data[2]  # side
    assert insert_tuple[2] == original_data[3]  # quantity
    assert insert_tuple[3] == original_data[4]  # avg_cost
    assert insert_tuple[4] == original_data[5]  # current_price
    assert insert_tuple[5] == original_data[6]  # realized_pnl
    assert insert_tuple[6] == original_data[7]  # unrealized_pnl
    assert insert_tuple[7] == original_data[8]  # entry_time
    assert insert_tuple[8] == original_data[9]  # last_update


@pytest.mark.unit
def test_system_state_complex_json():
    """Test SystemStateRow with complex nested JSON."""
    complex_json = {
        "settings": {
            "trading": {"enabled": True, "max_positions": 10},
            "risk": {"max_loss_pct": 0.02, "position_size_pct": 0.1},
        },
        "strategies": ["momentum", "mean_reversion"],
    }

    row = SystemStateRow(
        key="system_config",
        value=json.dumps(complex_json),
        value_type="json",
    )

    result = row.get_typed_value()

    assert isinstance(result, dict)
    assert result["settings"]["trading"]["enabled"] is True
    assert result["settings"]["risk"]["max_loss_pct"] == 0.02
    assert "momentum" in result["strategies"]
