"""Tests for BrokerSyncManager and position synchronization."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import pytest

from ordinis.adapters.broker.position_sync import (
    BrokerSyncManager,
    CachedAccountState,
    CachedPosition,
    Discrepancy,
    DiscrepancyType,
    PreTradeValidation,
    SyncResult,
    SyncStatus,
    ValidationResult,
)


# ==================== FIXTURES ====================


@dataclass
class MockAccountInfo:
    """Mock account info from broker."""

    account_id: str = "TEST123"
    equity: float = 100000.0
    cash: float = 50000.0
    buying_power: float = 100000.0
    portfolio_value: float = 100000.0
    is_paper: bool = True


@dataclass
class MockPosition:
    """Mock position from broker."""

    symbol: str
    quantity: float
    side: str
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float


@pytest.fixture
def mock_broker():
    """Create mock broker adapter."""
    broker = AsyncMock()
    broker.get_account = AsyncMock(return_value=MockAccountInfo())
    broker.get_positions = AsyncMock(return_value=[])
    return broker


@pytest.fixture
def mock_circuit_breaker():
    """Create mock circuit breaker."""
    cb = MagicMock()
    cb.is_open = False
    cb.record_account_failure = AsyncMock()
    return cb


@pytest.fixture
def sync_manager(mock_broker, mock_circuit_breaker):
    """Create BrokerSyncManager instance."""
    return BrokerSyncManager(
        broker=mock_broker,
        circuit_breaker=mock_circuit_breaker,
        sync_interval_seconds=1.0,
        stale_threshold_seconds=5.0,
        max_positions=20,
        min_buying_power=Decimal("1000"),
    )


# ==================== ENUM TESTS ====================


class TestSyncStatus:
    """Tests for SyncStatus enum."""

    def test_sync_status_values(self):
        """SyncStatus has expected values."""
        assert SyncStatus.SYNCED.name == "SYNCED"
        assert SyncStatus.STALE.name == "STALE"
        assert SyncStatus.SYNCING.name == "SYNCING"
        assert SyncStatus.ERROR.name == "ERROR"
        assert SyncStatus.NEVER_SYNCED.name == "NEVER_SYNCED"

    def test_sync_status_unique_values(self):
        """SyncStatus values are unique."""
        values = [s.value for s in SyncStatus]
        assert len(values) == len(set(values))


class TestDiscrepancyType:
    """Tests for DiscrepancyType enum."""

    def test_discrepancy_types(self):
        """DiscrepancyType has expected values."""
        assert DiscrepancyType.MISSING_LOCAL.name == "MISSING_LOCAL"
        assert DiscrepancyType.MISSING_BROKER.name == "MISSING_BROKER"
        assert DiscrepancyType.QUANTITY_MISMATCH.name == "QUANTITY_MISMATCH"
        assert DiscrepancyType.SIDE_MISMATCH.name == "SIDE_MISMATCH"
        assert DiscrepancyType.PRICE_MISMATCH.name == "PRICE_MISMATCH"


class TestValidationResult:
    """Tests for ValidationResult enum."""

    def test_validation_results(self):
        """ValidationResult has expected values."""
        assert ValidationResult.APPROVED.name == "APPROVED"
        assert ValidationResult.INSUFFICIENT_BUYING_POWER.name == "INSUFFICIENT_BUYING_POWER"
        assert ValidationResult.POSITION_LIMIT_EXCEEDED.name == "POSITION_LIMIT_EXCEEDED"
        assert ValidationResult.CIRCUIT_BREAKER_OPEN.name == "CIRCUIT_BREAKER_OPEN"


# ==================== DATACLASS TESTS ====================


class TestDiscrepancy:
    """Tests for Discrepancy dataclass."""

    def test_create_discrepancy(self):
        """Create discrepancy with all fields."""
        disc = Discrepancy(
            symbol="AAPL",
            type=DiscrepancyType.QUANTITY_MISMATCH,
            local_value=100,
            broker_value=110,
        )
        assert disc.symbol == "AAPL"
        assert disc.type == DiscrepancyType.QUANTITY_MISMATCH
        assert disc.local_value == 100
        assert disc.broker_value == 110
        assert disc.resolved is False
        assert disc.resolution_action is None

    def test_discrepancy_str(self):
        """Discrepancy has readable string representation."""
        disc = Discrepancy(
            symbol="AAPL",
            type=DiscrepancyType.QUANTITY_MISMATCH,
            local_value=100,
            broker_value=110,
        )
        result = str(disc)
        assert "QUANTITY_MISMATCH" in result
        assert "AAPL" in result
        assert "100" in result
        assert "110" in result


class TestPreTradeValidation:
    """Tests for PreTradeValidation dataclass."""

    def test_approved_validation(self):
        """Create approved validation."""
        validation = PreTradeValidation(
            result=ValidationResult.APPROVED,
            approved=True,
            reason="All checks passed",
        )
        assert validation.is_approved is True
        assert validation.approved is True

    def test_rejected_validation(self):
        """Create rejected validation."""
        validation = PreTradeValidation(
            result=ValidationResult.INSUFFICIENT_BUYING_POWER,
            approved=False,
            reason="Not enough funds",
            details={"buying_power": 500, "required": 1000},
        )
        assert validation.is_approved is False
        assert validation.approved is False
        assert validation.details["buying_power"] == 500


class TestCachedAccountState:
    """Tests for CachedAccountState dataclass."""

    def test_default_state(self):
        """Default state has expected values."""
        state = CachedAccountState()
        assert state.account_id == ""
        assert state.equity == Decimal("0")
        assert state.cash == Decimal("0")
        assert state.sync_status == SyncStatus.NEVER_SYNCED

    def test_is_stale_never_synced(self):
        """Never synced state is stale."""
        state = CachedAccountState()
        assert state.is_stale() is True

    def test_is_stale_recent_sync(self):
        """Recently synced state is not stale."""
        state = CachedAccountState(last_sync=datetime.utcnow())
        assert state.is_stale(max_age_seconds=10.0) is False

    def test_is_stale_old_sync(self):
        """Old sync is stale."""
        old_time = datetime.utcnow() - timedelta(seconds=20)
        state = CachedAccountState(last_sync=old_time)
        assert state.is_stale(max_age_seconds=10.0) is True

    def test_time_since_sync_never_synced(self):
        """Time since sync returns inf for never synced."""
        state = CachedAccountState()
        assert state.time_since_sync() == float("inf")

    def test_time_since_sync_with_sync(self):
        """Time since sync returns actual time difference."""
        state = CachedAccountState(last_sync=datetime.utcnow())
        time_since = state.time_since_sync()
        assert 0 <= time_since < 1  # Less than 1 second ago


class TestCachedPosition:
    """Tests for CachedPosition dataclass."""

    def test_create_position(self):
        """Create cached position."""
        pos = CachedPosition(
            symbol="AAPL",
            quantity=Decimal("100"),
            side="long",
            avg_entry_price=Decimal("150.00"),
            market_value=Decimal("15500.00"),
            unrealized_pnl=Decimal("500.00"),
        )
        assert pos.symbol == "AAPL"
        assert pos.quantity == Decimal("100")
        assert pos.side == "long"

    def test_has_discrepancy_no_match(self):
        """Position has discrepancy when quantities differ."""
        pos = CachedPosition(
            symbol="AAPL",
            quantity=Decimal("100"),
            side="long",
            avg_entry_price=Decimal("150.00"),
            market_value=Decimal("15000.00"),
            unrealized_pnl=Decimal("0"),
            local_quantity=Decimal("90"),
        )
        assert pos.has_discrepancy is True

    def test_has_discrepancy_match(self):
        """Position has no discrepancy when quantities match."""
        pos = CachedPosition(
            symbol="AAPL",
            quantity=Decimal("100"),
            side="long",
            avg_entry_price=Decimal("150.00"),
            market_value=Decimal("15000.00"),
            unrealized_pnl=Decimal("0"),
            local_quantity=Decimal("100"),
        )
        assert pos.has_discrepancy is False


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_successful_sync(self):
        """Create successful sync result."""
        result = SyncResult(
            success=True,
            status=SyncStatus.SYNCED,
            positions_synced=5,
            account_synced=True,
        )
        assert result.success is True
        assert result.has_discrepancies is False

    def test_sync_with_discrepancies(self):
        """Sync result with discrepancies."""
        disc = Discrepancy(
            symbol="AAPL",
            type=DiscrepancyType.QUANTITY_MISMATCH,
            local_value=100,
            broker_value=110,
        )
        result = SyncResult(
            success=True,
            status=SyncStatus.SYNCED,
            discrepancies=[disc],
        )
        assert result.has_discrepancies is True

    def test_failed_sync(self):
        """Create failed sync result."""
        result = SyncResult(
            success=False,
            status=SyncStatus.ERROR,
            error="API timeout",
        )
        assert result.success is False
        assert result.error == "API timeout"


# ==================== BROKER SYNC MANAGER TESTS ====================


class TestBrokerSyncManagerInit:
    """Tests for BrokerSyncManager initialization."""

    def test_init_default(self, mock_broker):
        """Initialize with defaults."""
        manager = BrokerSyncManager(broker=mock_broker)
        assert manager._broker == mock_broker
        assert manager._circuit_breaker is None
        assert manager._running is False

    def test_init_with_options(self, mock_broker, mock_circuit_breaker):
        """Initialize with custom options."""
        callback = MagicMock()
        manager = BrokerSyncManager(
            broker=mock_broker,
            circuit_breaker=mock_circuit_breaker,
            sync_interval_seconds=10.0,
            stale_threshold_seconds=30.0,
            max_positions=50,
            min_buying_power=Decimal("5000"),
            on_discrepancy=callback,
        )
        assert manager._circuit_breaker == mock_circuit_breaker
        assert manager._sync_interval == 10.0
        assert manager._stale_threshold == 30.0
        assert manager._max_positions == 50
        assert manager._min_buying_power == Decimal("5000")


class TestBrokerSyncManagerSync:
    """Tests for BrokerSyncManager sync operation."""

    @pytest.mark.asyncio
    async def test_sync_success(self, sync_manager, mock_broker):
        """Sync successfully fetches account and positions."""
        mock_broker.get_positions.return_value = [
            MockPosition("AAPL", 100, "long", 150.0, 15500.0, 500.0),
            MockPosition("GOOGL", 50, "long", 2800.0, 142500.0, 2500.0),
        ]

        result = await sync_manager.sync()

        assert result.success is True
        assert result.status == SyncStatus.SYNCED
        assert result.positions_synced == 2
        assert result.account_synced is True
        mock_broker.get_account.assert_called_once()
        mock_broker.get_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_updates_account(self, sync_manager, mock_broker):
        """Sync updates cached account state."""
        mock_broker.get_account.return_value = MockAccountInfo(
            equity=200000.0, buying_power=150000.0
        )

        await sync_manager.sync()

        assert sync_manager.account.equity == Decimal("200000")
        assert sync_manager.account.buying_power == Decimal("150000")
        assert sync_manager.account.sync_status == SyncStatus.SYNCED

    @pytest.mark.asyncio
    async def test_sync_circuit_breaker_open(self, sync_manager, mock_circuit_breaker):
        """Sync fails when circuit breaker is open."""
        mock_circuit_breaker.is_open = True

        result = await sync_manager.sync()

        assert result.success is False
        assert result.status == SyncStatus.ERROR
        assert "Circuit breaker" in result.error

    @pytest.mark.asyncio
    async def test_sync_api_error(self, sync_manager, mock_broker, mock_circuit_breaker):
        """Sync handles API errors gracefully."""
        mock_broker.get_account.side_effect = Exception("API timeout")

        result = await sync_manager.sync()

        assert result.success is False
        assert result.status == SyncStatus.ERROR
        assert "API timeout" in result.error
        mock_circuit_breaker.record_account_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_triggers_callbacks(self, mock_broker, mock_circuit_breaker):
        """Sync triggers callbacks."""
        on_complete = MagicMock()
        on_discrepancy = MagicMock()

        manager = BrokerSyncManager(
            broker=mock_broker,
            circuit_breaker=mock_circuit_breaker,
            on_sync_complete=on_complete,
            on_discrepancy=on_discrepancy,
        )

        await manager.sync()

        on_complete.assert_called_once()


class TestBrokerSyncManagerDiscrepancies:
    """Tests for discrepancy detection."""

    @pytest.mark.asyncio
    async def test_detect_missing_local(self, sync_manager, mock_broker):
        """Detect positions broker has that we don't track."""
        mock_broker.get_positions.return_value = [
            MockPosition("AAPL", 100, "long", 150.0, 15000.0, 0.0),
        ]

        result = await sync_manager.sync()

        assert result.has_discrepancies is True
        assert len(result.discrepancies) == 1
        assert result.discrepancies[0].type == DiscrepancyType.MISSING_LOCAL
        assert result.discrepancies[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_detect_missing_broker(self, sync_manager, mock_broker):
        """Detect positions we track that broker doesn't have."""
        sync_manager.update_local_position("TSLA", Decimal("50"))

        result = await sync_manager.sync()

        assert result.has_discrepancies is True
        disc = [d for d in result.discrepancies if d.type == DiscrepancyType.MISSING_BROKER]
        assert len(disc) == 1
        assert disc[0].symbol == "TSLA"

    @pytest.mark.asyncio
    async def test_detect_quantity_mismatch(self, sync_manager, mock_broker):
        """Detect quantity mismatches."""
        sync_manager.update_local_position("AAPL", Decimal("100"))
        mock_broker.get_positions.return_value = [
            MockPosition("AAPL", 110, "long", 150.0, 16500.0, 0.0),
        ]

        result = await sync_manager.sync()

        assert result.has_discrepancies is True
        disc = [d for d in result.discrepancies if d.type == DiscrepancyType.QUANTITY_MISMATCH]
        assert len(disc) == 1
        assert disc[0].local_value == 100.0
        assert disc[0].broker_value == 110.0


class TestBrokerSyncManagerValidation:
    """Tests for pre-trade validation."""

    @pytest.mark.asyncio
    async def test_validate_approved(self, sync_manager, mock_broker):
        """Validation approves valid order."""
        await sync_manager.sync()

        result = await sync_manager.validate_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            estimated_cost=Decimal("15000"),
        )

        assert result.is_approved is True
        assert result.result == ValidationResult.APPROVED

    @pytest.mark.asyncio
    async def test_validate_insufficient_buying_power(self, sync_manager, mock_broker):
        """Validation rejects when buying power too low."""
        mock_broker.get_account.return_value = MockAccountInfo(buying_power=500)
        await sync_manager.sync()

        result = await sync_manager.validate_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
        )

        assert result.is_approved is False
        assert result.result == ValidationResult.INSUFFICIENT_BUYING_POWER

    @pytest.mark.asyncio
    async def test_validate_position_limit_exceeded(self, mock_broker, mock_circuit_breaker):
        """Validation rejects when position limit exceeded."""
        # Create manager with max 2 positions
        manager = BrokerSyncManager(
            broker=mock_broker,
            circuit_breaker=mock_circuit_breaker,
            max_positions=2,
        )

        # Mock 2 existing positions
        mock_broker.get_positions.return_value = [
            MockPosition("AAPL", 100, "long", 150.0, 15000.0, 0.0),
            MockPosition("GOOGL", 50, "long", 2800.0, 140000.0, 0.0),
        ]
        await manager.sync()

        # Try to add a third position
        result = await manager.validate_order(
            symbol="MSFT",
            side="buy",
            quantity=Decimal("100"),
        )

        assert result.is_approved is False
        assert result.result == ValidationResult.POSITION_LIMIT_EXCEEDED

    @pytest.mark.asyncio
    async def test_validate_circuit_breaker_open(self, sync_manager, mock_circuit_breaker):
        """Validation rejects when circuit breaker is open."""
        await sync_manager.sync()
        mock_circuit_breaker.is_open = True

        result = await sync_manager.validate_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
        )

        assert result.is_approved is False
        assert result.result == ValidationResult.CIRCUIT_BREAKER_OPEN

    @pytest.mark.asyncio
    async def test_validate_order_cost_exceeds_buying_power(self, sync_manager, mock_broker):
        """Validation rejects when order cost exceeds buying power."""
        await sync_manager.sync()

        result = await sync_manager.validate_order(
            symbol="AAPL",
            side="buy",
            quantity=Decimal("1000"),
            estimated_cost=Decimal("200000"),  # More than buying power
        )

        assert result.is_approved is False
        assert result.result == ValidationResult.INSUFFICIENT_BUYING_POWER


class TestBrokerSyncManagerLocalPositions:
    """Tests for local position tracking."""

    def test_update_local_position_new(self, sync_manager):
        """Update local position for new symbol."""
        sync_manager.update_local_position("AAPL", Decimal("100"))
        assert sync_manager._local_positions["AAPL"] == Decimal("100")

    def test_update_local_position_add(self, sync_manager):
        """Update local position adds to existing."""
        sync_manager.update_local_position("AAPL", Decimal("100"))
        sync_manager.update_local_position("AAPL", Decimal("50"))
        assert sync_manager._local_positions["AAPL"] == Decimal("150")

    def test_update_local_position_removes_zero(self, sync_manager):
        """Update local position removes zero positions."""
        sync_manager.update_local_position("AAPL", Decimal("100"))
        sync_manager.update_local_position("AAPL", Decimal("-100"))
        assert "AAPL" not in sync_manager._local_positions

    def test_resolve_discrepancy_accept_broker(self, sync_manager):
        """Resolve discrepancy by accepting broker state."""
        sync_manager.update_local_position("AAPL", Decimal("100"))
        sync_manager._positions["AAPL"] = CachedPosition(
            symbol="AAPL",
            quantity=Decimal("110"),
            side="long",
            avg_entry_price=Decimal("150"),
            market_value=Decimal("16500"),
            unrealized_pnl=Decimal("0"),
        )
        sync_manager._discrepancies = [
            Discrepancy(
                symbol="AAPL",
                type=DiscrepancyType.QUANTITY_MISMATCH,
                local_value=100,
                broker_value=110,
            )
        ]

        sync_manager.resolve_discrepancy("AAPL", "accept_broker")

        assert sync_manager._local_positions["AAPL"] == Decimal("110")


class TestBrokerSyncManagerStartStop:
    """Tests for start/stop operations."""

    @pytest.mark.asyncio
    async def test_start(self, sync_manager, mock_broker):
        """Start performs initial sync and starts background loop."""
        result = await sync_manager.start()

        assert result.success is True
        assert sync_manager._running is True
        assert sync_manager._sync_task is not None

        # Clean up
        await sync_manager.stop()

    @pytest.mark.asyncio
    async def test_start_already_running(self, sync_manager):
        """Start returns immediately if already running."""
        await sync_manager.start()

        result = await sync_manager.start()

        assert result.success is True
        await sync_manager.stop()

    @pytest.mark.asyncio
    async def test_stop(self, sync_manager):
        """Stop cancels background loop."""
        await sync_manager.start()
        await sync_manager.stop()

        assert sync_manager._running is False
        assert sync_manager._sync_task is None


class TestBrokerSyncManagerProperties:
    """Tests for properties and getters."""

    @pytest.mark.asyncio
    async def test_account_property(self, sync_manager, mock_broker):
        """Account property returns cached state."""
        await sync_manager.sync()
        account = sync_manager.account
        assert isinstance(account, CachedAccountState)
        assert account.equity == Decimal("100000")

    @pytest.mark.asyncio
    async def test_positions_property(self, sync_manager, mock_broker):
        """Positions property returns copy of cached positions."""
        mock_broker.get_positions.return_value = [
            MockPosition("AAPL", 100, "long", 150.0, 15000.0, 0.0),
        ]
        await sync_manager.sync()

        positions = sync_manager.positions
        assert "AAPL" in positions
        # Verify it's a copy
        positions["TEST"] = MagicMock()
        assert "TEST" not in sync_manager._positions

    @pytest.mark.asyncio
    async def test_discrepancies_property(self, sync_manager, mock_broker):
        """Discrepancies property returns only unresolved."""
        mock_broker.get_positions.return_value = [
            MockPosition("AAPL", 100, "long", 150.0, 15000.0, 0.0),
        ]
        await sync_manager.sync()

        # Initially has discrepancy
        assert len(sync_manager.discrepancies) == 1

        # Resolve it
        sync_manager.resolve_discrepancy("AAPL", "accept_broker")
        assert len(sync_manager.discrepancies) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, sync_manager, mock_broker):
        """Get stats returns sync statistics."""
        await sync_manager.sync()
        stats = sync_manager.get_stats()

        assert stats["total_syncs"] == 1
        assert stats["failed_syncs"] == 0
        assert stats["success_rate"] == 100.0
        assert stats["sync_status"] == "SYNCED"
        assert "buying_power" in stats
