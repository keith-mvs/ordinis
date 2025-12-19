"""
Enhanced Broker Position Sync and Pre-Trade Validation.

Implements robust broker synchronization with:
- Position reconciliation
- Pre-trade buying power validation  
- Real-time account state caching
- Discrepancy detection and alerting

Step 2 of Trade Enhancement Roadmap (P0 Critical).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ordinis.adapters.broker.broker import (
        AccountInfo,
        BrokerAdapter,
        Order,
        Position,
    )
    from ordinis.safety.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Broker synchronization status."""
    
    SYNCED = auto()
    STALE = auto()
    SYNCING = auto()
    ERROR = auto()
    NEVER_SYNCED = auto()


class DiscrepancyType(Enum):
    """Types of position discrepancies."""
    
    MISSING_LOCAL = auto()  # Broker has position we don't track
    MISSING_BROKER = auto()  # We track position broker doesn't have
    QUANTITY_MISMATCH = auto()  # Quantity differs
    SIDE_MISMATCH = auto()  # Long vs short mismatch
    PRICE_MISMATCH = auto()  # Entry price differs significantly


class ValidationResult(Enum):
    """Pre-trade validation result."""
    
    APPROVED = auto()
    INSUFFICIENT_BUYING_POWER = auto()
    POSITION_LIMIT_EXCEEDED = auto()
    RISK_LIMIT_EXCEEDED = auto()
    MARKET_CLOSED = auto()
    KILL_SWITCH_ACTIVE = auto()
    CIRCUIT_BREAKER_OPEN = auto()
    ACCOUNT_RESTRICTED = auto()
    SYMBOL_RESTRICTED = auto()
    STALE_ACCOUNT_DATA = auto()


@dataclass
class Discrepancy:
    """Detected discrepancy between local and broker state."""
    
    symbol: str
    type: DiscrepancyType
    local_value: Any
    broker_value: Any
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution_action: str | None = None
    
    def __str__(self) -> str:
        return (
            f"{self.type.name}: {self.symbol} "
            f"(local={self.local_value}, broker={self.broker_value})"
        )


@dataclass
class PreTradeValidation:
    """Result of pre-trade validation."""
    
    result: ValidationResult
    approved: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_approved(self) -> bool:
        return self.result == ValidationResult.APPROVED


@dataclass
class CachedAccountState:
    """Thread-safe cached account state."""
    
    account_id: str = ""
    equity: Decimal = Decimal("0")
    cash: Decimal = Decimal("0")
    buying_power: Decimal = Decimal("0")
    portfolio_value: Decimal = Decimal("0")
    day_trades_remaining: int = 3
    pattern_day_trader: bool = False
    is_paper: bool = True
    last_sync: datetime | None = None
    sync_status: SyncStatus = SyncStatus.NEVER_SYNCED
    
    def is_stale(self, max_age_seconds: float = 5.0) -> bool:
        """Check if cached state is stale."""
        if self.last_sync is None:
            return True
        age = (datetime.utcnow() - self.last_sync).total_seconds()
        return age > max_age_seconds
        
    def time_since_sync(self) -> float:
        """Get seconds since last sync."""
        if self.last_sync is None:
            return float("inf")
        return (datetime.utcnow() - self.last_sync).total_seconds()


@dataclass
class CachedPosition:
    """Cached position with tracking metadata."""
    
    symbol: str
    quantity: Decimal
    side: str  # "long" or "short"
    avg_entry_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    cost_basis: Decimal = Decimal("0")
    last_sync: datetime = field(default_factory=datetime.utcnow)
    local_quantity: Decimal = Decimal("0")  # Our tracked quantity
    
    @property
    def has_discrepancy(self) -> bool:
        return self.quantity != self.local_quantity


@dataclass
class SyncResult:
    """Result of broker sync operation."""
    
    success: bool
    status: SyncStatus
    positions_synced: int = 0
    account_synced: bool = False
    discrepancies: list[Discrepancy] = field(default_factory=list)
    error: str | None = None
    duration_ms: float = 0.0
    
    @property
    def has_discrepancies(self) -> bool:
        return len(self.discrepancies) > 0


class BrokerSyncManager:
    """
    Manages broker state synchronization and pre-trade validation.
    
    Features:
    - Periodic background sync with configurable interval
    - Position reconciliation with discrepancy detection
    - Pre-trade validation for buying power and limits
    - Circuit breaker integration for API failures
    - Event callbacks for state changes
    
    Example:
        >>> sync_manager = BrokerSyncManager(broker_adapter)
        >>> await sync_manager.start()
        >>> 
        >>> # Validate before order
        >>> validation = await sync_manager.validate_order(order)
        >>> if validation.is_approved:
        ...     await broker.submit_order(order)
    """
    
    def __init__(
        self,
        broker: BrokerAdapter,
        circuit_breaker: CircuitBreaker | None = None,
        sync_interval_seconds: float = 5.0,
        stale_threshold_seconds: float = 10.0,
        max_positions: int = 20,
        min_buying_power: Decimal = Decimal("1000"),
        on_discrepancy: Callable[[Discrepancy], None] | None = None,
        on_sync_complete: Callable[[SyncResult], None] | None = None,
    ) -> None:
        """
        Initialize broker sync manager.
        
        Args:
            broker: Broker adapter for API calls
            circuit_breaker: Optional circuit breaker for API protection
            sync_interval_seconds: How often to sync (default: 5s)
            stale_threshold_seconds: When to consider data stale (default: 10s)
            max_positions: Maximum allowed concurrent positions
            min_buying_power: Minimum buying power to allow trading
            on_discrepancy: Callback when discrepancy detected
            on_sync_complete: Callback after each sync
        """
        self._broker = broker
        self._circuit_breaker = circuit_breaker
        self._sync_interval = sync_interval_seconds
        self._stale_threshold = stale_threshold_seconds
        self._max_positions = max_positions
        self._min_buying_power = min_buying_power
        self._on_discrepancy = on_discrepancy
        self._on_sync_complete = on_sync_complete
        
        # State
        self._account = CachedAccountState()
        self._positions: dict[str, CachedPosition] = {}
        self._local_positions: dict[str, Decimal] = {}  # Our tracked quantities
        self._discrepancies: list[Discrepancy] = []
        self._sync_history: list[SyncResult] = []
        
        # Background sync
        self._running = False
        self._sync_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        
        # Statistics
        self._total_syncs = 0
        self._failed_syncs = 0
        self._total_discrepancies = 0
        
    async def start(self) -> SyncResult:
        """
        Start background sync and perform initial sync.
        
        Returns:
            Result of initial sync
        """
        if self._running:
            return SyncResult(
                success=True,
                status=self._account.sync_status,
            )
            
        # Perform initial sync
        result = await self.sync()
        
        # Start background sync
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        
        logger.info(
            f"BrokerSyncManager started: {result.positions_synced} positions, "
            f"buying_power=${self._account.buying_power:,.2f}"
        )
        
        return result
        
    async def stop(self) -> None:
        """Stop background sync."""
        self._running = False
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
            
        logger.info("BrokerSyncManager stopped")
        
    async def sync(self) -> SyncResult:
        """
        Perform full broker state synchronization.
        
        Returns:
            Sync result with status and any discrepancies
        """
        start_time = datetime.utcnow()
        
        async with self._lock:
            self._account.sync_status = SyncStatus.SYNCING
            
            try:
                # Check circuit breaker
                if self._circuit_breaker and self._circuit_breaker.is_open:
                    return SyncResult(
                        success=False,
                        status=SyncStatus.ERROR,
                        error="Circuit breaker is open",
                    )
                    
                # Sync account
                account_synced = await self._sync_account()
                
                # Sync positions
                positions_synced = await self._sync_positions()
                
                # Detect discrepancies
                new_discrepancies = self._detect_discrepancies()
                
                self._account.sync_status = SyncStatus.SYNCED
                self._total_syncs += 1
                
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                result = SyncResult(
                    success=True,
                    status=SyncStatus.SYNCED,
                    positions_synced=positions_synced,
                    account_synced=account_synced,
                    discrepancies=new_discrepancies,
                    duration_ms=duration,
                )
                
                # Trigger callbacks
                for discrepancy in new_discrepancies:
                    self._total_discrepancies += 1
                    if self._on_discrepancy:
                        try:
                            self._on_discrepancy(discrepancy)
                        except Exception as e:
                            logger.error(f"Discrepancy callback error: {e}")
                            
                if self._on_sync_complete:
                    try:
                        self._on_sync_complete(result)
                    except Exception as e:
                        logger.error(f"Sync complete callback error: {e}")
                        
                # Keep history (last 100)
                self._sync_history.append(result)
                if len(self._sync_history) > 100:
                    self._sync_history.pop(0)
                    
                return result
                
            except Exception as e:
                self._failed_syncs += 1
                self._account.sync_status = SyncStatus.ERROR
                
                # Record failure in circuit breaker
                if self._circuit_breaker:
                    await self._circuit_breaker.record_account_failure(e)
                    
                logger.error(f"Broker sync failed: {e}")
                
                return SyncResult(
                    success=False,
                    status=SyncStatus.ERROR,
                    error=str(e),
                    duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                )
                
    async def _sync_account(self) -> bool:
        """Sync account state from broker."""
        account = await self._broker.get_account()
        
        self._account = CachedAccountState(
            account_id=account.account_id,
            equity=Decimal(str(account.equity)),
            cash=Decimal(str(account.cash)),
            buying_power=Decimal(str(account.buying_power)),
            portfolio_value=Decimal(str(account.portfolio_value)),
            is_paper=account.is_paper,
            last_sync=datetime.utcnow(),
            sync_status=SyncStatus.SYNCED,
        )
        
        return True
        
    async def _sync_positions(self) -> int:
        """Sync positions from broker."""
        broker_positions = await self._broker.get_positions()
        
        # Update cached positions
        old_symbols = set(self._positions.keys())
        new_symbols = set()
        
        for pos in broker_positions:
            symbol = pos.symbol
            new_symbols.add(symbol)
            
            self._positions[symbol] = CachedPosition(
                symbol=symbol,
                quantity=Decimal(str(pos.quantity)),
                side=pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                avg_entry_price=Decimal(str(pos.avg_entry_price)),
                market_value=Decimal(str(pos.market_value)),
                unrealized_pnl=Decimal(str(pos.unrealized_pnl)),
                last_sync=datetime.utcnow(),
                local_quantity=self._local_positions.get(symbol, Decimal("0")),
            )
            
        # Remove positions that no longer exist
        for symbol in old_symbols - new_symbols:
            del self._positions[symbol]
            
        return len(broker_positions)
        
    def _detect_discrepancies(self) -> list[Discrepancy]:
        """Detect discrepancies between local and broker state."""
        discrepancies: list[Discrepancy] = []
        
        broker_symbols = set(self._positions.keys())
        local_symbols = set(self._local_positions.keys())
        
        # Positions broker has that we don't track
        for symbol in broker_symbols - local_symbols:
            if self._positions[symbol].quantity != Decimal("0"):
                discrepancies.append(Discrepancy(
                    symbol=symbol,
                    type=DiscrepancyType.MISSING_LOCAL,
                    local_value=None,
                    broker_value=float(self._positions[symbol].quantity),
                ))
                
        # Positions we track that broker doesn't have
        for symbol in local_symbols - broker_symbols:
            if self._local_positions[symbol] != Decimal("0"):
                discrepancies.append(Discrepancy(
                    symbol=symbol,
                    type=DiscrepancyType.MISSING_BROKER,
                    local_value=float(self._local_positions[symbol]),
                    broker_value=None,
                ))
                
        # Quantity mismatches
        for symbol in broker_symbols & local_symbols:
            broker_qty = self._positions[symbol].quantity
            local_qty = self._local_positions[symbol]
            
            if broker_qty != local_qty:
                discrepancies.append(Discrepancy(
                    symbol=symbol,
                    type=DiscrepancyType.QUANTITY_MISMATCH,
                    local_value=float(local_qty),
                    broker_value=float(broker_qty),
                ))
                
        # Store for later reference
        self._discrepancies = discrepancies
        
        return discrepancies
        
    async def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        estimated_cost: Decimal | None = None,
    ) -> PreTradeValidation:
        """
        Validate order before submission.
        
        Args:
            symbol: Symbol to trade
            side: "buy" or "sell"
            quantity: Order quantity
            estimated_cost: Estimated order cost (for limit orders)
            
        Returns:
            PreTradeValidation result
        """
        # Check for stale data
        if self._account.is_stale(self._stale_threshold):
            # Try to refresh
            result = await self.sync()
            if not result.success:
                return PreTradeValidation(
                    result=ValidationResult.STALE_ACCOUNT_DATA,
                    approved=False,
                    reason=f"Account data is stale and sync failed: {result.error}",
                    details={"last_sync_age": self._account.time_since_sync()},
                )
                
        # Check circuit breaker
        if self._circuit_breaker and self._circuit_breaker.is_open:
            return PreTradeValidation(
                result=ValidationResult.CIRCUIT_BREAKER_OPEN,
                approved=False,
                reason="Circuit breaker is open due to API failures",
                details={"circuit_state": "open"},
            )
            
        # Check minimum buying power
        if self._account.buying_power < self._min_buying_power:
            return PreTradeValidation(
                result=ValidationResult.INSUFFICIENT_BUYING_POWER,
                approved=False,
                reason=f"Buying power ${self._account.buying_power:,.2f} below minimum ${self._min_buying_power:,.2f}",
                details={
                    "buying_power": float(self._account.buying_power),
                    "minimum": float(self._min_buying_power),
                },
            )
            
        # Check position limit
        current_positions = len([p for p in self._positions.values() if p.quantity != Decimal("0")])
        is_new_position = symbol not in self._positions or self._positions[symbol].quantity == Decimal("0")
        
        if side == "buy" and is_new_position and current_positions >= self._max_positions:
            return PreTradeValidation(
                result=ValidationResult.POSITION_LIMIT_EXCEEDED,
                approved=False,
                reason=f"Position limit of {self._max_positions} would be exceeded",
                details={
                    "current_positions": current_positions,
                    "max_positions": self._max_positions,
                },
            )
            
        # Estimate order cost for buy orders
        if side == "buy" and estimated_cost:
            if estimated_cost > self._account.buying_power:
                return PreTradeValidation(
                    result=ValidationResult.INSUFFICIENT_BUYING_POWER,
                    approved=False,
                    reason=f"Order cost ${estimated_cost:,.2f} exceeds buying power ${self._account.buying_power:,.2f}",
                    details={
                        "estimated_cost": float(estimated_cost),
                        "buying_power": float(self._account.buying_power),
                        "shortfall": float(estimated_cost - self._account.buying_power),
                    },
                )
                
        # All checks passed
        return PreTradeValidation(
            result=ValidationResult.APPROVED,
            approved=True,
            reason="All pre-trade checks passed",
            details={
                "buying_power": float(self._account.buying_power),
                "current_positions": current_positions,
                "account_age_seconds": self._account.time_since_sync(),
            },
        )
        
    def update_local_position(self, symbol: str, quantity_delta: Decimal) -> None:
        """
        Update local position tracking after order fill.
        
        Args:
            symbol: Symbol traded
            quantity_delta: Change in quantity (+buy, -sell)
        """
        current = self._local_positions.get(symbol, Decimal("0"))
        self._local_positions[symbol] = current + quantity_delta
        
        if self._local_positions[symbol] == Decimal("0"):
            del self._local_positions[symbol]
            
        logger.debug(f"Updated local position: {symbol} = {self._local_positions.get(symbol, 0)}")
        
    def resolve_discrepancy(
        self,
        symbol: str,
        action: str = "accept_broker",
    ) -> None:
        """
        Resolve a detected discrepancy.
        
        Args:
            symbol: Symbol with discrepancy
            action: Resolution action ("accept_broker" or "accept_local")
        """
        if action == "accept_broker":
            # Update local to match broker
            if symbol in self._positions:
                self._local_positions[symbol] = self._positions[symbol].quantity
            else:
                self._local_positions.pop(symbol, None)
        # else keep local (will show as discrepancy until broker syncs)
        
        # Mark as resolved
        for d in self._discrepancies:
            if d.symbol == symbol and not d.resolved:
                d.resolved = True
                d.resolution_action = action
                
        logger.info(f"Resolved discrepancy for {symbol}: {action}")
        
    async def _sync_loop(self) -> None:
        """Background sync loop."""
        while self._running:
            await asyncio.sleep(self._sync_interval)
            
            if self._running:
                try:
                    await self.sync()
                except Exception as e:
                    logger.error(f"Background sync error: {e}")
                    
    @property
    def account(self) -> CachedAccountState:
        """Get cached account state."""
        return self._account
        
    @property
    def positions(self) -> dict[str, CachedPosition]:
        """Get cached positions."""
        return self._positions.copy()
        
    @property
    def discrepancies(self) -> list[Discrepancy]:
        """Get unresolved discrepancies."""
        return [d for d in self._discrepancies if not d.resolved]
        
    def get_stats(self) -> dict[str, Any]:
        """Get sync statistics."""
        return {
            "total_syncs": self._total_syncs,
            "failed_syncs": self._failed_syncs,
            "success_rate": (
                (self._total_syncs - self._failed_syncs) / self._total_syncs * 100
                if self._total_syncs > 0 else 0
            ),
            "total_discrepancies": self._total_discrepancies,
            "current_discrepancies": len(self.discrepancies),
            "position_count": len(self._positions),
            "buying_power": float(self._account.buying_power),
            "sync_status": self._account.sync_status.name,
            "last_sync_age_seconds": self._account.time_since_sync(),
        }
