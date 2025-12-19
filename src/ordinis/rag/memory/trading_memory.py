"""
ChromaDB Trading Memory Collection and Synapse Integration.

Implements production-grade trading memory system:
- Trade decision memory collection
- Signal history and outcomes
- Session continuity across restarts
- Semantic retrieval for similar market conditions
- LLM-assisted trade analysis

Step 8 of Trade Enhancement Roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Attempt ChromaDB import with fallback
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. Memory features disabled.")


class MemoryType(Enum):
    """Types of trading memories."""
    
    TRADE_DECISION = auto()  # Trade entry/exit decision
    SIGNAL = auto()  # Signal generation event
    RISK_EVENT = auto()  # Risk limit triggered
    MARKET_CONDITION = auto()  # Market regime/condition
    STRATEGY_PERFORMANCE = auto()  # Strategy performance snapshot
    ERROR = auto()  # Error/failure event
    USER_ANNOTATION = auto()  # User-provided annotation


class OutcomeType(Enum):
    """Trade outcome classifications."""
    
    PROFITABLE = auto()
    BREAKEVEN = auto()
    LOSS = auto()
    STOPPED_OUT = auto()
    TIMED_OUT = auto()
    CANCELLED = auto()
    PENDING = auto()


@dataclass
class TradeMemory:
    """Single trade decision memory."""
    
    memory_id: str
    memory_type: MemoryType
    timestamp: datetime
    
    # Context
    symbol: str
    timeframe: str
    market_regime: str
    
    # Decision details
    decision: str  # "enter_long", "enter_short", "exit", etc.
    confidence: float
    reasoning: str
    
    # Market context at decision time
    price: Decimal
    indicators: dict[str, float]
    
    # Outcome (populated after trade closes)
    outcome: OutcomeType | None = None
    pnl: Decimal | None = None
    holding_period: timedelta | None = None
    
    # Signal source
    strategy_id: str | None = None
    signal_id: str | None = None
    
    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_document(self) -> str:
        """Convert to text document for embedding."""
        parts = [
            f"Trade Decision: {self.decision}",
            f"Symbol: {self.symbol}, Timeframe: {self.timeframe}",
            f"Market Regime: {self.market_regime}",
            f"Confidence: {self.confidence:.0%}",
            f"Reasoning: {self.reasoning}",
            f"Price: {self.price}",
        ]
        
        if self.indicators:
            indicator_str = ", ".join(f"{k}={v:.2f}" for k, v in self.indicators.items())
            parts.append(f"Indicators: {indicator_str}")
            
        if self.outcome:
            parts.append(f"Outcome: {self.outcome.name}")
            if self.pnl:
                parts.append(f"P&L: {self.pnl}")
                
        return "\n".join(parts)
        
    def to_metadata(self) -> dict[str, Any]:
        """Convert to ChromaDB metadata."""
        meta = {
            "memory_type": self.memory_type.name,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "market_regime": self.market_regime,
            "decision": self.decision,
            "confidence": self.confidence,
            "price": str(self.price),
        }
        
        if self.outcome:
            meta["outcome"] = self.outcome.name
        if self.pnl:
            meta["pnl"] = float(self.pnl)
        if self.strategy_id:
            meta["strategy_id"] = self.strategy_id
        if self.tags:
            meta["tags"] = ",".join(self.tags)
            
        return meta


@dataclass
class MemoryQuery:
    """Query for retrieving memories."""
    
    # Semantic search
    query_text: str | None = None
    
    # Filters
    symbol: str | None = None
    memory_types: list[MemoryType] | None = None
    market_regime: str | None = None
    outcome: OutcomeType | None = None
    strategy_id: str | None = None
    
    # Time range
    start_time: datetime | None = None
    end_time: datetime | None = None
    
    # Result options
    limit: int = 10
    include_pending: bool = False


@dataclass
class MemorySearchResult:
    """Search result with relevance score."""
    
    memory: TradeMemory
    relevance_score: float
    match_reasons: list[str]


@dataclass
class SessionState:
    """Session state for continuity."""
    
    session_id: str
    start_time: datetime
    last_activity: datetime
    
    # Active positions
    open_positions: list[str]
    
    # Recent decisions
    recent_decisions: list[str]  # Memory IDs
    
    # Session metrics
    trades_executed: int = 0
    signals_generated: int = 0
    errors_encountered: int = 0
    
    # Context
    market_regime: str = "unknown"
    risk_status: str = "normal"
    
    metadata: dict[str, Any] = field(default_factory=dict)


class TradingMemoryStore:
    """
    ChromaDB-backed trading memory store.
    
    Provides semantic search over trading decisions and outcomes.
    
    Example:
        >>> store = TradingMemoryStore(persist_path="./data/trading_memory")
        >>> store.add_memory(trade_memory)
        >>> 
        >>> # Find similar past decisions
        >>> results = store.search(MemoryQuery(
        ...     query_text="bullish breakout AAPL high volume",
        ...     limit=5,
        ... ))
    """
    
    COLLECTION_NAME = "trading_memories"
    SESSION_COLLECTION = "sessions"
    
    def __init__(
        self,
        persist_path: str | Path | None = None,
        embedding_model: str = "default",
    ) -> None:
        """Initialize memory store."""
        self.persist_path = Path(persist_path) if persist_path else None
        self.embedding_model = embedding_model
        self._client: chromadb.Client | None = None
        self._collection: Any = None
        self._session_collection: Any = None
        self._current_session: SessionState | None = None
        
        if CHROMADB_AVAILABLE:
            self._initialize_client()
            
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client."""
        if self.persist_path:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            settings = Settings(
                persist_directory=str(self.persist_path),
                anonymized_telemetry=False,
            )
            self._client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=settings,
            )
        else:
            self._client = chromadb.Client()
            
        # Get or create collections
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        
        self._session_collection = self._client.get_or_create_collection(
            name=self.SESSION_COLLECTION,
        )
        
        logger.info(
            f"Initialized trading memory store "
            f"({self._collection.count()} memories)"
        )
        
    def add_memory(self, memory: TradeMemory) -> str:
        """
        Add a trading memory.
        
        Returns:
            Memory ID
        """
        if not CHROMADB_AVAILABLE or not self._collection:
            logger.warning("ChromaDB not available")
            return memory.memory_id
            
        document = memory.to_document()
        metadata = memory.to_metadata()
        
        self._collection.add(
            ids=[memory.memory_id],
            documents=[document],
            metadatas=[metadata],
        )
        
        # Update session
        if self._current_session:
            self._current_session.recent_decisions.append(memory.memory_id)
            self._current_session.last_activity = datetime.utcnow()
            
            if memory.memory_type == MemoryType.TRADE_DECISION:
                self._current_session.trades_executed += 1
            elif memory.memory_type == MemoryType.SIGNAL:
                self._current_session.signals_generated += 1
            elif memory.memory_type == MemoryType.ERROR:
                self._current_session.errors_encountered += 1
                
        logger.debug(f"Added memory {memory.memory_id}: {memory.decision}")
        return memory.memory_id
        
    def update_memory_outcome(
        self,
        memory_id: str,
        outcome: OutcomeType,
        pnl: Decimal | None = None,
        holding_period: timedelta | None = None,
    ) -> None:
        """Update a memory with trade outcome."""
        if not CHROMADB_AVAILABLE or not self._collection:
            return
            
        # Get existing memory
        result = self._collection.get(ids=[memory_id], include=["metadatas"])
        
        if not result["ids"]:
            logger.warning(f"Memory {memory_id} not found")
            return
            
        metadata = result["metadatas"][0]
        metadata["outcome"] = outcome.name
        if pnl:
            metadata["pnl"] = float(pnl)
        if holding_period:
            metadata["holding_period_seconds"] = holding_period.total_seconds()
            
        self._collection.update(
            ids=[memory_id],
            metadatas=[metadata],
        )
        
        logger.debug(f"Updated memory {memory_id} outcome: {outcome.name}")
        
    def search(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """
        Search trading memories.
        
        Args:
            query: Search query with filters
            
        Returns:
            List of matching memories with relevance scores
        """
        if not CHROMADB_AVAILABLE or not self._collection:
            return []
            
        # Build where clause
        where = self._build_where_clause(query)
        
        if query.query_text:
            # Semantic search
            results = self._collection.query(
                query_texts=[query.query_text],
                n_results=query.limit,
                where=where if where else None,
                include=["documents", "metadatas", "distances"],
            )
        else:
            # Filter-only search
            results = self._collection.get(
                where=where if where else None,
                limit=query.limit,
                include=["documents", "metadatas"],
            )
            
        return self._parse_results(results)
        
    def _build_where_clause(self, query: MemoryQuery) -> dict[str, Any] | None:
        """Build ChromaDB where clause from query."""
        conditions = []
        
        if query.symbol:
            conditions.append({"symbol": query.symbol})
            
        if query.memory_types:
            type_names = [t.name for t in query.memory_types]
            if len(type_names) == 1:
                conditions.append({"memory_type": type_names[0]})
            else:
                conditions.append({"memory_type": {"$in": type_names}})
                
        if query.market_regime:
            conditions.append({"market_regime": query.market_regime})
            
        if query.outcome:
            conditions.append({"outcome": query.outcome.name})
            
        if query.strategy_id:
            conditions.append({"strategy_id": query.strategy_id})
            
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
            
    def _parse_results(self, results: dict) -> list[MemorySearchResult]:
        """Parse ChromaDB results into MemorySearchResults."""
        parsed = []
        
        ids = results.get("ids", [[]])[0] if "ids" in results else results.get("ids", [])
        documents = results.get("documents", [[]])[0] if "documents" in results else results.get("documents", [])
        metadatas = results.get("metadatas", [[]])[0] if "metadatas" in results else results.get("metadatas", [])
        distances = results.get("distances", [[]])[0] if "distances" in results else [1.0] * len(ids)
        
        for i, memory_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            
            # Reconstruct memory from metadata
            memory = TradeMemory(
                memory_id=memory_id,
                memory_type=MemoryType[metadata.get("memory_type", "TRADE_DECISION")],
                timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.utcnow().isoformat())),
                symbol=metadata.get("symbol", ""),
                timeframe=metadata.get("timeframe", ""),
                market_regime=metadata.get("market_regime", ""),
                decision=metadata.get("decision", ""),
                confidence=metadata.get("confidence", 0.0),
                reasoning=documents[i] if i < len(documents) else "",
                price=Decimal(str(metadata.get("price", "0"))),
                indicators={},
                outcome=OutcomeType[metadata["outcome"]] if "outcome" in metadata else None,
                pnl=Decimal(str(metadata["pnl"])) if "pnl" in metadata else None,
                strategy_id=metadata.get("strategy_id"),
            )
            
            # Relevance score (1 - distance for cosine)
            relevance = 1.0 - distance
            
            parsed.append(MemorySearchResult(
                memory=memory,
                relevance_score=relevance,
                match_reasons=[f"Semantic similarity: {relevance:.0%}"],
            ))
            
        return parsed
        
    def get_similar_trades(
        self,
        symbol: str,
        market_regime: str,
        indicators: dict[str, float],
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        """
        Find similar historical trades based on market conditions.
        
        This is used to inform current trade decisions.
        """
        # Build semantic query from current conditions
        indicator_str = ", ".join(f"{k}={v:.2f}" for k, v in indicators.items())
        query_text = f"Trade {symbol} in {market_regime} market with {indicator_str}"
        
        return self.search(MemoryQuery(
            query_text=query_text,
            symbol=symbol,
            market_regime=market_regime,
            limit=limit,
        ))
        
    def get_strategy_performance_history(
        self,
        strategy_id: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get performance history for a strategy."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        results = self.search(MemoryQuery(
            strategy_id=strategy_id,
            memory_types=[MemoryType.TRADE_DECISION],
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        ))
        
        outcomes = {"profitable": 0, "loss": 0, "breakeven": 0, "other": 0}
        total_pnl = Decimal("0")
        
        for r in results:
            if r.memory.outcome == OutcomeType.PROFITABLE:
                outcomes["profitable"] += 1
            elif r.memory.outcome == OutcomeType.LOSS:
                outcomes["loss"] += 1
            elif r.memory.outcome == OutcomeType.BREAKEVEN:
                outcomes["breakeven"] += 1
            else:
                outcomes["other"] += 1
                
            if r.memory.pnl:
                total_pnl += r.memory.pnl
                
        total_trades = sum(outcomes.values())
        
        return {
            "strategy_id": strategy_id,
            "period_days": days,
            "total_trades": total_trades,
            "win_rate": outcomes["profitable"] / total_trades if total_trades else 0,
            "total_pnl": float(total_pnl),
            "outcomes": outcomes,
        }


class SessionManager:
    """
    Manages trading session state for continuity.
    
    Persists session state to ChromaDB for recovery after restarts.
    """
    
    def __init__(self, memory_store: TradingMemoryStore) -> None:
        """Initialize session manager."""
        self.memory_store = memory_store
        self._session: SessionState | None = None
        
    def start_session(
        self,
        session_id: str | None = None,
        resume_latest: bool = True,
    ) -> SessionState:
        """
        Start a new session or resume existing.
        
        Args:
            session_id: Explicit session ID (auto-generated if None)
            resume_latest: Whether to resume the latest session
            
        Returns:
            Active SessionState
        """
        if resume_latest and session_id is None:
            # Try to resume latest session
            latest = self._get_latest_session()
            if latest:
                logger.info(f"Resuming session {latest.session_id}")
                self._session = latest
                self.memory_store._current_session = self._session
                return self._session
                
        # Create new session
        if session_id is None:
            session_id = self._generate_session_id()
            
        self._session = SessionState(
            session_id=session_id,
            start_time=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            open_positions=[],
            recent_decisions=[],
        )
        
        self.memory_store._current_session = self._session
        self._persist_session()
        
        logger.info(f"Started new session {session_id}")
        return self._session
        
    def get_current_session(self) -> SessionState | None:
        """Get current active session."""
        return self._session
        
    def update_session(
        self,
        open_positions: list[str] | None = None,
        market_regime: str | None = None,
        risk_status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update session state."""
        if not self._session:
            return
            
        self._session.last_activity = datetime.utcnow()
        
        if open_positions is not None:
            self._session.open_positions = open_positions
        if market_regime is not None:
            self._session.market_regime = market_regime
        if risk_status is not None:
            self._session.risk_status = risk_status
        if metadata is not None:
            self._session.metadata.update(metadata)
            
        self._persist_session()
        
    def end_session(self) -> SessionState | None:
        """End the current session."""
        if not self._session:
            return None
            
        session = self._session
        self._persist_session()
        
        logger.info(
            f"Ended session {session.session_id}: "
            f"{session.trades_executed} trades, "
            f"{session.signals_generated} signals, "
            f"{session.errors_encountered} errors"
        )
        
        self._session = None
        self.memory_store._current_session = None
        
        return session
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.utcnow().isoformat()
        hash_input = f"{timestamp}-{id(self)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
    def _persist_session(self) -> None:
        """Persist session state to ChromaDB."""
        if not CHROMADB_AVAILABLE or not self._session:
            return
            
        if not self.memory_store._session_collection:
            return
            
        session = self._session
        
        document = json.dumps({
            "session_id": session.session_id,
            "start_time": session.start_time.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "open_positions": session.open_positions,
            "recent_decisions": session.recent_decisions[-100:],  # Last 100
            "trades_executed": session.trades_executed,
            "signals_generated": session.signals_generated,
            "errors_encountered": session.errors_encountered,
            "market_regime": session.market_regime,
            "risk_status": session.risk_status,
            "metadata": session.metadata,
        })
        
        metadata = {
            "start_time": session.start_time.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "trades_executed": session.trades_executed,
        }
        
        self.memory_store._session_collection.upsert(
            ids=[session.session_id],
            documents=[document],
            metadatas=[metadata],
        )
        
    def _get_latest_session(self) -> SessionState | None:
        """Get the latest session from store."""
        if not CHROMADB_AVAILABLE:
            return None
            
        if not self.memory_store._session_collection:
            return None
            
        try:
            results = self.memory_store._session_collection.get(
                limit=1,
                include=["documents", "metadatas"],
            )
            
            if not results["ids"]:
                return None
                
            document = json.loads(results["documents"][0])
            
            return SessionState(
                session_id=document["session_id"],
                start_time=datetime.fromisoformat(document["start_time"]),
                last_activity=datetime.fromisoformat(document["last_activity"]),
                open_positions=document.get("open_positions", []),
                recent_decisions=document.get("recent_decisions", []),
                trades_executed=document.get("trades_executed", 0),
                signals_generated=document.get("signals_generated", 0),
                errors_encountered=document.get("errors_encountered", 0),
                market_regime=document.get("market_regime", "unknown"),
                risk_status=document.get("risk_status", "normal"),
                metadata=document.get("metadata", {}),
            )
        except Exception as e:
            logger.debug(f"Could not load session: {e}")
            return None


class SynapseMemoryIntegration:
    """
    Integration layer between TradingMemoryStore and Synapse RAG engine.
    
    Enables LLM-assisted trade analysis using memory context.
    """
    
    def __init__(
        self,
        memory_store: TradingMemoryStore,
    ) -> None:
        """Initialize integration."""
        self.memory_store = memory_store
        
    async def get_trade_context(
        self,
        symbol: str,
        decision: str,
        indicators: dict[str, float],
        limit: int = 5,
    ) -> str:
        """
        Get relevant historical context for a trade decision.
        
        Returns formatted context for LLM prompt.
        """
        # Find similar historical trades
        results = self.memory_store.get_similar_trades(
            symbol=symbol,
            market_regime=indicators.get("regime", "unknown"),
            indicators=indicators,
            limit=limit,
        )
        
        if not results:
            return "No similar historical trades found."
            
        context_parts = ["Historical similar trades:"]
        
        for i, r in enumerate(results, 1):
            m = r.memory
            outcome_str = m.outcome.name if m.outcome else "PENDING"
            pnl_str = f" (P&L: {m.pnl})" if m.pnl else ""
            
            context_parts.append(
                f"{i}. {m.symbol} {m.decision} at {m.price} "
                f"({m.market_regime}) → {outcome_str}{pnl_str}"
            )
            
        # Add summary statistics
        profitable = sum(1 for r in results if r.memory.outcome == OutcomeType.PROFITABLE)
        win_rate = profitable / len(results) if results else 0
        
        context_parts.append(f"\nHistorical win rate for similar conditions: {win_rate:.0%}")
        
        return "\n".join(context_parts)
        
    async def analyze_trade_decision(
        self,
        memory: TradeMemory,
    ) -> dict[str, Any]:
        """
        Analyze a trade decision using historical context.
        
        Returns analysis with recommendations.
        """
        # Get historical context
        context = await self.get_trade_context(
            symbol=memory.symbol,
            decision=memory.decision,
            indicators=memory.indicators,
        )
        
        # Calculate confidence adjustment based on history
        results = self.memory_store.get_similar_trades(
            symbol=memory.symbol,
            market_regime=memory.market_regime,
            indicators=memory.indicators,
            limit=10,
        )
        
        if results:
            profitable = sum(1 for r in results if r.memory.outcome == OutcomeType.PROFITABLE)
            historical_win_rate = profitable / len(results)
            
            # Adjust confidence based on history
            confidence_adjustment = (historical_win_rate - 0.5) * 0.2
            adjusted_confidence = min(1.0, max(0.0, memory.confidence + confidence_adjustment))
        else:
            historical_win_rate = 0.5
            adjusted_confidence = memory.confidence
            
        return {
            "original_confidence": memory.confidence,
            "adjusted_confidence": adjusted_confidence,
            "historical_win_rate": historical_win_rate,
            "similar_trades_count": len(results),
            "historical_context": context,
            "recommendation": "proceed" if adjusted_confidence > 0.6 else "review",
        }
