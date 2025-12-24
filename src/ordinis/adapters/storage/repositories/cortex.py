"""
Cortex output persistence models.

Pydantic models for storing CortexOutput and StrategyHypothesis in SQLite.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from ordinis.adapters.storage.database import DatabaseManager


class CortexOutputRow(BaseModel):
    """Database row for cortex_outputs table."""

    id: int | None = None
    output_id: str
    output_type: str
    content: str  # JSON serialized
    confidence: float
    reasoning: str
    requires_validation: bool = True
    model_used: str | None = None
    model_version: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace_id: str | None = None
    session_id: str | None = None
    metadata: str | None = None  # JSON serialized
    created_at: str
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "CortexOutputRow":
        """Create from database row tuple."""
        return cls(
            id=row[0],
            output_id=row[1],
            output_type=row[2],
            content=row[3],
            confidence=row[4],
            reasoning=row[5],
            requires_validation=bool(row[6]),
            model_used=row[7],
            model_version=row[8],
            prompt_tokens=row[9],
            completion_tokens=row[10],
            trace_id=row[11],
            session_id=row[12],
            metadata=row[13],
            created_at=row[14],
            updated_at=row[15],
        )

    def to_insert_tuple(self) -> tuple[Any, ...]:
        """Convert to tuple for INSERT."""
        return (
            self.output_id,
            self.output_type,
            self.content,
            self.confidence,
            self.reasoning,
            int(self.requires_validation),
            self.model_used,
            self.model_version,
            self.prompt_tokens,
            self.completion_tokens,
            self.trace_id,
            self.session_id,
            self.metadata,
            self.created_at,
        )

    def get_content_dict(self) -> dict[str, Any]:
        """Parse content JSON to dict."""
        return json.loads(self.content) if self.content else {}

    def get_metadata_dict(self) -> dict[str, Any]:
        """Parse metadata JSON to dict."""
        return json.loads(self.metadata) if self.metadata else {}


class StrategyHypothesisRow(BaseModel):
    """Database row for strategy_hypotheses table."""

    id: int | None = None
    hypothesis_id: str
    name: str
    description: str
    rationale: str
    instrument_class: str
    time_horizon: str
    strategy_type: str
    parameters: str  # JSON serialized
    entry_conditions: str  # JSON array
    exit_conditions: str  # JSON array
    max_position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    expected_sharpe: float
    expected_win_rate: float
    confidence: float
    validation_status: str = "pending"  # pending, validated, rejected
    validated_at: str | None = None
    validated_by: str | None = None
    trace_id: str | None = None
    session_id: str | None = None
    metadata: str | None = None  # JSON
    created_at: str
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "StrategyHypothesisRow":
        """Create from database row tuple."""
        return cls(
            id=row[0],
            hypothesis_id=row[1],
            name=row[2],
            description=row[3],
            rationale=row[4],
            instrument_class=row[5],
            time_horizon=row[6],
            strategy_type=row[7],
            parameters=row[8],
            entry_conditions=row[9],
            exit_conditions=row[10],
            max_position_size_pct=row[11],
            stop_loss_pct=row[12],
            take_profit_pct=row[13],
            expected_sharpe=row[14],
            expected_win_rate=row[15],
            confidence=row[16],
            validation_status=row[17],
            validated_at=row[18],
            validated_by=row[19],
            trace_id=row[20],
            session_id=row[21],
            metadata=row[22],
            created_at=row[23],
            updated_at=row[24],
        )

    def to_insert_tuple(self) -> tuple[Any, ...]:
        """Convert to tuple for INSERT."""
        return (
            self.hypothesis_id,
            self.name,
            self.description,
            self.rationale,
            self.instrument_class,
            self.time_horizon,
            self.strategy_type,
            self.parameters,
            self.entry_conditions,
            self.exit_conditions,
            self.max_position_size_pct,
            self.stop_loss_pct,
            self.take_profit_pct,
            self.expected_sharpe,
            self.expected_win_rate,
            self.confidence,
            self.validation_status,
            self.trace_id,
            self.session_id,
            self.metadata,
            self.created_at,
        )

    def get_parameters_dict(self) -> dict[str, Any]:
        """Parse parameters JSON to dict."""
        return json.loads(self.parameters) if self.parameters else {}

    def get_entry_conditions(self) -> list[str]:
        """Parse entry conditions JSON array."""
        return json.loads(self.entry_conditions) if self.entry_conditions else []

    def get_exit_conditions(self) -> list[str]:
        """Parse exit conditions JSON array."""
        return json.loads(self.exit_conditions) if self.exit_conditions else []


# Schema DDL for Cortex tables
CORTEX_SCHEMA_DDL = """
-- Cortex outputs table: stores all CortexOutput instances
CREATE TABLE IF NOT EXISTS cortex_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    output_id TEXT NOT NULL UNIQUE,
    output_type TEXT NOT NULL CHECK (output_type IN ('research', 'hypothesis', 'strategy_spec', 'param_proposal', 'review', 'code_analysis', 'market_insight')),
    content TEXT NOT NULL,  -- JSON
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    reasoning TEXT NOT NULL,
    requires_validation INTEGER NOT NULL DEFAULT 1,
    model_used TEXT,
    model_version TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    trace_id TEXT,
    session_id TEXT,
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cortex_outputs_output_id ON cortex_outputs(output_id);
CREATE INDEX IF NOT EXISTS idx_cortex_outputs_output_type ON cortex_outputs(output_type);
CREATE INDEX IF NOT EXISTS idx_cortex_outputs_trace_id ON cortex_outputs(trace_id);
CREATE INDEX IF NOT EXISTS idx_cortex_outputs_session_id ON cortex_outputs(session_id);
CREATE INDEX IF NOT EXISTS idx_cortex_outputs_created_at ON cortex_outputs(created_at);
CREATE INDEX IF NOT EXISTS idx_cortex_outputs_model_used ON cortex_outputs(model_used);

-- Strategy hypotheses table: stores StrategyHypothesis instances
CREATE TABLE IF NOT EXISTS strategy_hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    rationale TEXT NOT NULL,
    instrument_class TEXT NOT NULL,
    time_horizon TEXT NOT NULL,
    strategy_type TEXT NOT NULL,
    parameters TEXT NOT NULL,  -- JSON
    entry_conditions TEXT NOT NULL,  -- JSON array
    exit_conditions TEXT NOT NULL,  -- JSON array
    max_position_size_pct REAL NOT NULL,
    stop_loss_pct REAL NOT NULL,
    take_profit_pct REAL NOT NULL,
    expected_sharpe REAL NOT NULL,
    expected_win_rate REAL NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    validation_status TEXT NOT NULL DEFAULT 'pending' CHECK (validation_status IN ('pending', 'validated', 'rejected')),
    validated_at TEXT,
    validated_by TEXT,
    trace_id TEXT,
    session_id TEXT,
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_hypotheses_hypothesis_id ON strategy_hypotheses(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_strategy_type ON strategy_hypotheses(strategy_type);
CREATE INDEX IF NOT EXISTS idx_hypotheses_validation_status ON strategy_hypotheses(validation_status);
CREATE INDEX IF NOT EXISTS idx_hypotheses_confidence ON strategy_hypotheses(confidence);
CREATE INDEX IF NOT EXISTS idx_hypotheses_trace_id ON strategy_hypotheses(trace_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_session_id ON strategy_hypotheses(session_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_created_at ON strategy_hypotheses(created_at);
"""


class CortexRepository:
    """
    Repository for Cortex output and hypothesis persistence.

    Provides CRUD operations for:
    - CortexOutput instances (research, code analysis, reviews, etc.)
    - StrategyHypothesis instances
    """

    def __init__(self, db: "DatabaseManager"):
        """
        Initialize Cortex repository.

        Args:
            db: Database manager instance
        """
        self.db = db

    async def initialize_schema(self) -> None:
        """Create Cortex tables if they don't exist."""
        # Execute DDL statements one at a time
        for statement in CORTEX_SCHEMA_DDL.split(";"):
            statement = statement.strip()
            if statement:
                await self.db.execute(statement)

    # ==================== OUTPUT OPERATIONS ====================

    async def save_output(self, output: CortexOutputRow) -> int | None:
        """
        Save a CortexOutput to the database.

        Args:
            output: CortexOutputRow to save

        Returns:
            Row ID of inserted record or None
        """
        cursor = await self.db.execute(
            """
            INSERT INTO cortex_outputs (
                output_id, output_type, content, confidence, reasoning,
                requires_validation, model_used, model_version,
                prompt_tokens, completion_tokens, trace_id, session_id,
                metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            output.to_insert_tuple(),
        )
        return cursor.lastrowid

    async def get_output_by_id(self, output_id: str) -> CortexOutputRow | None:
        """Get output by output_id."""
        row = await self.db.fetch_one(
            "SELECT * FROM cortex_outputs WHERE output_id = ?",
            (output_id,),
        )
        return CortexOutputRow.from_row(row) if row else None

    async def get_outputs_by_type(
        self, output_type: str, limit: int = 100
    ) -> list[CortexOutputRow]:
        """Get outputs by type."""
        rows = await self.db.fetch_all(
            """
            SELECT * FROM cortex_outputs
            WHERE output_type = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (output_type, limit),
        )
        return [CortexOutputRow.from_row(row) for row in rows]

    async def get_outputs_by_trace(self, trace_id: str) -> list[CortexOutputRow]:
        """Get all outputs for a trace."""
        rows = await self.db.fetch_all(
            "SELECT * FROM cortex_outputs WHERE trace_id = ? ORDER BY created_at",
            (trace_id,),
        )
        return [CortexOutputRow.from_row(row) for row in rows]

    async def get_recent_outputs(self, limit: int = 50) -> list[CortexOutputRow]:
        """Get most recent outputs."""
        rows = await self.db.fetch_all(
            "SELECT * FROM cortex_outputs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [CortexOutputRow.from_row(row) for row in rows]

    async def delete_outputs_older_than(self, days: int) -> int:
        """
        Delete outputs older than specified days (TTL cleanup).

        Args:
            days: Number of days to retain

        Returns:
            Number of deleted rows
        """
        cursor = await self.db.execute(
            """
            DELETE FROM cortex_outputs
            WHERE created_at < datetime('now', '-' || ? || ' days')
            """,
            (days,),
        )
        return cursor.rowcount

    # ==================== HYPOTHESIS OPERATIONS ====================

    async def save_hypothesis(self, hypothesis: StrategyHypothesisRow) -> int | None:
        """
        Save a StrategyHypothesis to the database.

        Args:
            hypothesis: StrategyHypothesisRow to save

        Returns:
            Row ID of inserted record or None
        """
        cursor = await self.db.execute(
            """
            INSERT INTO strategy_hypotheses (
                hypothesis_id, name, description, rationale, instrument_class,
                time_horizon, strategy_type, parameters, entry_conditions,
                exit_conditions, max_position_size_pct, stop_loss_pct,
                take_profit_pct, expected_sharpe, expected_win_rate,
                confidence, validation_status, trace_id, session_id,
                metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            hypothesis.to_insert_tuple(),
        )
        return cursor.lastrowid

    async def get_hypothesis_by_id(
        self, hypothesis_id: str
    ) -> StrategyHypothesisRow | None:
        """Get hypothesis by hypothesis_id."""
        row = await self.db.fetch_one(
            "SELECT * FROM strategy_hypotheses WHERE hypothesis_id = ?",
            (hypothesis_id,),
        )
        return StrategyHypothesisRow.from_row(row) if row else None

    async def get_hypotheses_by_status(
        self, status: str, limit: int = 100
    ) -> list[StrategyHypothesisRow]:
        """Get hypotheses by validation status."""
        rows = await self.db.fetch_all(
            """
            SELECT * FROM strategy_hypotheses
            WHERE validation_status = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (status, limit),
        )
        return [StrategyHypothesisRow.from_row(row) for row in rows]

    async def get_hypotheses_by_type(
        self, strategy_type: str, limit: int = 100
    ) -> list[StrategyHypothesisRow]:
        """Get hypotheses by strategy type."""
        rows = await self.db.fetch_all(
            """
            SELECT * FROM strategy_hypotheses
            WHERE strategy_type = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (strategy_type, limit),
        )
        return [StrategyHypothesisRow.from_row(row) for row in rows]

    async def get_hypotheses_above_confidence(
        self, min_confidence: float, limit: int = 100
    ) -> list[StrategyHypothesisRow]:
        """Get hypotheses above minimum confidence threshold."""
        rows = await self.db.fetch_all(
            """
            SELECT * FROM strategy_hypotheses
            WHERE confidence >= ?
            ORDER BY confidence DESC
            LIMIT ?
            """,
            (min_confidence, limit),
        )
        return [StrategyHypothesisRow.from_row(row) for row in rows]

    async def update_validation_status(
        self,
        hypothesis_id: str,
        status: str,
        validated_by: str | None = None,
    ) -> bool:
        """
        Update hypothesis validation status.

        Args:
            hypothesis_id: Hypothesis ID
            status: New status (pending, validated, rejected)
            validated_by: Who/what validated it

        Returns:
            True if updated, False if not found
        """
        validated_at = (
            datetime.now().astimezone().isoformat() if status != "pending" else None
        )
        cursor = await self.db.execute(
            """
            UPDATE strategy_hypotheses
            SET validation_status = ?,
                validated_at = ?,
                validated_by = ?,
                updated_at = datetime('now')
            WHERE hypothesis_id = ?
            """,
            (status, validated_at, validated_by, hypothesis_id),
        )
        return cursor.rowcount > 0

    async def get_recent_hypotheses(
        self, limit: int = 50
    ) -> list[StrategyHypothesisRow]:
        """Get most recent hypotheses."""
        rows = await self.db.fetch_all(
            "SELECT * FROM strategy_hypotheses ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [StrategyHypothesisRow.from_row(row) for row in rows]

    async def delete_hypotheses_older_than(self, days: int) -> int:
        """
        Delete hypotheses older than specified days (TTL cleanup).

        Only deletes non-validated hypotheses.

        Args:
            days: Number of days to retain

        Returns:
            Number of deleted rows
        """
        cursor = await self.db.execute(
            """
            DELETE FROM strategy_hypotheses
            WHERE created_at < datetime('now', '-' || ? || ' days')
            AND validation_status = 'pending'
            """,
            (days,),
        )
        return cursor.rowcount

    # ==================== STATISTICS ====================

    async def get_output_stats(self) -> dict[str, Any]:
        """Get statistics about stored outputs."""
        type_counts = await self.db.fetch_all(
            """
            SELECT output_type, COUNT(*) as count
            FROM cortex_outputs
            GROUP BY output_type
            """
        )

        total_tokens = await self.db.fetch_one(
            """
            SELECT
                SUM(prompt_tokens) as total_prompt,
                SUM(completion_tokens) as total_completion
            FROM cortex_outputs
            """
        )

        return {
            "by_type": {row[0]: row[1] for row in type_counts},
            "total_prompt_tokens": total_tokens[0] if total_tokens else 0,
            "total_completion_tokens": total_tokens[1] if total_tokens else 0,
        }

    async def get_hypothesis_stats(self) -> dict[str, Any]:
        """Get statistics about stored hypotheses."""
        status_counts = await self.db.fetch_all(
            """
            SELECT validation_status, COUNT(*) as count
            FROM strategy_hypotheses
            GROUP BY validation_status
            """
        )

        type_counts = await self.db.fetch_all(
            """
            SELECT strategy_type, COUNT(*) as count
            FROM strategy_hypotheses
            GROUP BY strategy_type
            """
        )

        avg_confidence = await self.db.fetch_one(
            "SELECT AVG(confidence) FROM strategy_hypotheses"
        )

        return {
            "by_status": {row[0]: row[1] for row in status_counts},
            "by_type": {row[0]: row[1] for row in type_counts},
            "avg_confidence": avg_confidence[0] if avg_confidence else 0.0,
        }


# Import at bottom to avoid circular import
if TYPE_CHECKING:
    from ordinis.adapters.storage.database import DatabaseManager
