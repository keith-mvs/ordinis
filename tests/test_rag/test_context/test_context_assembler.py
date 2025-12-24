from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ordinis.rag.context.assembler import (
    AssembledContext,
    ContextAssembler,
    ContextChunk,
    ContextPriority,
    ContextSource,
)


class _DummyCollection:
    def query(self, *, query_embeddings: list[Any], n_results: int, include: list[str], where=None):
        # Return different docs depending on where-clause.
        if where == {"entity_type": "trade"}:
            docs = [f"trade-{i}" for i in range(n_results)]
            metas = [{"entity_type": "trade", "id": i} for i in range(n_results)]
            dists = [0.1 + i * 0.05 for i in range(n_results)]
        elif where and where.get("entity_type"):
            docs = [f"kb-{i}" for i in range(n_results)]
            metas = [{"entity_type": "kb", "title": f"Doc {i}"} for i in range(n_results)]
            dists = [0.2 + i * 0.05 for i in range(n_results)]
        else:
            docs = [f"doc-{i}" for i in range(n_results)]
            metas = [{"id": i} for i in range(n_results)]
            dists = [0.05 + i * 0.05 for i in range(n_results)]

        return {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }


class _DummyChromaClient:
    def get_text_collection(self):
        return _DummyCollection()


class _DummyEmbedder:
    def embed_texts(self, texts: list[str]):
        # Return numpy arrays with a .tolist() method.
        return [np.array([0.0, 1.0, 2.0], dtype=float) for _ in texts]


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "rag_test.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE system_state (key TEXT, value TEXT, updated_at TEXT)"
        )
        cur.execute(
            "CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, created_at TEXT)"
        )
        cur.execute(
            "CREATE TABLE session_summaries (session_id TEXT, content TEXT, summary_type TEXT, message_end_idx INTEGER)"
        )

        now = datetime.now(timezone.utc)

        cur.executemany(
            "INSERT INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
            [
                ("kill_switch", "off", now.isoformat()),
                ("circuit_breaker", "healthy", now.isoformat()),
            ],
        )

        # Insert messages out of order to validate ORDER BY created_at DESC + reverse.
        cur.executemany(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            [
                ("s1", "user", "First", (now - timedelta(minutes=2)).isoformat()),
                ("s1", "assistant", "Second", (now - timedelta(minutes=1)).isoformat()),
                ("s1", "user", "Third", now.isoformat()),
            ],
        )

        # Provide both session-level and chunk-level summaries.
        cur.executemany(
            "INSERT INTO session_summaries (session_id, content, summary_type, message_end_idx) VALUES (?, ?, ?, ?)",
            [
                ("s1", "Chunk summary", "chunk", 5),
                ("s1", "Session summary", "session", 2),
            ],
        )

        conn.commit()
    finally:
        conn.close()

    return db_path


@pytest.mark.unit
@pytest.mark.asyncio
async def test_assemble_context_happy_path_includes_sources(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path)

    assembler = ContextAssembler(
        chroma_client=_DummyChromaClient(),
        db_path=db_path,
        text_embedder=_DummyEmbedder(),
        default_max_tokens=500,
        chars_per_token=4,
    )

    assembled = await assembler.assemble_context(
        query="What about trades?",
        session_id="s1",
        max_tokens=500,
        include_recent_messages=10,
        include_trade_history=2,
        retrieve_n_docs=2,
    )

    assert isinstance(assembled, AssembledContext)
    # We should see headings for the sources we gathered.
    assert "### System State" in assembled.content
    assert "### Recent Messages" in assembled.content
    assert "### Session Summary" in assembled.content
    assert "### Retrieved Docs" in assembled.content
    assert "### Trade History" in assembled.content
    assert "### Knowledge Base" in assembled.content

    # Verify chronological message formatting.
    assert "[USER]: First" in assembled.content
    assert "[ASSISTANT]: Second" in assembled.content
    assert "[USER]: Third" in assembled.content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_assemble_context_cache_hit_skips_gather(tmp_path: Path, monkeypatch) -> None:
    db_path = _make_db(tmp_path)

    assembler = ContextAssembler(
        chroma_client=_DummyChromaClient(),
        db_path=db_path,
        text_embedder=_DummyEmbedder(),
        default_max_tokens=300,
        chars_per_token=4,
    )

    _ = await assembler.assemble_context(
        query="cache-me",
        session_id="s1",
        max_tokens=300,
        retrieve_n_docs=1,
        include_trade_history=1,
    )

    # If cache is used, these should never be called.
    async def _boom(*args, **kwargs):  # pragma: no cover
        raise AssertionError("gather should not be called on cache hit")

    monkeypatch.setattr(assembler, "_gather_system_state", _boom)
    monkeypatch.setattr(assembler, "_gather_recent_messages", _boom)
    monkeypatch.setattr(assembler, "_gather_session_summary", _boom)
    monkeypatch.setattr(assembler, "_gather_retrieved_docs", _boom)
    monkeypatch.setattr(assembler, "_gather_trade_history", _boom)
    monkeypatch.setattr(assembler, "_gather_knowledge_base", _boom)

    assembled2 = await assembler.assemble_context(
        query="cache-me",
        session_id="s1",
        max_tokens=300,
        retrieve_n_docs=1,
        include_trade_history=1,
    )

    assert assembled2.total_tokens >= 0


@pytest.mark.unit
def test_pack_chunks_respects_budget_and_truncates_critical() -> None:
    assembler = ContextAssembler(
        chroma_client=_DummyChromaClient(),
        db_path=":memory:",
        text_embedder=_DummyEmbedder(),
        default_max_tokens=200,
        chars_per_token=1,  # make token estimation deterministic in this test
    )

    critical = ContextChunk(
        source=ContextSource.SYSTEM_STATE,
        priority=ContextPriority.CRITICAL,
        content="X" * 500,
        token_count=500,
    )
    high = ContextChunk(
        source=ContextSource.RECENT_MESSAGES,
        priority=ContextPriority.HIGH,
        content="Y" * 50,
        token_count=50,
        score=0.5,
    )

    assembled = assembler._pack_chunks([high, critical], max_tokens=200)

    assert assembled.total_tokens <= 200
    # Critical chunk must be included even if truncated.
    assert ContextSource.SYSTEM_STATE in assembled.sources_used
    assert "..." in assembled.content


@pytest.mark.unit
@pytest.mark.asyncio
async def test_preload_session_context_does_not_retrieve_docs(tmp_path: Path, monkeypatch) -> None:
    db_path = _make_db(tmp_path)

    assembler = ContextAssembler(
        chroma_client=_DummyChromaClient(),
        db_path=db_path,
        text_embedder=_DummyEmbedder(),
        default_max_tokens=300,
        chars_per_token=4,
    )

    async def _boom(*args, **kwargs):  # pragma: no cover
        raise AssertionError("retrieval should not occur during preload")

    monkeypatch.setattr(assembler, "_gather_retrieved_docs", _boom)

    assembled = await assembler.preload_session_context("s1", max_tokens=300)

    assert "### System State" in assembled.content
    assert "### Recent Messages" in assembled.content
    assert "### Session Summary" in assembled.content
