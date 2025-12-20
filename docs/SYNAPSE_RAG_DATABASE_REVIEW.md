# Synapse RAG Database Architecture Review

**Review Date:** December 19, 2025  
**Scope:** Synapse/RAGEngine architecture, SQLite persistence, ChromaDB vector storage, ingestion pipelines, cross-engine workflows  
**Reviewer:** System Architect

---

## 1. Executive Summary

### Top 5 Critical Issues

| Priority | Issue | Impact | Location |
|----------|-------|--------|----------|
| **P0** | **No linkage between SQLite trades/orders and Chroma vectors** | Trading data in SQLite is never ingested into Chroma for semantic retrieval; lost learning opportunity | [schema.py](src/ordinis/adapters/storage/schema.py), [trading_memory.py](src/ordinis/rag/memory/trading_memory.py) |
| **P0** | **Dual-write inconsistency: no transaction coordination** | TradingMemoryStore writes to Chroma independently; failures leave orphan records with no reconciliation | [trading_memory.py#L265-L285](src/ordinis/rag/memory/trading_memory.py#L265-L285) |
| **P1** | **Missing session_id linkage across data stores** | Sessions exist in Chroma (`session_logs` collection) but not tracked in SQLite; cannot correlate trades to sessions | [session_indexer.py](src/ordinis/rag/pipeline/session_indexer.py), [schema.py](src/ordinis/adapters/storage/schema.py) |
| **P1** | **No embedding version tracking or migration strategy** | Re-embedding with a different model produces incompatible vectors; no versioning metadata stored | [chroma_client.py](src/ordinis/rag/vectordb/chroma_client.py) |
| **P2** | **ID generation is fragile and non-deterministic** | Auto-generated IDs like `text_{collection.count() + i}` can collide after deletes or concurrent writes | [chroma_client.py#L95-L97](src/ordinis/rag/vectordb/chroma_client.py#L95-L97) |

### Top 5 Recommended Fixes

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| **P0** | Add `session_id` column to SQLite `trades` and `orders` tables; create ingestion pipeline from SQLite → Chroma | Medium | Enables trade-session correlation and semantic search over trading history |
| **P0** | Implement `DualWriteManager` with saga pattern: write SQLite first, then Chroma, with compensating rollback | Medium | Guarantees consistency or detectable inconsistency |
| **P1** | Store embedding model version and dimension in collection metadata; add migration script to re-embed on version change | Low | Prevents silent retrieval degradation after model updates |
| **P1** | Introduce deterministic ID generation: `{entity_type}:{source_hash}:{chunk_index}` using SHA-256 | Low | Eliminates ID collisions; enables idempotent upserts |
| **P2** | Add FTS5 virtual table on `trades.metadata` for hybrid retrieval (keyword + vector) | Low | Improves recall for structured queries |

---

## 2. System Map: Synapse/RAGEngine in Context

### 2.1 Architecture Diagram (Text-Based)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    WRITERS                                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  SignalCore      RiskGuard      FlowRoute      PortfolioEngine      Analytics            │
│      │               │              │              │              │                 │
│      └───────────────┴──────────────┴──────────────┴──────────────┘                 │
│                                      │                                              │
│                          ┌───────────▼───────────┐                                  │
│                          │   OrchestrationEngine │                                  │
│                          │   (Trading Pipeline)  │                                  │
│                          └───────────┬───────────┘                                  │
│                                      │                                              │
│         ┌────────────────────────────┼────────────────────────────┐                 │
│         ▼                            ▼                            ▼                 │
│  ┌─────────────┐            ┌─────────────────┐          ┌───────────────┐          │
│  │ SQLite DB   │            │ LearningEngine  │          │ TradingMemory │          │
│  │ (ordinis.db)│            │ (event buffer)  │          │ Store (Chroma)│          │
│  │             │            │                 │          │               │          │
│  │ • positions │            │ • signals       │          │ • decisions   │          │
│  │ • orders    │◄───────────┤ • executions    │          │ • outcomes    │          │
│  │ • trades    │   (TODO)   │ • portfolio     │          │ • sessions    │          │
│  │ • fills     │            │ • predictions   │          │               │          │
│  └──────┬──────┘            └────────┬────────┘          └───────┬───────┘          │
│         │                            │                           │                  │
│         │                            │                           │                  │
│         ▼                            ▼                           ▼                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                              DATA STORES                                    │    │
│  │  ┌─────────────────┐    ┌─────────────────────────────────────────────────┐ │    │
│  │  │   SQLite        │    │                 ChromaDB                        │ │    │
│  │  │   (Relational)  │    │              (Vector Store)                     │ │    │
│  │  │                 │    │                                                 │ │    │
│  │  │ Tables:         │    │ Collections:                                    │ │    │
│  │  │ • positions     │    │ • kb_text (knowledge base docs)                 │ │    │
│  │  │ • orders        │    │ • codebase (source code embeddings)             │ │    │
│  │  │ • fills         │    │ • session_logs (conversation history)           │ │    │
│  │  │ • trades        │    │ • trading_memories (trade decisions)            │ │    │
│  │  │ • system_state  │    │ • sessions (session state)                      │ │    │
│  │  │ • audit_log     │    │ • publications (research papers)                │ │    │
│  │  │ • snapshots     │    │ • _metadata_schemas (schema registry)           │ │    │
│  │  └────────┬────────┘    └───────────────────────┬─────────────────────────┘ │    │
│  │           │                                     │                           │    │
│  │           │     ⚠️ NO LINKAGE ⚠️                 │                           │    │
│  │           └─────────────────────────────────────┘                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                    READERS                                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                          ┌───────────────────────┐                                  │
│                          │       SYNAPSE         │                                  │
│                          │   (RAG Retrieval)     │                                  │
│                          │                       │                                  │
│                          │ • retrieve()          │                                  │
│                          │ • synthesize()        │                                  │
│                          │ • search_code()       │                                  │
│                          │ • search_docs()       │                                  │
│                          └───────────┬───────────┘                                  │
│                                      │                                              │
│                          ┌───────────▼───────────┐                                  │
│                          │   RetrievalEngine     │                                  │
│                          │                       │                                  │
│                          │ • query()             │                                  │
│                          │ • _query_text()       │                                  │
│                          │ • _query_code()       │                                  │
│                          │ • _rerank_results()   │                                  │
│                          └───────────┬───────────┘                                  │
│                                      │                                              │
│         ┌────────────────────────────┼────────────────────────────┐                 │
│         ▼                            ▼                            ▼                 │
│    Cortex (LLM)              Helix (LLM Facade)           CodeGenService           │
│    (code review)             (model dispatch)             (AI code gen)             │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Summary

| Source Engine | Data Type | SQLite Table | Chroma Collection | Linkage |
|---------------|-----------|--------------|-------------------|---------|
| OrchestrationEngine | Trade executions | `trades`, `orders`, `fills` | ❌ None | **MISSING** |
| LearningEngine | Learning events | ❌ File-based flush | ❌ None | **MISSING** |
| TradingMemoryStore | Trade decisions | ❌ None | `trading_memories` | **PARTIAL** (no SQLite record) |
| SessionIndexer | Session logs | ❌ None | `session_logs` | **PARTIAL** (no SQLite record) |
| KBIndexer | Knowledge base docs | ❌ None | `kb_text` | OK (static content) |
| CodeIndexer | Source code | ❌ None | `codebase` | OK (static content) |

---

## 3. Data Model Inventory

### 3.1 SQLite Entities (Source of Truth for Trading)

**Location:** [src/ordinis/adapters/storage/schema.py](src/ordinis/adapters/storage/schema.py)

| Table | Key Columns | Indexes | Foreign Keys | Notes |
|-------|-------------|---------|--------------|-------|
| `positions` | `symbol` (UNIQUE), `side`, `quantity`, `avg_cost` | `idx_positions_symbol`, `idx_positions_side` | None | Current portfolio state |
| `orders` | `order_id` (UNIQUE), `symbol`, `status`, `strategy_id`, `signal_id` | `idx_orders_order_id`, `idx_orders_symbol`, `idx_orders_status`, `idx_orders_created_at` | None | Order lifecycle |
| `fills` | `fill_id` (UNIQUE), `order_id` | `idx_fills_fill_id`, `idx_fills_order_id`, `idx_fills_symbol`, `idx_fills_timestamp` | `order_id` → `orders.order_id` | Individual fills |
| `trades` | `trade_id` (UNIQUE), `symbol`, `strategy_id` | `idx_trades_trade_id`, `idx_trades_symbol`, `idx_trades_entry_time`, `idx_trades_exit_time` | None | Completed trades |
| `system_state` | `key` (UNIQUE), `value`, `value_type` | `idx_system_state_key` | None | Runtime config |
| `persistence_audit` | `entity_type`, `entity_id`, `action`, `timestamp` | `idx_persistence_audit_entity`, `idx_persistence_audit_timestamp` | None | Audit trail |
| `portfolio_snapshots` | `snapshot_date` (UNIQUE), `positions_json` | `idx_portfolio_snapshots_date` | None | Daily snapshots |
| `schema_version` | `version`, `applied_at` | None | None | Migration tracking |

**Critical Gap:** No `session_id` column in any table to correlate trades/orders with Chroma session data.

### 3.2 ChromaDB Collections (Vector Store)

**Location:** [src/ordinis/rag/vectordb/chroma_client.py](src/ordinis/rag/vectordb/chroma_client.py)

| Collection | Metadata Schema | ID Format | Embedding Model | Notes |
|------------|-----------------|-----------|-----------------|-------|
| `kb_text` | `domain`, `source`, `chunk_index`, `publication_id`, `section` | `{source}_chunk{i}` | `nvidia/llama-3.2-nemoretriever-300m-embed-v2` | Knowledge base documents |
| `codebase` | `file_path`, `function_name`, `class_name`, `engine`, `line_start`, `line_end` | `{file_path}_chunk{i}` | Same | Source code |
| `session_logs` | `session_id`, `export_time`, `source_file`, `content_type`, `chunk_index`, `total_chunks` | `session_{session_id}_chunk{i}` | Same | Conversation history |
| `trading_memories` | `memory_type`, `timestamp`, `symbol`, `timeframe`, `market_regime`, `decision`, `confidence`, `price`, `outcome`, `pnl`, `strategy_id`, `tags` | `{memory_id}` (UUID-like hash) | Default Chroma | Trade decisions |
| `sessions` | `start_time`, `last_activity`, `trades_executed` | `{session_id}` | Default Chroma | Session state blobs |
| `publications` | `source_type`, `topic` | Dynamic | Same | Research papers |
| `_metadata_schemas` | `collection`, `timestamp` | `{collection_name}` | N/A | Schema registry |

**Critical Gap:** No embedding model version or dimension stored in collection metadata.

### 3.3 LearningEngine Events (In-Memory Buffer)

**Location:** [src/ordinis/engines/learning/core/engine.py](src/ordinis/engines/learning/core/engine.py)

| Event Type | Source | Persisted To | Notes |
|------------|--------|--------------|-------|
| `SIGNAL_GENERATED` | SignalCore | ❌ Memory only | Flushed to file on shutdown |
| `SIGNAL_ACCURACY` | Analytics | ❌ Memory only | No Chroma ingestion |
| `ORDER_SUBMITTED` | FlowRoute | ❌ Memory only | Duplicates SQLite data |
| `ORDER_FILLED` | FlowRoute | ❌ Memory only | Duplicates SQLite data |
| `REBALANCE_EXECUTED` | Portfolio | ❌ Memory only | No Chroma ingestion |
| `POSITION_OPENED/CLOSED` | Portfolio | ❌ Memory only | No Chroma ingestion |
| `MODEL_PREDICTION` | SignalCore | ❌ Memory only | No Chroma ingestion |

**Critical Gap:** LearningEngine captures rich event data but never writes to Chroma for semantic retrieval.

---

## 4. SQLite: Schema, Indexing, Migrations, and Query Review

### 4.1 Schema Assessment

**Strengths:**

- Well-normalized tables with clear separation of concerns
- Appropriate use of `UNIQUE` constraints on identifiers
- JSON columns (`metadata`, `broker_response`) for flexibility
- `CHECK` constraints on enums (`side`, `order_type`, `status`)
- WAL mode enabled for concurrent reads

**Weaknesses:**

| Issue | Location | Severity | Recommendation |
|-------|----------|----------|----------------|
| No `session_id` in `trades`/`orders` | [schema.py#L46-L68](src/ordinis/adapters/storage/schema.py#L46-L68) | High | Add `session_id TEXT` column |
| No FTS5 for `metadata` JSON search | All tables | Medium | Add FTS5 virtual table |
| `persistence_audit.entity_id` is TEXT but not indexed alone | [schema.py#L114-L127](src/ordinis/adapters/storage/schema.py#L114-L127) | Low | Add single-column index |
| No partial indexes for common filters | All tables | Low | Add `WHERE status = 'open'` partial index on `orders` |

### 4.2 Index Coverage Analysis

**Current Indexes:**

```sql
-- Positions
idx_positions_symbol ON positions(symbol)
idx_positions_side ON positions(side)

-- Orders
idx_orders_order_id ON orders(order_id)
idx_orders_symbol ON orders(symbol)
idx_orders_status ON orders(status)
idx_orders_created_at ON orders(created_at)
idx_orders_broker_order_id ON orders(broker_order_id)

-- Fills
idx_fills_fill_id ON fills(fill_id)
idx_fills_order_id ON fills(order_id)
idx_fills_symbol ON fills(symbol)
idx_fills_timestamp ON fills(timestamp)

-- Trades
idx_trades_trade_id ON trades(trade_id)
idx_trades_symbol ON trades(symbol)
idx_trades_entry_time ON trades(entry_time)
idx_trades_exit_time ON trades(exit_time)
```

**Missing Indexes:**

```sql
-- Recommended additions
CREATE INDEX idx_orders_strategy_id ON orders(strategy_id) WHERE strategy_id IS NOT NULL;
CREATE INDEX idx_trades_strategy_id ON trades(strategy_id) WHERE strategy_id IS NOT NULL;
CREATE INDEX idx_orders_status_created ON orders(status, created_at);  -- Composite for open orders query
```

### 4.3 Query Pattern Analysis

**Common Queries (from repositories):**

| Query | Table | Indexes Used | Recommendation |
|-------|-------|--------------|----------------|
| `SELECT * FROM trades WHERE trade_id = ?` | trades | ✅ `idx_trades_trade_id` | OK |
| `SELECT * FROM trades WHERE symbol = ? ORDER BY exit_time DESC` | trades | ✅ `idx_trades_symbol` | OK |
| `SELECT * FROM trades WHERE exit_time >= ? AND exit_time <= ?` | trades | ✅ `idx_trades_exit_time` | OK |
| `SELECT * FROM orders WHERE status = 'open' ORDER BY created_at` | orders | ⚠️ Partial scan | Add composite index |
| `SELECT * FROM orders WHERE strategy_id = ?` | orders | ❌ Full scan | Add `idx_orders_strategy_id` |

### 4.4 Migration Strategy

**Current State:** Schema version tracking exists via `schema_version` table.

**Missing:**

- No migration files (only DDL in `SCHEMA_DDL` string)
- No rollback capability
- No Alembic or similar tool

**Recommendation:** Adopt Alembic for structured migrations with up/down scripts.

---

## 5. Chroma: Collections, Metadata, Embeddings, and Retrieval Review

### 5.1 Collection Configuration

**Current Configuration:**

```python
# From chroma_client.py
client = chromadb.PersistentClient(
    path=str(self.persist_directory),
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True,
    ),
)
```

**Issues:**

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| No HNSW tuning parameters | Medium | Set `hnsw:space=cosine`, tune `hnsw:M`, `hnsw:ef_construction` |
| No collection-level metadata for versioning | High | Store `embedding_model`, `embedding_dim`, `created_at` |
| `allow_reset=True` in production | Low | Disable in production config |

### 5.2 ID Strategy Analysis

**Current ID Generation:**

| Collection | ID Format | Generation Method | Collision Risk |
|------------|-----------|-------------------|----------------|
| `kb_text` | `{relative_path}_chunk{i}` | Sequential counter | Medium (rename/reorder) |
| `codebase` | `{file_path}_chunk{i}` | Sequential counter | Medium (refactor) |
| `session_logs` | `session_{session_id}_chunk{i}` | Sequential counter | Low (session-scoped) |
| `trading_memories` | `{memory_id}` (SHA-256 hash) | Content hash | Low |
| Default | `text_{count()}` or `code_{count()}` | Collection count | **High** (delete + add) |

**Recommended ID Strategy:**

```python
def generate_deterministic_id(entity_type: str, content: str, chunk_index: int) -> str:
    """Generate collision-resistant, deterministic ID."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{entity_type}:{content_hash}:{chunk_index}"
```

### 5.3 Metadata Schema Analysis

**kb_text Metadata (TextChunkMetadata):**

```python
domain: int | None      # 1-11 for KB domains
source: str             # File path
chunk_index: int        # Chunk position
publication_id: str | None
section: str | None
```

**codebase Metadata (CodeChunkMetadata):**

```python
file_path: str
function_name: str | None
class_name: str | None
engine: str | None      # Engine name (cortex, signalcore, etc.)
line_start: int | None
line_end: int | None
```

**Issues:**

- No `embedding_model` or `embedding_version` in any metadata
- No `indexed_at` timestamp for staleness detection
- No `checksum` for change detection

### 5.4 Retrieval Quality Assessment

**Query Pipeline:**

```
Query → Embed → Chroma Search (top_k_retrieval=20) → Threshold Filter → Rerank (top_k_rerank=5)
```

**Reranking:**

```python
# From engine.py
self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
```

**Issues:**

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| No BM25/FTS hybrid retrieval | Medium | Add FTS fallback when vector similarity is low |
| No recency weighting | Medium | Add `indexed_at` decay factor |
| No session scoping in default queries | High | Always filter by `session_id` for session-aware retrieval |
| L2 distance → cosine conversion is approximate | Low | Use `collection.metadata={"hnsw:space": "cosine"}` |

### 5.5 Embedding Pipeline

**Current Flow:**

```
Document → Tokenize (tiktoken cl100k_base) → Chunk (512 tokens, 50 overlap) → Embed (NVIDIA API) → Store
```

**Issues:**

- No local embedding cache (re-embeds on every index run)
- No batching optimization for NVIDIA API (uses default batch_size=32)
- No embedding dimension reduction (Matryoshka support configured but unused)

---

## 6. Ingestion Pipelines and Dual-Write Consistency

### 6.1 Current Ingestion Flows

| Pipeline | Source | Destination | Trigger | Consistency |
|----------|--------|-------------|---------|-------------|
| KBIndexer | `docs/knowledge-base/*.md` | `kb_text` | Manual script | N/A (static) |
| CodeIndexer | `src/**/*.py` | `codebase` | Manual script | N/A (static) |
| SessionLogIndexer | Session export files | `session_logs` | Manual call | N/A (event-driven) |
| TradingMemoryStore | Trading decisions | `trading_memories` | Real-time | ⚠️ No SQLite record |

### 6.2 Missing Ingestion Flows

| Data Source | Should Go To | Current State | Impact |
|-------------|--------------|---------------|--------|
| SQLite `trades` | Chroma `trading_memories` or new `trade_vectors` | **Not connected** | Cannot semantically search trade history |
| LearningEngine events | Chroma `learning_events` | **Not connected** | Cannot retrieve similar learning patterns |
| GovernanceEngine audit | Chroma `governance_audit` | **Not connected** | Cannot search policy decisions |

### 6.3 Dual-Write Consistency Analysis

**Current State:**

- TradingMemoryStore writes to Chroma only (no SQLite counterpart)
- SQLite repositories write to SQLite only (no Chroma sync)
- No transaction coordination between stores

**Failure Modes:**

1. Chroma write succeeds, application crashes → SQLite has no record
2. SQLite write succeeds, Chroma write fails → Orphan relational record
3. Concurrent writes to same entity → Race condition in metadata

**Recommended Pattern:**

```python
async def dual_write_trade_memory(memory: TradeMemory, db: DatabaseManager, chroma: ChromaClient):
    """Saga pattern for dual-write consistency."""
    # Phase 1: Write to SQLite (source of truth)
    sqlite_row_id = await db.execute(
        "INSERT INTO trade_memories (...) VALUES (...)",
        memory.to_sqlite_tuple()
    )
    await db.commit()
    
    try:
        # Phase 2: Write to Chroma
        chroma.add_documents(
            texts=[memory.to_document()],
            embeddings=[embed(memory.to_document())],
            metadata=[memory.to_metadata()],
            collection_name="trading_memories",
            ids=[f"tm:{sqlite_row_id}"]  # Link via ID
        )
    except Exception as e:
        # Compensating action: mark SQLite record as unsynced
        await db.execute(
            "UPDATE trade_memories SET chroma_synced = 0 WHERE id = ?",
            (sqlite_row_id,)
        )
        await db.commit()
        raise DualWriteError(f"Chroma sync failed: {e}")
```

---

## 7. Cross-Engine Interfaces and Workflows

### 7.1 Current Synapse Consumers

| Consumer | Interface | Data Consumed | Notes |
|----------|-----------|---------------|-------|
| Cortex | `synapse.retrieve_for_codegen()` | Code + doc snippets | Used for AI code review |
| Helix | `synapse.retrieve_for_prompt()` | Context string | Used for LLM prompts |
| CLI | `synapse.search_docs()`, `synapse.search_code()` | Snippets | Interactive search |
| Dashboard | `synapse.get_stats()` | Statistics | Monitoring |

### 7.2 Missing Cross-Engine Integrations

| Integration | Source Engine | Target Engine | Data Flow | Benefit |
|-------------|---------------|---------------|-----------|---------|
| Trade → Memory | OrchestrationEngine | Synapse | `trades` → `trading_memories` | Semantic trade history search |
| Signal → Memory | SignalCore | Synapse | Signals → `signal_vectors` | Find similar signal patterns |
| Risk Event → Memory | RiskGuard | Synapse | Risk violations → `risk_events` | Learn from risk incidents |
| Session → SQLite | SessionManager | SQLite | Session state → `sessions` table | Durable session tracking |

### 7.3 LearningEngine Integration Gap

**Current State:**

- LearningEngine captures events in memory
- Flushes to files on shutdown
- No Chroma ingestion

**Recommended Integration:**

```python
# In LearningEngine._flush_events()
async def _flush_events(self):
    """Flush events to both file and Chroma."""
    for event in self._events:
        # 1. Write to file (existing)
        await self._write_event_to_file(event)
        
        # 2. Index to Chroma for semantic retrieval
        if self.config.enable_chroma_sync:
            await self._synapse.index_learning_event(event)
```

---

## 8. Gaps, Risks, and Failure Modes

### 8.1 Critical Gaps

| Gap | Category | Description | Risk Level |
|-----|----------|-------------|------------|
| **G1** | Linkage | SQLite trades/orders not linked to Chroma vectors | Critical |
| **G2** | Consistency | No dual-write transaction coordination | Critical |
| **G3** | Versioning | No embedding model version in metadata | High |
| **G4** | Session | No `session_id` in SQLite trading tables | High |
| **G5** | Deletion | Chroma vectors not deleted when SQLite records deleted | Medium |
| **G6** | Observability | No explain/tracing for retrieval decisions | Medium |
| **G7** | FTS | No full-text search fallback for keyword queries | Low |

### 8.2 Failure Modes

| Failure Mode | Trigger | Current Behavior | Recommended Fix |
|--------------|---------|------------------|-----------------|
| **F1** | Chroma unavailable | Synapse throws exception | Graceful degradation to cached results |
| **F2** | Embedding API timeout | Request fails | Retry with exponential backoff; local fallback |
| **F3** | ID collision | Delete + add same content | Upsert fails silently | Use deterministic content-hash IDs |
| **F4** | Schema migration | Add column to SQLite | Requires manual DDL | Use Alembic migrations |
| **F5** | Embedding model change | New vectors incompatible | Silent retrieval degradation | Track version; trigger re-index |

### 8.3 Data Lineage Gaps

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT STATE                                      │
│                                                                              │
│   Signal ──► Order ──► Fill ──► Trade                                       │
│     │          │         │        │                                         │
│     ▼          ▼         ▼        ▼                                         │
│   ❌ N/A     SQLite   SQLite   SQLite   ← Source of Truth                   │
│                                   │                                         │
│                                   ▼                                         │
│                                 ❌ N/A   ← No Chroma ingestion              │
│                                                                              │
│   Session Log ──────────────► Chroma (session_logs)                         │
│                                   │                                         │
│                                   ▼                                         │
│                                 ❌ N/A   ← No SQLite record                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           TARGET STATE                                       │
│                                                                              │
│   Signal ──► Order ──► Fill ──► Trade                                       │
│     │          │         │        │                                         │
│     ▼          ▼         ▼        ▼                                         │
│   SQLite    SQLite   SQLite   SQLite    ← Source of Truth                   │
│   (signals) (orders) (fills)  (trades)                                      │
│     │          │         │        │                                         │
│     ▼          ▼         ▼        ▼                                         │
│  Chroma    Chroma    Chroma   Chroma    ← Vector Store                      │
│  (linked via sqlite_id in metadata)                                         │
│                                                                              │
│   Session ───────► SQLite (sessions) ──► Chroma (session_logs)              │
│                          │                       │                          │
│                          └───────────────────────┘                          │
│                            (linked via session_id)                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Recommendations (P0/P1/P2)

### 9.1 Priority Matrix

| ID | Priority | Change | Impact | Effort | Risk | Owner |
|----|----------|--------|--------|--------|------|-------|
| **R1** | P0 | Add `session_id` to SQLite `trades`/`orders` tables | High | Medium | Low | Storage |
| **R2** | P0 | Create `TradeVectorIngester` pipeline (SQLite → Chroma) | Critical | Medium | Medium | RAG |
| **R3** | P0 | Implement `DualWriteManager` with saga pattern | Critical | Medium | Medium | Core |
| **R4** | P1 | Add `embedding_model`, `embedding_dim` to collection metadata | High | Low | Low | RAG |
| **R5** | P1 | Implement deterministic ID generation for all collections | High | Low | Low | RAG |
| **R6** | P1 | Add `indexed_at`, `checksum` to vector metadata | Medium | Low | Low | RAG |
| **R7** | P1 | Create SQLite `sessions` table with FK to trades/orders | High | Medium | Low | Storage |
| **R8** | P2 | Add FTS5 virtual table on `trades.metadata` | Medium | Low | Low | Storage |
| **R9** | P2 | Implement Alembic migrations | Medium | Medium | Low | Storage |
| **R10** | P2 | Add retrieval tracing/explain capability | Medium | Medium | Low | RAG |

### 9.2 Concrete Implementation Snippets

#### R1: Add session_id to trades/orders

```sql
-- Migration: add_session_id_to_trades.sql
ALTER TABLE trades ADD COLUMN session_id TEXT;
ALTER TABLE orders ADD COLUMN session_id TEXT;

CREATE INDEX idx_trades_session_id ON trades(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_orders_session_id ON orders(session_id) WHERE session_id IS NOT NULL;
```

#### R2: TradeVectorIngester

```python
# src/ordinis/rag/pipeline/trade_ingester.py
class TradeVectorIngester:
    """Ingest SQLite trades into Chroma for semantic retrieval."""
    
    COLLECTION = "trade_vectors"
    
    async def ingest_trade(self, trade: TradeRow) -> str:
        """Ingest a single trade into Chroma."""
        document = self._trade_to_document(trade)
        metadata = {
            "sqlite_id": trade.id,
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "side": trade.side,
            "strategy_id": trade.strategy_id,
            "pnl": trade.pnl,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "indexed_at": datetime.utcnow().isoformat(),
        }
        
        embedding = self.embedder.embed([document])[0]
        vector_id = f"trade:{trade.trade_id}"
        
        self.chroma.add_documents(
            texts=[document],
            embeddings=[embedding.tolist()],
            metadata=[metadata],
            collection_name=self.COLLECTION,
            ids=[vector_id],
        )
        return vector_id
    
    def _trade_to_document(self, trade: TradeRow) -> str:
        """Convert trade to natural language document."""
        return (
            f"Trade {trade.symbol} {trade.side}: "
            f"Entry ${trade.entry_price:.2f} at {trade.entry_time}, "
            f"Exit ${trade.exit_price:.2f} at {trade.exit_time}. "
            f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.1f}%). "
            f"Strategy: {trade.strategy_id or 'manual'}."
        )
```

#### R3: DualWriteManager

```python
# src/ordinis/core/dual_write.py
from contextlib import asynccontextmanager

class DualWriteManager:
    """Coordinate writes across SQLite and Chroma."""
    
    def __init__(self, db: DatabaseManager, chroma: ChromaClient):
        self.db = db
        self.chroma = chroma
        self._pending_compensations: list[Callable] = []
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for dual-write transaction."""
        await self.db.begin_transaction()
        self._pending_compensations = []
        try:
            yield self
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            await self._execute_compensations()
            raise
    
    async def write_with_vector(
        self,
        sql: str,
        params: tuple,
        document: str,
        metadata: dict,
        collection: str,
        vector_id: str,
    ):
        """Write to SQLite and Chroma atomically."""
        # Phase 1: SQLite
        await self.db.execute(sql, params)
        
        # Phase 2: Chroma (with compensation)
        try:
            embedding = self._embed(document)
            self.chroma.add_documents(
                texts=[document],
                embeddings=[embedding],
                metadata=[metadata],
                collection_name=collection,
                ids=[vector_id],
            )
        except Exception as e:
            # Register compensation
            self._pending_compensations.append(
                lambda: self.chroma.delete_documents([vector_id], collection)
            )
            raise DualWriteError(f"Vector write failed: {e}")
    
    async def _execute_compensations(self):
        """Execute compensating actions on failure."""
        for action in reversed(self._pending_compensations):
            try:
                action()
            except Exception as e:
                logger.error(f"Compensation failed: {e}")
```

#### R4: Collection Metadata Versioning

```python
# src/ordinis/rag/vectordb/chroma_client.py

def create_collection(
    self,
    collection_name: str,
    dimension: int = 1024,
    embedding_model: str = "nvidia/llama-3.2-nemoretriever-300m-embed-v2",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Create a new collection with versioning metadata."""
    versioned_metadata = {
        "embedding_model": embedding_model,
        "embedding_dim": dimension,
        "created_at": datetime.utcnow().isoformat(),
        "schema_version": "1.0",
        "hnsw:space": "cosine",
        **(metadata or {}),
    }
    
    self.client.get_or_create_collection(
        name=collection_name,
        metadata=versioned_metadata,
    )
```

#### R5: Deterministic ID Generation

```python
# src/ordinis/rag/vectordb/id_generator.py
import hashlib

def generate_vector_id(
    entity_type: str,
    source_id: str,
    content: str,
    chunk_index: int = 0,
) -> str:
    """Generate deterministic, collision-resistant vector ID.
    
    Format: {entity_type}:{source_id}:{content_hash}:{chunk_index}
    
    Args:
        entity_type: Type of entity (trade, session, kb, code)
        source_id: Source identifier (trade_id, session_id, file_path)
        content: Content being indexed (for change detection)
        chunk_index: Chunk index within document
    
    Returns:
        Deterministic ID string
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{entity_type}:{source_id}:{content_hash}:{chunk_index}"
```

---

## 10. Validation Strategy

### 10.1 Query Plan Verification

```sql
-- Verify index usage for common queries
EXPLAIN QUERY PLAN
SELECT * FROM trades WHERE symbol = 'AAPL' ORDER BY exit_time DESC LIMIT 100;
-- Expected: SEARCH trades USING INDEX idx_trades_symbol

EXPLAIN QUERY PLAN
SELECT * FROM orders WHERE status = 'open' ORDER BY created_at;
-- Expected: SEARCH orders USING INDEX (requires new composite index)

EXPLAIN QUERY PLAN
SELECT * FROM trades WHERE session_id = 'sess_123' AND strategy_id = 'momentum';
-- Expected: SEARCH trades USING INDEX idx_trades_session_id (after R1)
```

### 10.2 Load Test Scenarios

| Scenario | Target | Metric | Acceptance |
|----------|--------|--------|------------|
| Concurrent reads | 50 sessions reading trades | p99 latency | ≤ 50ms |
| Write throughput | 100 trades/second | Sustained rate | 100% success |
| Vector search | 10k vectors, 20 concurrent queries | p99 latency | ≤ 200ms |
| Dual-write | 50 trades/second with Chroma sync | Consistency | 0 orphans |
| Session index | 1MB session export | Index time | ≤ 5s |

### 10.3 Retrieval Evaluation Harness

```python
# tests/test_rag/test_retrieval_eval.py
class RetrievalEvaluator:
    """Evaluate retrieval quality against ground truth."""
    
    def __init__(self, synapse: Synapse, ground_truth: list[dict]):
        self.synapse = synapse
        self.ground_truth = ground_truth  # [{query, expected_ids, min_recall}]
    
    async def evaluate(self) -> dict:
        """Run evaluation and return metrics."""
        results = {
            "recall@5": [],
            "precision@5": [],
            "mrr": [],  # Mean Reciprocal Rank
        }
        
        for case in self.ground_truth:
            retrieved = await self.synapse.retrieve(case["query"], top_k=5)
            retrieved_ids = {r.id for r in retrieved}
            expected_ids = set(case["expected_ids"])
            
            # Calculate metrics
            hits = retrieved_ids & expected_ids
            recall = len(hits) / len(expected_ids) if expected_ids else 0
            precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0
            
            # MRR: position of first relevant result
            mrr = 0
            for i, r in enumerate(retrieved):
                if r.id in expected_ids:
                    mrr = 1 / (i + 1)
                    break
            
            results["recall@5"].append(recall)
            results["precision@5"].append(precision)
            results["mrr"].append(mrr)
        
        return {
            "recall@5": sum(results["recall@5"]) / len(results["recall@5"]),
            "precision@5": sum(results["precision@5"]) / len(results["precision@5"]),
            "mrr": sum(results["mrr"]) / len(results["mrr"]),
        }
```

### 10.4 Correctness Checks

| Check | Query | Expected | Severity |
|-------|-------|----------|----------|
| Orphan vectors | `SELECT COUNT(*) FROM chroma WHERE sqlite_id NOT IN (SELECT id FROM trades)` | 0 | Critical |
| Unsynced records | `SELECT COUNT(*) FROM trades WHERE chroma_synced = 0 AND created_at < NOW() - INTERVAL '1 hour'` | 0 | High |
| Duplicate IDs | `SELECT id, COUNT(*) FROM chroma GROUP BY id HAVING COUNT(*) > 1` | 0 | Critical |
| Schema drift | `SELECT embedding_model FROM collection_metadata WHERE embedding_model != ?` | 0 | High |

---

## 11. Implementation Plan

### 11.1 Next Sprint (Week 1-2)

| Task | Owner | Dependencies | Deliverable |
|------|-------|--------------|-------------|
| Add `session_id` to SQLite schema | Storage | None | Migration script |
| Implement deterministic ID generation | RAG | None | `id_generator.py` |
| Add embedding version to collection metadata | RAG | None | Updated `create_collection()` |
| Create `TradeVectorIngester` pipeline | RAG | Schema change | `trade_ingester.py` |

### 10.2 30-Day Milestone

- [ ] SQLite schema updated with `session_id` in `trades`, `orders`
- [ ] New SQLite `sessions` table with FK relationships
- [ ] `TradeVectorIngester` operational with backfill capability
- [ ] `DualWriteManager` implemented and tested
- [ ] All collections have versioned metadata

### 10.3 60-Day Milestone

- [ ] LearningEngine events synced to Chroma
- [ ] FTS5 virtual table on `trades.metadata`
- [ ] Hybrid retrieval (vector + FTS) implemented
- [ ] Alembic migrations set up
- [ ] Retrieval tracing/explain capability

### 10.4 90-Day Milestone

- [ ] Full data lineage: Signal → Trade → Vector → Session
- [ ] Deletion consistency (cascade from SQLite to Chroma)
- [ ] Embedding model upgrade path tested
- [ ] Load tests completed (10k trades, 100k vectors)
- [ ] Retrieval evaluation harness with ground-truth dataset

---

## Appendix: Evidence (Paths/Symbols)

### A.1 Key Source Files

| File | Key Symbols | Lines |
|------|-------------|-------|
| [src/ordinis/ai/synapse/engine.py](src/ordinis/ai/synapse/engine.py) | `Synapse`, `retrieve()`, `synthesize()`, `_detect_intent()` | 1-735 |
| [src/ordinis/rag/retrieval/engine.py](src/ordinis/rag/retrieval/engine.py) | `RetrievalEngine`, `query()`, `_rerank_results()`, `safe_query()` | 1-597 |
| [src/ordinis/rag/vectordb/chroma_client.py](src/ordinis/rag/vectordb/chroma_client.py) | `ChromaClient`, `add_texts()`, `query_texts()`, `_format_results()` | 1-426 |
| [src/ordinis/rag/vectordb/schema.py](src/ordinis/rag/vectordb/schema.py) | `TextChunkMetadata`, `CodeChunkMetadata`, `RetrievalResult` | 1-56 |
| [src/ordinis/adapters/storage/schema.py](src/ordinis/adapters/storage/schema.py) | `SCHEMA_DDL`, `INITIAL_SYSTEM_STATE` | 1-165 |
| [src/ordinis/adapters/storage/database.py](src/ordinis/adapters/storage/database.py) | `DatabaseManager`, `initialize()`, `execute()`, `fetch_all()` | 1-394 |
| [src/ordinis/rag/memory/trading_memory.py](src/ordinis/rag/memory/trading_memory.py) | `TradingMemoryStore`, `TradeMemory`, `SessionManager`, `SynapseMemoryIntegration` | 1-790 |
| [src/ordinis/rag/pipeline/session_indexer.py](src/ordinis/rag/pipeline/session_indexer.py) | `SessionLogIndexer`, `index_session_log()`, `search_sessions()` | 1-288 |
| [src/ordinis/rag/pipeline/kb_indexer.py](src/ordinis/rag/pipeline/kb_indexer.py) | `KBIndexer`, `index_directory()`, `_chunk_text()` | 1-211 |
| [src/ordinis/rag/pipeline/code_indexer.py](src/ordinis/rag/pipeline/code_indexer.py) | `CodeIndexer`, `index_directory()`, `_process_file()` | 1-241 |
| [src/ordinis/engines/learning/core/engine.py](src/ordinis/engines/learning/core/engine.py) | `LearningEngine`, `record_event()`, `_flush_events()` | 1-623 |

### A.2 Configuration Files

| File | Purpose |
|------|---------|
| [src/ordinis/rag/config.py](src/ordinis/rag/config.py) | RAG system configuration (chunk sizes, models, paths) |
| [src/ordinis/ai/synapse/config.py](src/ordinis/ai/synapse/config.py) | Synapse-specific settings (thresholds, cache, Chroma paths) |
| [src/ordinis/engines/learning/core/config.py](src/ordinis/engines/learning/core/config.py) | LearningEngine settings (event collection, training, drift detection) |

### A.3 Test Files

| File | Coverage |
|------|----------|
| [tests/test_rag/test_integration.py](tests/test_rag/test_integration.py) | RAG integration tests (skipped without data) |
| [tests/test_ai/test_synapse/test_config.py](tests/test_ai/test_synapse/test_config.py) | Synapse configuration tests |

---

## Appendix A: Memory & Context Integration Enhancements

### A.1 Current Memory Architecture Gaps

The current implementation has separate memory tiers that are not integrated:

| Tier | Current State | Gap |
|------|---------------|-----|
| **Raw Messages** | Session logs indexed to Chroma | No SQLite record; cannot join with trades |
| **Trade Decisions** | `TradingMemoryStore` in Chroma only | No SQLite backup; no linkage to session |
| **Session State** | `SessionManager` in-memory + Chroma | Lost on restart if Chroma unavailable |
| **Summaries** | Not implemented | No rolling summarization |
| **Artifacts** | File-based storage | Not indexed for semantic retrieval |

### A.2 Proposed Tiered Memory Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            TIERED MEMORY ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   TIER 1: HOT CONTEXT (Last N messages, in-memory + SQLite)                        │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  • Last 20 messages (configurable)                                          │   │
│   │  • Active session state                                                     │   │
│   │  • Current positions and open orders                                        │   │
│   │  • Store: SQLite `messages` table + in-memory cache                         │   │
│   │  • TTL: Session duration                                                    │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                             │
│                                       ▼                                             │
│   TIER 2: WARM CONTEXT (Session summary, SQLite + Chroma)                          │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  • Session summary (generated every 50 messages or on session end)          │   │
│   │  • Key decisions and outcomes                                               │   │
│   │  • User preferences and corrections                                         │   │
│   │  • Store: SQLite `session_summaries` + Chroma `session_summaries`           │   │
│   │  • TTL: 90 days (configurable)                                              │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                             │
│                                       ▼                                             │
│   TIER 3: COLD CONTEXT (Semantic chunks, Chroma only)                              │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  • Historical session chunks (older than 7 days)                            │   │
│   │  • Trade reasoning and outcomes                                             │   │
│   │  • Strategy documentation                                                   │   │
│   │  • Store: Chroma `session_logs`, `trading_memories`                         │   │
│   │  • TTL: 1 year (configurable, with compaction)                              │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                             │
│                                       ▼                                             │
│   TIER 4: ARCHIVE (Compressed transcripts, Object storage)                         │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  • Full session transcripts (markdown/jsonl)                                │   │
│   │  • Compressed and encrypted                                                 │   │
│   │  • On-demand rehydration to Chroma                                          │   │
│   │  • Store: data/archives/*.jsonl.gz                                          │   │
│   │  • TTL: Indefinite (regulatory compliance)                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### A.3 Transcript Export and Compilation Strategy

#### A.3.1 Transcript Schema

```python
# src/ordinis/rag/memory/transcript.py
@dataclass
class TranscriptEntry:
    """Single entry in a session transcript."""
    
    entry_id: str          # Deterministic: {session_id}:{sequence}
    session_id: str
    sequence: int          # Monotonic within session
    timestamp: datetime
    role: str              # "user", "assistant", "system", "tool"
    content: str
    
    # Context at time of entry
    active_positions: list[str] | None = None
    market_regime: str | None = None
    
    # Metadata
    tokens: int = 0
    tool_calls: list[str] | None = None
    citations: list[str] | None = None


@dataclass
class CompiledTranscript:
    """Full session transcript for export."""
    
    session_id: str
    start_time: datetime
    end_time: datetime
    
    entries: list[TranscriptEntry]
    
    # Session summary
    summary: str | None = None
    key_decisions: list[str] | None = None
    outcomes: dict[str, Any] | None = None
    
    # Metrics
    total_tokens: int = 0
    trades_executed: int = 0
    signals_generated: int = 0
    
    def to_markdown(self) -> str:
        """Export to chronological markdown."""
        lines = [
            f"# Session {self.session_id}",
            f"**Period:** {self.start_time} - {self.end_time}",
            f"**Trades:** {self.trades_executed} | **Signals:** {self.signals_generated}",
            "",
            "---",
            "",
        ]
        
        for entry in self.entries:
            lines.append(f"### [{entry.timestamp}] {entry.role.upper()}")
            lines.append("")
            lines.append(entry.content)
            lines.append("")
        
        if self.summary:
            lines.append("---")
            lines.append("## Session Summary")
            lines.append(self.summary)
        
        return "\n".join(lines)
    
    def to_jsonl(self) -> str:
        """Export to JSONL for archival."""
        import json
        lines = []
        for entry in self.entries:
            lines.append(json.dumps({
                "entry_id": entry.entry_id,
                "session_id": entry.session_id,
                "sequence": entry.sequence,
                "timestamp": entry.timestamp.isoformat(),
                "role": entry.role,
                "content": entry.content,
                "tokens": entry.tokens,
            }))
        return "\n".join(lines)
```

#### A.3.2 SQLite Schema for Messages

```sql
-- Migration: add_messages_table.sql

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT UNIQUE NOT NULL,     -- {session_id}:{sequence}
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tokens INTEGER DEFAULT 0,
    
    -- Context snapshot
    active_positions TEXT,              -- JSON array
    market_regime TEXT,
    
    -- Metadata
    tool_calls TEXT,                    -- JSON array
    citations TEXT,                     -- JSON array
    
    -- Indexing flags
    chroma_synced INTEGER DEFAULT 0,
    summary_included INTEGER DEFAULT 0,
    
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_messages_session_seq ON messages(session_id, sequence);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);
CREATE INDEX idx_messages_role ON messages(role);
CREATE INDEX idx_messages_unsynced ON messages(chroma_synced) WHERE chroma_synced = 0;

-- Session summaries table
CREATE TABLE IF NOT EXISTS session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary_type TEXT NOT NULL CHECK(summary_type IN ('rolling', 'final', 'key_facts')),
    content TEXT NOT NULL,
    
    -- Time range covered
    start_sequence INTEGER NOT NULL,
    end_sequence INTEGER NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    
    -- Metadata
    tokens INTEGER DEFAULT 0,
    model_used TEXT,
    
    -- Sync status
    chroma_synced INTEGER DEFAULT 0,
    chroma_id TEXT,
    
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_summaries_session ON session_summaries(session_id);
CREATE INDEX idx_summaries_type ON session_summaries(session_id, summary_type);
CREATE UNIQUE INDEX idx_summaries_range ON session_summaries(session_id, summary_type, start_sequence, end_sequence);
```

### A.4 Rolling Summary and Compaction Strategy

#### A.4.1 Summary Generation Policy

| Trigger | Summary Type | Coverage | Retention |
|---------|--------------|----------|-----------|
| Every 50 messages | `rolling` | Last 50 messages | 90 days |
| Session end | `final` | Entire session | 1 year |
| Explicit user request | `key_facts` | Specific topics | Indefinite |
| Weekly compaction job | `rolling` | Older summaries merged | 1 year |

#### A.4.2 Summary Generator Implementation

```python
# src/ordinis/rag/memory/summarizer.py
class SessionSummarizer:
    """Generate rolling summaries for long sessions."""
    
    ROLLING_WINDOW = 50  # messages
    
    def __init__(self, helix: Helix, db: DatabaseManager):
        self.helix = helix
        self.db = db
    
    async def generate_rolling_summary(
        self,
        session_id: str,
        start_sequence: int,
        end_sequence: int,
    ) -> str:
        """Generate summary for message range."""
        
        # Fetch messages
        messages = await self.db.fetch_all(
            """
            SELECT role, content, timestamp, tokens
            FROM messages
            WHERE session_id = ? AND sequence BETWEEN ? AND ?
            ORDER BY sequence
            """,
            (session_id, start_sequence, end_sequence),
        )
        
        # Build conversation text
        conversation = "\n\n".join(
            f"[{m['role'].upper()}] {m['content']}"
            for m in messages
        )
        
        # Generate summary via LLM
        prompt = f"""Summarize this trading session conversation segment.
Focus on:
1. Key decisions made (buy/sell signals, risk adjustments)
2. Market conditions discussed
3. User preferences or corrections
4. Outcomes of trades mentioned

Conversation:
{conversation}

Summary (max 200 words):"""
        
        response = await self.helix.generate(
            messages=[{"role": "user", "content": prompt}],
            model_id="nemotron-8b-v3.1",
            max_tokens=300,
        )
        
        summary = response.content
        
        # Store summary
        await self.db.execute(
            """
            INSERT INTO session_summaries 
            (session_id, summary_type, content, start_sequence, end_sequence, start_time, end_time, model_used)
            VALUES (?, 'rolling', ?, ?, ?, ?, ?, ?)
            """,
            (session_id, summary, start_sequence, end_sequence, 
             messages[0]['timestamp'], messages[-1]['timestamp'], "nemotron-8b-v3.1"),
        )
        await self.db.commit()
        
        return summary
    
    async def check_and_summarize(self, session_id: str) -> str | None:
        """Check if summary needed and generate if so."""
        
        # Get last summarized sequence
        last_summary = await self.db.fetch_one(
            """
            SELECT MAX(end_sequence) as last_seq
            FROM session_summaries
            WHERE session_id = ? AND summary_type = 'rolling'
            """,
            (session_id,),
        )
        
        last_seq = last_summary['last_seq'] or 0
        
        # Count messages since last summary
        count = await self.db.fetch_one(
            """
            SELECT COUNT(*) as cnt, MAX(sequence) as max_seq
            FROM messages
            WHERE session_id = ? AND sequence > ?
            """,
            (session_id, last_seq),
        )
        
        if count['cnt'] >= self.ROLLING_WINDOW:
            return await self.generate_rolling_summary(
                session_id,
                last_seq + 1,
                count['max_seq'],
            )
        
        return None
```

### A.5 Context Assembly Pipeline

```python
# src/ordinis/rag/context/assembler.py
class ContextAssembler:
    """Assemble context from tiered memory for LLM calls."""
    
    def __init__(
        self,
        db: DatabaseManager,
        synapse: Synapse,
        max_tokens: int = 8000,
    ):
        self.db = db
        self.synapse = synapse
        self.max_tokens = max_tokens
    
    async def assemble(
        self,
        session_id: str,
        query: str,
        include_trades: bool = True,
        cross_session: bool = False,
    ) -> AssembledContext:
        """Assemble context from all memory tiers."""
        
        context_parts = []
        token_budget = self.max_tokens
        
        # TIER 1: Recent messages (always included, from SQLite)
        recent = await self._get_recent_messages(session_id, limit=20)
        recent_tokens = sum(m['tokens'] for m in recent)
        context_parts.append(ContextPart(
            tier=1,
            type="recent_messages",
            content=self._format_messages(recent),
            tokens=recent_tokens,
            source="sqlite:messages",
        ))
        token_budget -= recent_tokens
        
        # TIER 2: Session summaries (if budget allows)
        if token_budget > 500:
            summaries = await self._get_session_summaries(session_id)
            summary_text = "\n\n".join(s['content'] for s in summaries)
            summary_tokens = self._count_tokens(summary_text)
            
            if summary_tokens <= token_budget * 0.3:  # Max 30% for summaries
                context_parts.append(ContextPart(
                    tier=2,
                    type="session_summaries",
                    content=summary_text,
                    tokens=summary_tokens,
                    source="sqlite:session_summaries",
                ))
                token_budget -= summary_tokens
        
        # TIER 3: Semantic retrieval (remaining budget)
        if token_budget > 200 and query:
            # Scope to session unless cross_session=True
            session_filter = None if cross_session else session_id
            
            retrieved = await self.synapse.retrieve(
                query=query,
                top_k=min(10, token_budget // 200),
                session_id=session_filter,
            )
            
            for chunk in retrieved:
                chunk_tokens = self._count_tokens(chunk.text)
                if chunk_tokens <= token_budget:
                    context_parts.append(ContextPart(
                        tier=3,
                        type="semantic_chunk",
                        content=chunk.text,
                        tokens=chunk_tokens,
                        source=f"chroma:{chunk.metadata.get('source', 'unknown')}",
                        chunk_id=chunk.id,
                        score=chunk.score,
                    ))
                    token_budget -= chunk_tokens
        
        # TIER 4: Trade context (if requested)
        if include_trades and token_budget > 300:
            trades = await self._get_recent_trades(session_id, limit=5)
            trades_text = self._format_trades(trades)
            trades_tokens = self._count_tokens(trades_text)
            
            if trades_tokens <= token_budget:
                context_parts.append(ContextPart(
                    tier=4,
                    type="trade_context",
                    content=trades_text,
                    tokens=trades_tokens,
                    source="sqlite:trades",
                ))
        
        return AssembledContext(
            session_id=session_id,
            query=query,
            parts=context_parts,
            total_tokens=self.max_tokens - token_budget,
            cross_session=cross_session,
        )
```

### A.6 Retention and Governance Policies

| Data Type | SQLite Retention | Chroma Retention | Deletion Policy |
|-----------|------------------|------------------|-----------------|
| Messages | 90 days | N/A (embedded in chunks) | Hard delete with audit log |
| Session summaries | 1 year | 1 year | Hard delete both stores |
| Session chunks | N/A | 1 year | Soft delete (tombstone) |
| Trade memories | Indefinite | Indefinite | Soft delete |
| Archived transcripts | 7 years | N/A | Regulatory hold |

#### A.6.1 Cascade Delete Implementation

```python
# src/ordinis/rag/memory/retention.py
class RetentionManager:
    """Manage data retention across SQLite and Chroma."""
    
    async def delete_session(
        self,
        session_id: str,
        audit_reason: str,
    ) -> dict:
        """Delete session data from all stores with audit trail."""
        
        results = {"sqlite": {}, "chroma": {}}
        
        async with self.db.transaction():
            # 1. Audit log entry
            await self.db.execute(
                """
                INSERT INTO retention_audit 
                (entity_type, entity_id, action, reason, timestamp)
                VALUES ('session', ?, 'delete', ?, datetime('now'))
                """,
                (session_id, audit_reason),
            )
            
            # 2. Delete SQLite messages
            result = await self.db.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            results["sqlite"]["messages"] = result.rowcount
            
            # 3. Delete SQLite summaries
            result = await self.db.execute(
                "DELETE FROM session_summaries WHERE session_id = ?",
                (session_id,),
            )
            results["sqlite"]["summaries"] = result.rowcount
            
            # 4. Delete Chroma vectors
            try:
                deleted = await self._delete_chroma_by_session(session_id)
                results["chroma"]["vectors"] = deleted
            except Exception as e:
                logger.error(f"Chroma delete failed: {e}")
                results["chroma"]["error"] = str(e)
                # Mark for reconciliation
                await self.db.execute(
                    """
                    INSERT INTO chroma_cleanup_queue 
                    (session_id, action, created_at)
                    VALUES (?, 'delete', datetime('now'))
                    """,
                    (session_id,),
                )
        
        return results
```

---

## Appendix B: Reference Notes (SQLite + Chroma Best Practices Applied)

### B.1 SQLite Best Practices Applied

| Best Practice | Current State | Recommendation | Reference |
|---------------|---------------|----------------|-----------|
| **Parameterized queries** | ✅ Used throughout | Maintain | [sqlite3 docs: Placeholders](https://docs.python.org/3/library/sqlite3.html#sqlite3-placeholders) |
| **WAL mode** | ✅ Enabled in `database.py:L67` | Maintain | [SQLite WAL](https://www.sqlite.org/wal.html) |
| **Foreign keys** | ⚠️ Enabled but underused | Add FK from trades→sessions | [sqlite3 docs: Foreign keys](https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.execute) |
| **Explicit transactions** | ⚠️ Partial | Wrap dual-writes in transactions | [sqlite3 docs: Transactions](https://docs.python.org/3/library/sqlite3.html#sqlite3-controlling-transactions) |
| **busy_timeout** | ❌ Not set | Add `PRAGMA busy_timeout = 5000;` | [SQLite PRAGMA](https://www.sqlite.org/pragma.html#pragma_busy_timeout) |
| **Index alignment** | ⚠️ Missing composite indexes | Add per Section 4.2 | [SQLite Query Planner](https://www.sqlite.org/queryplanner.html) |
| **FTS5** | ❌ Not implemented | Add for message search | [SQLite FTS5](https://www.sqlite.org/fts5.html) |
| **Connection pooling** | ⚠️ Single connection | Use aiosqlite pool for concurrent writes | [aiosqlite docs](https://aiosqlite.omnilib.dev/) |

### B.2 ChromaDB Best Practices Applied

| Best Practice | Current State | Recommendation | Reference |
|---------------|---------------|----------------|-----------|
| **Stable IDs** | ⚠️ Fragile (count-based) | Use content-hash IDs per R5 | [Chroma: IDs](https://docs.trychroma.com/docs/collections/add-data#ids) |
| **Metadata for scoping** | ✅ `session_id`, `source` present | Add `workspace_id`, `engine` | [Chroma: Metadata filtering](https://docs.trychroma.com/docs/collections/query-data#filtering-by-metadata) |
| **Provenance fields** | ⚠️ Missing `source_table`, `source_id` | Add for SQLite linkage | [Chroma: Metadata](https://docs.trychroma.com/docs/collections/add-data#metadata) |
| **Embedding versioning** | ❌ Not tracked | Store `embedding_model_version` | [Chroma: Collection metadata](https://docs.trychroma.com/docs/collections/configure#collection-configuration) |
| **HNSW tuning** | ⚠️ Default parameters | Set `hnsw:space=cosine`, tune M/ef | [Chroma: HNSW](https://docs.trychroma.com/docs/collections/configure#hnsw-configuration) |
| **Upsert support** | ✅ Available | Use for idempotent writes | [Chroma: Upsert](https://docs.trychroma.com/docs/collections/add-data#upserting-data) |
| **Tenant isolation** | ❌ Single tenant | Add multi-tenant config | [Chroma: Multi-tenancy](https://docs.trychroma.com/docs/production/multitenancy) |
| **Persistence** | ✅ PersistentClient | Maintain | [Chroma: Persistence](https://docs.trychroma.com/docs/run-chroma/persistent-client) |

### B.3 Dual-Write Patterns Applied

| Pattern | Implementation | Reference |
|---------|----------------|-----------|
| **Outbox pattern** | Proposed via `chroma_synced` flag in SQLite | [Microservices Patterns: Outbox](https://microservices.io/patterns/data/transactional-outbox.html) |
| **Saga with compensation** | Proposed via `DualWriteManager` | [Saga Pattern](https://microservices.io/patterns/data/saga.html) |
| **Reconciliation** | Proposed via `chroma_cleanup_queue` table | Best practice for eventual consistency |
| **Idempotent writes** | Deterministic IDs enable safe retries | [Chroma: Upsert](https://docs.trychroma.com/docs/collections/add-data#upserting-data) |

### B.4 Memory Integration Patterns

| Pattern | Implementation | Benefit |
|---------|----------------|---------|
| **Tiered memory** | Tier 1-4 architecture (Section A.2) | Balances freshness vs. cost |
| **Rolling summaries** | 50-message windows (Section A.4) | Compresses old context |
| **Session scoping** | Default filter by `session_id` | Prevents context bleed |
| **Cross-session retrieval** | Explicit flag with audit | Enables learning across sessions |
| **Deterministic ordering** | `sequence` column in messages | Reproducible context assembly |

---

## Appendix C: Evidence (Paths/Symbols/Line Ranges)

### C.1 Core Source Files with Line Ranges

| File | Key Symbols | Lines | Notes |
|------|-------------|-------|-------|
| [src/ordinis/ai/synapse/engine.py](src/ordinis/ai/synapse/engine.py) | `Synapse.__init__`, `retrieve()`, `synthesize()`, `_detect_intent()`, `_ensure_rag_engine()` | 1-735 | Main RAG facade; lazy-loads RetrievalEngine at L145-180 |
| [src/ordinis/rag/retrieval/engine.py](src/ordinis/rag/retrieval/engine.py) | `RetrievalEngine.query()`, `_rerank_results()`, `_query_text()`, `_query_code()`, `safe_query()` | 1-597 | Core retrieval; reranking at L320-380; threshold filtering at L250-280 |
| [src/ordinis/rag/vectordb/chroma_client.py](src/ordinis/rag/vectordb/chroma_client.py) | `ChromaClient.__init__`, `add_texts()`, `query_texts()`, `_format_results()`, `reset()` | 1-426 | Chroma wrapper; ID generation at L95-97 (fragile); L2→cosine at L218-220 |
| [src/ordinis/rag/vectordb/schema.py](src/ordinis/rag/vectordb/schema.py) | `TextChunkMetadata`, `CodeChunkMetadata`, `RetrievalResult` | 1-56 | Pydantic models for metadata |
| [src/ordinis/adapters/storage/schema.py](src/ordinis/adapters/storage/schema.py) | `SCHEMA_DDL`, `INITIAL_SYSTEM_STATE` | 1-165 | SQLite DDL; trades table at L46-68; orders at L70-95 |
| [src/ordinis/adapters/storage/database.py](src/ordinis/adapters/storage/database.py) | `DatabaseManager.__init__`, `initialize()`, `execute()`, `fetch_all()`, `_enable_wal()` | 1-394 | WAL at L67-72; backup at L200-220; connection at L50-65 |
| [src/ordinis/rag/memory/trading_memory.py](src/ordinis/rag/memory/trading_memory.py) | `TradingMemoryStore`, `TradeMemory`, `SessionManager`, `add_memory()`, `search()` | 1-790 | Chroma-only writes at L265-285; session state at L680-720 |
| [src/ordinis/rag/pipeline/session_indexer.py](src/ordinis/rag/pipeline/session_indexer.py) | `SessionLogIndexer.__init__`, `index_session_log()`, `search_sessions()`, `_chunk_text()` | 1-288 | Chunking at L200-250; metadata at L110-135; ID generation at L123-125 |
| [src/ordinis/rag/pipeline/kb_indexer.py](src/ordinis/rag/pipeline/kb_indexer.py) | `KBIndexer.index_directory()`, `_process_file()`, `_chunk_text()` | 1-211 | Domain detection at L80-100; chunking at L140-170 |
| [src/ordinis/rag/pipeline/code_indexer.py](src/ordinis/rag/pipeline/code_indexer.py) | `CodeIndexer.index_directory()`, `_process_file()`, `_extract_functions()` | 1-241 | AST parsing at L100-140; metadata at L150-180 |
| [src/ordinis/engines/learning/core/engine.py](src/ordinis/engines/learning/core/engine.py) | `LearningEngine.__init__`, `record_event()`, `_flush_events()`, `train()` | 1-623 | Event buffer at L100-130; flush at L450-500; no Chroma sync |
| [src/ordinis/engines/learning/core/config.py](src/ordinis/engines/learning/core/config.py) | `LearningEngineConfig`, `event_retention_days`, `model_version_retention` | 1-93 | Retention at L66-68; validation at L75-92 |

### C.2 Configuration Files

| File | Key Settings | Notes |
|------|--------------|-------|
| [src/ordinis/rag/config.py](src/ordinis/rag/config.py) | `RAGConfig`, `chunk_size=512`, `chunk_overlap=50`, `embedding_model` | Central RAG config |
| [src/ordinis/ai/synapse/config.py](src/ordinis/ai/synapse/config.py) | `SynapseConfig`, `similarity_threshold`, `top_k_retrieval`, `top_k_rerank` | Synapse-specific thresholds |
| [src/ordinis/adapters/storage/schema.py](src/ordinis/adapters/storage/schema.py) | `SCHEMA_VERSION=1`, `SCHEMA_DDL` | SQLite schema definition |

### C.3 Test Files

| File | Coverage | Notes |
|------|----------|-------|
| [tests/test_rag/test_integration.py](tests/test_rag/test_integration.py) | RAG end-to-end | Skipped without data |
| [tests/test_rag/test_retrieval.py](tests/test_rag/test_retrieval.py) | RetrievalEngine unit tests | Mocked Chroma |
| [tests/test_ai/test_synapse/test_config.py](tests/test_ai/test_synapse/test_config.py) | Synapse config validation | Unit tests |
| [tests/test_adapters/test_storage/](tests/test_adapters/test_storage/) | SQLite repositories | CRUD operations |

### C.4 Identified Code Hotspots

| Location | Issue | Fix |
|----------|-------|-----|
| [chroma_client.py#L95-97](src/ordinis/rag/vectordb/chroma_client.py#L95-L97) | ID generation uses `collection.count()` | Use deterministic content-hash |
| [chroma_client.py#L218-220](src/ordinis/rag/vectordb/chroma_client.py#L218-L220) | L2 → cosine approximation | Set `hnsw:space=cosine` in collection metadata |
| [trading_memory.py#L265-285](src/ordinis/rag/memory/trading_memory.py#L265-L285) | Writes to Chroma only, no SQLite | Add dual-write with DualWriteManager |
| [schema.py#L46-68](src/ordinis/adapters/storage/schema.py#L46-L68) | `trades` table missing `session_id` | Add column per R1 |
| [learning/engine.py#L450-500](src/ordinis/engines/learning/core/engine.py#L450-L500) | `_flush_events()` writes files only | Add Chroma sync per Section 7.3 |
| [session_indexer.py#L123-125](src/ordinis/rag/pipeline/session_indexer.py#L123-L125) | ID uses sequential counter | Use `session_{session_id}:{chunk_hash}` |

---

**End of Review**
