# Synapse (Retrieval Engine) — RAG System Architecture & Implementation

## Overview

Synapse is the Ordinis retrieval subsystem. It implements a RAG (Retrieval Augmented Generation) pipeline that provides semantic search and contextual retrieval over trading knowledge and codebase documentation.

**Implementation Status:** Phase 1 Complete | Phases 2-4 Planned

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAG SYSTEM                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   Indexing   │────▶│   Vector DB  │◀────│  Retrieval   │                │
│  │  Pipelines   │     │  (ChromaDB)  │     │   Engine     │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│         │                    │                    │                         │
│         ▼                    ▼                    ▼                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │  Embedders   │     │ text_chunks  │     │   API/UI     │                │
│  │ Text │ Code  │     │ code_chunks  │     │  Interface   │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│                                                                              │
│  Integration Points:                                                         │
│  ├── Cortex: Hypothesis generation with KB context                          │
│  ├── RiskGuard: Trade explanations with risk principles                     │
│  ├── ProofBench: Backtest narration with strategy context                   │
│  └── SignalCore: Signal interpretation with TA concepts                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Location | Description |
|-----------|----------|-------------|
| **Embedders** | `src/rag/embedders/` | NVIDIA NeMo Retriever models for text/code |
| **Vector Database** | `src/rag/vectordb/` | ChromaDB with dual collections |
| **Retrieval Engine** | `src/rag/retrieval/` | Query classification and search |
| **Indexing Pipelines** | `src/rag/pipeline/` | KB and codebase indexers |
| **API Server** | `src/rag/api/` | FastAPI REST endpoints |
| **Engine Integration** | `src/engines/*/rag/` | Per-engine RAG adapters |

---

## Implementation Phases

### Phase 1: Foundation (COMPLETE)

**Status:** Complete

**Implemented:**

1. **Embedders**
   - `TextEmbedder`: NVIDIA NeMo Retriever 300M
   - `CodeEmbedder`: NVIDIA nv-embedcode-7B
   - Local GPU and API fallback support
   - Matryoshka compression (768 → 384 dims)

2. **Vector Database**
   - ChromaDB persistent storage
   - Dual collections: `text_chunks`, `code_chunks`
   - Metadata filtering support

3. **Retrieval Engine**
   - Auto query type detection (text/code/hybrid)
   - Multi-collection search and merging
   - Similarity threshold filtering

4. **Indexing Pipelines**
   - `KBIndexer`: Markdown knowledge base
   - `CodeIndexer`: Python codebase with AST parsing

**Known Limitations:**
- Using placeholder models pending NVIDIA NIM integration
- No reranking (Phase 2)
- Keyword-based query classifier

### Phase 2: Standalone API (PLANNED)

**Target:** FastAPI server with REST endpoints

- `/api/v1/query` - Semantic search
- `/health` - System health check
- `/stats` - Collection statistics
- Web UI for interactive testing
- Benchmark suite (100-query dataset)

### Phase 3: Cortex Enhancement (PLANNED)

- Integrate RAG into Cortex engine
- Enhance `generate_hypothesis()` with KB context
- Enhance `analyze_code()` with architecture patterns
- A/B testing framework

### Phase 4: Full Engine Integration (PLANNED)

- RiskGuard: Trade explanation with risk principles
- ProofBench: Backtest narration with strategy context
- SignalCore: Signal interpretation with TA concepts

---

## Configuration

### RAGConfig

```python
from rag.config import get_config, set_config, RAGConfig

config = RAGConfig(
    # Embedding models
    text_embedding_model="nvidia/llama-3.2-nemoretriever-300m-embed-v2",
    code_embedding_model="nvidia/nv-embedcode-7b-v1",

    # Deployment mode
    use_local_embeddings=True,  # False for API-only

    # Retrieval settings
    top_k_retrieval=20,
    top_k_rerank=5,
    similarity_threshold=0.7,

    # Chunking
    text_chunk_size=512,   # tokens
    code_chunk_size=100,   # lines
)
set_config(config)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_local_embeddings` | True | Use local GPU vs NVIDIA API |
| `top_k_retrieval` | 20 | Candidates to retrieve |
| `top_k_rerank` | 5 | Results after reranking |
| `similarity_threshold` | 0.7 | Minimum similarity (0-1) |
| `text_chunk_size` | 512 | Token size for text chunks |
| `code_chunk_size` | 100 | Max lines for code chunks |

---

## Usage

### Indexing

```python
from rag.pipeline.kb_indexer import KBIndexer
from rag.pipeline.code_indexer import CodeIndexer
from pathlib import Path

# Index knowledge base
kb_indexer = KBIndexer()
kb_indexer.index_directory(
    kb_path=Path("docs/knowledge-base"),
    collection_name="text_chunks",
)

# Index codebase
code_indexer = CodeIndexer()
code_indexer.index_directory(
    code_base=Path("src"),
    collection_name="code_chunks",
    file_patterns=["**/*.py"],
    exclude_patterns=["**/__pycache__/**"],
)
```

### Querying

```python
from rag.retrieval.engine import RetrievalEngine

engine = RetrievalEngine()

# Text query
response = engine.query(
    query="What is RSI mean reversion strategy?",
    query_type="text",
    top_k=5,
    filters={"domain": 2},  # Domain 2 = Technical Analysis
)

# Code query
response = engine.query(
    query="strategy implementation example",
    query_type="code",
    top_k=3,
)

# Hybrid query
response = engine.query(
    query="trading system architecture",
    query_type="hybrid",
    top_k=5,
)

# Process results
for result in response.results:
    print(f"Score: {result.score:.2f}")
    print(f"Source: {result.metadata.get('source')}")
    print(f"Text: {result.text[:200]}...")
```

### Cortex Integration

```python
from engines.cortex.core.engine import CortexEngine

cortex = CortexEngine(
    nvidia_api_key="your-api-key",
    rag_enabled=True,
)

# Hypothesis with RAG context
hypothesis = cortex.generate_hypothesis(
    market_context={"regime": "trending", "volatility": "low"},
    constraints={"max_position_pct": 0.10},
)
```

---

## API Endpoints

### Query

```
POST /api/v1/query
Content-Type: application/json

{
  "query": "What is momentum trading?",
  "query_type": "text",
  "top_k": 5,
  "filters": {"domain": 2}
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "...",
      "score": 0.92,
      "metadata": {"source": "02-technical-analysis.md", "domain": 2}
    }
  ],
  "query_type": "text",
  "latency_ms": 142.3
}
```

### Health Check

```
GET /health

{"status": "healthy", "text_embedder_available": true, "code_embedder_available": true}
```

### Statistics

```
GET /stats

{
  "chroma": {
    "text_chunks": {"count": 1250},
    "code_chunks": {"count": 856}
  }
}
```

---

## Knowledge Base Structure

### Domain Organization

| Domain | Content |
|--------|---------|
| 01 | Market Fundamentals |
| 02 | Technical Analysis |
| 03 | Market Microstructure |
| 04 | Options & Derivatives |
| 05 | Quantitative Methods |
| 06 | Behavioral Finance |
| 07 | Risk Management |
| 08 | Strategy Design |
| 09 | Backtesting & Validation |
| 10 | Execution & Infrastructure |

### Metadata Fields

**Text Chunks:**
- `domain`: Domain number (1-10)
- `source`: Filename
- `chunk_index`: Position in document
- `publication_id`: Optional publication reference

**Code Chunks:**
- `file_path`: Relative path
- `module`: Module name
- `function_name`, `class_name`: Code element
- `line_start`, `line_end`: Line numbers

---

## Deployment Options

### Option 1: Full Local (GPU Required)

```python
config = RAGConfig(
    use_local_embeddings=True,
    max_vram_usage_gb=9.0,
)
```

**Requirements:** NVIDIA GPU 11GB+ VRAM, CUDA 12.1+

### Option 2: Hybrid (Recommended)

```python
config = RAGConfig(
    use_local_embeddings=True,  # Text local
    # Code falls back to API (7B model)
)
```

**Requirements:** NVIDIA GPU 6GB+ VRAM, NVIDIA API key

### Option 3: Full API

```python
config = RAGConfig(
    use_local_embeddings=False,
    nvidia_api_key="your-key",
)
```

**Requirements:** NVIDIA API key, internet connection

---

## Performance

### Latency Targets

| Operation | Target | Status |
|-----------|--------|--------|
| Text Embedding (local) | <2s | Pending NIM |
| Code Embedding (local) | <2s | Pending NIM |
| Vector Search | <100ms | Achieved |
| End-to-End Retrieval | <3s | Pending NIM |

### Cost Estimate

**NVIDIA API (1000 queries/day):**
- Text queries: ~$0.60/month
- Code queries: ~$3.00/month
- LLM synthesis: ~$9.00/month
- **Total: ~$13/month**

**With Local Embeddings:** ~$9/month (LLM only)

---

## Testing

```bash
# Run all RAG tests
pytest tests/test_rag/ -v

# Unit tests only (fast)
pytest tests/test_rag/ -v -m "not integration"

# Integration tests (requires ChromaDB)
pytest tests/test_rag/ -v -m integration
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| VRAM OOM | Set `use_local_embeddings=False` or reduce `max_vram_usage_gb` |
| ChromaDB not found | Check `config.chroma_persist_directory` exists |
| No results | Verify collections are populated via `/stats` |
| Slow queries | Reduce `top_k_retrieval` or enable local embeddings |

---

## Related Documentation

| Document | Relationship |
|----------|--------------|
| [NVIDIA Integration](nvidia-integration.md) | LLM model setup for Cortex |
| [NVIDIA Blueprint Integration](nvidia-blueprint-integration.md) | Distillery for model optimization |
| [Layered System Architecture](layered-system-architecture.md) | RAG's role in orchestration |
| [SignalCore System](signalcore-system.md) | Engine integration targets |

---

## References

- [NVIDIA Build Platform](https://build.nvidia.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Matryoshka Embeddings](https://arxiv.org/abs/2205.13147)
