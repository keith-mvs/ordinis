# RAG System Documentation

## Overview

The Intelligent Investor RAG (Retrieval Augmented Generation) system provides semantic search and contextual retrieval capabilities for trading knowledge and codebase documentation.

## Architecture

### Components

1. **Embedders** (`src/rag/embedders/`)
   - `TextEmbedder`: NVIDIA NeMo Retriever 300M for text embedding
   - `CodeEmbedder`: NVIDIA nv-embedcode-7B for code embedding
   - Supports both local GPU and NVIDIA API deployment

2. **Vector Database** (`src/rag/vectordb/`)
   - ChromaDB for persistent vector storage
   - Dual collections: `text_chunks` and `code_chunks`
   - Metadata filtering support

3. **Retrieval Engine** (`src/rag/retrieval/`)
   - Query classification (text/code/hybrid)
   - Semantic search with similarity scoring
   - Reranking support (future)

4. **Indexing Pipelines** (`src/rag/pipeline/`)
   - `KBIndexer`: Index markdown knowledge base documents
   - `CodeIndexer`: Index Python codebase with AST parsing

5. **API Server** (`src/rag/api/`)
   - FastAPI REST API for RAG queries
   - Health check and statistics endpoints
   - Web UI for interactive testing

6. **Engine Integration** (`src/engines/*/rag/`)
   - Cortex: Hypothesis generation and code analysis
   - RiskGuard: Trade explanations with best practices
   - ProofBench: Backtesting context
   - SignalCore: Signal generation patterns

## Configuration

### RAGConfig (`src/rag/config.py`)

```python
from rag.config import get_config, set_config, RAGConfig

# Get current config
config = get_config()

# Modify config
config.use_local_embeddings = False  # Use NVIDIA API instead of local
config.top_k_retrieval = 10
config.similarity_threshold = 0.75

# Or create new config
custom_config = RAGConfig(
    text_embedding_model="nvidia/llama-3.2-nemoretriever-300m-embed-v2",
    code_embedding_model="nvidia/nv-embedcode-7b-v1",
    use_local_embeddings=True,
    top_k_retrieval=20,
    top_k_rerank=5,
    similarity_threshold=0.7,
)
set_config(custom_config)
```

### Key Parameters

- `use_local_embeddings`: Use local GPU (True) or NVIDIA API (False)
- `top_k_retrieval`: Number of candidates to retrieve
- `top_k_rerank`: Number of results after reranking
- `similarity_threshold`: Minimum similarity score (0.0-1.0)
- `text_chunk_size`: Token size for text chunks (default: 512)
- `code_chunk_size`: Max lines for code chunks (default: 100)

## Usage

### 1. Indexing Knowledge Base

```python
from rag.pipeline.kb_indexer import KBIndexer
from pathlib import Path

# Index knowledge base documents
indexer = KBIndexer()
indexer.index_directory(
    kb_path=Path("knowledge_base"),
    collection_name="text_chunks",
)

print(f"Indexed {indexer.stats['chunks_indexed']} chunks")
```

### 2. Indexing Codebase

```python
from rag.pipeline.code_indexer import CodeIndexer
from pathlib import Path

# Index Python codebase
indexer = CodeIndexer()
indexer.index_directory(
    code_base=Path("src"),
    collection_name="code_chunks",
    file_patterns=["**/*.py"],
    exclude_patterns=["**/__pycache__/**", "**/tests/**"],
)

print(f"Indexed {indexer.stats['chunks_indexed']} code chunks")
```

### 3. Querying the RAG System

```python
from rag.retrieval.engine import RetrievalEngine

# Create engine
engine = RetrievalEngine()

# Text query
response = engine.query(
    query="What is RSI mean reversion strategy?",
    query_type="text",  # or None for auto-detection
    top_k=5,
    filters={"domain": 2},  # Domain 2 = Technical Analysis
)

# Code query
response = engine.query(
    query="strategy implementation example",
    query_type="code",
    top_k=3,
)

# Hybrid query (both KB and code)
response = engine.query(
    query="trading system architecture design",
    query_type="hybrid",
    top_k=5,
)

# Process results
for result in response.results:
    print(f"Score: {result.score:.2f}")
    print(f"Source: {result.metadata.get('source')}")
    print(f"Text: {result.text[:200]}...")
```

### 4. Using the API Server

```bash
# Start the server
python scripts/start_rag_server.py --port 8000

# Or with auto-reload for development
python scripts/start_rag_server.py --reload
```

Access the web UI at: http://localhost:8000/ui
API documentation at: http://localhost:8000/docs

### 5. Cortex Integration

```python
from engines.cortex.core.engine import CortexEngine

# Create Cortex with RAG enabled
cortex = CortexEngine(
    nvidia_api_key="your-api-key",
    usd_code_enabled=True,
    rag_enabled=True,  # Enable RAG integration
)

# Generate hypothesis with RAG context
hypothesis = cortex.generate_hypothesis(
    market_context={"regime": "trending", "volatility": "low"},
    constraints={"max_position_pct": 0.10},
)

# Analyze code with RAG best practices
analysis = cortex.analyze_code(
    code=strategy_code,
    analysis_type="review",
)
```

## API Endpoints

### Query Endpoint

```
POST /api/v1/query
Content-Type: application/json

{
  "query": "What is momentum trading?",
  "query_type": "text",  // optional: "text", "code", "hybrid", or null for auto-detect
  "top_k": 5,
  "filters": {
    "domain": 2  // optional: filter by domain
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "...",
      "score": 0.92,
      "metadata": {
        "source": "02-technical-analysis.md",
        "domain": 2,
        "chunk_index": 0
      }
    }
  ],
  "query_type": "text",
  "total_candidates": 15,
  "latency_ms": 142.3
}
```

### Health Check

```
GET /health

{
  "status": "healthy",
  "text_embedder_available": true,
  "code_embedder_available": true,
  "chroma_persist_directory": "C:\\...\\chroma_data"
}
```

### Statistics

```
GET /stats

{
  "chroma": {
    "text_chunks": {"count": 1250, ...},
    "code_chunks": {"count": 856, ...}
  },
  "text_embedder_available": true,
  "code_embedder_available": true,
  "config": {...}
}
```

## Knowledge Base Structure

### Domain Organization

Knowledge base documents are organized by domain:

1. **01 - Market Fundamentals**: Core trading concepts
2. **02 - Technical Analysis**: Indicators, patterns
3. **03 - Market Microstructure**: Order flow, liquidity
4. **04 - Options & Derivatives**: Options strategies
5. **05 - Quantitative Methods**: Mathematical models
6. **06 - Behavioral Finance**: Psychology, biases
7. **07 - Risk Management**: Position sizing, stops
8. **08 - Strategy Design**: Strategy development
9. **09 - Backtesting & Validation**: Testing methodologies
10. **10 - Execution & Infrastructure**: Trading systems

### Metadata Fields

**Text Chunks:**
- `domain`: Domain number (1-10)
- `source`: Filename (e.g., "02-technical-analysis.md")
- `chunk_index`: Chunk position in document
- `publication_id`: Optional publication identifier

**Code Chunks:**
- `file_path`: Relative file path
- `module`: Module name
- `function_name`: Function/method name
- `class_name`: Class name (if applicable)
- `line_start`: Starting line number
- `line_end`: Ending line number

## Performance

### Embedding Performance

- **Text Embedding**: ~50-100 ms per query (API)
- **Code Embedding**: ~100-200 ms per query (7B model, API)
- **Local GPU**: 2-5x faster with RTX 2080 Ti (11GB VRAM)

### Retrieval Performance

- **Text Query**: ~100-200 ms (including embedding + search)
- **Code Query**: ~150-300 ms
- **Hybrid Query**: ~200-400 ms

### Optimization

- Use Matryoshka embeddings (768 → 384 dims) for 35x storage reduction
- Enable local embeddings for lower latency
- Adjust `top_k_retrieval` vs `top_k_rerank` for speed/quality tradeoff

## Deployment Options

### Option 1: Full Local (GPU Required)

```python
config = RAGConfig(
    use_local_embeddings=True,
    max_vram_usage_gb=9.0,  # Reserve 2GB for system
)
```

**Requirements:**
- NVIDIA GPU with 11GB+ VRAM
- CUDA 12.1+
- sentence-transformers library

### Option 2: Hybrid (Recommended)

```python
config = RAGConfig(
    use_local_embeddings=True,  # Text embedding local
    # Code embedding falls back to API (7B model)
)
```

**Requirements:**
- NVIDIA GPU with 6GB+ VRAM
- NVIDIA API key for code embedding

### Option 3: Full API

```python
config = RAGConfig(
    use_local_embeddings=False,
    nvidia_api_key="your-api-key",
)
```

**Requirements:**
- NVIDIA API key
- Internet connection

## Testing

```bash
# Run all RAG tests
pytest tests/test_rag/ -v

# Run only unit tests (fast)
pytest tests/test_rag/ -v -m "not integration"

# Run integration tests (requires ChromaDB data)
pytest tests/test_rag/ -v -m integration
```

## Troubleshooting

### Common Issues

**1. VRAM Out of Memory**
```python
# Solution: Use API embeddings or reduce VRAM limit
config.use_local_embeddings = False
# OR
config.max_vram_usage_gb = 6.0
```

**2. ChromaDB Not Found**
```python
# Solution: Check persist directory
config = get_config()
print(config.chroma_persist_directory)
# Ensure directory exists and is writable
```

**3. No Results Returned**
```python
# Solution: Check if database is populated
from rag.vectordb.chroma_client import ChromaClient
client = ChromaClient()
stats = client.get_collection_stats("text_chunks")
print(f"Text chunks: {stats['count']}")
```

**4. Slow Query Performance**
```python
# Solution: Reduce top_k or enable local embeddings
config.top_k_retrieval = 10  # Reduce from default 20
config.use_local_embeddings = True  # If GPU available
```

## Future Enhancements

1. **Reranking**: Implement NVIDIA NeMo Retriever reranking model
2. **Hybrid Search**: Combine semantic + keyword search
3. **Caching**: Add diskcache for frequent queries
4. **Multi-modal**: Support images, charts, tables
5. **Fine-tuning**: Domain-specific embedding fine-tuning
6. **Real-time Updates**: Incremental indexing for new documents

## Cost Estimate

**NVIDIA API Pricing** (as of 2025):
- Text Embedding: ~$0.02 per 1M tokens
- Code Embedding: ~$0.10 per 1M tokens
- LLM (70B): ~$0.30 per 1M tokens

**Monthly Estimate** (1000 queries/day):
- Text queries: ~$0.60/month
- Code queries: ~$3.00/month
- LLM synthesis: ~$9.00/month
- **Total: ~$13/month**

**With Local Embeddings**: ~$9/month (LLM only)

## References

- [NVIDIA Build Platform](https://build.nvidia.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [NeMo Retriever Technical Report](https://arxiv.org/abs/...)
- [Matryoshka Embeddings Paper](https://arxiv.org/abs/2205.13147)
