# L2-22 — SYN — Synapse RAG Engine

## 1. Identifier
- **ID:** `L2-22`
- **Component:** SYN — Synapse RAG Engine
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Retrieval-augmented grounding for Cortex/CodeGen prompts.
- Indexing of repo/docs/runbooks and retrieval with citations.
- Sensitivity tagging and policy-enforced retrieval constraints.
- Incremental indexing and deterministic retrieval baselines (DB1 FAISS/local).

### Out of scope
- Internet browsing retrieval (no network RAG in DB1 unless explicitly added).
- Enterprise document connectors (Confluence/Drive) (P2).

## 3. Responsibilities
- Own indexing pipeline (chunking → embedding → vector store).
- Serve retrieve(query, context) returning citation-grade snippets.
- Enforce source allowlists/deny lists and sensitivity tags.
- Emit retrieval metrics and coverage stats.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    locator: str           # file path + line range / section
    text: str
    score: float
    tags: Dict[str, str]

@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunks: List[RetrievedChunk]
    latency_ms: float
    index_version: str

class Synapse:
    def index(self, sources: List[str]) -> Dict[str, Any]: ...
    def retrieve(self, query: str, context: Dict[str, Any], k: int = 8) -> RetrievalResult: ...
    def health(self) -> Dict[str, Any]: ...
```

### Events
**Consumes**
- ordinis.syn.index.requested
- ordinis.ai.request (when retrieval is part of workflow)

**Emits**
- ordinis.syn.index.completed
- ordinis.syn.retrieve.completed
- ordinis.syn.retrieve.denied

### Schemas
- RetrievedChunk schema
- RetrievalResult schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_SYN_INDEX_FAILED` | Index build failed | Yes |
| `E_SYN_RETRIEVE_DENIED` | Policy denied retrieval of requested sources | No |
| `E_SYN_VECTORSTORE_UNAVAILABLE` | Vector store unavailable | Yes |

## 5. Dependencies
### Upstream
- CFG for source roots, chunking params, vector store path
- SEC/GOV for sensitivity policy enforcement
- HEL for embedding model calls (or local embedder)
- PERSIST for storing index artifacts

### Downstream
- CTX (Cortex) for grounded reasoning
- CG (CodeGen) for repo context selection

### External services / libraries
- FAISS (DB1 local vector index) or equivalent
- Filesystem access to repo/docs

## 6. Data & State
### Storage
- Index artifacts stored under `./data/synapse/index/` (DB1) and referenced via PERSIST artifacts table.
- Chunk metadata store (SQLite `syn_chunks`) optional; file-based manifest acceptable DB1.

### Caching
- In-process LRU cache for recent retrieval results (by query hash) (optional).

### Retention
- Index artifacts retained until superseded; keep last N index versions (configurable).

### Migrations / Versioning
- Index version increments when chunking rules or embedding model changes.

## 7. Algorithms / Logic
### Indexing pipeline
1. Enumerate source files under allowed roots.
2. Chunk content deterministically (fixed size + overlap + semantic boundaries where possible).
3. Compute embeddings for each chunk via HEL (embed model) or local embedder.
4. Build vector index (FAISS) and persist:
   - vectors
   - chunk metadata (doc_id, locator, tags, hash)
   - index_version

### Retrieval pipeline
1. Validate query and caller context (policy, sensitivity).
2. Embed query.
3. Vector search top-k.
4. Package results with locator and confidence scores.

### Edge cases
- Repo changes during indexing: use file hashes; reindex incremental only changed files.
- Large files: chunk streaming; hard cap chunk count per file to prevent index blowup.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| SYN_SOURCE_ROOTS | list[str] | ["./docs","./src"] | P0 | Roots to index. |
| SYN_CHUNK_SIZE | int | 1200 | P0 | Characters or tokens (choose and fix). |
| SYN_CHUNK_OVERLAP | int | 200 | P0 | Overlap size. |
| SYN_TOP_K | int | 8 | P0 | Retrieval depth. |
| SYN_INDEX_PATH | string | ./data/synapse/index | P0 | Local index path. |
| SYN_SENSITIVITY_ENABLE | bool | true | P0 | Enforce sensitivity tags. |

**Environment variables**
- `ORDINIS_SYN_INDEX_PATH`

## 9. Non-Functional Requirements
- Retrieval latency: p95 < 200ms for local FAISS index (<1M chunks).
- Indexing determinism: identical inputs yield identical chunk hashes and index_version.
- Citation-grade results: every returned snippet includes doc locator.

## 10. Observability
**Metrics**
- `syn_index_chunks_total`
- `syn_index_latency_ms`
- `syn_retrieve_latency_ms`
- `syn_retrieve_denied_total{reason}`
- `syn_retrieval_precision_at_k` (requires labeled eval set; LEARN)

**Alerts**
- Index build failures
- Retrieval denial spikes (misconfigured policies)

## 11. Failure Modes & Recovery
- Embedding model unavailable: fall back to lexical search (optional DB1), or deny retrieval with clear error.
- Vector store corrupt: rebuild index; keep last known good index for fallback.
- Policy denies: return empty results and emit denied event (do not leak restricted content).

## 12. Test Plan
### Unit tests
- Chunking determinism and locator correctness.
- Sensitivity tagging and enforcement.

### Integration tests
- Cortex workflow calls Synapse and receives cited chunks.

### Acceptance criteria
- Retrieval returns relevant snippets for known queries against repo/docs and includes correct citations.

## 13. Open Questions / Risks
- Embed model choice and deployment (local vs remote).
- Token-based chunking vs character-based (token-based recommended but requires tokenizer).

## 14. Traceability
### Parent
- DB1 System SRS: Synapse retrieves contextual knowledge to ground LLM outputs.

### Children / Related
- [L3-09 SYN-INDEX-RETRIEVE — Indexing & Retrieval Pipeline](../L3/L3-09_SYN-INDEX-RETRIEVE_Indexing_and_Retrieval_Pipeline.md)
- [L2-21 HEL — Helix Provider](L2-21_HEL_Helix_LLM_Provider.md)

### Originating system requirements
- SYN-IF-001, SYN-FR-001..003
