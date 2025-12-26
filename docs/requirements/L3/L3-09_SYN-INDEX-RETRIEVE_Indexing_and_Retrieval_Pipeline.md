# L3-09 — SYN-INDEX-RETRIEVE — Indexing & Retrieval Pipeline

## 1. Identifier
- **ID:** `L3-09`
- **Component:** SYN-INDEX-RETRIEVE — Indexing & Retrieval Pipeline
- **Level:** L3

## 2. Purpose / Scope
### In scope
- Chunking rules and document locator format.
- Embedding model usage and index_version calculation.
- Vector store layout and manifest schema.
- Retrieval scoring, filtering, and citation formatting.

### Out of scope
- External connectors (Confluence/Drive) (P2).
- Internet browsing retrieval (explicitly out of DB1).

## 3. Responsibilities
- Define deterministic chunking and hashing rules.
- Define index_version derivation and when reindex is required.
- Define retrieval scoring normalization and thresholding.
- Define citation format used by Cortex and CodeGen.

## 4. Interfaces
### Public APIs / SDKs
Synapse APIs are exposed by [L2-22 SYN](../L2/L2-22_SYN_Synapse_RAG_Engine.md). This spec defines pipeline internals.

**Document locator format**
- `locator = "<path>#L<start>-L<end>"` for source code/text
- `locator = "<path>#<section_id>"` for markdown headings if parsed

**Chunk hash**
- `chunk_hash = H(doc_id + locator + normalized_text + tags + chunking_params_hash)`

### Events
**Consumes**
_None._

**Emits**
_None._

### Schemas
- IndexManifest schema: {index_version, embed_model_id, chunking_params, chunk_count, hashes}
- ChunkMetadata schema: {doc_id, locator, tags, chunk_hash}

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_SYN_CHUNK_PARSE_FAILED` | Unable to parse/segment a file | No |
| `E_SYN_EMBED_FAILED` | Embedding call failed | Yes |
| `E_SYN_INDEX_VERSION_MISMATCH` | Index version required but missing | No |

## 5. Dependencies
### Upstream
- CFG for chunking parameters and source roots
- HEL or local embedder for embeddings
- SEC/GOV for sensitivity tagging rules

### Downstream
- Synapse runtime
- Cortex and CodeGen consumers

### External services / libraries
- FAISS/local vector store backend

## 6. Data & State
### Storage
- Index directory contains:
- - `manifest.json` (IndexManifest)
- - `vectors.faiss` (FAISS index)
- - `chunks.parquet` (ChunkMetadata table) OR `chunks.jsonl`

### Caching
- LRU cache of query embeddings and recent retrieval results (optional).

### Retention
- Keep last N index versions (default 3) for rollback and reproducibility.

### Migrations / Versioning
- Any change to embed_model_id or chunking params increments index_version.

## 7. Algorithms / Logic
### Deterministic chunking rules (DB1 baseline)
- Normalize line endings to LF.
- Strip trailing whitespace.
- Chunk by tokens if tokenizer available; else by characters with fixed size.
- Respect natural boundaries:
  - markdown headings
  - code function/class boundaries when parsable (best-effort)
- Overlap fixed at SYN_CHUNK_OVERLAP.

### Retrieval scoring
- Use cosine similarity (or FAISS default) and return raw score.
- Apply optional filters:
  - source allowlist (path prefixes)
  - sensitivity tag check (deny restricted)
  - minimum score threshold (SYN_MIN_SCORE)
- Return top-k after filtering; if <k, return fewer.

### Citation formatting
- Each returned chunk includes locator and doc_id.
- Cortex must reference locator for any claim that originates from retrieval.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| SYN_CHUNK_SIZE | int | 1200 | P0 | tokens/chars (choose and fix). |
| SYN_CHUNK_OVERLAP | int | 200 | P0 | Overlap. |
| SYN_MIN_SCORE | float | 0.2 | P0 | Minimum similarity threshold. |
| SYN_TOP_K | int | 8 | P0 | Retrieval depth. |

## 9. Non-Functional Requirements
- Index build is deterministic given identical source set and config.
- Locator correctness: must allow engineers to navigate to exact content quickly.

## 10. Observability
**Metrics**
- `syn_chunks_total{tag}`
- `syn_retrieve_results_count`
- `syn_retrieve_empty_total`
- `syn_retrieve_min_score` (distribution)

**Alerts**
- Frequent empty retrievals for known queries indicates stale index or bad embed model.

## 11. Failure Modes & Recovery
- Tokenizer mismatch: chunking changes unexpectedly; treat as config change and reindex.
- Embed failure: retry/backoff; if persistent, fail indexing and keep last known good index.

## 12. Test Plan
### Unit tests
- Locator generation for code and markdown.
- Chunk hash determinism.
- Filter enforcement (allowlist/denylist).

### Integration tests
- Build index on fixture repo and assert retrieval returns expected chunk for known queries.

### Acceptance criteria
- Index rebuild on identical sources yields identical manifest hashes.

## 13. Open Questions / Risks
- Tokenization approach selection for DB1 (ship tokenizer or char-based).
- How to tag sensitivity automatically (path-based vs content-based classifier).

## 14. Traceability
### Parent
- [L2-22 SYN — Synapse RAG Engine](../L2/L2-22_SYN_Synapse_RAG_Engine.md)

### Originating system requirements
- SYN-FR-001..003
