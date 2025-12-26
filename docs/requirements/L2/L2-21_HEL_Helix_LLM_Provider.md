# L2-21 — HEL — Helix LLM Provider

## 1. Identifier
- **ID:** `L2-21`
- **Component:** HEL — Helix LLM Provider
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Unified adapter layer for LLM calls (request/response normalization).
- Model routing between heavy/fast models based on task_type and profile.
- Allowlist enforcement, safety profiles, deterministic mode.
- Usage metering and drift telemetry emission.

### Out of scope
- User interface / chat UX (not a UI).
- Training/fine-tuning (owned by LEARN).

## 3. Responsibilities
- Provide a single generate() API for Cortex/Synapse/CodeGen.
- Normalize provider-specific responses into standard AIResponse schema.
- Enforce model allowlist and safety policy from SEC/GOV/CFG.
- Emit token usage, latency, refusal rates, and drift signals.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

@dataclass(frozen=True)
class AIResponse:
    model_used: str
    latency_ms: float
    token_usage: Dict[str, int]
    output: Dict[str, Any]          # structured output, if requested
    text: Optional[str]             # raw text
    citations: List[Dict[str, Any]]
    safety_findings: List[Dict[str, Any]]
    request_hash: str

class HelixProvider:
    def generate(self, messages: List[Dict[str, str]], model_id: Optional[str] = None, opts: Dict[str, Any] = None) -> AIResponse: ...
    def list_models(self) -> List[str]: ...
    def health(self) -> Dict[str, Any]: ...
```

### Events
**Consumes**
- ordinis.ai.request (optional; if requests are event-driven)

**Emits**
- ordinis.ai.response
- ordinis.ai.drift.signal
- ordinis.ops.ai.denied

### Schemas
- EVT-DATA-021 AIRequest/AIResponse
- Safety findings schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_HEL_MODEL_DENIED` | Model not allowlisted | No |
| `E_HEL_TIMEOUT` | Provider call timed out | Yes |
| `E_HEL_RATE_LIMIT` | Provider rate limited | Yes |
| `E_HEL_PROVIDER_ERROR` | Provider error | Yes |

## 5. Dependencies
### Upstream
- SEC for model allowlist and secret keys
- CFG for routing and defaults
- CB for circuit breaking provider calls
- OBS for tracing and metrics

### Downstream
- CTX (Cortex)
- SYN (Synapse) for embeddings/LLM answer modes
- CG (CodeGen) for code generation

### External services / libraries
- NVIDIA API client or HTTP endpoint adapter (default per prior design)
- Optional local inference runtime for small models (later)

## 6. Data & State
### Storage
- Optional request/response artifact storage (redacted) for reproducibility and eval; default off in prod.

### Caching
- Prompt/response cache keyed by request_hash (optional; disabled for non-deterministic or sensitive requests).

### Retention
- If stored, AI artifacts retained short (e.g., 30 days) unless explicitly marked for eval/training.

### Migrations / Versioning
- AIResponse schema versioned; backward compatible additions only.

## 7. Algorithms / Logic
### Routing
- Inputs include `task_type` and `safety_profile`.
- ModelRouter selects:
  - heavy model for codegen/research synthesis
  - fast model for narration/explanations
- Explicit model_id overrides are allowed only if allowlisted.

### Deterministic mode
- Default for codegen and eval: temperature=0.
- request_hash includes model_id + messages + opts + Synapse context refs (if any).

### Drift telemetry (minimum)
- Track:
  - refusal rate
  - latency distribution
  - output embedding distribution shift (if embeddings available)
  - token usage drift

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| HEL_DEFAULT_MODEL_HEAVY | string | nvidia/nemotron-49b | P0 | Placeholder id; must match allowlist. |
| HEL_DEFAULT_MODEL_FAST | string | nvidia/nemotron-8b | P0 | Placeholder id; must match allowlist. |
| HEL_TIMEOUT_MS | int | 30000 | P0 | Provider timeout. |
| HEL_DETERMINISTIC_DEFAULT | bool | true | P0 | For codegen/evals. |
| HEL_CACHE_ENABLE | bool | false | P0 | Response cache. |

**Environment variables**
- `NVIDIA_API_KEY`
- `ORDINIS_HEL_DEFAULT_MODEL_HEAVY`

## 9. Non-Functional Requirements
- p95 Helix overhead (excluding provider inference): < 10ms.
- Provider timeouts enforced; no blocking beyond timeout.
- All AI calls are auditable: model_used, request_hash, and safety findings captured.

## 10. Observability
**Metrics**
- `hel_requests_total{task_type,model,success}`
- `hel_latency_ms{model}`
- `hel_tokens_total{model,type=prompt|completion}`
- `hel_refusals_total{model}`
- `hel_drift_score{model,metric}`

**Alerts**
- Provider error spikes / rate limits
- Drift score breaches

## 11. Failure Modes & Recovery
- Provider unavailable: CB opens; degrade AI features; do not block trading loop (unless AI is required for dev-plane tasks only).
- Allowlist denies: return E_HEL_MODEL_DENIED; escalate to config fix.
- Timeout/rate limit: retry with backoff; optionally fall back to fast model if allowed.

## 12. Test Plan
### Unit tests
- Request hashing determinism.
- Allowlist enforcement.
- Routing decisions for task_type.

### Integration tests
- Cortex calls Helix and receives normalized AIResponse.

### Acceptance criteria
- Helix returns stable schema and captures model_used and token_usage for every call.

## 13. Open Questions / Risks
- Finalize exact NVIDIA model IDs and verify availability/licensing in the target environment.
- Decide whether to store prompts/responses by default in dev vs prod.

## 14. Traceability
### Parent
- DB1 System SRS: Helix abstracts LLM calls (not a UI).

### Children / Related
- [L3-08 HEL-MODEL-ROUTER — Model Routing & Allowlisting](../L3/L3-08_HEL-MODEL-ROUTER_Model_Routing_and_Allowlisting.md)
- [L2-23 CTX — Cortex](L2-23_CTX_Cortex_LLM_Engine.md)

### Originating system requirements
- HEL-IF-001, HEL-FR-001..004
