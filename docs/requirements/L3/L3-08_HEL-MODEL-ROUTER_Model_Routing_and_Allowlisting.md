# L3-08 — HEL-MODEL-ROUTER — Model Routing & Allowlisting

## 1. Identifier
- **ID:** `L3-08`
- **Component:** HEL-MODEL-ROUTER — Model Routing & Allowlisting
- **Level:** L3

## 2. Purpose / Scope
### In scope
- Task-aware model selection rules (heavy vs fast).
- Allowlist enforcement and fallback strategy.
- Safety profile mapping (temperature, max tokens, refusal policy).
- Budget controls (token caps per task and per run).

### Out of scope
- Provider-specific API integration (HEL owns).
- Governance policy definition (GOV owns overall policy, router enforces).

## 3. Responsibilities
- Define the routing decision algorithm and inputs.
- Enforce allowlists from SEC/CFG and deny unknown models.
- Provide deterministic routing decisions for audit/reproducibility.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Literal

TaskType = Literal["codegen","analysis","narrative","retrieval_embed","eval"]

@dataclass(frozen=True)
class RouteDecision:
    model_id: str
    safety_profile: str
    opts: Dict[str, object]
    reason: str

class ModelRouter:
    def route(self, task_type: TaskType, requested_model: Optional[str], profile: str, context: Dict[str, object]) -> RouteDecision: ...
```

### Events
**Consumes**
_None._

**Emits**
- ordinis.ops.ai.denied
- ordinis.ops.ai.routed

### Schemas
- RouteDecision schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_ROUTER_MODEL_DENIED` | Requested model not in allowlist | No |
| `E_ROUTER_NO_FALLBACK` | No fallback model available | No |

## 5. Dependencies
### Upstream
- CFG provides default heavy/fast models and budgets
- SEC provides allowlist
- GOV may provide additional deny rules

### Downstream
- HEL uses router to select model_id and opts

### External services / libraries
_None._

## 6. Data & State
### Storage
- No persistent state required; decisions are emitted to audit/ops logs.

### Caching
- In-memory cache of allowlist and resolved model mapping per profile.

### Retention
- Routing events retained per ops log retention; audit references stored by GOV.

### Migrations / Versioning
- RouteDecision schema versioned; additive changes only.

## 7. Algorithms / Logic
### Routing rules (DB1 baseline)
- Inputs: task_type, requested_model (optional), env profile, context tags.
- If requested_model is set:
  - if allowlisted → use it
  - else → deny with E_ROUTER_MODEL_DENIED
- Else select default:
  - codegen/eval → heavy model
  - analysis/narrative → fast model
  - retrieval_embed → embedding model (may be separate)
- Apply safety_profile:
  - codegen/eval: deterministic (temperature=0), max_tokens capped, no tools unless explicit
  - narrative: temperature small but bounded (default 0.2)
  - analysis: deterministic by default in DB1

### Budgets
- Per-task token caps and per-run daily caps.
- If budget exceeded: deny or degrade to fast model depending on config.

### Edge cases
- Heavy model unavailable: fallback to fast model only for non-codegen tasks; codegen denies by default (avoid bad patches).

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| HEL_DEFAULT_MODEL_HEAVY | string | nvidia/nemotron-49b | P0 | Example. |
| HEL_DEFAULT_MODEL_FAST | string | nvidia/nemotron-8b | P0 | Example. |
| HEL_DEFAULT_MODEL_EMBED | string | nvidia/embed-v1 | P0 | Example. |
| HEL_TASK_TOKEN_CAPS | map | see defaults | P0 | token caps per task_type. |
| HEL_RUN_TOKEN_CAP | int | 500000 | P0 | total tokens per run/day. |

## 9. Non-Functional Requirements
- Routing is O(1) and does not call external services.
- Allowlist enforcement cannot be bypassed by requested_model.

## 10. Observability
**Metrics**
- `router_decisions_total{task_type,model,reason}`
- `router_denied_total{task_type}`
- `router_budget_exceeded_total{task_type}`

**Audit**
- RouteDecision recorded in GOV audit for every HEL call.

## 11. Failure Modes & Recovery
- Allowlist misconfig empties allowlist: all requests denied; degrade dev-plane; alert.
- Budget misconfig too low: frequent denies; alert.

## 12. Test Plan
### Unit tests
- Task_type routing to heavy/fast models.
- Allowlist enforcement.
- Budget exceeded behavior.

### Acceptance criteria
- For each task_type, router produces deterministic RouteDecision and enforces allowlists.

## 13. Open Questions / Risks
- Final model IDs and embed model selection for target environment.
- Temperature defaults for narrative tasks in DB1 (keep low).

## 14. Traceability
### Parent
- [L2-21 HEL — Helix LLM Provider](../L2/L2-21_HEL_Helix_LLM_Provider.md)

### Originating system requirements
- HEL-FR-001..004, SEC-FR-003
