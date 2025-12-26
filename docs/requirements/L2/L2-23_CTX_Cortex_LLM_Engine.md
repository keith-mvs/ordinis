# L2-23 — CTX — Cortex LLM Engine

## 1. Identifier
- **ID:** `L2-23`
- **Component:** CTX — Cortex LLM Engine
- **Level:** L2

## 2. Purpose / Scope
### In scope
- LLM-driven reasoning, code analysis, and research synthesis for development operations.
- Strictly non-authoritative for trading decisions in DB1 (cannot emit executable signals/orders).
- Structured outputs with schemas and model_used metadata.
- Synapse-grounded prompting with citations.

### Out of scope
- Trading signal generation (explicitly out-of-scope).
- Autonomous code deployment without governance/human approval (CG/GOV).

## 3. Responsibilities
- Own task workflows (analyze_code, synthesize_research, critique, plan) and output schemas.
- Call Helix for model inference and Synapse for retrieval grounding.
- Enforce grounding and refusal behavior via GOV/SEC policies.
- Emit AI audit records including prompts (hashed), citations, and model_used.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

@dataclass(frozen=True)
class CortexResult:
    task: str
    model_used: str
    output: Dict[str, Any]
    citations: List[Dict[str, Any]]
    latency_ms: float
    request_hash: str

class Cortex:
    def analyze_code(self, code: str, analysis_type: str, context: Dict[str, Any] = None) -> CortexResult: ...
    def synthesize_research(self, query: str, sources: List[str] = None, context: Dict[str, Any] = None) -> CortexResult: ...
    def plan_task(self, goal: str, context: Dict[str, Any] = None) -> CortexResult: ...
```

### Events
**Consumes**
- ordinis.ai.request (optional)
- ordinis.cg.patch.proposed (optional critique workflow)

**Emits**
- ordinis.ai.response
- ordinis.ai.refusal
- ordinis.ai.audit

### Schemas
- CortexResult schema per task type (see L3-10)
- AI audit schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_CTX_INVALID_TASK` | Unknown analysis task or schema | No |
| `E_CTX_OUTPUT_SCHEMA_INVALID` | Model output failed schema validation | No |
| `E_CTX_GROUNDING_REQUIRED` | Task requires citations but none available | No |

## 5. Dependencies
### Upstream
- HEL for inference
- SYN for retrieval grounding
- GOV for model/prompt policies
- SEC for redaction and allowlists

### Downstream
- CG consumes Cortex results for code patch generation/review
- ANA may consume Cortex for narration (optional)

### External services / libraries
- None beyond HEL and SYN

## 6. Data & State
### Storage
- Optional: persist CortexResult artifacts (redacted) for eval and reproducibility.

### Caching
- Request_hash cache to dedupe identical dev tasks (optional).

### Retention
- If stored, dev-plane outputs retained per LEARN policy; default 30-90 days.

### Migrations / Versioning
- Output schemas are versioned per task; changes require explicit version bump and CI updates.

## 7. Algorithms / Logic
### Workflow model
- Each Cortex method maps to a workflow:
  1. Build prompt from PromptLibrary template
  2. Optionally call Synapse.retrieve for grounding
  3. Call Helix.generate with deterministic opts (default)
  4. Validate structured output against task schema
  5. Emit audit record with hashes and citations

### Guardrails (DB1 hard requirements)
- Non-authoritative: Cortex must not emit trading instructions as executable events.
- Grounding: tasks that claim factual repo/doc assertions must include citations from Synapse.
- Output validation: invalid outputs are rejected and optionally re-asked once with self-correction prompt.

### Edge cases
- Synapse returns empty: either refuse (if grounding required) or proceed with explicit `uncertain=true`.
- Helix refusal: propagate refusal with reason codes; do not retry unless policy allows.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| CTX_REQUIRE_GROUNDING | bool | true | P0 | For code/repo claims. |
| CTX_MAX_SELF_REPAIR | int | 1 | P0 | Output schema self-correction attempts. |
| CTX_TASK_ALLOWLIST | list[str] | ["analyze_code","synthesize_research","plan_task"] | P0 | Allowed tasks. |
| CTX_DEFAULT_TASK_MODEL | string | helix_router | P0 | Use Helix router by default. |

## 9. Non-Functional Requirements
- Schema-valid outputs: > 99% after self-repair attempt on known eval set.
- Latency: p95 < provider-dependent; Cortex adds < 20ms overhead excluding LLM inference.
- No leakage: prompts and outputs redacted before persistence/audit.

## 10. Observability
**Metrics**
- `ctx_tasks_total{task,result}`
- `ctx_latency_ms{task}`
- `ctx_schema_fail_total{task}`
- `ctx_grounding_empty_total{task}`

**Alerts**
- Schema failure spike
- Grounding empty spike (index stale)

## 11. Failure Modes & Recovery
- Invalid output schema: self-repair once; if still invalid, fail task with E_CTX_OUTPUT_SCHEMA_INVALID.
- Helix provider down: fail dev-plane tasks; do not block trading loop.
- Synapse index missing: refuse if grounding required; otherwise proceed with explicit uncertainty.

## 12. Test Plan
### Unit tests
- Prompt template rendering.
- Output schema validation and self-repair behavior (with mocked Helix).

### Integration tests
- Full Cortex workflow with Synapse retrieval (local index) and Helix mocked responses.

### Acceptance criteria
- For a curated eval set, Cortex outputs pass schema validation and include citations when required.

## 13. Open Questions / Risks
- Exact structured schemas for each task (define in L3-10).
- How to log prompts safely: store hashes + retrieval refs vs full prompts (default).

## 14. Traceability
### Parent
- DB1 System SRS: Cortex provides LLM-driven reasoning and code analysis.

### Children / Related
- [L3-10 CTX-WORKFLOWS — Cortex Workflows & Output Schemas](../L3/L3-10_CTX-WORKFLOWS_Cortex_Workflows_and_Output_Schemas.md)
- [L2-22 SYN — Synapse](L2-22_SYN_Synapse_RAG_Engine.md)
- [L2-21 HEL — Helix](L2-21_HEL_Helix_LLM_Provider.md)

### Originating system requirements
- CTX-IF-001, CTX-FR-001..004
