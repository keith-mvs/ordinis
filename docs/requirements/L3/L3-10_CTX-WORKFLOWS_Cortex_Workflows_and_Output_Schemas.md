# L3-10 — CTX-WORKFLOWS — Cortex Workflows & Output Schemas

## 1. Identifier
- **ID:** `L3-10`
- **Component:** CTX-WORKFLOWS — Cortex Workflows & Output Schemas
- **Level:** L3

## 2. Purpose / Scope
### In scope
- Definition of Cortex workflows (analyze_code, synthesize_research, plan_task).
- Structured output schemas for each workflow and schema validation rules.
- Grounding/citation requirements per workflow.
- Self-repair (one retry) and refusal semantics.

### Out of scope
- Helix provider integration (HEL).
- Synapse indexing internals (SYN).

## 3. Responsibilities
- Define workflow-specific prompts (as templates) and required inputs.
- Define JSON schemas for outputs and guarantee they are machine-consumable.
- Define which workflows require citations and the minimum citation coverage.

## 4. Interfaces
### Public APIs / SDKs
Cortex exposes APIs in [L2-23 CTX](../L2/L2-23_CTX_Cortex_LLM_Engine.md). This spec defines the workflow contracts.

### Workflow: analyze_code
**Input**
- `code` (string or file refs)
- `analysis_type`: `bug_find|security_review|performance_review|architecture_review`
- `context` (optional): retrieved chunks + metadata

**Output schema (excerpt)**
```json
{
  "schema_version": "1.0.0",
  "findings": [
    {
      "severity": "low|medium|high",
      "title": "string",
      "evidence": [{"locator":"...","quote":"..."}],
      "recommendation": "string"
    }
  ],
  "risk": {"score": 0.0, "notes": "string"},
  "next_actions": ["string"]
}
```

### Workflow: synthesize_research
**Output schema (excerpt)**
```json
{
  "schema_version": "1.0.0",
  "answer": "string",
  "claims": [
    {"claim":"string","citations":[{"doc_id":"...","locator":"..."}]}
  ],
  "uncertainties": ["string"]
}
```

### Workflow: plan_task
**Output schema (excerpt)**
```json
{
  "schema_version": "1.0.0",
  "plan": [{"step":1,"action":"string","owner":"human|system","artifacts":["string"]}],
  "risks": ["string"]
}
```

### Events
**Consumes**
_None._

**Emits**
_None._

### Schemas
- analyze_code output schema v1
- synthesize_research output schema v1
- plan_task output schema v1

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_CTX_SCHEMA_INVALID` | Workflow output failed schema validation | No |
| `E_CTX_CITATIONS_MISSING` | Citations required but missing | No |
| `E_CTX_REFUSAL` | Model refused the task | No |

## 5. Dependencies
### Upstream
- SYN retrieval results for grounding
- HEL model inference
- GOV policy for task allowlist and safety rules

### Downstream
- CG uses analyze_code + plan_task outputs
- ANA may use synthesize_research for narrative (optional)

### External services / libraries
_None._

## 6. Data & State
### Storage
- Workflow outputs stored as artifacts optionally; otherwise ephemeral.

### Caching
- Request_hash cache optional (dedupe identical workflow inputs).

### Retention
- If stored, outputs retained short (30-90 days) unless marked for eval/training.

### Migrations / Versioning
- Schema versions are explicit; any non-additive change increments major.

## 7. Algorithms / Logic
### Citation requirements
- analyze_code:
  - evidence entries must reference locators if claim is about repo content.
  - minimum: 1 evidence item per high-severity finding.
- synthesize_research:
  - each claim must have ≥1 citation unless explicitly listed in uncertainties.
- plan_task:
  - citations optional; include when referencing repo constraints.

### Self-repair
- If output fails schema validation:
  - run at most 1 repair attempt with the validation errors embedded in prompt.
  - if still invalid: fail with E_CTX_SCHEMA_INVALID.

### Refusals
- If model refuses:
  - return structured refusal object with `refusal_reason` and `policy_ref`.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| CTX_MAX_SELF_REPAIR | int | 1 | P0 | Max repair attempts. |
| CTX_REQUIRE_GROUNDING | bool | true | P0 | Require citations for code claims. |
| CTX_MIN_CITATIONS_PER_CLAIM | int | 1 | P0 | synthesize_research. |

## 9. Non-Functional Requirements
- Schema-valid outputs ≥ 99% on eval set after repair.
- Citations coverage enforced for grounded workflows.

## 10. Observability
**Metrics**
- `ctx_schema_valid_total{workflow}`
- `ctx_citations_missing_total{workflow}`
- `ctx_refusal_total{workflow,reason}`

## 11. Failure Modes & Recovery
- Synapse returns empty when required: refuse with E_CTX_CITATIONS_MISSING.
- Schema invalid after repair: fail and record validation errors (redacted).

## 12. Test Plan
### Unit tests
- Validate sample outputs against schemas.
- Ensure self-repair is attempted exactly once.

### Integration tests
- Run workflows with mocked Helix outputs and real schema validation.

### Acceptance criteria
- Downstream consumers (CG) can parse outputs without custom string parsing.

## 13. Open Questions / Risks
- Exact severity definitions and risk scoring rubric for analyze_code.
- How to store evidence quotes without violating the 25-word quoting policy (for internal system it's fine; for user-facing output still constrain).

## 14. Traceability
### Parent
- [L2-23 CTX — Cortex](../L2/L2-23_CTX_Cortex_LLM_Engine.md)

### Originating system requirements
- CTX-FR-001..004
