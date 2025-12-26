# L2-24 — CG — CodeGenService

## 1. Identifier
- **ID:** `L2-24`
- **Component:** CG — CodeGenService
- **Level:** L2

## 2. Purpose / Scope
### In scope
- LLM-assisted code change proposals (patch generation) for the repo.
- Repo-context retrieval via Synapse; generation via Cortex/Helix.
- Static scanning (secrets/license/path allowlists) and test execution before approval.
- Human-in-loop approval workflow integrated with GOV.

### Out of scope
- Autonomous deployment to production (explicitly out-of-scope).
- Direct modification of user secrets or external accounts.

## 3. Responsibilities
- Own propose_change() API producing a patch + validation results.
- Guarantee safety controls: file allowlists, forbidden patterns, no secret leakage.
- Run lint/unit tests in a sandboxed environment and record results as artifacts.
- Emit patch proposal events and audit records for all generated changes.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class PatchProposal:
    proposal_id: str
    base_commit: str
    files_changed: List[str]
    diff_unified: str
    tests_run: List[str]
    test_results: Dict[str, Any]
    static_scan: Dict[str, Any]
    model_used: str
    citations: List[Dict[str, Any]]

class CodeGenService:
    def propose_change(self, prompt: str, files: List[str], context: Dict[str, Any] = None) -> PatchProposal: ...
```

### Events
**Consumes**
- ordinis.cg.patch.requested
- ordinis.gov.change.approved|denied

**Emits**
- ordinis.cg.patch.proposed
- ordinis.cg.patch.approved
- ordinis.cg.patch.denied
- ordinis.cg.patch.tests.completed

### Schemas
- PatchProposal schema
- Static scan report schema
- Test report schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_CG_FORBIDDEN_PATH` | Proposed change touches forbidden file/path | No |
| `E_CG_SECRET_DETECTED` | Secrets detected in diff or context | No |
| `E_CG_TESTS_FAILED` | Generated patch failed tests | No |
| `E_CG_GOV_DENIED` | Governance denied patch promotion | No |

## 5. Dependencies
### Upstream
- SYN for retrieval context
- CTX/HEL for generation and code analysis
- SEC for redaction and secret scanning
- GOV for approval workflow and audit
- OBS for logs/traces

### Downstream
- Repo/workspace (filesystem) for applying patches in a sandbox
- CI harness for tests/lint (local runner in DB1)

### External services / libraries
- Git CLI or library for diff/apply
- Linters/test frameworks (pytest, ruff/flake8, mypy optional)

## 6. Data & State
### Storage
- Persist PatchProposal artifacts and test reports via PERSIST artifact store.
- Optional `patch_proposals` table for workflow state.

### Caching
- Cache Synapse retrieval context for a proposal to keep citations stable.

### Retention
- Patch proposals retained 180 days (default); longer if merged into main branch with commit reference.

### Migrations / Versioning
- PatchProposal schema versioned; backward compatible additions only.

## 7. Algorithms / Logic
### Patch lifecycle (DB1)
1. Receive prompt + file list.
2. Synapse retrieves relevant context (docs/code snippets) and returns citations.
3. Cortex/Helix generates a candidate patch (unified diff).
4. Static scans:
   - file/path allowlist
   - secret detection
   - license header checks
   - forbidden imports/syscalls (optional)
5. Apply patch in sandbox checkout and run:
   - formatter/lint
   - unit tests (targeted + full depending on config)
6. If all green, emit `cg.patch.proposed` and await GOV/HumanGate approval.

### Edge cases
- Non-deterministic tests: rerun once; if still flaky, mark as `flaky` and require manual approval.
- Large diffs: cap diff size; require human review for >N LOC changes.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| CG_ALLOWLIST_PATHS | list[str] | ["src/","docs/"] | P0 | Allowed to change. |
| CG_DENYLIST_PATHS | list[str] | [".env","secrets/"] | P0 | Forbidden. |
| CG_MAX_DIFF_LOC | int | 500 | P0 | Cap change size. |
| CG_RUN_TESTS | bool | true | P0 | Must be true for proposals. |
| CG_TEST_COMMAND | string | "pytest -q" | P0 | Test runner. |
| CG_LINT_COMMAND | string | "ruff check ." | P0 | Lint runner. |

## 9. Non-Functional Requirements
- No patch can be marked approved without passing static scans and tests (hard requirement).
- Patch generation is reproducible: model_used + request_hash recorded; context citations stable.
- Sandbox isolation: patch/test execution cannot access external network by default (DB1).

## 10. Observability
**Metrics**
- `cg_proposals_total{result}`
- `cg_tests_total{result}`
- `cg_static_scan_fail_total{type}`
- `cg_latency_ms{stage}`

**Alerts**
- Secrets detected
- Repeated failing proposals (could indicate prompt drift or broken tests)

## 11. Failure Modes & Recovery
- Synapse retrieval empty: proceed only if prompt explicitly allows; otherwise deny proposal.
- Tests fail: proposal denied with E_CG_TESTS_FAILED; attach logs.
- Governance denied: proposal stored but not applied; requires human intervention.

## 12. Test Plan
### Unit tests
- Diff parsing and file allowlist enforcement.
- Secret scanner on diff text.

### Integration tests
- End-to-end proposal on a small fixture repo with mocked Helix response.

### Acceptance criteria
- A generated patch cannot bypass scans/tests and produces auditable artifacts.

## 13. Open Questions / Risks
- Define human approval process and required reviewers for different change classes (risk engine changes should require stricter review).
- Sandbox strategy on Windows (venv vs container) for deterministic tests.

## 14. Traceability
### Parent
- DB1 System SRS: CodeGenService generates/refines code with governance checks.

### Children / Related
- [L3-11 CG-PATCH-LIFECYCLE — Patch Lifecycle & Guardrails](../L3/L3-11_CG-PATCH-LIFECYCLE_Patch_Lifecycle_and_Guardrails.md)
- [L2-18 GOV — GovernanceEngine](L2-18_GOV_GovernanceEngine.md)

### Originating system requirements
- CG-IF-001, CG-FR-001..005
