# L3-11 — CG-PATCH-LIFECYCLE — Patch Lifecycle & Guardrails

## 1. Identifier
- **ID:** `L3-11`
- **Component:** CG-PATCH-LIFECYCLE — Patch Lifecycle & Guardrails
- **Level:** L3

## 2. Purpose / Scope
### In scope
- Patch proposal states and transitions (requested → proposed → approved/denied → applied/merged).
- Static scans and forbidden patterns (secrets, dangerous syscalls, path denylist).
- Test execution policy and artifact capture.
- Governance and human approval requirements by change category.

### Out of scope
- Actual git merge/deployment mechanics (external CI/CD).
- Runtime feature flag rollout (CFG/GOV).

## 3. Responsibilities
- Define the patch lifecycle state machine and required artifacts per transition.
- Define guardrails that must pass before a patch can be approved.
- Define required reviewers and governance rules per change class.

## 4. Interfaces
### Public APIs / SDKs
### Patch lifecycle state machine
- `REQUESTED` → `PROPOSED` (generated diff + scans + tests)
- `PROPOSED` → `DENIED` (scan/test fail or GOV deny)
- `PROPOSED` → `APPROVED` (HumanGate + GOV approve)
- `APPROVED` → `APPLIED` (patch applied to workspace/branch)
- `APPLIED` → `MERGED` (external git merge/CI)

### Required artifacts
- unified diff
- static scan report
- test report (stdout/stderr, exit codes)
- model provenance (model_used, request_hash, citations)
- approvals (actor ids, timestamps, policy_version)

### Events
**Consumes**
- ordinis.cg.patch.requested
- ordinis.gov.change.approved|denied

**Emits**
- ordinis.cg.patch.proposed
- ordinis.cg.patch.denied
- ordinis.cg.patch.approved
- ordinis.cg.patch.applied

### Schemas
- PatchProposal schema (L2-24)
- StaticScanReport schema
- TestReport schema
- ApprovalRecord schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_CG_SCAN_FORBIDDEN` | Static scan found forbidden pattern | No |
| `E_CG_TESTS_FAILED` | Tests failed | No |
| `E_CG_APPROVAL_REQUIRED` | Attempt to apply without approvals | No |

## 5. Dependencies
### Upstream
- SEC for secret scanning rules
- GOV for approvals and audit
- OBS for artifact/log capture

### Downstream
- CG runtime

### External services / libraries
- Git tooling
- Test runners / linters

## 6. Data & State
### Storage
- All patch lifecycle artifacts stored in PERSIST artifact store and referenced from audit log.

### Caching
- N/A

### Retention
- Patch artifacts retained 180 days by default; merge commit ref can extend retention.

### Migrations / Versioning
- Guardrail rules are versioned; changes require policy_version bump and documentation.

## 7. Algorithms / Logic
### Guardrails (minimum)
- Path allowlist enforced (`src/`, `docs/` by default).
- Path denylist enforced (`.env`, `secrets/`, deployment scripts).
- Secret detection:
  - regex patterns (API keys, tokens)
  - high-entropy string detection (optional)
- Forbidden operations:
  - `os.system`, `subprocess.Popen` without allowlist
  - network calls in test sandbox (default blocked)
- Diff size cap: max LOC changed.

### Approval policy (recommended)
- Any change under `src/risk/` or `src/exec/` requires:
  - 2 human approvals (or 1 human + governance override)
  - successful full test suite (not just targeted)
- Docs-only changes require 1 human approval + lint.

### Edge cases
- Flaky tests: rerun once; if still failing mark flaky and require human justification.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| CG_MAX_DIFF_LOC | int | 500 | P0 | Change size cap. |
| CG_REQUIRE_FULL_TESTS_FOR_CRITICAL | bool | true | P0 | risk/exec changes. |
| CG_CRITICAL_PATHS | list[str] | ["src/risk/","src/exec/"] | P0 | Stricter review. |

## 9. Non-Functional Requirements
- No patch can reach APPROVED without required artifacts and approvals (hard requirement).
- Guardrail checks must complete < 30s for typical diffs and repos (excluding full test runs).

## 10. Observability
**Metrics**
- `cg_guardrail_fail_total{type}`
- `cg_patch_state_total{state}`
- `cg_patch_cycle_time_ms{state_transition}`

## 11. Failure Modes & Recovery
- Static scan tool failure: fail closed (deny) for critical changes; warn for docs-only (configurable).
- Artifact store unavailable: fail closed (cannot approve without artifacts).

## 12. Test Plan
### Unit tests
- Path allow/deny checks.
- Secret detection patterns.
- Forbidden API detection in diff.

### Integration tests
- Simulate a critical patch and ensure approval rules enforce full tests and 2 approvals.

### Acceptance criteria
- CG cannot bypass governance approvals or guardrails.

## 13. Open Questions / Risks
- Define the exact set of forbidden APIs and allowed exceptions.
- Do we treat config-only changes as critical if they impact risk thresholds?

## 14. Traceability
### Parent
- [L2-24 CG — CodeGenService](../L2/L2-24_CG_CodeGenService.md)

### Originating system requirements
- CG-FR-001..005, GOV-FR-001..003
