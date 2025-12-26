# L3-15 — SEC-SECRETS — Secrets & Key Management

## 1. Identifier
- **ID:** `L3-15`
- **Component:** SEC-SECRETS — Secrets & Key Management
- **Level:** L3

## 2. Purpose / Scope
### In scope
- SecretProvider implementations and selection rules (env in DB1).
- Secret naming conventions and rotation procedures.
- Redaction pattern configuration and high-entropy detection (optional).
- Model allowlist and sensitive config handling.

### Out of scope
- Enterprise vault integrations (P1+).
- SSO and user authentication UX (P2).

## 3. Responsibilities
- Define how secrets are named, fetched, and injected into components.
- Define redaction rules and how to test for secret leakage.
- Define rotation playbook and how to verify after rotation.

## 4. Interfaces
### Public APIs / SDKs
Secrets APIs are in [L2-04 SEC](../L2/L2-04_SEC_Security_Identity_and_Secrets.md). This spec defines operational conventions.

### Naming conventions
- All secrets referenced by logical key, not raw values.
- Examples:
  - `NVIDIA_API_KEY`
  - `BROKER_API_KEY`
  - `BROKER_API_SECRET`
  - `DATA_VENDOR_KEY`

### Provider selection (DB1)
- `SEC_PROVIDER=env` reads from environment variables only.
- Secrets are never stored in SQLite in DB1.

### Events
**Consumes**
_None._

**Emits**
- ordinis.sec.redaction.applied (optional)
- ordinis.sec.secret.missing

### Schemas
- RedactionPolicy schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_SEC_SECRET_MISSING` | Secret not found in provider | No |
| `E_SEC_SECRET_INVALID` | Secret malformed/invalid format | No |

## 5. Dependencies
### Upstream
- CFG selects provider and redaction patterns

### Downstream
- All adapters needing credentials (ING/EXEC/HEL)
- OBS and GOV for redaction enforcement

### External services / libraries
_None._

## 6. Data & State
### Storage
- No secret storage; only non-secret references stored in snapshots.

### Caching
- In-memory cache of secret presence (has()) to fail fast at startup; do not cache secret values unless necessary.

### Retention
- Redaction policies retained with config snapshots; secrets never retained.

### Migrations / Versioning
- Redaction pattern changes are config changes; require audit in prod.

## 7. Algorithms / Logic
### Secret access rules
- Fetch secrets at startup for mandatory dependencies; fail startup if missing.
- Fetch secrets on-demand for optional dependencies; return clear error if missing.
- Never log secret values; redactor must remove known keys and patterns.

### Rotation procedure (baseline)
1. Add new secret value in deployment env (side-by-side if supported).
2. Restart services (or reload if supported).
3. Verify health checks for adapters (broker/data/helix).
4. Remove old secret value.

### Leakage test
- In CI, inject a canary secret value and scan logs/artifacts after tests.
- Build fails if canary appears anywhere.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| SEC_PROVIDER | string | env | P0 | Only env in DB1. |
| SEC_REDACTION_PATTERNS | list[str] | built-in | P0 | Regex patterns. |
| SEC_SECRET_CANARY | string | "" | P1 | CI only; used for leakage tests. |

## 9. Non-Functional Requirements
- No secret leakage in logs/events/audit at rest (hard requirement).
- Startup fails fast if required secrets missing.

## 10. Observability
**Metrics**
- `sec_secret_missing_total{key}`
- `sec_redactions_total`

**Alerts**
- Any secret leakage detection is SEV1.

## 11. Failure Modes & Recovery
- Missing secret: component cannot start; ORCH should not run cycles if critical adapters missing.
- Redaction misconfig: treat as critical; fail closed for emission/persistence in prod.

## 12. Test Plan
### Unit tests
- Redactor removes key-based and pattern-based secrets.
- SecretProvider returns missing for unset env keys.

### Integration tests
- Inject canary secret; run workflow; scan logs/artifacts.

### Acceptance criteria
- Canary secret never appears in any persisted artifact or log.

## 13. Open Questions / Risks
- Whether to implement encrypted local secrets file for DB1 convenience (still risky).
- High-entropy detection: false positives vs value for leakage prevention.

## 14. Traceability
### Parent
- [L2-04 SEC — Security](../L2/L2-04_SEC_Security_Identity_and_Secrets.md)

### Originating system requirements
- SEC-FR-001..003
