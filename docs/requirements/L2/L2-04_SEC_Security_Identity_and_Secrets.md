# L2-04 — SEC — Security, Identity & Secrets

## 1. Identifier
- **ID:** `L2-04`
- **Component:** SEC — Security, Identity & Secrets
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Secrets access abstraction (env provider in DB1; extendable).
- Secret redaction for logs/events/audit attachments.
- Basic identity model for internal actions (actor, role) used by GOV audit.
- Model allowlist enforcement inputs for HEL/CTX/CG.
- Optional event signing (HMAC) for integrity (DB1 optional).

### Out of scope
- Full enterprise IAM/SSO integration (SAML/OIDC) (P2).
- Network perimeter controls (handled by deployment stack).

## 3. Responsibilities
- Provide SecretProvider interface and DB1 EnvSecretProvider.
- Provide Redactor policy used by OBS logging and GOV audit attachments.
- Provide minimal AuthZ primitives (roles/scopes) for internal APIs.
- Define secure defaults (deny-by-default model usage, no secrets in snapshots).

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any, List

class SecretProvider(Protocol):
    provider_id: str
    def get(self, key: str) -> str: ...
    def has(self, key: str) -> bool: ...

@dataclass(frozen=True)
class Actor:
    actor_id: str
    roles: List[str]
    scopes: List[str]

class Redactor(Protocol):
    def redact_str(self, s: str) -> str: ...
    def redact_obj(self, obj: Any) -> Any: ...

class Signer(Protocol):
    def sign(self, bytes_: bytes) -> str: ...
    def verify(self, bytes_: bytes, sig: str) -> bool: ...
```

### Events
**Consumes**
_None._

**Emits**
- ordinis.sec.redaction.applied (optional)
- ordinis.sec.signature.invalid (optional)

### Schemas
- Actor schema (used in GOV AuditRecord)
- Redaction policy configuration schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_SEC_SECRET_MISSING` | Requested secret not available | No |
| `E_SEC_REDACTION_FAILED` | Redactor failed; action must fail closed for audit/log emission | No |
| `E_SEC_MODEL_NOT_ALLOWED` | Requested model_id is not in allowlist | No |
| `E_SEC_SIGNATURE_INVALID` | Signature verification failed | No |

## 5. Dependencies
### Upstream
- CFG for selecting secrets provider and allowlists

### Downstream
- OBS (log redaction)
- GOV (audit redaction + actor model)
- HEL (model allowlist enforcement)
- All external adapters requiring credentials (ING, EXEC, HEL backend adapters)

### External services / libraries
- OS environment variables (DB1)
- Optional cryptography library for HMAC signing (P1)

## 6. Data & State
### Storage
- DB1: secrets are not persisted (env only).
- Optional: store redaction patterns and allowlists in config snapshots (excluding secrets).

### Caching
- In-process caching of allowlists and redaction regexes (reload on config snapshot change).

### Retention
- Redaction patterns retained with config snapshots per retention policy.

### Migrations / Versioning
- Actor and redaction schema versioned; must remain backward compatible for audit replays.

## 7. Algorithms / Logic
### Redaction rules (DB1 baseline)
- Redact known secret keys (API keys, tokens) by key name match and by pattern match (e.g., `sk-...`).
- Redact values in:
  - log messages
  - event payload attachments
  - GOV audit attachments
- Redaction is applied **before** persistence or emission.

### Model allowlist enforcement
- HEL/CTX/CG must call `SEC` allowlist check prior to generating requests.
- Default posture: deny unknown model_id, allow only configured allowlist per profile.

### Edge cases
- Accidental secret embedding in exception traces: wrap and redact stack traces before logging.
- Large objects: redact in streaming fashion (avoid copying huge payloads into memory).

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| SEC_PROVIDER | string | env | P0 | `env` in DB1. |
| SEC_REDACTION_ENABLE | bool | true | P0 | Master redaction switch (must be true in prod). |
| SEC_REDACTION_PATTERNS | list[str] | built-in | P0 | Regex patterns for secrets. |
| SEC_MODEL_ALLOWLIST | list[str] | [] | P0 | Allowlisted model IDs for HEL. |

**Environment variables**
- `NVIDIA_API_KEY` (example; provider-specific)

## 9. Non-Functional Requirements
- No secret bytes in logs/events/audit at rest (best-effort with deny-on-redaction-failure).
- Redaction overhead: p95 < 2ms per log/event emission for typical payloads.
- Model allowlist checks: p95 < 0.1ms (in-memory).

## 10. Observability
**Metrics**
- `sec_redactions_total{type=log|event|audit}`
- `sec_redaction_fail_total`
- `sec_model_denied_total{model_id}`

**Alerts**
- Any redaction failure (must page in prod).

## 11. Failure Modes & Recovery
- Secret missing: fail the dependent component startup if required secret is absent.
- Redactor failure: fail closed for emission/persistence; emit minimal safe error record.
- Allowlist misconfig: denies all model calls; system should degrade AI features but keep trading loop operational (DB1 intent).

## 12. Test Plan
### Unit tests
- Redact known patterns and key names in nested objects.
- Ensure secrets are removed from exception traces.
- Allowlist enforcement for HEL/CTX.

### Integration tests
- Attempt to log/config snapshot with a secret and confirm persisted artifact is redacted.

### Acceptance criteria
- No test run produces logs containing a known injected secret token.

## 13. Open Questions / Risks
- DB1 decision: enable event signing now or defer (signing adds complexity but helps audit integrity).
- Secret storage roadmap (encrypted local file vs Vault).

## 14. Traceability
### Parent
- DB1 System SRS: Governance and audit require safe handling of secrets.

### Children / Related
- [L3-15 SEC-SECRETS — Secrets & Key Management](../L3/L3-15_SEC-SECRETS_Secrets_and_Key_Management.md)
- [L2-21 HEL — Helix LLM Provider](L2-21_HEL_Helix_LLM_Provider.md)

### Originating system requirements
- SEC-FR-001, SEC-FR-002, SEC-FR-003
