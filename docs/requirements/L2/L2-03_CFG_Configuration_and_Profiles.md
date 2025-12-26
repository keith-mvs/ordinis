# L2-03 — CFG — Configuration & Profiles

## 1. Identifier
- **ID:** `L2-03`
- **Component:** CFG — Configuration & Profiles
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Profile-based configuration (dev/paper/prod) with deterministic wiring.
- Typed config loading (file + env overrides) and validation.
- Immutable run_config_snapshot persisted per run/cycle.
- Governed runtime overrides (feature flags, thresholds) with audit trail.

### Out of scope
- UI for editing configs (operator tooling).
- Complex distributed config propagation (single-node DB1).

## 3. Responsibilities
- Own configuration file formats, env var mapping, and defaults.
- Validate configuration at startup and on change request.
- Emit config snapshots to PERSIST and GOV audit on every run.
- Provide a stable `Config` API to all engines (no ad-hoc env reads).

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

Profile = Literal["dev","paper","prod"]

@dataclass(frozen=True)
class Config:
    profile: Profile
    bus_backend: str
    broker_backend: str
    storage_dsn: str
    model_allowlist: list[str]
    feature_flags: Dict[str, bool]
    thresholds: Dict[str, float]
    # ... additional engine configs ...

class ConfigManager:
    def load(self, profile: Profile) -> Config: ...
    def get(self) -> Config: ...
    def snapshot(self, correlation_id: str) -> str: ...  # returns snapshot_ref
    def request_override(self, key: str, value: Any, actor: str) -> str: ...  # change_request_id
    def apply_override(self, change_request_id: str) -> None: ...            # requires GOV approval
```

### Events
**Consumes**
- ordinis.cfg.change.requested
- ordinis.gov.change.approved|denied

**Emits**
- ordinis.cfg.loaded
- ordinis.cfg.snapshot.created
- ordinis.cfg.change.applied
- ordinis.cfg.change.denied

### Schemas
- Config schema (typed; stored as JSON in run_config_snapshot)
- EVT-DATA-020 AuditRecord (config changes are audited)

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_CFG_INVALID` | Config validation failed (missing/invalid fields) | No |
| `E_CFG_CHANGE_DENIED` | Governance denied config change | No |
| `E_CFG_SNAPSHOT_FAILED` | Unable to persist config snapshot | Yes |

## 5. Dependencies
### Upstream
- SEC for secret values (API keys, broker creds) injection without logging

### Downstream
- All engines (read config via ConfigManager)
- GOV for approval and audit
- PERSIST for snapshot storage

### External services / libraries
- Config file parser (YAML/TOML/JSON); DB1 recommendation: TOML or YAML with strict schema validation

## 6. Data & State
### Storage
- SQLite table `run_config_snapshot` keyed by correlation_id/run_id.
- Optional `config_changes` table for pending override requests.

### Caching
- In-memory singleton Config object; hot reload only via governed apply_override.

### Retention
- Config snapshots retained for the lifetime of audit retention (default: 7y for regulated; DB1 default: 1y configurable).

### Migrations / Versioning
- Config schema versioned; snapshots include `config_version` and git commit hash.

## 7. Algorithms / Logic
### Loading precedence
1. Base defaults (embedded)
2. Profile file (e.g., `config/dev.yaml`)
3. Environment variables (only whitelisted keys)
4. Governed runtime overrides (stored + applied)

### Override governance flow
1. `request_override` emits `cfg.change.requested` with proposed key/value.
2. GOV evaluates policy and emits approved/denied.
3. If approved, `apply_override` updates active config, increments config_version, persists snapshot.

### Edge cases
- Conflicting overrides: resolved by monotonic change_request_id order; last approved wins (explicitly audited).
- Secret keys: never stored in snapshots; store a reference to SEC provider key name instead.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| CFG_PROFILE | string | dev | P0 | Active profile. |
| CFG_PATH | string | ./config | P0 | Directory containing profile files. |
| CFG_ENV_PREFIX | string | ORDINIS_ | P0 | Prefix for env overrides. |
| CFG_ALLOW_KEYS | list | fixed | P0 | Allowlist of env override keys. |

**Environment variables**
- `ORDINIS_PROFILE`
- `ORDINIS_CFG_PATH`

## 9. Non-Functional Requirements
- Startup validation completes < 1s for typical configs (< 5k keys).
- Config load is deterministic and reproducible given file+env+approved overrides.
- No secrets in snapshots or logs.

## 10. Observability
**Metrics**
- `cfg_load_total{profile,result}`
- `cfg_override_requests_total{result}`
- `cfg_snapshot_write_latency_ms`

**Logs**
- Config version, snapshot_ref, profile, and git commit hash.

**Alerts**
- Snapshot persistence failures (blocks auditability).

## 11. Failure Modes & Recovery
- Invalid config: fail startup hard (no partial boot).
- Snapshot failure: block run (if strict) or continue with degraded audit (not recommended).
- GOV unavailable: deny overrides and continue with last known good config.

## 12. Test Plan
### Unit tests
- Precedence rules (defaults < profile < env < overrides).
- Validation errors include key path and expected type.
- Secret redaction / exclusion from snapshot.

### Integration tests
- Override request emits event, requires GOV approval, and results in new snapshot.

### Acceptance criteria
- A correlation_id always has an immutable config snapshot reference.

## 13. Open Questions / Risks
- Pick config format: YAML vs TOML (TOML tends to be less foot-gun).
- Retention policy defaults for regulated vs non-regulated runs.

## 14. Traceability
### Parent
- DB1 System SRS: Single source of truth configuration with governed overrides.

### Related
- [L2-18 GOV — GovernanceEngine](L2-18_GOV_GovernanceEngine.md)
- [L2-05 PERSIST — Persistence Layer](L2-05_PERSIST_Persistence_Layer.md)

### Originating system requirements
- CFG-FR-001, CFG-FR-002, CFG-FR-003
