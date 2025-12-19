# {ENGINE_NAME} - Design Specification

> **Document ID:** {ENGINE_ID}-DESIGN-001
> **Version:** {VERSION}
> **Last Updated:** {DATE}
> **Author:** {AUTHOR}
> **Status:** Draft | Review | Approved
> **Implements:** {ENGINE_ID}-SPEC-001

---

## 1. High-Level Design

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        {ENGINE_NAME}                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐                           ┌──────────────┐   │
│   │  StreamingBus│◄─────────────────────────►│  Helix/LLM   │   │
│   │   Events     │                           │  (optional)  │   │
│   └──────┬───────┘                           └──────────────┘   │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │   Input     │  │    Core     │  │   Output    │            │
│   │  Adapters   │─►│   Logic     │─►│  Adapters   │            │
│   └─────────────┘  └──────┬──────┘  └─────────────┘            │
│                           │                                      │
│                    ┌──────▼──────┐                              │
│                    │  Governance │                              │
│                    │    Hooks    │                              │
│                    └─────────────┘                              │
│                           │                                      │
│                    ┌──────▼──────┐                              │
│                    │   Metrics   │                              │
│                    │  & Logging  │                              │
│                    └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Diagram

| Component | Responsibility | Location | Dependencies |
|-----------|---------------|----------|--------------|
| Engine | Main orchestration | `core/engine.py` | BaseEngine |
| Config | Configuration | `core/config.py` | BaseEngineConfig |
| Models | Data structures | `core/models.py` | Pydantic |
| Hooks | Governance | `hooks/governance.py` | GovernanceHook |
| Adapters | External integrations | `adapters/` | Protocol-specific |

### 1.3 Data Flow

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌───────────┐    ┌────────┐
│  Input  │───►│ Validate │───►│Preflight│───►│  Process  │───►│ Output │
└─────────┘    └──────────┘    └────┬────┘    └─────┬─────┘    └────────┘
                                    │               │
                                    ▼               ▼
                               ┌─────────┐    ┌─────────┐
                               │  Audit  │    │  Audit  │
                               │ (start) │    │  (end)  │
                               └─────────┘    └─────────┘
```

### 1.4 Sequence Diagram - Primary Operation

```
┌──────┐       ┌────────┐       ┌──────────┐       ┌────────┐
│Client│       │ Engine │       │Governance│       │  Bus   │
└──┬───┘       └───┬────┘       └────┬─────┘       └───┬────┘
   │               │                 │                 │
   │ request()     │                 │                 │
   │──────────────►│                 │                 │
   │               │ preflight()     │                 │
   │               │────────────────►│                 │
   │               │     Decision    │                 │
   │               │◄────────────────│                 │
   │               │                 │                 │
   │               │ [if APPROVE]    │                 │
   │               │ _do_operation() │                 │
   │               │─────────┐       │                 │
   │               │         │       │                 │
   │               │◄────────┘       │                 │
   │               │                 │                 │
   │               │ audit()         │                 │
   │               │────────────────►│                 │
   │               │                 │                 │
   │               │ publish(event)  │                 │
   │               │─────────────────────────────────►│
   │               │                 │                 │
   │   response    │                 │                 │
   │◄──────────────│                 │                 │
   │               │                 │                 │
```

### 1.5 State Diagram

```
                    ┌─────────────┐
                    │UNINITIALIZED│
                    └──────┬──────┘
                           │ initialize()
                           ▼
                    ┌─────────────┐
              ┌────►│INITIALIZING │
              │     └──────┬──────┘
              │            │ success
              │            ▼
              │     ┌─────────────┐
              │     │    READY    │◄─────────────┐
              │     └──────┬──────┘              │
              │            │ start()             │ pause()
              │            ▼                     │
              │     ┌─────────────┐              │
              │     │   RUNNING   │──────────────┘
              │     └──────┬──────┘
              │            │ stop() / error
              │            ▼
              │     ┌─────────────┐
              │     │ STOPPING    │
              │     └──────┬──────┘
              │            │ complete
              │            ▼
              │     ┌─────────────┐
              └─────│  STOPPED    │
                    └──────┬──────┘
                           │ error (unrecoverable)
                           ▼
                    ┌─────────────┐
                    │   FAILED    │
                    └─────────────┘
```

**State Transitions:**
| From | Event | To | Action |
|------|-------|-----|--------|
| UNINITIALIZED | initialize() | INITIALIZING | Load config, connect dependencies |
| INITIALIZING | success | READY | Register with orchestrator |
| READY | start() | RUNNING | Begin processing |
| RUNNING | pause() | READY | Drain current work |
| RUNNING | stop() | STOPPING | Graceful shutdown |
| STOPPING | complete | STOPPED | Release resources |
| Any | error | FAILED | Log error, alert |

---

## 2. Low-Level Design

### 2.1 Class Diagram

```python
from ordinis.engines.base import BaseEngine, BaseEngineConfig
from ordinis.engines.base.hooks import GovernanceHook

class {ENGINE_NAME}Config(BaseEngineConfig):
    """Engine configuration."""

    # Required fields
    engine_id: str = "{ENGINE_ID}"
    engine_name: str = "{ENGINE_NAME}"

    # Domain-specific configuration
    {config_field}: {type} = {default}

    def validate(self) -> list[str]:
        """Validate configuration."""
        errors = super().validate()
        # Add domain-specific validation
        return errors


class {ENGINE_NAME}(BaseEngine[{ENGINE_NAME}Config]):
    """
    {ENGINE_NAME} - {Brief description}

    Implements: {ENGINE_ID}-SPEC-001
    """

    def __init__(
        self,
        config: {ENGINE_NAME}Config,
        governance_hook: GovernanceHook | None = None,
    ):
        super().__init__(config, governance_hook)
        # Domain-specific initialization
        self._state: dict = {}

    async def _do_initialize(self) -> None:
        """Initialize engine resources."""
        ...

    async def _do_shutdown(self) -> None:
        """Release engine resources."""
        ...

    async def _do_health_check(self) -> HealthStatus:
        """Check engine health."""
        ...

    # Domain methods
    async def {method_name}(self, {params}) -> {return_type}:
        """
        {Method description}

        Implements: {ENGINE_ID}-FUNC-001
        """
        async with self.track_operation("{method_name}", {"{param}": {param}}):
            return await self._do_{method_name}({params})

    async def _do_{method_name}(self, {params}) -> {return_type}:
        """Internal implementation."""
        ...
```

### 2.2 Method Specifications

#### `{method_name}({params}) -> {return_type}`

**Purpose:** {What this method does}

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| {param} | {type} | Yes/No | {Description} |

**Returns:** {Description of return value}

**Raises:**
| Exception | Condition |
|-----------|-----------|
| {Exception} | {When raised} |

**Algorithm:**
```
1. Validate inputs
2. Run preflight check
3. {Step 3}
4. {Step 4}
5. Record audit
6. Return result
```

**Complexity:** O({complexity})

### 2.3 Data Models

#### {ModelName}
```python
@dataclass
class {ModelName}:
    """{Description}"""
    {field}: {type}  # {Description}
```

---

## 3. Integration Points

### 3.1 StreamingBus Events

#### Published Events
| Event Type | Schema | Trigger |
|------------|--------|---------|
| `{engine}.{event}` | `{Schema}` | {When published} |

#### Subscribed Events
| Event Type | Handler | Action |
|------------|---------|--------|
| `{source}.{event}` | `_handle_{event}` | {What happens} |

### 3.2 API Endpoints (if applicable)

| Method | Endpoint | Request | Response |
|--------|----------|---------|----------|
| POST | `/api/{engine}/{action}` | `{RequestSchema}` | `{ResponseSchema}` |

---

## 4. Error Handling

### 4.1 Error Codes

| Code | Message | Recoverable | Action |
|------|---------|-------------|--------|
| `{ENGINE_ID}_001` | {Message} | Yes/No | {Recovery action} |

### 4.2 Retry Strategy

| Error Type | Max Retries | Backoff | Timeout |
|------------|-------------|---------|---------|
| Transient | 3 | Exponential | 30s |
| Permanent | 0 | N/A | N/A |

---

## 5. Configuration

### 5.1 Configuration Schema

```python
@dataclass
class {ENGINE_NAME}Config(BaseEngineConfig):
    # Required
    {field}: {type}

    # Optional with defaults
    {field}: {type} = {default}
```

### 5.2 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `{ENGINE_ID}_ENABLED` | No | `true` | Enable/disable engine |

---

## 6. Design Patterns

### 6.1 Patterns Used

| Pattern | Usage | Rationale |
|---------|-------|-----------|
| Template Method | BaseEngine lifecycle | Enforce consistent initialization/shutdown |
| Strategy | GovernanceHook | Pluggable governance policies |
| Observer | StreamingBus | Decoupled event-driven communication |
| Factory | Config.validate() | Consistent validation across engines |
| Circuit Breaker | Error handling | Prevent cascade failures |

### 6.2 Template Method Pattern

```
BaseEngine                          {ENGINE_NAME}
────────────                        ───────────────
initialize()  ──────────────────►   _do_initialize()
    │                                    │
    ├── set state INITIALIZING           ├── connect dependencies
    │                                    │
    ├── call _do_initialize() ──────────►├── load resources
    │                                    │
    └── set state READY                  └── register handlers
```

### 6.3 Strategy Pattern - Governance

```
┌─────────────────┐
│ GovernanceHook  │ (Protocol)
├─────────────────┤
│ + preflight()   │
│ + audit()       │
└────────┬────────┘
         │
    ┌────┴────┬────────────────┐
    │         │                │
┌───▼───┐ ┌───▼───┐       ┌────▼────┐
│Default│ │Strict │       │Composite│
│ Hook  │ │ Hook  │       │  Hook   │
└───────┘ └───────┘       └─────────┘
```

---

## 7. Design Decisions

### 7.1 Decision Log

| ID | Decision | Rationale | Alternatives Considered | Date |
|----|----------|-----------|------------------------|------|
| DD-001 | Async-first API | Trading requires non-blocking I/O | Sync with threads | {DATE} |
| DD-002 | Protocol-based contracts | Enable static type checking | ABC inheritance | {DATE} |
| DD-003 | Governance hooks required | Compliance and auditability | Optional hooks | {DATE} |

### 7.2 Architecture Decision Records (ADRs)

#### ADR-001: {Decision Title}

**Status:** Accepted | Proposed | Deprecated

**Context:** {What is the issue we're trying to solve?}

**Decision:** {What is the change we're proposing?}

**Consequences:**
- Positive: {Benefits}
- Negative: {Tradeoffs}
- Neutral: {Side effects}

---

## 8. Security Considerations

### 8.1 Threat Model

| Threat | Impact | Mitigation |
|--------|--------|------------|
| Invalid input | Data corruption | Schema validation |
| Unauthorized access | Data breach | RBAC + audit logs |
| Denial of service | Service outage | Rate limiting |

### 8.2 Data Protection

- Sensitive fields masked in logs
- No credentials in config files (use env vars)
- Audit trail for all mutations

---

## 9. Appendices

### 9.1 Glossary

| Term | Definition |
|------|------------|
| Preflight | Pre-execution governance check |
| Audit | Post-execution record for compliance |
| Track Operation | Context manager for metrics/governance |

### 9.2 References

- `{ENGINE_ID}-SPEC.md` - Requirements specification
- `{ENGINE_ID}-INTERFACE.md` - Interface control document
- BaseEngine Framework - `src/ordinis/engines/base/`

---

## 10. Revision History

| Version | Date | Author | Changes | Reviewed By |
|---------|------|--------|---------|-------------|
| 1.0.0 | {DATE} | {AUTHOR} | Initial design | {Reviewer} |
