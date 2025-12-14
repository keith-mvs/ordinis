# {ENGINE_NAME} - Interface Control Document

> **Document ID:** {ENGINE_ID}-ICD-001
> **Version:** {VERSION}
> **Last Updated:** {DATE}
> **Status:** Draft | Review | Approved

---

## 1. Overview

### 1.1 Purpose
This document defines all interfaces exposed by and consumed by {ENGINE_NAME}.

### 1.2 Interface Summary

| Interface | Type | Direction | Description |
|-----------|------|-----------|-------------|
| {Interface} | Sync/Async | In/Out | {Description} |

---

## 2. Public API

### 2.1 Primary Interface

```python
class I{ENGINE_NAME}(Protocol):
    """Protocol defining {ENGINE_NAME} interface."""

    @property
    def name(self) -> str:
        """Engine name."""
        ...

    async def {method}(self, {params}) -> {return}:
        """{Description}"""
        ...
```

### 2.2 Method Contracts

#### `{method_name}`

**Signature:**
```python
async def {method_name}(
    self,
    {param}: {type},
) -> {return_type}:
```

**Preconditions:**
- Engine must be in `READY` or `RUNNING` state
- {Additional precondition}

**Postconditions:**
- {Postcondition 1}
- Audit record emitted

**Invariants:**
- {Invariant}

---

## 3. Data Contracts

### 3.1 Input Schemas

#### {InputSchemaName}
```python
@dataclass(frozen=True)
class {InputSchemaName}:
    """{Description}"""

    {field}: {type}

    def validate(self) -> bool:
        """Validate input."""
        ...
```

**Validation Rules:**
| Field | Rule | Error Code |
|-------|------|------------|
| {field} | {Rule} | `INVALID_{FIELD}` |

### 3.2 Output Schemas

#### {OutputSchemaName}
```python
@dataclass(frozen=True)
class {OutputSchemaName}:
    """{Description}"""

    {field}: {type}
```

**Guarantees:**
- {Guarantee 1}
- {Guarantee 2}

---

## 4. Event Interfaces

### 4.1 Published Events

#### `{engine}.{event_name}`
```python
@dataclass
class {EventName}Event:
    """Published when {condition}."""

    timestamp: datetime
    trace_id: str
    {payload_field}: {type}
```

**Trigger:** {When this event is published}
**Consumers:** {Who subscribes}

### 4.2 Subscribed Events

#### `{source}.{event_name}`
**Handler:** `_on_{event_name}`
**Action:** {What the engine does when this event arrives}

---

## 5. Dependencies

### 5.1 Required Interfaces

| Dependency | Interface | Methods Used |
|------------|-----------|--------------|
| {Engine} | `I{Engine}` | `{method}` |

### 5.2 Optional Interfaces

| Dependency | Interface | Fallback |
|------------|-----------|----------|
| {Engine} | `I{Engine}` | {Default behavior} |

---

## 6. Error Contracts

### 6.1 Exceptions

| Exception | When Raised | Handling |
|-----------|-------------|----------|
| `{Exception}` | {Condition} | {How caller should handle} |

### 6.2 Error Responses

```python
@dataclass
class {ENGINE_NAME}Error:
    code: str           # e.g., "{ENGINE_ID}_VALIDATION_001"
    message: str        # Human-readable message
    details: dict       # Additional context
    recoverable: bool   # Can caller retry?
```

---

## 7. Versioning

### 7.1 API Version
- Current: `v1`
- Supported: `v1`
- Deprecated: None

### 7.2 Breaking Changes Policy
- Major version bump for breaking changes
- Minimum 2 release deprecation period
- Migration guide required

---

## 8. Examples

### 8.1 Basic Usage
```python
from ordinis.engines.{engine} import {ENGINE_NAME}, {ENGINE_NAME}Config

config = {ENGINE_NAME}Config(
    {param}={value},
)
engine = {ENGINE_NAME}(config)
await engine.initialize()

result = await engine.{method}({params})
print(result)

await engine.shutdown()
```

### 8.2 With Governance
```python
async with engine.track_operation("{action}", {"{param}": value}) as ctx:
    result = await engine._do_{action}({params})
    ctx["outputs"] = result
```

---

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | {DATE} | {AUTHOR} | Initial interface specification |
