# {ENGINE_NAME} - Interface Control Document

> **Document ID:** {ENGINE_ID}-ICD-001
> **Version:** {VERSION}
> **Last Updated:** {DATE}
> **Author:** {AUTHOR}
> **Status:** Draft | Review | Approved
> **Implements:** {ENGINE_ID}-SPEC-001

---

## 1. Overview

### 1.1 Purpose
This document defines all interfaces exposed by and consumed by {ENGINE_NAME}, including sync/async variants, batch operations, and streaming interfaces.

### 1.2 Interface Summary

| Interface | Type | Direction | Async | Batch | Streaming |
|-----------|------|-----------|-------|-------|-----------|
| I{ENGINE_NAME} | Protocol | Exposed | Yes | No | No |
| I{ENGINE_NAME}Sync | Protocol | Exposed | No | No | No |
| I{ENGINE_NAME}Batch | Protocol | Exposed | Yes | Yes | No |
| I{ENGINE_NAME}Stream | Protocol | Exposed | Yes | No | Yes |

### 1.3 Interface Stability

| Level | Meaning |
|-------|---------|
| Stable | No breaking changes without major version bump |
| Evolving | May change with deprecation notice |
| Experimental | May change without notice |

---

## 2. Public API

### 2.1 Primary Interface (Async)

```python
from typing import Protocol, AsyncIterator

class I{ENGINE_NAME}(Protocol):
    """
    Protocol defining {ENGINE_NAME} async interface.

    Stability: Stable
    Implements: {ENGINE_ID}-INT-001
    """

    @property
    def name(self) -> str:
        """Engine name."""
        ...

    @property
    def state(self) -> EngineState:
        """Current engine state."""
        ...

    async def {method}(self, {params}) -> {return}:
        """
        {Description}

        Implements: {ENGINE_ID}-FUNC-001
        """
        ...
```

### 2.2 Synchronous Interface

```python
class I{ENGINE_NAME}Sync(Protocol):
    """
    Protocol defining {ENGINE_NAME} sync interface.

    Stability: Stable
    Note: Wraps async methods for sync contexts.
    """

    def {method}_sync(self, {params}) -> {return}:
        """
        Synchronous wrapper for {method}.

        Note: Blocks the calling thread until complete.
        """
        ...
```

### 2.3 Batch Interface

```python
class I{ENGINE_NAME}Batch(Protocol):
    """
    Protocol for batch operations.

    Stability: Evolving
    Use for: Processing multiple items efficiently.
    """

    async def {method}_batch(
        self,
        items: list[{InputType}],
        *,
        max_concurrency: int = 10,
        fail_fast: bool = False,
    ) -> list[{return} | Exception]:
        """
        Process multiple items concurrently.

        Args:
            items: Items to process
            max_concurrency: Maximum parallel operations
            fail_fast: Stop on first failure

        Returns:
            List of results or exceptions for each item
        """
        ...
```

### 2.4 Streaming Interface

```python
class I{ENGINE_NAME}Stream(Protocol):
    """
    Protocol for streaming operations.

    Stability: Experimental
    Use for: Real-time data processing, large datasets.
    """

    async def {method}_stream(
        self,
        {params},
    ) -> AsyncIterator[{chunk_type}]:
        """
        Stream results as they become available.

        Yields:
            {chunk_type}: Incremental results
        """
        ...
```

### 2.5 Method Contracts

#### `{method_name}`

**Signature:**
```python
async def {method_name}(
    self,
    {param}: {type},
    *,
    timeout: float | None = None,
    trace_id: str | None = None,
) -> {return_type}:
```

**Preconditions:**
- Engine must be in `READY` or `RUNNING` state
- Input must pass schema validation
- {Additional precondition}

**Postconditions:**
- Result conforms to output schema
- Audit record emitted to governance hook
- Metrics updated (latency, success/failure)

**Invariants:**
- Engine state unchanged on error
- No side effects on validation failure

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

### 8.1 Basic Usage (Async)
```python
import asyncio
from ordinis.engines.{engine} import {ENGINE_NAME}, {ENGINE_NAME}Config

async def main():
    config = {ENGINE_NAME}Config(
        engine_id="{ENGINE_ID}",
        {param}={value},
    )
    engine = {ENGINE_NAME}(config)
    await engine.initialize()

    try:
        result = await engine.{method}({params})
        print(f"Result: {result}")
    finally:
        await engine.shutdown()

asyncio.run(main())
```

### 8.2 Synchronous Usage
```python
from ordinis.engines.{engine} import {ENGINE_NAME}, {ENGINE_NAME}Config

config = {ENGINE_NAME}Config({param}={value})
engine = {ENGINE_NAME}(config)
engine.initialize_sync()

result = engine.{method}_sync({params})
print(f"Result: {result}")

engine.shutdown_sync()
```

### 8.3 Batch Processing
```python
items = [{InputType}(...) for _ in range(100)]

results = await engine.{method}_batch(
    items,
    max_concurrency=10,
    fail_fast=False,
)

successes = [r for r in results if not isinstance(r, Exception)]
failures = [r for r in results if isinstance(r, Exception)]
print(f"Success: {len(successes)}, Failed: {len(failures)}")
```

### 8.4 Streaming
```python
async for chunk in engine.{method}_stream({params}):
    print(f"Received: {chunk}")
    # Process incrementally
```

### 8.5 With Governance Hook
```python
from ordinis.engines.base.hooks import CompositeGovernanceHook

# Custom governance hook
class CustomHook(BaseGovernanceHook):
    async def preflight(self, context: PreflightContext) -> PreflightResult:
        # Add custom checks
        return PreflightResult(decision=Decision.APPROVE)

engine = {ENGINE_NAME}(
    config,
    governance_hook=CompositeGovernanceHook([CustomHook()]),
)
```

### 8.6 Error Handling
```python
from ordinis.engines.{engine}.errors import {ENGINE_NAME}Error

try:
    result = await engine.{method}({params})
except {ENGINE_NAME}Error as e:
    if e.recoverable:
        # Retry logic
        pass
    else:
        # Log and alert
        raise
```

---

## 9. Testing Contracts

### 9.1 Contract Tests

```python
import pytest
from ordinis.engines.base.requirements import verifies

@verifies("{ENGINE_ID}-INT-001")
async def test_{method}_contract():
    """Verify {method} adheres to interface contract."""
    engine = {ENGINE_NAME}(config)
    await engine.initialize()

    # Precondition: engine ready
    assert engine.state == EngineState.READY

    result = await engine.{method}({params})

    # Postcondition: valid output
    assert isinstance(result, {return_type})

    # Invariant: state unchanged
    assert engine.state == EngineState.READY
```

### 9.2 Mock Implementation

```python
class Mock{ENGINE_NAME}(I{ENGINE_NAME}):
    """Mock implementation for testing."""

    async def {method}(self, {params}) -> {return}:
        return {return}(...)  # Canned response
```

---

## 10. Revision History

| Version | Date | Author | Changes | Reviewed By |
|---------|------|--------|---------|-------------|
| 1.0.0 | {DATE} | {AUTHOR} | Initial interface specification | {Reviewer} |
