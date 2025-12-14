# {ENGINE_NAME} - Design Specification

> **Document ID:** {ENGINE_ID}-DESIGN-001
> **Version:** {VERSION}
> **Last Updated:** {DATE}
> **Status:** Draft | Review | Approved

---

## 1. High-Level Design

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    {ENGINE_NAME}                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Input     │  │    Core     │  │   Output    │     │
│  │  Adapters   │→ │   Logic     │→ │  Adapters   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         ↑                ↑                ↓             │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Governance Hooks                    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Component Diagram

| Component | Responsibility | Location |
|-----------|---------------|----------|
| Engine | Main orchestration | `core/engine.py` |
| Config | Configuration | `core/config.py` |
| Models | Data structures | `core/models.py` |
| Hooks | Governance | `hooks/governance.py` |

### 1.3 Data Flow

```
Input → Validate → Process → Transform → Output
          ↓           ↓          ↓
       Preflight    Audit      Audit
```

---

## 2. Low-Level Design

### 2.1 Class Diagram

```python
class {ENGINE_NAME}Config(BaseEngineConfig):
    """Engine configuration."""
    {config_field}: {type} = {default}

class {ENGINE_NAME}(BaseEngine[{ENGINE_NAME}Config]):
    """Main engine class."""

    async def _do_initialize(self) -> None: ...
    async def _do_shutdown(self) -> None: ...
    async def _do_health_check(self) -> HealthStatus: ...

    # Domain methods
    async def {method_name}(self, {params}) -> {return_type}: ...
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

## 6. Design Decisions

### 6.1 Decision Log

| ID | Decision | Rationale | Alternatives Considered |
|----|----------|-----------|------------------------|
| DD-001 | {Decision} | {Why} | {What else was considered} |

---

## 7. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | {DATE} | {AUTHOR} | Initial design |
