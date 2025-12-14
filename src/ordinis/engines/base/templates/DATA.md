# {ENGINE_NAME} - Data Dictionary

> **Document ID:** {ENGINE_ID}-DATA-001
> **Version:** {VERSION}
> **Last Updated:** {DATE}
> **Status:** Draft | Review | Approved

---

## 1. Overview

### 1.1 Purpose
This document defines all data structures, schemas, and persistence requirements for {ENGINE_NAME}.

### 1.2 Data Flow Summary

```
Input Sources          Engine Processing          Output Destinations
─────────────          ─────────────────          ───────────────────
{Source 1} ──────►┌─────────────────────┐──────► {Destination 1}
{Source 2} ──────►│   {ENGINE_NAME}     │──────► {Destination 2}
{Source 3} ──────►└─────────────────────┘──────► Audit Log
```

---

## 2. Data Models

### 2.1 Core Models

#### {ModelName}

**Purpose:** {What this model represents}

**Definition:**
```python
@dataclass(frozen=True)
class {ModelName}:
    """{Description}"""

    # Primary identifier
    id: str

    # Required fields
    {field_name}: {type}  # {Description}

    # Optional fields
    {field_name}: {type} | None = None  # {Description}

    # Computed properties
    @property
    def {property_name}(self) -> {type}:
        """{Description}"""
        ...
```

**Field Specifications:**

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `id` | `str` | Yes | UUID format | Unique identifier |
| `{field}` | `{type}` | Yes/No | {constraints} | {Description} |

**Validation Rules:**
- {Rule 1}
- {Rule 2}

**Example:**
```python
{ModelName}(
    id="550e8400-e29b-41d4-a716-446655440000",
    {field}={value},
)
```

---

## 3. Input Schemas

### 3.1 {InputName}

**Source:** {Where this input comes from}

**Schema:**
```python
@dataclass
class {InputName}:
    {field}: {type}
```

**Validation:**
| Field | Rule | Error |
|-------|------|-------|
| `{field}` | {Rule} | `{ERROR_CODE}` |

**Transformations:**
- {Input} → {Internal representation}

---

## 4. Output Schemas

### 4.1 {OutputName}

**Destination:** {Where this output goes}

**Schema:**
```python
@dataclass(frozen=True)
class {OutputName}:
    {field}: {type}
```

**Guarantees:**
- Immutable after creation
- All fields populated (no None for required)
- Timestamps in UTC

---

## 5. Event Schemas

### 5.1 Published Events

#### `{engine}.{event_name}`
```python
@dataclass
class {EventName}Payload:
    """Payload for {event_name} event."""

    timestamp: datetime      # UTC timestamp
    trace_id: str           # Correlation ID
    engine: str             # Engine name
    {field}: {type}         # {Description}
```

### 5.2 Consumed Events

#### `{source}.{event_name}`
**Expected Schema:**
```python
{
    "timestamp": "ISO8601",
    "trace_id": "string",
    "{field}": "{type}"
}
```

---

## 6. Persistence

### 6.1 Database Tables (if applicable)

#### `{table_name}`

**Purpose:** {What this table stores}

**Schema:**
```sql
CREATE TABLE {table_name} (
    id              UUID PRIMARY KEY,
    {column}        {SQL_TYPE} {CONSTRAINTS},
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_{table}_{column} ON {table_name}({column});
```

**Retention:** {Retention policy}

### 6.2 Cache Structures

| Cache Key | Value Type | TTL | Invalidation |
|-----------|------------|-----|--------------|
| `{engine}:{key}` | `{type}` | {duration} | {When invalidated} |

---

## 7. Data Quality

### 7.1 Quality Rules

| Rule ID | Field | Rule | Severity |
|---------|-------|------|----------|
| DQ-001 | `{field}` | {Rule description} | Error/Warning |

### 7.2 Data Lineage

```
Source System → Ingestion → Validation → Transformation → Storage
     │              │            │             │            │
     └──────────────┴────────────┴─────────────┴────────────┘
                              Audit Trail
```

---

## 8. Serialization

### 8.1 JSON Serialization

```python
def to_json(model: {ModelName}) -> str:
    """Serialize model to JSON."""
    return json.dumps({
        "id": model.id,
        "{field}": model.{field},
    })

def from_json(data: str) -> {ModelName}:
    """Deserialize model from JSON."""
    obj = json.loads(data)
    return {ModelName}(**obj)
```

### 8.2 Protocol Buffers (if applicable)

```protobuf
message {ModelName} {
    string id = 1;
    {proto_type} {field} = 2;
}
```

---

## 9. Data Migration

### 9.1 Schema Versions

| Version | Date | Changes | Migration |
|---------|------|---------|-----------|
| 1.0.0 | {DATE} | Initial schema | N/A |
| 1.1.0 | {DATE} | Added {field} | `migrate_v1_to_v1_1.py` |

### 9.2 Backward Compatibility

- Old format supported for: {duration}
- Migration strategy: {Strategy}

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | {DATE} | {AUTHOR} | Initial data dictionary |
