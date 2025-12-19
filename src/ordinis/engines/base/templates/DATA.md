# {ENGINE_NAME} - Data Dictionary

> **Document ID:** {ENGINE_ID}-DATA-001
> **Version:** {VERSION}
> **Last Updated:** {DATE}
> **Author:** {AUTHOR}
> **Status:** Draft | Review | Approved
> **Implements:** {ENGINE_ID}-SPEC-001

---

## 1. Overview

### 1.1 Purpose
This document defines all data structures, schemas, persistence requirements, data governance policies, and data contracts for {ENGINE_NAME}.

### 1.2 Data Flow Summary

```
Input Sources          Engine Processing          Output Destinations
─────────────          ─────────────────          ───────────────────
{Source 1} ──────►┌─────────────────────┐──────► {Destination 1}
{Source 2} ──────►│   {ENGINE_NAME}     │──────► {Destination 2}
{Source 3} ──────►│                     │──────► StreamingBus
                  │  ┌───────────────┐  │──────► Audit Log
                  │  │ Data Contract │  │──────► Metrics
                  │  │  Validation   │  │
                  │  └───────────────┘  │
                  └─────────────────────┘
```

### 1.3 Data Classification

| Level | Description | Handling |
|-------|-------------|----------|
| Public | Non-sensitive, shareable | No restrictions |
| Internal | Business data | Access control required |
| Confidential | Sensitive financial | Encryption + audit |
| Restricted | PII, credentials | Encryption + masking + audit |

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

| Version | Date | Changes | Migration Script |
|---------|------|---------|------------------|
| 1.0.0 | {DATE} | Initial schema | N/A |
| 1.1.0 | {DATE} | Added {field} | `migrate_v1_to_v1_1.py` |

### 9.2 Migration Strategy

**Approach:** Blue-Green / Rolling / Big Bang

**Steps:**
1. Deploy new version with backward compatibility
2. Run migration script during maintenance window
3. Validate migrated data
4. Remove backward compatibility after {duration}

### 9.3 Backward Compatibility

- Old format supported for: {duration}
- Deprecation notice: {X} releases before removal
- Migration tooling: `ordinis migrate --engine {ENGINE_ID}`

---

## 10. Data Governance

### 10.1 Data Ownership

| Data Entity | Owner | Steward |
|-------------|-------|---------|
| {ModelName} | {Team/Role} | {Person/Role} |

### 10.2 Data Retention

| Data Type | Retention Period | Archive Policy | Deletion |
|-----------|------------------|----------------|----------|
| Operational | 90 days | Move to cold storage | Automated |
| Audit | 7 years | Immutable archive | Manual approval |
| Metrics | 30 days | Aggregate and archive | Automated |

### 10.3 Access Control

| Role | Read | Write | Delete | Export |
|------|------|-------|--------|--------|
| Admin | Yes | Yes | Yes | Yes |
| Operator | Yes | Yes | No | Yes |
| Analyst | Yes | No | No | Yes |
| Service Account | Yes | Yes | No | No |

### 10.4 Audit Requirements

```python
@dataclass
class DataAuditRecord:
    """Audit record for data operations."""

    timestamp: datetime
    operation: str  # CREATE, READ, UPDATE, DELETE
    entity_type: str
    entity_id: str
    user_id: str
    trace_id: str
    old_value: dict | None  # For UPDATE/DELETE
    new_value: dict | None  # For CREATE/UPDATE
    outcome: str  # SUCCESS, FAILURE
```

---

## 11. PII Handling

### 11.1 PII Inventory

| Field | PII Type | Classification | Engine |
|-------|----------|----------------|--------|
| `account_id` | Account Number | Confidential | Masked in logs |
| `api_key` | Credential | Restricted | Never logged |
| `email` | Contact | Confidential | Hashed for lookup |

### 11.2 Masking Rules

```python
MASKING_RULES = {
    "account_id": lambda v: f"***{v[-4:]}",
    "api_key": lambda _: "[REDACTED]",
    "email": lambda v: f"{v[0]}***@{v.split('@')[1]}",
}
```

### 11.3 Encryption

| Data State | Method | Key Management |
|------------|--------|----------------|
| At Rest | AES-256 | AWS KMS / Azure Key Vault |
| In Transit | TLS 1.3 | Certificate rotation |
| In Memory | N/A | Process isolation |

### 11.4 Data Subject Rights

| Right | Implementation |
|-------|----------------|
| Access | Export API with audit log |
| Rectification | Update API with validation |
| Erasure | Soft delete with 30-day purge |
| Portability | Standard JSON export format |

---

## 12. Data Contracts

### 12.1 Producer Contract

```python
@dataclass
class {ENGINE_NAME}DataContract:
    """
    Data contract for {ENGINE_NAME} outputs.

    Version: 1.0.0
    Stability: Stable
    """

    # Schema definition
    schema_version: str = "1.0.0"

    # Field guarantees
    fields: dict = field(default_factory=lambda: {
        "{field}": {
            "type": "{type}",
            "nullable": False,
            "constraints": "{constraints}",
        },
    })

    # Quality guarantees
    quality: dict = field(default_factory=lambda: {
        "completeness": 0.99,  # 99% fields non-null
        "freshness": "< 1 minute",
        "accuracy": "validated against source",
    })

    # SLA
    sla: dict = field(default_factory=lambda: {
        "availability": "99.9%",
        "latency_p95": "< 100ms",
    })
```

### 12.2 Consumer Contract

```python
@dataclass
class {ENGINE_NAME}ConsumerContract:
    """
    Contract for consuming {ENGINE_NAME} data.

    Required by: {ConsumerEngine}
    """

    # Expected schema
    required_fields: list[str] = field(default_factory=lambda: [
        "{field_1}",
        "{field_2}",
    ])

    # Consumption pattern
    pattern: str = "streaming"  # batch | streaming | request-response

    # Retry behavior
    on_missing_data: str = "retry"  # retry | skip | fail
```

### 12.3 Contract Validation

```python
from ordinis.data.contracts import validate_contract

def validate_output(data: dict) -> bool:
    """Validate output against data contract."""
    contract = {ENGINE_NAME}DataContract()
    return validate_contract(data, contract)
```

### 12.4 Contract Evolution

| Version | Breaking | Deprecation | Migration |
|---------|----------|-------------|-----------|
| 1.0 → 1.1 | No | N/A | Auto-compatible |
| 1.x → 2.0 | Yes | 2 releases | Migration guide |

---

## 13. Appendices

### 13.1 Glossary

| Term | Definition |
|------|------------|
| PII | Personally Identifiable Information |
| Data Contract | Agreement between producer and consumer |
| Data Lineage | Tracking data from source to destination |
| Cold Storage | Low-cost archival storage |

### 13.2 References

- `{ENGINE_ID}-SPEC.md` - Requirements specification
- `{ENGINE_ID}-INTERFACE.md` - Interface control document
- Data Governance Policy - `docs/governance/data-policy.md`

---

## 14. Revision History

| Version | Date | Author | Changes | Reviewed By |
|---------|------|--------|---------|-------------|
| 1.0.0 | {DATE} | {AUTHOR} | Initial data dictionary | {Reviewer} |
