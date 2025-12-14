# {ENGINE_NAME} - Security Specification

> **Document ID:** {ENGINE_ID}-SEC-001
> **Version:** {VERSION}
> **Last Updated:** {DATE}
> **Status:** Draft | Review | Approved
> **Classification:** Internal / Confidential

---

## 1. Security Overview

### 1.1 Data Classification

| Data Type | Classification | Handling Requirements |
|-----------|---------------|----------------------|
| Market data | Internal | Standard encryption |
| Trading signals | Confidential | Encrypted at rest/transit |
| API keys | Secret | Vault storage, never logged |
| User data | PII | GDPR compliant |
| Audit logs | Compliance | Immutable, retained 7 years |

### 1.2 Security Requirements

| Req ID | Requirement | Implementation | Status |
|--------|-------------|----------------|--------|
| {ENGINE_ID}-SEC-001 | Encrypt data at rest | AES-256 | Implemented |
| {ENGINE_ID}-SEC-002 | Encrypt data in transit | TLS 1.3 | Implemented |
| {ENGINE_ID}-SEC-003 | Audit all operations | Governance hooks | Implemented |

---

## 2. Threat Model

### 2.1 Trust Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│                    TRUSTED ZONE                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Internal Network                      │   │
│  │   ┌─────────┐      ┌─────────┐      ┌─────────┐      │   │
│  │   │{ENGINE} │◄────►│ Other   │◄────►│ Database│      │   │
│  │   │         │      │ Engines │      │         │      │   │
│  │   └─────────┘      └─────────┘      └─────────┘      │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ▲                                  │
│                           │ TLS + Auth                       │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    API Gateway                        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ HTTPS
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   UNTRUSTED ZONE                             │
│                   (External clients)                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Threat Catalog

#### T-001: Unauthorized Access

**Threat:** Attacker gains unauthorized access to engine

**Attack Vectors:**
- Stolen credentials
- API key exposure
- Session hijacking

**Controls:**
- [ ] API key rotation (90 days)
- [ ] IP allowlisting
- [ ] Rate limiting
- [ ] Failed auth alerting

**Residual Risk:** Low

#### T-002: Data Exfiltration

**Threat:** Sensitive data leaked externally

**Attack Vectors:**
- Log exposure
- Unencrypted backup
- Insider threat

**Controls:**
- [ ] Data masking in logs
- [ ] Encrypted backups
- [ ] Access auditing
- [ ] DLP monitoring

**Residual Risk:** Low

#### T-003: Injection Attacks

**Threat:** Malicious input causes unintended behavior

**Attack Vectors:**
- SQL injection
- Command injection
- Code injection

**Controls:**
- [ ] Parameterized queries
- [ ] Input validation
- [ ] Output encoding
- [ ] Least privilege

**Residual Risk:** Low

#### T-004: Denial of Service

**Threat:** Engine becomes unavailable

**Attack Vectors:**
- Resource exhaustion
- Amplification attacks
- Malformed requests

**Controls:**
- [ ] Rate limiting
- [ ] Request size limits
- [ ] Resource quotas
- [ ] Circuit breakers

**Residual Risk:** Medium

---

## 3. Authentication & Authorization

### 3.1 Authentication Methods

| Method | Use Case | Configuration |
|--------|----------|---------------|
| API Key | Service-to-service | Header: `X-API-Key` |
| JWT | User requests | Header: `Authorization: Bearer` |
| mTLS | High-security paths | Certificate validation |

### 3.2 Authorization Model

**RBAC Roles:**

| Role | Permissions | Scope |
|------|-------------|-------|
| `admin` | Full access | All operations |
| `operator` | Read + execute | Operations only |
| `viewer` | Read only | Monitoring only |
| `service` | API access | Defined endpoints |

**Permission Matrix:**

| Operation | Admin | Operator | Viewer | Service |
|-----------|-------|----------|--------|---------|
| Read config | ✓ | ✓ | ✓ | - |
| Update config | ✓ | - | - | - |
| Execute operations | ✓ | ✓ | - | ✓ |
| View metrics | ✓ | ✓ | ✓ | ✓ |
| View audit logs | ✓ | ✓ | - | - |

---

## 4. Data Protection

### 4.1 Encryption

| Data State | Method | Key Management |
|------------|--------|----------------|
| At rest | AES-256-GCM | Vault |
| In transit | TLS 1.3 | Certificate Manager |
| In memory | N/A | Process isolation |

### 4.2 Key Management

```
┌─────────────────┐
│   Vault/KMS     │
│                 │
│  ┌───────────┐  │
│  │Master Key │  │
│  └─────┬─────┘  │
│        │        │
│  ┌─────▼─────┐  │
│  │ Data Keys │  │
│  └───────────┘  │
└─────────────────┘
```

**Key Rotation:**
- Master key: Annual
- Data encryption keys: 90 days
- API keys: 90 days (or on compromise)

### 4.3 Data Masking

**Fields to mask in logs:**
- API keys: `sk-****1234`
- Account numbers: `****5678`
- IP addresses: `192.168.xxx.xxx` (last octet)

**Implementation:**
```python
MASKED_FIELDS = ["api_key", "password", "token", "secret"]

def mask_sensitive(data: dict) -> dict:
    """Mask sensitive fields for logging."""
    for field in MASKED_FIELDS:
        if field in data:
            data[field] = "****" + str(data[field])[-4:]
    return data
```

---

## 5. Audit & Compliance

### 5.1 Audit Events

| Event | Logged Fields | Retention |
|-------|--------------|-----------|
| Authentication | user, ip, result, timestamp | 2 years |
| Authorization | user, resource, action, result | 2 years |
| Data access | user, data_type, operation | 7 years |
| Configuration change | user, before, after | 7 years |
| Engine operations | action, inputs, outputs, trace_id | 1 year |

### 5.2 Audit Log Format

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "event_type": "operation",
  "engine": "{ENGINE_NAME}",
  "action": "{action}",
  "user": "{user_id}",
  "trace_id": "{trace_id}",
  "inputs": { /* sanitized */ },
  "outputs": { /* sanitized */ },
  "result": "success|failure",
  "duration_ms": 123,
  "source_ip": "192.168.xxx.xxx"
}
```

### 5.3 Compliance Requirements

| Standard | Requirement | Implementation |
|----------|-------------|----------------|
| SOC 2 | Access controls | RBAC + audit |
| GDPR | Data protection | Encryption + consent |
| PCI-DSS | Secure transmission | TLS 1.3 |
| SEC 17a-4 | Record retention | 7-year audit logs |

---

## 6. Secure Development

### 6.1 Security Checklist

**Code Review:**
- [ ] No hardcoded secrets
- [ ] Input validation on all entry points
- [ ] Output encoding for external data
- [ ] Parameterized queries only
- [ ] Error messages don't leak info
- [ ] Logging doesn't include secrets

**Dependencies:**
- [ ] No known vulnerabilities (CVE scan)
- [ ] Pinned versions
- [ ] Minimal dependencies

### 6.2 Security Testing

| Test Type | Frequency | Tool |
|-----------|-----------|------|
| SAST | Every commit | Semgrep, Bandit |
| DAST | Weekly | OWASP ZAP |
| Dependency scan | Daily | Dependabot |
| Penetration test | Annual | External firm |

---

## 7. Incident Response

### 7.1 Security Incident Classification

| Severity | Examples | Response Time |
|----------|----------|---------------|
| Critical | Data breach, active attack | Immediate |
| High | Vulnerability exploited | < 4 hours |
| Medium | Policy violation | < 24 hours |
| Low | Audit finding | < 1 week |

### 7.2 Incident Response Steps

1. **Detect** - Alert triggered or reported
2. **Contain** - Isolate affected systems
3. **Investigate** - Determine scope and cause
4. **Eradicate** - Remove threat
5. **Recover** - Restore normal operations
6. **Review** - Post-incident analysis

### 7.3 Security Contacts

| Role | Contact |
|------|---------|
| Security Team | security@{company}.com |
| Incident Hotline | {Phone} |
| CISO | {Name} |

---

## 8. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | {DATE} | {AUTHOR} | Initial security specification |
