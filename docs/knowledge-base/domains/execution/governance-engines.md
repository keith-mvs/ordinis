# Governance Engines Framework

## Overview

This document specifies the governance, compliance, and ethical oversight engines for the Ordinis trading system. These engines ensure regulatory compliance, protect sensitive data, maintain audit trails, enforce ethical trading practices, and respect broker terms of service.

**Implementation Status: COMPLETE**
- Location: `src/engines/governance/`
- Tests: `tests/test_engines/test_governance/`

---

## OECD AI Principles Integration (2024)

The governance framework implements the OECD AI Principles (2024 Update) for responsible automated decision-making:

### Principle 1: Inclusive Growth & Sustainable Development
- ESG scoring and compliance checking
- Sector exclusion lists for controversial industries
- Human wellbeing considerations in trading decisions

### Principle 2: Human Rights & Democratic Values
- Fair trading practices enforcement
- Market manipulation detection and prevention
- Privacy protection through PPI engine

### Principle 3: Transparency & Explainability
- Signal explanation quality scoring
- Decision tracing and audit trails
- Clear policy documentation

### Principle 4: Robustness, Security & Safety
- Hash-chained immutable audit logs
- PPI detection and masking
- Kill switch integration

### Principle 5: Accountability & Human Oversight
- Human review workflows for large trades
- Approval queues with reviewer levels
- Complete audit chain verification

**Reference**: https://oecd.ai/en/ai-principles

---

## 1. Audit Engine

### 1.1 Purpose

The Audit Engine maintains a complete, immutable record of all system decisions, trades, and state changes for regulatory compliance and internal review.

### 1.2 Implementation

**Location**: `src/engines/governance/core/audit.py`

```python
from src.engines.governance import AuditEngine, AuditEvent, AuditEventType

# Initialize engine
engine = AuditEngine(environment="production")

# Log a trading event
event = engine.log_event(
    event_type=AuditEventType.SIGNAL_GENERATED,
    actor="SignalCore",
    action="Generated buy signal for AAPL",
    details={"confidence": 0.85, "strategy": "SMA_Crossover"},
    affected_symbols=["AAPL"],
)

# Convenience method for signals
signal_event = engine.log_signal(
    symbol="AAPL",
    signal_type="buy",
    confidence=0.9,
    source="Momentum_Strategy",
)

# Verify chain integrity
is_valid, errors = engine.verify_chain_integrity()
```

### 1.3 Event Types

```python
class AuditEventType(Enum):
    # Trading Lifecycle
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_FILTERED = "signal_filtered"
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_MODIFIED = "order_modified"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"

    # Risk Events
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"

    # Governance Events
    POLICY_EVALUATED = "policy_evaluated"
    POLICY_VIOLATION = "policy_violation"
    ETHICS_CHECK_PASSED = "ethics_check_passed"
    ETHICS_CHECK_FAILED = "ethics_check_failed"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    HUMAN_REVIEW_COMPLETED = "human_review_completed"

    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGED = "config_changed"
    MODEL_UPDATED = "model_updated"

    # AI/Model Events (OECD Principle 3)
    AI_DECISION_MADE = "ai_decision_made"
    AI_EXPLANATION_GENERATED = "ai_explanation_generated"
```

### 1.4 Hash Chaining

Events are cryptographically linked using SHA-256:

```
Event 1 → Hash(E1) → Event 2.previous_hash → Hash(E2) → Event 3.previous_hash → ...
```

This creates an immutable, tamper-evident audit trail that can be verified at any time.

---

## 2. Governance Engine

### 2.1 Purpose

The Governance Engine orchestrates all governance components and enforces organizational policies, trading rules, and regulatory compliance requirements.

### 2.2 Implementation

**Location**: `src/engines/governance/core/governance.py`

```python
from src.engines.governance import GovernanceEngine, Policy, PolicyType, PolicyAction

# Initialize with all sub-engines
engine = GovernanceEngine()

# Evaluate a trade
decision = engine.evaluate_trade(
    symbol="AAPL",
    action="buy",
    quantity=100,
    price=150.0,
    strategy="Momentum",
    signal_explanation="RSI oversold with volume confirmation",
    account_value=100000,
    current_position_value=5000,
)

if decision.action == PolicyAction.ALLOW:
    # Proceed with trade
    pass
elif decision.action == PolicyAction.REVIEW:
    # Queue for human approval
    pass
elif decision.action == PolicyAction.BLOCK:
    # Reject trade
    print(f"Blocked: {decision.reason}")
```

### 2.3 Policy Types

```python
class PolicyType(Enum):
    TRADING = "trading"        # Order restrictions, position limits
    RISK = "risk"              # Drawdown limits, concentration
    COMPLIANCE = "compliance"  # Regulatory requirements
    ETHICS = "ethics"          # ESG, sector exclusions
    PRIVACY = "privacy"        # PPI protection
    OPERATIONAL = "operational" # System limits, rate limiting
```

### 2.4 Policy Actions

```python
class PolicyAction(Enum):
    ALLOW = "allow"      # Trade approved
    WARN = "warn"        # Trade allowed with warning
    REVIEW = "review"    # Requires human approval
    BLOCK = "block"      # Trade rejected
    ESCALATE = "escalate" # Escalate to senior reviewer
```

### 2.5 Default Policies

| Policy | Type | Threshold | Action |
|--------|------|-----------|--------|
| Max Position Size | Risk | 10% of portfolio | Block |
| Daily Trade Limit | Trading | 100 trades | Warn |
| Max Drawdown | Risk | -15% | Block |
| ESG Compliance | Ethics | Score >= 50 | Warn |
| PPI Protection | Privacy | Any detection | Block transmission |

---

## 3. PPI (Private Personal Information) Engine

### 3.1 Purpose

The PPI Engine protects sensitive personal and financial information through detection, masking, and access control.

### 3.2 Implementation

**Location**: `src/engines/governance/core/ppi.py`

```python
from src.engines.governance import PPIEngine, PPICategory, MaskingMethod

engine = PPIEngine()

# Scan text for PPI
text = "Contact john.doe@example.com or call 555-123-4567"
masked, detections = engine.scan_text(text)
# masked = "Contact j***@example.com or call ***-***-4567"

# Scan nested data structures
data = {
    "user": {"ssn": "123-45-6789", "email": "test@test.com"},
    "order": {"symbol": "AAPL", "quantity": 100}
}
masked_data, detections = engine.scan_dict(data)

# Check if transmission should be blocked
should_block, categories = engine.should_block_transmission(detections)
```

### 3.3 Detection Categories

```python
class PPICategory(Enum):
    SSN = "ssn"                    # Social Security Numbers
    CREDIT_CARD = "credit_card"   # Credit/debit cards
    EMAIL = "email"               # Email addresses
    PHONE = "phone"               # Phone numbers
    ADDRESS = "address"           # Physical addresses
    API_KEY = "api_key"           # API keys/secrets
    PASSWORD = "password"         # Passwords
    ACCOUNT_NUMBER = "account"    # Financial accounts
    CUSTOM = "custom"             # User-defined patterns
```

### 3.4 Masking Methods

| Method | Description | Example |
|--------|-------------|---------|
| FULL | Complete replacement | `***-**-****` |
| PARTIAL | Show last 4 | `***-**-6789` |
| HASH | Deterministic hash | `[HASH:a1b2c3d4]` |
| TOKENIZE | Reversible token | `TOK-SSN-ABC123` |
| REDACT | Category label | `[REDACTED-SSN]` |

---

## 4. Ethics Engine

### 4.1 Purpose

The Ethics Engine ensures trading activities align with OECD AI Principles, ESG standards, and responsible investing practices.

### 4.2 Implementation

**Location**: `src/engines/governance/core/ethics.py`

```python
from src.engines.governance import EthicsEngine, OECDPrinciple, ESGScore

engine = EthicsEngine()

# Comprehensive trade ethics check
approved, results = engine.check_trade(
    symbol="AAPL",
    action="buy",
    quantity=100,
    price=150.0,
    strategy="Momentum",
    signal_explanation="RSI oversold with volume confirmation and MACD crossover",
)

# Check ESG compliance
engine.set_esg_score("AAPL", ESGScore(
    symbol="AAPL",
    environmental=75.0,
    social=80.0,
    governance=85.0,
    overall=80.0,
    data_source="provider",
    last_updated=datetime.utcnow(),
))
esg_result = engine.check_esg_compliance("AAPL")

# Check sector exclusions
result = engine.check_sector_exclusion("PM")  # Philip Morris - tobacco
# result.passed = False, result.blocking = True

# Check manipulation risk
manipulation_result = engine.check_manipulation_risk(
    symbol="TEST",
    action="buy",
    quantity=100,
    price=10.0,
)
```

### 4.3 OECD Principles Enumeration

```python
class OECDPrinciple(Enum):
    # Principle 1: Inclusive Growth
    INCLUSIVE_GROWTH = "inclusive_growth"
    SUSTAINABLE_DEVELOPMENT = "sustainable_development"
    HUMAN_WELLBEING = "human_wellbeing"

    # Principle 2: Human Rights
    HUMAN_RIGHTS = "human_rights"
    FAIRNESS = "fairness"
    PRIVACY = "privacy"

    # Principle 3: Transparency
    TRANSPARENCY = "transparency"
    EXPLAINABILITY = "explainability"

    # Principle 4: Robustness
    ROBUSTNESS = "robustness"
    SECURITY = "security"
    SAFETY = "safety"

    # Principle 5: Accountability
    ACCOUNTABILITY = "accountability"
    TRACEABILITY = "traceability"
    HUMAN_OVERSIGHT = "human_oversight"
```

### 4.4 Manipulation Detection

The ethics engine detects these manipulation patterns:
- **Wash Trading**: Simultaneous buy/sell for same quantity
- **Layering**: Multiple orders at different price levels
- **Momentum Ignition**: Rapid-fire orders to move price

### 4.5 Human Oversight Thresholds

| Trade Size | Review Requirement |
|------------|-------------------|
| < $100,000 | Automatic approval |
| $100,000 - $500,000 | Standard review |
| > $500,000 | Senior review required |

---

## 5. Broker Compliance Engine

### 5.1 Purpose

The Broker Compliance Engine ensures trading operations comply with broker/API agreements and terms of service.

### 5.2 Implementation

**Location**: `src/engines/governance/core/broker_compliance.py`

```python
from src.engines.governance import BrokerComplianceEngine, Broker

# Initialize for Alpaca paper trading
engine = BrokerComplianceEngine(
    broker=Broker.ALPACA_PAPER,
    account_type="paper",
)

# Check rate limits before API call
result = engine.check_rate_limit("api")
if not result.passed:
    # Wait or queue request
    pass

# Check PDT compliance
pdt_result = engine.check_pdt_compliance(
    account_value=15000,  # Under $25k
    is_day_trade=True,
)

# Comprehensive order check
passed, results = engine.check_order(
    symbol="AAPL",
    side="buy",
    quantity=100,
    order_type="market",
    account_info={
        "buying_power": 50000,
        "account_value": 100000,
        "has_margin": False,
        "positions": {},
    },
)
```

### 5.3 Supported Brokers

```python
class Broker(Enum):
    ALPACA = "alpaca"
    ALPACA_PAPER = "alpaca_paper"
    INTERACTIVE_BROKERS = "interactive_brokers"
    TD_AMERITRADE = "td_ameritrade"
    CUSTOM = "custom"
```

### 5.4 Compliance Categories

| Category | Description |
|----------|-------------|
| RATE_LIMITING | API call quotas (200/min for Alpaca) |
| DATA_USAGE | Market data redistribution restrictions |
| ORDER_RESTRICTIONS | PDT rule, position limits |
| ACCOUNT_REQUIREMENTS | Margin approval, buying power |
| TRADING_PATTERNS | Wash trade prevention, manipulation |
| API_USAGE | Key security, endpoint usage |

### 5.5 Alpaca-Specific Policies

| Policy | Limit | Consequence |
|--------|-------|-------------|
| API Rate Limit | 200 requests/minute | Request blocked |
| Order Rate Limit | 100 orders/minute | Order blocked |
| PDT Rule | 3 day trades/5 days (< $25k) | Trade blocked |
| Short Selling | Requires margin approval | Trade blocked |
| Data Redistribution | Prohibited | Violation logged |

### 5.6 Pattern Day Trader (PDT) Compliance

```python
# Automatic PDT tracking
engine.record_day_trade(
    symbol="AAPL",
    buy_time=datetime.utcnow(),
    sell_time=datetime.utcnow(),
    profit_loss=150.0,
)

# Check before day trade
result = engine.check_pdt_compliance(
    account_value=15000,  # Under $25k threshold
    is_day_trade=True,
)

if not result.passed:
    print("PDT limit reached - cannot execute day trade")
```

---

## 6. Integration Architecture

### 6.1 Engine Integration Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ORDINIS TRADING SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Signal Generated (SignalCore)                                          │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────┐                                                │
│  │   ETHICS ENGINE     │ ← OECD Principles, ESG, Sector Exclusions      │
│  │   (OECDPrinciple)   │                                                │
│  └──────────┬──────────┘                                                │
│             │ Ethics Passed                                             │
│             ▼                                                           │
│  ┌─────────────────────┐                                                │
│  │ BROKER COMPLIANCE   │ ← Rate Limits, PDT, Terms of Service           │
│  │   (Alpaca, IB)      │                                                │
│  └──────────┬──────────┘                                                │
│             │ Compliance Passed                                         │
│             ▼                                                           │
│  ┌─────────────────────┐                                                │
│  │ GOVERNANCE ENGINE   │ ← Policy Enforcement, Human Review             │
│  │   (PolicyDecision)  │                                                │
│  └──────────┬──────────┘                                                │
│             │ Policy Approved                                           │
│             ▼                                                           │
│  ┌─────────────────────┐                                                │
│  │    PPI ENGINE       │ ← Mask Sensitive Data Before Logging           │
│  │   (PPICategory)     │                                                │
│  └──────────┬──────────┘                                                │
│             │ Data Sanitized                                            │
│             ▼                                                           │
│  ┌─────────────────────┐                                                │
│  │   AUDIT ENGINE      │ ← Immutable Hash-Chained Log                   │
│  │   (AuditEvent)      │                                                │
│  └──────────┬──────────┘                                                │
│             │ Event Logged                                              │
│             ▼                                                           │
│     RiskGuard Check → FlowRoute Execution                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Usage Example

```python
from src.engines.governance import (
    GovernanceEngine,
    BrokerComplianceEngine,
    Broker,
    PolicyAction,
)

# Initialize engines
governance = GovernanceEngine()
broker_compliance = BrokerComplianceEngine(broker=Broker.ALPACA_PAPER)

def evaluate_trade_opportunity(signal: dict, account: dict) -> bool:
    """Comprehensive trade evaluation with all governance checks."""

    # Step 1: Check broker compliance
    broker_passed, broker_results = broker_compliance.check_order(
        symbol=signal["symbol"],
        side=signal["action"],
        quantity=signal["quantity"],
        order_type="market",
        account_info=account,
    )

    if not broker_passed:
        print(f"Broker compliance failed: {broker_results}")
        return False

    # Step 2: Governance evaluation (includes ethics, PPI, audit)
    decision = governance.evaluate_trade(
        symbol=signal["symbol"],
        action=signal["action"],
        quantity=signal["quantity"],
        price=signal["price"],
        strategy=signal["strategy"],
        signal_explanation=signal["explanation"],
        account_value=account["account_value"],
        current_position_value=account.get("positions", {}).get(signal["symbol"], 0),
    )

    if decision.action == PolicyAction.ALLOW:
        return True
    elif decision.action == PolicyAction.REVIEW:
        # Queue for human approval
        print(f"Trade requires review: {decision.reason}")
        return False
    else:
        print(f"Trade blocked: {decision.reason}")
        return False
```

### 6.3 Configuration

```python
GOVERNANCE_CONFIG = {
    "audit": {
        "enabled": True,
        "retention_days": 2555,  # ~7 years for regulatory compliance
        "environment": "production",
    },
    "governance": {
        "enabled": True,
        "human_review_threshold": 100000,  # $100k requires review
        "senior_review_threshold": 500000,  # $500k requires senior review
    },
    "ppi": {
        "enabled": True,
        "default_masking": "partial",
        "block_transmission_categories": ["ssn", "credit_card", "api_key"],
    },
    "ethics": {
        "enabled": True,
        "esg_threshold": 50.0,
        "excluded_sectors": ["tobacco", "weapons", "gambling", "private_prisons"],
    },
    "broker_compliance": {
        "broker": "alpaca_paper",
        "enforce_pdt": True,
        "enforce_rate_limits": True,
    },
}
```

---

## 7. Testing

### 7.1 Test Coverage

| Engine | Test File | Coverage |
|--------|-----------|----------|
| Audit | `test_audit.py` | Event creation, hash chaining, integrity verification |
| PPI | `test_ppi.py` | Detection patterns, masking methods, tokenization |
| Ethics | `test_ethics.py` | OECD principles, ESG, manipulation detection |
| Governance | `test_governance.py` | Policy enforcement, human review workflow |
| Broker Compliance | `test_broker_compliance.py` | Rate limits, PDT, order checks |

### 7.2 Running Tests

```bash
# Run all governance tests
pytest tests/test_engines/test_governance/ -v

# Run specific engine tests
pytest tests/test_engines/test_governance/test_ethics.py -v
pytest tests/test_engines/test_governance/test_broker_compliance.py -v
```

---

## 8. Regulatory References

- **SEC Rule 17a-4**: Broker-dealer record retention
- **FINRA Rule 4511**: Books and records requirements
- **FINRA Rule 4210**: Pattern Day Trader requirements
- **MiFID II**: Transaction reporting and record keeping
- **GDPR**: Personal data protection (European operations)
- **SOX Act**: Internal controls and audit trails
- **CCPA**: California Consumer Privacy Act
- **OECD AI Principles (2024)**: Responsible AI framework

---

## 9. Broker Terms of Service References

- **Alpaca Markets**: https://alpaca.markets/docs/api-references/
  - API Rate Limits: 200 requests/minute
  - Data Usage: Redistribution prohibited without license
  - PDT Rules: Standard SEC/FINRA requirements

- **Interactive Brokers**: https://www.interactivebrokers.com/en/trading/tos.php
  - API Message Limits: 50 messages/second
  - Market Data: Exchange-specific agreements required

---

## Appendix: Complete Module Exports

```python
from src.engines.governance import (
    # Audit
    AuditEngine,
    AuditEvent,
    AuditEventType,

    # PPI
    PPIEngine,
    PPICategory,
    MaskingMethod,

    # Ethics
    EthicsEngine,
    OECDPrinciple,
    EthicsCheckResult,

    # Governance
    GovernanceEngine,
    Policy,
    PolicyDecision,

    # Broker Compliance
    BrokerComplianceEngine,
    Broker,
    BrokerPolicy,
    ComplianceCategory,
    ComplianceCheckResult,
)
```
