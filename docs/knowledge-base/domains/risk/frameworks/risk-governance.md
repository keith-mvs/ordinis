# Risk Governance Framework

**Section**: 03_risk/frameworks
**Last Updated**: 2025-12-12
**Version**: 1.0

---

## Overview

Risk governance establishes the structures, processes, and accountabilities for effective risk management across the organization. This framework defines how risk decisions are made, escalated, and overseen.

---

## 1. Governance Structure

### 1.1 Three Lines of Defense Model

```
┌─────────────────────────────────────────────────────────────┐
│                    GOVERNING BODY / BOARD                   │
│    Sets risk appetite • Oversight • Strategic direction    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      SENIOR MANAGEMENT                       │
│    Implements strategy • Allocates resources • Accountable  │
└─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
┌───▼───┐              ┌──────▼──────┐           ┌──────▼──────┐
│  1st  │              │     2nd     │           │     3rd     │
│ LINE  │              │    LINE     │           │    LINE     │
├───────┤              ├─────────────┤           ├─────────────┤
│Business│             │Risk Mgmt    │           │Internal     │
│Units  │              │& Compliance │           │Audit        │
├───────┤              ├─────────────┤           ├─────────────┤
│• Own  │              │• Design     │           │• Independent│
│  risks│              │  frameworks │           │  assurance  │
│• Apply│              │• Monitor    │           │• Evaluate   │
│  ctrls│              │• Challenge  │           │  governance │
│• Report│             │• Report     │           │• Report     │
└───────┘              └─────────────┘           └─────────────┘
```

### 1.2 Key Governance Bodies

| Body | Composition | Frequency | Key Responsibilities |
|------|-------------|-----------|---------------------|
| **Board/Risk Committee** | Directors, NEDs | Quarterly | Appetite, strategy, oversight |
| **Executive Risk Committee** | C-suite | Monthly | Resource allocation, escalations |
| **Operational Risk Forum** | Department heads | Bi-weekly | Operational risk review |
| **Risk Working Groups** | Subject matter experts | As needed | Specific risk investigations |

---

## 2. Risk Appetite Framework

### 2.1 Risk Appetite Statement Components

| Component | Description | Example |
|-----------|-------------|---------|
| **Appetite** | Amount of risk willing to accept | "Accept up to $5M annual loss from operational failures" |
| **Tolerance** | Acceptable variance from appetite | "+/- 20% of stated appetite" |
| **Capacity** | Maximum risk bearable | "Cannot sustain losses exceeding $50M" |
| **Limit** | Hard constraint not to be exceeded | "No single counterparty exposure > $10M" |

### 2.2 Risk Appetite by Category

| Category | Appetite Level | Rationale |
|----------|----------------|-----------|
| Strategic | Moderate-High | Growth requires calculated risks |
| Operational | Low | Efficiency and reliability paramount |
| Compliance | Very Low | Regulatory violations unacceptable |
| Reputational | Low | Trust is core asset |
| Financial | Moderate | Aligned with return objectives |

### 2.3 Risk Appetite Cascade

```
Board-Level Appetite Statement
    │
    ├── Category-Level Limits
    │       │
    │       ├── Business Unit Allocations
    │       │       │
    │       │       └── Individual Risk Limits
    │       │
    │       └── Monitoring Thresholds
    │
    └── Escalation Triggers
```

---

## 3. Roles and Responsibilities

### 3.1 RACI Matrix

| Activity | Board | Exec | Risk Mgmt | Bus Unit | Audit |
|----------|-------|------|-----------|----------|-------|
| Set risk appetite | A | C | R | I | I |
| Define frameworks | I | A | R | C | C |
| Identify risks | I | I | C | R | I |
| Assess risks | I | I | R | R | C |
| Implement controls | I | A | C | R | I |
| Monitor risks | I | I | R | R | C |
| Report to Board | I | R | R | C | A |
| Independent assurance | I | I | I | I | R |

**Key**: R = Responsible, A = Accountable, C = Consulted, I = Informed

### 3.2 Key Risk Roles

| Role | Primary Responsibility | Reporting Line |
|------|----------------------|----------------|
| **Chief Risk Officer** | Enterprise risk oversight | CEO / Board |
| **Risk Manager** | Day-to-day risk management | CRO |
| **Risk Owner** | Specific risk accountability | Business Unit Head |
| **Control Owner** | Control operation and effectiveness | Risk Owner |
| **Risk Analyst** | Assessment and reporting | Risk Manager |

### 3.3 Role Descriptions

**Chief Risk Officer (CRO)**
- Sets risk strategy and frameworks
- Reports to Board on enterprise risk
- Chairs Executive Risk Committee
- Allocates risk management resources
- Authorizes risk acceptance beyond thresholds

**Risk Owner**
- Accountable for specific risks in their domain
- Ensures controls are designed and operating
- Reports material risk changes
- Implements risk treatment plans
- Participates in risk assessments

---

## 4. Risk Decision Framework

### 4.1 Decision Authority Matrix

| Risk Score | Decision Authority | Approval Required |
|------------|-------------------|-------------------|
| 1-6 (Low) | Risk Owner | Risk Manager notification |
| 7-12 (Medium) | Risk Manager | Risk Owner sign-off |
| 13-18 (High) | CRO | Executive Risk Committee review |
| 19-25 (Critical) | Board Risk Committee | CEO endorsement |

### 4.2 Risk Acceptance Criteria

For risks to be formally accepted:

| Criterion | Requirement |
|-----------|-------------|
| Business justification | Documented rationale for accepting |
| Cost-benefit analysis | Control cost vs. risk reduction |
| Authority level | Approved at appropriate level |
| Time-bound | Acceptance period defined |
| Review trigger | Conditions for re-assessment |
| Residual risk | Below capacity threshold |

### 4.3 Risk Acceptance Record

```python
@dataclass
class RiskAcceptance:
    risk_id: str
    risk_description: str
    residual_score: int
    business_justification: str
    accepting_authority: str
    approval_date: date
    expiry_date: date
    review_triggers: List[str]
    conditions: List[str]
    status: str  # Active, Expired, Revoked
```

---

## 5. Escalation Procedures

### 5.1 Escalation Triggers

| Trigger Type | Threshold | Action |
|--------------|-----------|--------|
| **Score breach** | Risk score increases to next level | Escalate to next authority |
| **Appetite breach** | Exceeds stated appetite | Immediate CRO notification |
| **Limit breach** | Hard limit exceeded | Stop activity, notify Board |
| **Incident** | Material loss or near-miss | Incident response + escalation |
| **External event** | Market/regulatory change | Ad-hoc risk assessment |

### 5.2 Escalation Paths

```
Risk Owner identifies issue
    │
    ├── Routine (within tolerance)
    │       └── Document and monitor
    │
    ├── Elevated (approaching limit)
    │       └── Risk Manager → CRO briefing
    │
    └── Critical (limit breach / incident)
            └── Immediate CRO → CEO → Board
```

### 5.3 Escalation Timing

| Severity | Initial Notification | Full Report |
|----------|---------------------|-------------|
| Critical | Within 1 hour | Within 24 hours |
| High | Within 4 hours | Within 3 days |
| Medium | Within 24 hours | Within 1 week |
| Low | Next reporting cycle | Standard report |

---

## 6. Reporting Framework

### 6.1 Reporting Calendar

| Report | Audience | Frequency | Content |
|--------|----------|-----------|---------|
| Risk Dashboard | Executive | Weekly | Key metrics, alerts |
| Risk Report | Risk Committee | Monthly | Full risk profile |
| Board Risk Report | Board | Quarterly | Strategic risks, appetite |
| Annual Risk Review | Board | Annual | Comprehensive assessment |
| Incident Reports | As required | Event-driven | Incident details, lessons |

### 6.2 Report Content Standards

**Executive Dashboard (Weekly)**
- Top 10 risks by score
- New/emerging risks
- Limit breaches
- Key risk indicators (KRIs)
- Trend summary

**Monthly Risk Report**
- Full risk register summary
- Heat map
- Control effectiveness
- Risk movements
- Incident summary
- KRI trends
- Action item status

**Board Report (Quarterly)**
- Executive summary
- Strategic risk landscape
- Appetite utilization
- Emerging risks
- Significant incidents
- Regulatory developments
- Risk culture assessment

---

## 7. Policy and Standards

### 7.1 Policy Hierarchy

```
Enterprise Risk Management Policy
    │
    ├── Risk Assessment Standard
    ├── Risk Reporting Standard
    ├── Risk Acceptance Standard
    ├── Incident Management Standard
    │
    └── Category-Specific Policies
            ├── Operational Risk Policy
            ├── IT Risk Policy
            ├── Compliance Risk Policy
            └── Third-Party Risk Policy
```

### 7.2 Policy Governance

| Activity | Frequency | Responsibility |
|----------|-----------|----------------|
| Policy review | Annual | Policy Owner |
| Policy approval | As updated | Risk Committee |
| Exception approval | As requested | CRO |
| Compliance assessment | Annual | Risk Management |
| Audit of compliance | Per audit plan | Internal Audit |

---

## 8. Risk Culture

### 8.1 Risk Culture Indicators

| Indicator | Positive Signs | Warning Signs |
|-----------|----------------|---------------|
| **Tone at top** | Leadership discusses risk openly | Risk seen as compliance burden |
| **Accountability** | Clear ownership, consequences | Blame culture, finger-pointing |
| **Communication** | Open reporting, no fear | Incidents hidden, late reporting |
| **Incentives** | Risk-adjusted performance | Risk-taking without regard |
| **Learning** | Post-incident reviews, improvements | Repeat incidents, no lessons |

### 8.2 Risk Culture Assessment

Annual assessment across dimensions:

- Leadership commitment to risk management
- Risk awareness and training
- Open communication and reporting
- Accountability and consequences
- Learning from experience
- Resource allocation to risk management

---

## 9. Continuous Improvement

### 9.1 Maturity Assessment

| Level | Characteristics |
|-------|-----------------|
| **1 - Initial** | Ad-hoc, reactive, individual efforts |
| **2 - Developing** | Basic processes, some standardization |
| **3 - Defined** | Documented processes, roles defined |
| **4 - Managed** | Quantitative management, metrics |
| **5 - Optimizing** | Continuous improvement, best practices |

### 9.2 Improvement Process

```
Assess Current State
    │
    ├── Gap Analysis
    │       │
    │       ├── Define Target State
    │       │       │
    │       │       ├── Develop Improvement Plan
    │       │       │       │
    │       │       │       ├── Implement
    │       │       │       │       │
    │       │       │       │       └── Measure & Review
    │       │       │       │               │
    │       │       │       │               └── [Iterate]
```

---

## Cross-References

- [Risk Taxonomy](risk_taxonomy.md)
- [Risk Assessment Methodology](risk_assessment_methodology.md)
- [Risk Scoring and Heat Maps](risk_scoring.md)

---

**Template**: Enterprise Risk Management v1.0
