# Risk Assessment Methodology

**Section**: 03_risk/frameworks
**Last Updated**: 2025-12-12
**Version**: 1.0

---

## Overview

Risk assessment is the systematic process of identifying, analyzing, and evaluating risks. This document defines the methodology for conducting consistent, repeatable risk assessments across all risk categories.

---

## 1. Assessment Framework

### 1.1 Assessment Phases

```
Phase 1: Preparation
    └── Define scope, objectives, criteria
Phase 2: Identification
    └── Discover risks using structured methods
Phase 3: Analysis
    └── Assess probability and impact
Phase 4: Evaluation
    └── Compare against criteria, prioritize
Phase 5: Treatment Selection
    └── Determine response strategies
Phase 6: Documentation
    └── Record findings and decisions
```

### 1.2 Assessment Types

| Type | Frequency | Depth | Trigger |
|------|-----------|-------|---------|
| **Comprehensive** | Annual | Full coverage | Scheduled |
| **Targeted** | Quarterly | Specific domains | Scheduled |
| **Rapid** | Ad-hoc | Focused | Event-driven |
| **Continuous** | Ongoing | Automated metrics | Threshold breach |

---

## 2. Probability Assessment

### 2.1 Probability Scale (5-Point)

| Level | Rating | Description | Frequency Equivalent |
|-------|--------|-------------|---------------------|
| 1 | Rare | Exceptional circumstances | < 1% per year |
| 2 | Unlikely | Could occur | 1-10% per year |
| 3 | Possible | Might occur | 10-50% per year |
| 4 | Likely | Will probably occur | 50-90% per year |
| 5 | Almost Certain | Expected to occur | > 90% per year |

### 2.2 Probability Evidence Sources

| Evidence Type | Weight | Description |
|--------------|--------|-------------|
| **Historical data** | High | Past occurrence frequency |
| **Industry benchmarks** | Medium | Peer organization rates |
| **Expert judgment** | Medium | Specialist assessment |
| **Near-miss data** | Medium | Close calls, prevented incidents |
| **Theoretical analysis** | Low | Scenario probability modeling |

### 2.3 Probability Assessment Questions

- How often has this occurred in the past 3 years?
- What is the industry benchmark for this risk?
- What environmental factors affect likelihood?
- Have near-misses occurred?
- What controls are in place?

---

## 3. Impact Assessment

### 3.1 Impact Scale (5-Point)

| Level | Rating | Financial | Operational | Reputational |
|-------|--------|-----------|-------------|--------------|
| 1 | Negligible | < $10K | < 1 hour disruption | Internal only |
| 2 | Minor | $10K - $100K | Hours of disruption | Limited external |
| 3 | Moderate | $100K - $1M | Days of disruption | Regional coverage |
| 4 | Major | $1M - $10M | Weeks of disruption | National coverage |
| 5 | Catastrophic | > $10M | Extended outage | International, legal |

### 3.2 Impact Dimensions

| Dimension | Assessment Criteria |
|-----------|---------------------|
| **Financial** | Direct losses, remediation costs, fines |
| **Operational** | Service disruption, productivity loss |
| **Reputational** | Brand damage, customer trust |
| **Legal** | Regulatory action, litigation |
| **Strategic** | Market position, competitive impact |
| **Safety** | Injury, loss of life |

### 3.3 Impact Assessment Matrix

For multi-dimensional impacts, use the highest severity across dimensions:

```python
def calculate_impact(dimensions: Dict[str, int]) -> int:
    """
    Calculate overall impact as max across dimensions.
    Conservative approach: worst-case dimension drives rating.
    """
    return max(dimensions.values())
```

---

## 4. Risk Scoring

### 4.1 Risk Score Calculation

```
Risk Score = Probability × Impact
```

| Range | Level | Response Priority |
|-------|-------|-------------------|
| 1-3 | Low | Accept or routine controls |
| 4-6 | Medium | Enhanced monitoring |
| 9-12 | High | Active management required |
| 15-25 | Critical | Immediate action required |

### 4.2 Risk Score Matrix (Heat Map)

```
Impact
  5 │  5 │ 10 │ 15 │ 20 │ 25 │
  4 │  4 │  8 │ 12 │ 16 │ 20 │
  3 │  3 │  6 │  9 │ 12 │ 15 │
  2 │  2 │  4 │  6 │  8 │ 10 │
  1 │  1 │  2 │  3 │  4 │  5 │
    └────┴────┴────┴────┴────┘
       1    2    3    4    5
              Probability
```

### 4.3 Velocity Factor (Optional)

For dynamic environments, incorporate velocity (speed of impact):

| Velocity | Multiplier | Description |
|----------|------------|-------------|
| Slow | 0.8 | Months to years for impact |
| Moderate | 1.0 | Weeks to months |
| Fast | 1.25 | Days to weeks |
| Immediate | 1.5 | Hours to days |

```
Adjusted Score = Risk Score × Velocity Multiplier
```

---

## 5. Qualitative vs. Quantitative Analysis

### 5.1 Qualitative Analysis

**When to Use:**
- Initial screening of risks
- Limited historical data available
- Rapid assessment needed
- Comparing diverse risk types

**Approach:**
- Use probability/impact scales
- Leverage expert judgment
- Apply consistent criteria
- Document assumptions

### 5.2 Quantitative Analysis

**When to Use:**
- High-stakes decisions
- Sufficient historical data
- Financial impact dominates
- Regulatory requirements

**Techniques:**

| Technique | Application | Complexity |
|-----------|-------------|------------|
| **Monte Carlo** | Portfolio risk, aggregate exposure | High |
| **VaR/CVaR** | Market risk, trading losses | Medium |
| **Loss Distribution** | Operational loss modeling | High |
| **Bayesian Networks** | Complex dependencies | High |
| **Scenario Analysis** | Strategic risks | Medium |

### 5.3 Hybrid Approach

For most organizations, combine qualitative screening with quantitative deep dives:

```
1. Qualitative screening → All risks (broad coverage)
2. Prioritization → Top 20% by score
3. Quantitative analysis → High-priority risks only
4. Capital allocation → Based on quantitative results
```

---

## 6. Control Effectiveness Assessment

### 6.1 Control Rating Scale

| Rating | Description | Residual Factor |
|--------|-------------|-----------------|
| **Effective** | Controls working as designed | 0.25 |
| **Partially Effective** | Some gaps or inconsistencies | 0.50 |
| **Weak** | Significant deficiencies | 0.75 |
| **Ineffective** | Controls not operating | 1.00 |

### 6.2 Residual Risk Calculation

```
Residual Score = Inherent Score × Control Factor
```

Example:
- Inherent: Probability 4 × Impact 4 = 16 (Critical)
- Control Rating: Effective (0.25)
- Residual: 16 × 0.25 = 4 (Medium)

### 6.3 Control Assessment Questions

- Is the control designed appropriately?
- Is the control operating consistently?
- When was the control last tested?
- Are there known bypass mechanisms?
- What monitoring exists for control failures?

---

## 7. Assessment Documentation

### 7.1 Risk Assessment Record

| Field | Description | Required |
|-------|-------------|----------|
| Assessment ID | Unique identifier | Yes |
| Date | Assessment date | Yes |
| Assessor | Who performed assessment | Yes |
| Risk ID | Reference to risk register | Yes |
| Inherent Probability | Pre-control probability | Yes |
| Inherent Impact | Pre-control impact | Yes |
| Control Description | Summary of controls | Yes |
| Control Effectiveness | Control rating | Yes |
| Residual Probability | Post-control probability | Yes |
| Residual Impact | Post-control impact | Yes |
| Assumptions | Key assumptions made | Yes |
| Confidence Level | Assessor confidence | Recommended |
| Next Review | Scheduled review date | Yes |

### 7.2 Assessment Quality Standards

- **Consistency**: Same risk assessed similarly by different assessors
- **Repeatability**: Same assessor reaches same conclusion over time
- **Transparency**: Rationale documented for all ratings
- **Timeliness**: Assessments completed within defined windows
- **Independence**: Assessor appropriate for risk type

---

## 8. Assessment Governance

### 8.1 Roles and Responsibilities

| Role | Responsibility |
|------|----------------|
| **Risk Owner** | Provide risk context, validate findings |
| **Risk Assessor** | Conduct assessment, document findings |
| **Risk Manager** | Review methodology, quality assurance |
| **Risk Committee** | Approve methodology, review results |

### 8.2 Assessment Schedule

| Assessment Type | Frequency | Scope | Approver |
|-----------------|-----------|-------|----------|
| Full Assessment | Annual | All risks | Risk Committee |
| Refresh | Quarterly | Top 20 risks | Risk Manager |
| New Risk | As identified | Single risk | Risk Owner |
| Post-Incident | Within 5 days | Incident-related | Risk Manager |

### 8.3 Quality Assurance

- **Peer Review**: High-risk assessments reviewed by second assessor
- **Calibration Sessions**: Periodic alignment across assessors
- **Back-Testing**: Compare predicted vs. actual outcomes
- **External Validation**: Annual methodology review

---

## 9. Integration with Decision-Making

### 9.1 Risk-Based Prioritization

```python
def prioritize_risks(risks: List[Risk]) -> List[Risk]:
    """
    Prioritize by risk score, then velocity.
    """
    return sorted(
        risks,
        key=lambda r: (r.residual_score, r.velocity),
        reverse=True
    )
```

### 9.2 Risk-Informed Decisions

| Decision Type | Risk Input |
|--------------|------------|
| Capital allocation | Quantitative risk estimates |
| Control investment | Risk reduction potential |
| Strategy selection | Strategic risk implications |
| Vendor selection | Third-party risk profiles |
| Project approval | Implementation risk assessment |

---

## Cross-References

- [Risk Taxonomy](risk_taxonomy.md)
- [Risk Scoring and Heat Maps](risk_scoring.md)
- [Risk Governance Framework](risk_governance.md)

---

**Template**: Enterprise Risk Management v1.0
