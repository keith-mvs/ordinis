# Operational Risk

**Section**: 03_risk/operational
**Last Updated**: 2025-12-12
**Version**: 1.0

---

## Definition and Scope

**Operational risk** is the risk of loss resulting from inadequate or failed internal processes, people, systems, or from external events.

This definition, aligned with Basel II/III for financial institutions, encompasses:
- Process failures
- Human errors
- System outages
- External incidents

It excludes strategic risk and reputational risk (though operational failures may trigger these).

---

## 1. Risk Categories (Basel Classification)

### 1.1 Internal Fraud

**Definition**: Losses due to intentional misrepresentation, misappropriation of property, or circumvention of regulations by internal parties.

| Subcategory | Examples |
|-------------|----------|
| Unauthorized trading | Rogue trader positions |
| Theft | Asset misappropriation |
| Forgery | Falsified documents |
| Account fraud | Unauthorized transactions |

### 1.2 External Fraud

**Definition**: Losses due to intentional deception by third parties.

| Subcategory | Examples |
|-------------|----------|
| Theft/robbery | Physical asset theft |
| Forgery | Check fraud, counterfeit |
| Systems intrusion | Hacking, data theft |
| Identity theft | Account takeover |

### 1.3 Employment Practices

**Definition**: Losses from employment-related issues.

| Subcategory | Examples |
|-------------|----------|
| Discrimination | Harassment claims |
| Workplace safety | Injury, accidents |
| Labor relations | Wrongful termination |
| Compensation disputes | Benefit claims |

### 1.4 Clients, Products, and Business Practices

**Definition**: Losses from unintentional or negligent failure to meet obligations.

| Subcategory | Examples |
|-------------|----------|
| Suitability | Inappropriate advice |
| Disclosure | Inadequate information |
| Fiduciary | Breach of duty |
| Product defects | Design flaws |

### 1.5 Damage to Physical Assets

**Definition**: Losses from damage to physical assets.

| Subcategory | Examples |
|-------------|----------|
| Natural disasters | Flood, earthquake, fire |
| Vandalism | Property damage |
| Terrorism | Physical attack |

### 1.6 Business Disruption and System Failures

**Definition**: Losses from disruption of business or system failures.

| Subcategory | Examples |
|-------------|----------|
| Hardware failure | Server crash |
| Software failure | Application bugs |
| Telecommunications | Network outage |
| Utility failure | Power outage |

### 1.7 Execution, Delivery, and Process Management

**Definition**: Losses from failed transaction processing.

| Subcategory | Examples |
|-------------|----------|
| Data entry errors | Incorrect input |
| Delivery failure | Missed deadlines |
| Accounting errors | Incorrect postings |
| Vendor disputes | Service failures |

---

## 2. Common Sources and Triggers

### 2.1 People-Related

| Source | Trigger Examples |
|--------|------------------|
| Skills gaps | Undertrained staff on new systems |
| Fatigue | Extended hours, alert fatigue |
| Incentive misalignment | Targets encouraging shortcuts |
| Turnover | Knowledge loss, training gaps |
| Misconduct | Intentional policy violations |

### 2.2 Process-Related

| Source | Trigger Examples |
|--------|------------------|
| Complexity | Too many manual steps |
| Ambiguity | Unclear procedures |
| Bottlenecks | Single point of failure |
| Exceptions | Non-standard handling |
| Changes | Process updates without controls |

### 2.3 Technology-Related

| Source | Trigger Examples |
|--------|------------------|
| Legacy systems | Outdated, unsupported software |
| Integration | System interface errors |
| Capacity | Insufficient resources |
| Releases | Deployment failures |
| Dependencies | Third-party system issues |

### 2.4 External

| Source | Trigger Examples |
|--------|------------------|
| Vendor failure | Supplier outage |
| Regulatory change | New requirements |
| Natural events | Weather, disasters |
| Market stress | Volume spikes |
| Infrastructure | Utility failures |

---

## 3. Risk Indicators and Early Warning

### 3.1 Key Risk Indicators (KRIs)

| Indicator | Green | Amber | Red |
|-----------|-------|-------|-----|
| Transaction error rate | < 0.1% | 0.1-0.5% | > 0.5% |
| System uptime | > 99.9% | 99-99.9% | < 99% |
| Staff turnover | < 10% | 10-20% | > 20% |
| Outstanding audit issues | < 5 | 5-10 | > 10 |
| Near-miss reports | Trend down | Stable | Trend up |
| Control test failures | < 5% | 5-15% | > 15% |
| Vendor SLA breaches | < 2/month | 2-5/month | > 5/month |

### 3.2 Leading Indicators

- Increased overtime/workload
- Postponed maintenance windows
- Staff complaints
- Process exceptions trending up
- Training completion delays
- Vendor performance deterioration

### 3.3 Lagging Indicators

- Loss events
- Customer complaints
- Regulatory findings
- Audit issues
- System outages
- Legal claims

---

## 4. Impact Analysis

### 4.1 Direct Financial Losses

| Category | Typical Range | Example |
|----------|---------------|---------|
| Error correction | $10K - $1M | Reprocessing costs |
| Regulatory fines | $100K - $100M+ | Compliance failures |
| Legal settlements | $1M - $100M+ | Class actions |
| Fraud losses | $100K - $10M | Internal fraud |
| System recovery | $100K - $10M | Major outage |

### 4.2 Indirect Impacts

| Impact | Description |
|--------|-------------|
| Business disruption | Lost revenue during outage |
| Customer attrition | Churn from service failures |
| Regulatory scrutiny | Enhanced supervision |
| Insurance premiums | Higher costs post-loss |
| Staff morale | Impact of incident stress |

---

## 5. Mitigation Strategies

### 5.1 Process Controls

| Control Type | Description | Examples |
|--------------|-------------|----------|
| **Segregation of duties** | Split critical functions | Approval vs. execution |
| **Dual control** | Two-person requirement | High-value transactions |
| **Reconciliation** | Independent verification | Daily position checks |
| **Limits** | Transaction/authority caps | Approval thresholds |
| **Exception handling** | Defined escalation | Non-standard procedures |

### 5.2 Technology Controls

| Control Type | Description | Examples |
|--------------|-------------|----------|
| **Validation** | Input/output checks | Format validation |
| **Automation** | Reduce manual intervention | Straight-through processing |
| **Monitoring** | Real-time alerting | Threshold triggers |
| **Redundancy** | Backup systems | Failover capability |
| **Change management** | Controlled releases | Testing, approval |

### 5.3 People Controls

| Control Type | Description | Examples |
|--------------|-------------|----------|
| **Training** | Skills development | Certification programs |
| **Competency** | Qualification requirements | Role prerequisites |
| **Supervision** | Oversight and review | Management review |
| **Incentives** | Aligned compensation | Risk-adjusted performance |
| **Culture** | Risk awareness | Reporting encouragement |

### 5.4 Recovery Controls

| Control Type | Description | Examples |
|--------------|-------------|----------|
| **BCP** | Business continuity | Disaster recovery |
| **Insurance** | Risk transfer | Operational loss policies |
| **Incident response** | Defined procedures | Escalation protocols |
| **Root cause analysis** | Post-incident learning | Process improvements |

---

## 6. Loss Data Collection

### 6.1 Loss Event Categories

| Field | Description |
|-------|-------------|
| Event ID | Unique identifier |
| Discovery date | When identified |
| Occurrence date | When occurred |
| Description | What happened |
| Category | Basel classification |
| Business line | Affected area |
| Gross loss | Total loss amount |
| Recoveries | Insurance, other |
| Net loss | Gross - recoveries |
| Root cause | Why it happened |
| Controls failed | What didn't work |
| Remediation | Actions taken |

### 6.2 Reporting Thresholds

| Threshold | Reporting Requirement |
|-----------|----------------------|
| > $10K | Business unit reporting |
| > $100K | Risk management review |
| > $500K | Executive notification |
| > $1M | Board reporting |

---

## 7. Risk and Control Self-Assessment (RCSA)

### 7.1 RCSA Process

```
1. Identify processes and risks
    │
    ├── 2. Assess inherent risk
    │       │
    │       ├── 3. Document controls
    │       │       │
    │       │       ├── 4. Assess control effectiveness
    │       │       │       │
    │       │       │       ├── 5. Calculate residual risk
    │       │       │       │       │
    │       │       │       │       └── 6. Identify action items
```

### 7.2 RCSA Frequency

| Risk Level | Assessment Frequency |
|------------|---------------------|
| Critical | Quarterly |
| High | Semi-annually |
| Medium | Annually |
| Low | Every 2 years |

---

## 8. Real-World Examples

### 8.1 Knight Capital (2012)

**Event**: Trading software deployment error caused $440M loss in 45 minutes.

**Category**: Execution, Delivery, Process Management + Systems

**Root Causes**:
- Inadequate deployment procedures
- Recycled code not fully removed
- Insufficient testing
- No kill switch activated in time

**Lessons**:
- Rigorous change management
- Automated deployment testing
- Real-time position monitoring
- Circuit breakers essential

### 8.2 Société Générale (2008)

**Event**: Rogue trader Jerome Kerviel caused €4.9B loss.

**Category**: Internal Fraud + Execution Failures

**Root Causes**:
- Control circumvention
- Inadequate supervision
- Failed reconciliation
- Risk management gaps

**Lessons**:
- Independent control verification
- Dual control on large positions
- Investigation of profits, not just losses
- Cultural red flags

---

## 9. Residual Risk Considerations

After controls, residual risk remains due to:

- Control failures (human error, system issues)
- Unknown scenarios (novel events)
- Control gaps (incomplete coverage)
- External factors (beyond control)

**Acceptance requires**:
- Documented control framework
- Regular testing evidence
- Monitoring in place
- Incident response ready
- Within risk appetite

---

## Cross-References

- [Risk Taxonomy](../frameworks/risk_taxonomy.md)
- [Technical Risk](../technical/technical_risk.md)
- [Business Continuity](business_continuity.md)

---

**Template**: Enterprise Risk Management v1.0
