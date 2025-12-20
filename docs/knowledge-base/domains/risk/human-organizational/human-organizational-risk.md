# Human and Organizational Risk

**Section**: 03_risk/human_organizational
**Last Updated**: 2025-12-12
**Version**: 1.0

---

## Definition and Scope

**Human risk** encompasses the potential for losses or adverse outcomes arising from human behavior, decisions, errors, and misconduct.

**Organizational risk** addresses structural factors including culture, governance, and organizational design that can amplify or mitigate individual human risks.

---

## 1. Human Risk Categories

### 1.1 Key Person Risk

**Definition**: Risk that critical knowledge, relationships, or capabilities are concentrated in individuals.

| Factor | Risk Manifestation | Impact |
|--------|-------------------|--------|
| **Expertise concentration** | Sole person understands system | Knowledge loss on departure |
| **Relationship dependency** | Client ties to individual | Revenue loss on departure |
| **Decision authority** | Single point of control | Bottleneck or poor decisions |
| **Succession gaps** | No backup identified | Transition disruption |

**Key Person Assessment**:
```python
KEY_PERSON_FACTORS = {
    'knowledge': {
        'unique_skills': 'Only person with capability',
        'documentation': 'Knowledge documented?',
        'training': 'Others trained?'
    },
    'relationships': {
        'client_contacts': 'Primary relationship holder',
        'vendor_contacts': 'Key vendor relationships',
        'regulatory': 'Regulatory relationships'
    },
    'authority': {
        'signing_authority': 'Approval capabilities',
        'system_access': 'Privileged access',
        'decision_rights': 'Sole decision maker'
    }
}
```

### 1.2 Error Risk

**Definition**: Risk of losses from unintentional human mistakes.

| Error Type | Examples | Contributing Factors |
|------------|----------|---------------------|
| **Execution errors** | Wrong quantity, wrong symbol | Workload, interface design |
| **Judgment errors** | Poor decisions | Cognitive bias, fatigue |
| **Procedural errors** | Steps skipped, wrong sequence | Complexity, training |
| **Communication errors** | Misunderstanding, omission | Ambiguity, time pressure |

**Error Rate Factors**:
| Factor | Impact on Error Rate |
|--------|---------------------|
| Workload | High workload → higher errors |
| Fatigue | Extended hours → degraded performance |
| Complexity | More steps → more failure points |
| Training | Inadequate training → more mistakes |
| Tools | Poor interfaces → user error |
| Stress | High stress → impaired judgment |

### 1.3 Misconduct Risk

**Definition**: Risk of intentional wrongdoing by individuals.

| Type | Examples | Indicators |
|------|----------|------------|
| **Fraud** | Unauthorized trading, theft | Unexplained profits, lifestyle |
| **Market abuse** | Insider trading, manipulation | Trading patterns, communications |
| **Policy violations** | Circumventing controls | Workarounds, exceptions |
| **Conflicts of interest** | Self-dealing | Related party transactions |

### 1.4 Competence Risk

**Definition**: Risk that individuals lack required capabilities.

| Gap Type | Description | Consequences |
|----------|-------------|--------------|
| **Technical skills** | Insufficient expertise | Poor execution, errors |
| **Market knowledge** | Limited understanding | Wrong decisions |
| **Risk awareness** | Underestimating dangers | Excessive risk-taking |
| **Regulatory knowledge** | Compliance gaps | Violations |

---

## 2. Organizational Risk Categories

### 2.1 Culture Risk

**Definition**: Risk that organizational culture enables negative outcomes.

| Culture Dimension | Negative Manifestation | Consequences |
|-------------------|----------------------|--------------|
| **Risk culture** | Risk-taking rewarded without controls | Excessive risk |
| **Speak-up culture** | Issues suppressed | Late detection |
| **Accountability** | Blame-shifting | Repeat failures |
| **Ethics** | Win-at-all-costs | Misconduct |
| **Learning** | Punishment for failure | Hidden errors |

**Culture Assessment Areas**:
- Tone at the top
- Risk awareness and ownership
- Escalation effectiveness
- Consequence management
- Learning from mistakes

### 2.2 Governance Risk

**Definition**: Risk from inadequate oversight structures.

| Gap | Description | Impact |
|-----|-------------|--------|
| **Board oversight** | Insufficient expertise/attention | Strategic drift |
| **Committee structure** | Unclear mandates | Coverage gaps |
| **Information flow** | Inadequate reporting | Delayed awareness |
| **Decision rights** | Unclear authority | Delays, conflicts |
| **Challenge function** | Insufficient scrutiny | Groupthink |

### 2.3 Structure Risk

**Definition**: Risk from organizational design weaknesses.

| Issue | Description | Impact |
|-------|-------------|--------|
| **Silos** | Poor cross-functional coordination | Information gaps |
| **Span of control** | Over-stretched management | Inadequate oversight |
| **Reporting lines** | Conflicts, ambiguity | Confusion |
| **Centralization** | Excessive centralization/decentralization | Speed vs. control |
| **Complexity** | Too many layers, entities | Coordination failures |

### 2.4 Change Risk

**Definition**: Risk from organizational transitions.

| Change Type | Risks | Mitigation |
|-------------|-------|------------|
| **Restructuring** | Knowledge loss, confusion | Transition planning |
| **M&A** | Integration failures | Due diligence, integration management |
| **Leadership change** | Direction shifts | Succession planning |
| **Rapid growth** | Control strain | Scalable processes |
| **Downsizing** | Capability loss | Critical skill retention |

---

## 3. Early Warning Signals

### 3.1 Human Risk Indicators

| Indicator | Green | Amber | Red |
|-----------|-------|-------|-----|
| Staff turnover | < 10% | 10-20% | > 20% |
| Key person backup | All covered | Some gaps | Critical gaps |
| Training completion | > 95% | 80-95% | < 80% |
| Error rates | Trend down | Stable | Trend up |
| Near-misses | Reported, declining | Reported, stable | Not reported |
| Overtime hours | < 10% extra | 10-25% | > 25% |

### 3.2 Organizational Risk Indicators

| Indicator | Green | Amber | Red |
|-----------|-------|-------|-----|
| Employee engagement | > 75% | 60-75% | < 60% |
| Speak-up reports | Trend up | Stable | Trend down |
| Audit findings | < 5 open | 5-15 | > 15 |
| Escalation timeliness | < 24h for critical | 24-48h | > 48h |
| Meeting effectiveness | Decisions made | Some delay | Gridlock |
| Strategic alignment | Clear | Partially clear | Unclear |

---

## 4. Impact Analysis

### 4.1 Key Person Impact

| Scenario | Immediate Impact | Long-term Impact |
|----------|------------------|------------------|
| CTO departure | Technical decisions stall | Technology strategy drift |
| Lead quant departure | Model development stops | Alpha erosion |
| Head of trading departure | Trading capacity reduced | Client relationships |
| Compliance officer departure | Regulatory gap | Examination risk |

### 4.2 Misconduct Impact

| Type | Financial | Non-Financial |
|------|-----------|---------------|
| Rogue trading | Trading losses, legal costs | Reputation, regulatory |
| Fraud | Direct losses, fines | Trust, morale |
| Market abuse | Fines, disgorgement | License, reputation |
| Policy circumvention | Losses enabled | Control environment |

---

## 5. Mitigation Strategies

### 5.1 Key Person Risk Mitigation

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Succession planning** | Identify and develop successors | Talent reviews |
| **Cross-training** | Multiple people for critical roles | Rotation, shadowing |
| **Documentation** | Capture knowledge | Runbooks, procedures |
| **Retention** | Keep critical people | Compensation, engagement |
| **Transition planning** | Orderly handover | Notice periods, vesting |

### 5.2 Error Mitigation

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Automation** | Remove manual steps | Straight-through processing |
| **Validation** | Check inputs and outputs | Automated controls |
| **Dual control** | Two-person verification | Approval workflows |
| **Checklists** | Structured procedures | Process documentation |
| **Interface design** | User-friendly systems | UX design principles |
| **Training** | Skill development | Certification, practice |
| **Workload management** | Sustainable workloads | Capacity planning |

### 5.3 Misconduct Mitigation

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Screening** | Pre-employment checks | Background verification |
| **Monitoring** | Detect anomalies | Surveillance, analytics |
| **Speak-up channels** | Report concerns | Hotlines, ombudsman |
| **Consequences** | Deter misconduct | Consistent enforcement |
| **Culture** | Ethical environment | Tone from top, training |

### 5.4 Organizational Risk Mitigation

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Governance design** | Clear structures | Charters, mandates |
| **Board composition** | Diverse expertise | Refreshment, skills matrix |
| **Information flow** | Effective reporting | Dashboards, escalation |
| **Decision frameworks** | Clear authority | RACI, delegation |
| **Culture assessment** | Measure and monitor | Surveys, focus groups |

---

## 6. Performance Management

### 6.1 Balanced Scorecard

| Dimension | Metrics |
|-----------|---------|
| **Financial** | Revenue, costs, risk-adjusted returns |
| **Customer** | Satisfaction, retention |
| **Process** | Quality, efficiency, compliance |
| **People** | Engagement, development, retention |

### 6.2 Risk-Adjusted Compensation

| Principle | Implementation |
|-----------|----------------|
| **Risk alignment** | Defer compensation, clawback provisions |
| **Balanced metrics** | Not just financial results |
| **Behavior assessment** | How results achieved |
| **Compliance consideration** | Factor in compliance record |

---

## 7. Succession Planning

### 7.1 Succession Framework

```
1. Role Criticality Assessment
   └── Identify critical roles

2. Incumbent Risk Assessment
   └── Evaluate departure probability

3. Successor Identification
   └── Internal and external candidates

4. Development Planning
   └── Close readiness gaps

5. Transition Readiness
   └── Document for emergency succession
```

### 7.2 Successor Readiness

| Readiness Level | Definition | Action Required |
|-----------------|------------|-----------------|
| **Ready now** | Can step in immediately | Document, maintain |
| **Ready 1-2 years** | Needs development | Accelerated development |
| **Ready 3-5 years** | Longer-term potential | Longer-term development |
| **No successor** | Gap exists | External search, interim plan |

---

## 8. Culture Management

### 8.1 Risk Culture Dimensions

| Dimension | Positive Attributes | Negative Attributes |
|-----------|---------------------|---------------------|
| **Awareness** | Risks understood, discussed | Risks ignored, minimized |
| **Ownership** | Clear accountability | Blame-shifting |
| **Escalation** | Open, timely | Suppressed, delayed |
| **Challenge** | Constructive debate | Groupthink, deference |
| **Learning** | Improvement-focused | Punishment-focused |

### 8.2 Culture Change Levers

| Lever | Application |
|-------|-------------|
| **Leadership** | Model desired behaviors |
| **Communication** | Reinforce key messages |
| **Training** | Build capabilities and awareness |
| **Incentives** | Align rewards with behaviors |
| **Consequences** | Consistent enforcement |
| **Processes** | Embed in ways of working |
| **Metrics** | Measure and report |

---

## 9. Real-World Examples

### 9.1 Barings Bank (1995)

**Event**: Rogue trader Nick Leeson caused $1.3B loss.

**Human/Org Factors**:
- Inadequate supervision
- Trader controlled both front and back office
- Culture of deference to profitable trader
- Weak escalation

**Lessons**:
- Segregation of duties
- Independent oversight
- Culture of challenge
- Escalation encouragement

### 9.2 Wells Fargo (2016)

**Event**: Widespread fake account creation.

**Human/Org Factors**:
- Aggressive sales targets
- Fear-based culture
- Insufficient speak-up
- Inadequate board oversight

**Lessons**:
- Balanced incentives
- Speak-up channels
- Board scrutiny
- Culture monitoring

---

## 10. Residual Risk

Human and organizational risks remain due to:

- Human fallibility
- Unpredictable behavior
- Cultural complexity
- Organizational inertia
- External pressures

**Acceptance requires**:
- Key roles have succession
- Controls for high-risk activities
- Culture actively monitored
- Governance operating effectively
- Incident response capability

---

## Cross-References

- [Operational Risk](../operational/operational_risk.md)
- [Risk Governance](../frameworks/risk_governance.md)
- [Compliance Risk](../compliance/compliance_risk.md)

---

**Template**: Enterprise Risk Management v1.0
