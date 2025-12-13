# Due Diligence Framework

**Section**: 04_strategy
**Last Updated**: 2025-12-12
**Source Skills**: [due-diligence](../../../.claude/skills/due-diligence/SKILL.md)

---

## Overview

Systematic framework for conducting thorough due diligence that produces audit-ready documentation with complete source attribution and risk analysis.

---

## Core Workflow

### Phase 1: Scope Definition

| Element | Description |
|---------|-------------|
| Objective | What decision does this research inform? |
| Scope | Areas requiring investigation (technical, financial, market, legal) |
| Success Criteria | Specific questions that must be answered |
| Timeline | Decision deadline and interim milestones |

**Scope Documentation Template**:
```markdown
- Primary research question:
- Critical success factors:
- Known constraints:
- Stakeholders:
```

### Phase 2: Data Collection

**Source Categories**:
- Documents (uploaded files, contracts, reports)
- Databases (internal systems, market data)
- Web sources (company sites, news, filings)
- Interviews (subject matter experts)

**Source Tracking**:
```markdown
| Source ID | Type | Access Date | Relevance | Restrictions |
|-----------|------|-------------|-----------|--------------|
| S001 | SEC Filing | 2025-12-12 | Critical | Public |
| S002 | Interview | 2025-12-11 | Important | Confidential |
```

### Phase 3: Analysis & Synthesis

**Critical Analysis Dimensions**:

| Dimension | Questions |
|-----------|-----------|
| Accuracy | Can claims be verified? Do sources agree? |
| Completeness | What's missing? Where are gaps? |
| Context | What assumptions underlie the data? |
| Implications | What does this mean for the decision? |

**Cross-Reference Protocol**:
- Verify claims across multiple sources
- Flag discrepancies explicitly
- Note confidence level for each finding

### Phase 4: Risk Assessment

**Risk Documentation Format**:
```markdown
**Risk**: [Specific risk statement]
**Category**: Technical / Financial / Operational / Legal / Market
**Probability**: Low / Medium / High (based on [evidence])
**Impact**: Low / Medium / High / Critical (because [reasoning])
**Evidence**: [Source citations]
**Mitigation**: [Recommended actions]
**Residual Risk**: [Remaining exposure after mitigation]
```

**Risk Matrix**:

|  | Low Impact | Medium Impact | High Impact | Critical |
|--|------------|---------------|-------------|----------|
| **High Prob** | Monitor | Mitigate | Mitigate | Escalate |
| **Med Prob** | Accept | Monitor | Mitigate | Mitigate |
| **Low Prob** | Accept | Accept | Monitor | Monitor |

### Phase 5: Report Generation

**Standard Report Structure**:
```markdown
# [Project] Due Diligence Report

**Date**: YYYY-MM-DD
**Recommendation**: [Clear recommendation + confidence level]

## Executive Summary
- Key findings (3-5 bullets)
- Top risks (3 items)
- Critical action items

## Findings by Category

### [Category 1]
**Finding**: [Insight]
**Evidence**: [Citations]
**Implications**: [Impact]

## Risk Assessment
### Critical Risks
### Significant Risks
### Minor Risks

## Source Attribution
[Complete bibliography]

## Appendices
[Supporting materials]
```

### Phase 6: Quality Assurance

**Pre-Delivery Checklist**:
- [ ] All critical questions answered
- [ ] Every claim traced to source
- [ ] Conclusions follow from evidence
- [ ] Gaps identified and assessed
- [ ] Stakeholder needs addressed

---

## Domain-Specific Considerations

### Technical Due Diligence

| Focus Area | Key Questions |
|------------|---------------|
| Architecture | Is it scalable? Maintainable? |
| Tech Debt | What's the remediation cost? |
| Security | What vulnerabilities exist? |
| Team | What expertise is available? |

### Financial Due Diligence

| Focus Area | Key Questions |
|------------|---------------|
| Revenue Quality | Recurring vs. one-time? |
| Unit Economics | LTV/CAC ratio? |
| Projections | Assumptions reasonable? |
| Liabilities | Hidden obligations? |

### Market Analysis

| Focus Area | Key Questions |
|------------|---------------|
| Market Size | TAM/SAM/SOM? |
| Competition | Positioning and moats? |
| Trends | Growth drivers? |
| Barriers | Entry/exit barriers? |

---

## Best Practices

### Source Management

- Maintain running source log
- Note access timestamps (information can become stale)
- Distinguish primary (first-hand) from secondary (analysis)
- Flag access-restricted sources

### Citation Standards

| Type | Requirement |
|------|-------------|
| Direct Quote | Quotation marks + precise attribution |
| Paraphrase | Citation required |
| Multiple Sources | Strengthens credibility |
| Contradictions | Explicit discussion required |

### Footnote Vigilance

- Material limitations often buried in footnotes
- Financial projection assumptions typically in footnotes
- Legal disclaimers may reveal unstated risks
- Always read fine print

### Bias Recognition

| Bias | Mitigation |
|------|------------|
| Confirmation | Seek disconfirming evidence |
| Selection | Look for survivorship bias |
| Recency | Balance with historical data |
| Authority | Verify expert claims |

---

## Anti-Patterns to Avoid

| Pattern | Problem |
|---------|---------|
| Confirmation Bias | Only seeking supporting data |
| Analysis Paralysis | Endless research, no conclusions |
| False Precision | Uncertain estimates as exact figures |
| Sunk Cost | Continuing beyond diminishing returns |

---

## Efficiency Techniques

### Rapid Baseline

1. Comprehensive web search on key topics
2. Identify authoritative sources
3. Map information landscape before deep dive

### Progressive Refinement

1. Start with broad strokes
2. Identify decision-critical questions
3. Narrow focus based on findings
4. Stop when additional info won't change recommendation

---

## Output Formats

| Format | Use Case |
|--------|----------|
| Full Report | Comprehensive analysis |
| Executive Presentation | Key findings + risk matrix |
| Decision Memo | Recommendation + rationale |
| Comparison Matrix | Side-by-side evaluation |
| Risk Register | Structured risk inventory |

---

## Cross-References

- [Strategy Formulation](strategy_formulation_framework.md)
- [Backtesting Requirements](backtesting-requirements.md)
- [Financial Analysis](../02_signals/fundamental/skill_integration.md)

---

**Template**: KB Skills Integration v1.0
