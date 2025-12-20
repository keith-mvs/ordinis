---
name: due-diligence
description: Comprehensive due diligence and research framework that synthesizes data across multiple sources, identifies critical details in complex documents, builds compliance-ready audit trails, and accelerates analysis from weeks to days. Use when conducting technical assessments, market analysis, vendor evaluations, competitive research, investment analysis, risk assessments, or any situation requiring systematic investigation with documented findings and source attribution.
---

# Due Diligence & Research Framework

Systematic framework for conducting thorough due diligence that produces audit-ready documentation with complete source attribution and risk analysis.

## Core Workflow

### Phase 1: Scope Definition

**Establish investigation parameters:**

1. **Define objective** - What decision does this research inform? (e.g., vendor selection, technology adoption, market entry, investment decision)
2. **Identify scope** - What areas require investigation? (technical, financial, market, operational, legal, competitive)
3. **Set success criteria** - What specific questions must be answered? What evidence would validate or invalidate the hypothesis?
4. **Determine timeline** - What's the decision deadline? When are interim reports needed?

**Document scope in structured format:**
- Primary research question
- Critical success factors (must-haves vs. nice-to-haves)
- Known constraints (budget, time, access limitations)
- Stakeholders and their information needs

### Phase 2: Data Collection

**Systematic multi-source gathering:**

1. **Catalog available sources** - Identify what data exists (documents, databases, APIs, web sources, interviews)
2. **Prioritize by impact** - Focus first on sources that answer critical questions
3. **Establish baseline** - Document what's already known vs. what needs discovery
4. **Execute collection** - Gather data systematically, tracking source metadata

**For each source, capture:**
- Source identifier (URL, document name, API endpoint, interview subject)
- Collection timestamp
- Data type and format
- Relevance score (critical / important / supplementary)
- Access restrictions or limitations

**Use appropriate tools:**
- `web_search` and `web_fetch` for online research
- `Filesystem:read_file` for uploaded documents
- `bash_tool` for data processing and analysis
- MCP tools if integrated data sources are available

### Phase 3: Analysis & Synthesis

**Transform raw data into insights:**

1. **Extract key facts** - Pull specific claims, metrics, dates, names, relationships
2. **Cross-reference** - Verify claims across multiple sources; flag discrepancies
3. **Identify patterns** - Look for trends, correlations, red flags, or gaps
4. **Assess credibility** - Evaluate source reliability, bias, timeliness, completeness
5. **Synthesize findings** - Connect disparate information into coherent narrative

**Critical analysis dimensions:**
- **Factual accuracy** - Can claims be verified? Do sources agree?
- **Completeness** - What's missing? Where are the gaps?
- **Context** - What assumptions underlie the data? What's not being said?
- **Implications** - What does this mean for the decision at hand?

**Flag critical details:**
- Footnotes, disclaimers, or caveats that materially affect interpretation
- Temporal factors (expiration dates, outdated information, pending changes)
- Dependencies or assumptions that could invalidate findings
- Contradictions between sources requiring reconciliation

### Phase 4: Risk Assessment

**Systematic risk identification and evaluation:**

1. **Identify risk categories** - Technical, financial, operational, legal, market, reputational
2. **Document specific risks** - What could go wrong? What's the evidence?
3. **Assess probability** - How likely is this risk to materialize? (Low / Medium / High)
4. **Evaluate impact** - If it occurs, what's the damage? (Low / Medium / High / Critical)
5. **Propose mitigations** - What actions reduce exposure? What's the residual risk?

**Risk documentation format:**
```
Risk: [Specific risk statement]
Category: [Risk type]
Probability: [L/M/H] based on [evidence]
Impact: [L/M/H/C] because [reasoning]
Evidence: [Source citations]
Mitigation: [Recommended actions]
Residual Risk: [Remaining exposure after mitigation]
```

### Phase 5: Report Generation

**Produce compliance-ready documentation:**

1. **Executive summary** - Decision recommendation, key findings, critical risks (1-2 pages)
2. **Detailed findings** - Organized by category with full context and source attribution
3. **Risk matrix** - Visual summary of probability/impact assessment
4. **Source bibliography** - Complete list of sources with access dates and relevance notes
5. **Appendices** - Supporting data, detailed analyses, methodology notes

**Report structure:**

```markdown
# [Project Name] Due Diligence Report

**Date**: [Completion date]
**Analyst**: Claude
**Decision Deadline**: [If applicable]
**Recommendation**: [Clear recommendation with confidence level]

## Executive Summary

[3-5 key findings that drive the recommendation]
[Top 3 risks requiring attention]
[Critical action items]

## Findings by Category

### [Category 1: e.g., Technical Assessment]

**Key Finding**: [Insight]
**Evidence**: [Source citations]
**Implications**: [What this means]

[Repeat for each finding]

### [Category 2: e.g., Market Analysis]

[Structure as above]

## Risk Assessment

### Critical Risks
[High probability + High impact risks]

### Significant Risks
[Other material risks requiring mitigation]

### Minor Risks
[Lower priority risks for monitoring]

## Source Attribution

[Complete bibliography with access metadata]

## Appendices

[Supporting materials]
```

**Ensure audit trail integrity:**
- Every claim links to specific source(s)
- Timestamps document when information was current
- Methodology section explains analytical approach
- Limitations section acknowledges gaps or uncertainties

### Phase 6: Quality Assurance

**Pre-delivery validation:**

1. **Completeness check** - Are all critical questions answered?
2. **Source verification** - Can every key claim be traced to a source?
3. **Logic review** - Do conclusions follow from evidence?
4. **Gap identification** - What remains unknown? Is it material?
5. **Stakeholder alignment** - Does the report address stated needs?

**Common quality issues to prevent:**
- Unsourced claims presented as fact
- Outdated information without temporal context
- Cherry-picking data that supports predetermined conclusion
- Missing critical footnotes or caveats
- Recommendations that don't address identified risks

## Domain-Specific Workflows

For specialized due diligence types, see reference files:

- **Technical Due Diligence**: See [technical-dd.md](references/technical-dd.md) - Code quality, architecture, tech debt, scalability
- **Market Analysis**: See [market-analysis.md](references/market-analysis.md) - Competitive landscape, market size, trends, positioning
- **Financial Due Diligence**: See [financial-dd.md](references/financial-dd.md) - Revenue quality, unit economics, cap table, projections
- **Vendor Evaluation**: See [vendor-evaluation.md](references/vendor-evaluation.md) - Capability assessment, reference checks, contract terms
- **Compliance Review**: See [compliance-review.md](references/compliance-review.md) - Regulatory requirements, certifications, audit readiness

## Best Practices

**Source Management:**
- Maintain a running source log throughout investigation
- Note when sources were accessed (information can become stale)
- Flag paywalled or access-restricted sources
- Distinguish between primary sources (first-hand data) and secondary sources (analysis of data)

**Citation Standards:**
- Direct quotes require quotation marks and precise attribution
- Paraphrased information still requires citation
- Multiple sources supporting same fact strengthens credibility
- Contradictory sources require explicit discussion

**Footnote Vigilance:**
- Read all footnotes, disclaimers, and fine print carefully
- Material limitations often buried in footnotes
- Assumptions underlying financial projections typically in footnotes
- Legal disclaimers may reveal unstated risks

**Temporal Awareness:**
- Date all information (when was this true?)
- Flag upcoming events that could change analysis (expirations, pending regulations, competitive announcements)
- Distinguish between historical data, current state, and forward-looking projections

**Bias Recognition:**
- Consider source incentives (vendor marketing vs. independent review)
- Identify selection bias (survivorship bias in case studies)
- Recognize confirmation bias in your own analysis
- Seek disconfirming evidence deliberately

**Gap Management:**
- Document what you don't know
- Assess materiality of gaps
- Recommend additional investigation where critical
- Don't speculate to fill gaps; acknowledge uncertainty

## Output Formats

**Standard deliverable**: Full markdown report with embedded citations

**Alternative formats available:**
- Executive presentation (key findings + risk matrix)
- Decision memo (recommendation + supporting rationale)
- Comparison matrix (side-by-side vendor/option evaluation)
- Risk register (structured risk inventory with mitigations)
- Source-annotated document review (line-by-line analysis of contracts, proposals, technical docs)

**Request specific format in initial scope discussion.**

## Anti-Patterns to Avoid

- **Confirmation bias**: Seeking only information that supports initial hypothesis
- **Analysis paralysis**: Endless research without reaching conclusions
- **False precision**: Presenting uncertain estimates as exact figures
- **Recency bias**: Over-weighting recent information vs. long-term patterns
- **Authority bias**: Accepting expert claims without verification
- **Sunk cost fallacy**: Continuing investigation beyond point of diminishing returns

## Efficiency Techniques

**Rapid baseline establishment:**
- Start with comprehensive web search on key topics
- Identify authoritative sources and industry standards
- Map the information landscape before deep diving

**Parallel processing:**
- Investigate multiple areas simultaneously where possible
- Use structured templates to ensure consistent coverage
- Batch similar tasks (e.g., all web research, then all document review)

**Progressive refinement:**
- Start with broad strokes; narrow focus based on findings
- Identify decision-critical questions early; prioritize those
- Stop investigating when additional information won't change recommendation

**Leverage previous work:**
- Reference existing analyses where valid
- Build on prior due diligence rather than starting from scratch
- Adapt frameworks from similar projects

## Stakeholder Communication

**Interim updates:**
- Share preliminary findings early to course-correct if needed
- Flag critical risks immediately, don't wait for final report
- Request additional information or access when gaps discovered

**Final presentation:**
- Lead with recommendation and confidence level
- Support with 3-5 key findings
- Address anticipated objections
- Provide clear next steps

**Post-delivery:**
- Be available for questions and clarification
- Provide additional analysis if new information emerges
- Document lessons learned for future due diligence projects
