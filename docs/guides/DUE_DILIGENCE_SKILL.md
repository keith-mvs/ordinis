# Due Diligence & Research Skill

> Accelerate due diligence from weeks to days. Synthesize data across all your sources, catch footnotes that matter, and build audit trails that survive compliance reviews.

[![Skill Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/yourusername/due-diligence-skill)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Claude Compatible](https://img.shields.io/badge/claude-compatible-orange.svg)](https://claude.ai)

## 🎯 Overview

The **Due Diligence & Research** skill transforms Claude into a systematic analyst capable of conducting comprehensive, audit-ready research across multiple domains. Whether you're evaluating vendors, assessing market opportunities, conducting technical reviews, or preparing for compliance audits, this skill provides structured workflows and domain expertise to deliver professional-grade analysis.

### Key Benefits

- **⚡ Speed**: Reduce investigation timelines from weeks to days
- **🔍 Thoroughness**: Catch critical details hidden in footnotes, disclaimers, and fine print
- **📊 Synthesis**: Combine insights from web research, documents, databases, and interviews
- **✅ Audit-Ready**: Complete source attribution and compliance-grade documentation
- **⚖️ Risk-Aware**: Systematic risk identification with probability/impact assessment
- **🎯 Decision-Focused**: Clear recommendations backed by evidence

## 📦 What's Included

### Core Framework
Six-phase systematic workflow for conducting due diligence:
1. **Scope Definition** - Establish investigation parameters and success criteria
2. **Data Collection** - Multi-source systematic gathering with metadata tracking
3. **Analysis & Synthesis** - Transform raw data into actionable insights
4. **Risk Assessment** - Identify and evaluate risks with mitigation strategies
5. **Report Generation** - Produce compliance-ready professional documentation
6. **Quality Assurance** - Validate completeness, accuracy, and logical consistency

### Domain-Specific Expertise

**🔧 Technical Due Diligence**
- Architecture assessment & scalability analysis
- Code quality evaluation & technical debt quantification
- Technology stack viability & team capability
- Security posture & vulnerability management
- Performance benchmarking & capacity planning

**📈 Market Analysis**
- TAM/SAM/SOM market sizing with multiple methodologies
- Competitive landscape mapping & positioning strategy
- Customer segmentation & needs analysis
- Porter's Five Forces & SWOT frameworks
- Growth dynamics & barrier assessment

**💰 Financial Due Diligence**
- Revenue quality & composition analysis
- Unit economics (LTV:CAC, payback, Rule of 40)
- Profitability path & cash flow forecasting
- Cap table & equity structure review
- Valuation analysis with multiple methodologies

**🤝 Vendor Evaluation**
- Capability assessment & scope alignment
- Financial stability & business viability
- Performance track record & reference checks
- Contract terms & commercial comparison
- Security, compliance, & risk evaluation

**📋 Compliance Review**
- Regulatory landscape mapping (GDPR, HIPAA, SOC 2, etc.)
- Data privacy & protection assessment
- Security certification status & audit readiness
- Employment, IP, and contractual compliance
- Corporate governance & control evaluation

### Professional Templates
- **Report Template**: Markdown structure for executive summaries, detailed findings, risk matrices, and appendices
- **Comparison Matrices**: Side-by-side vendor/option evaluation frameworks
- **Risk Registers**: Structured risk documentation with mitigation plans

## 🚀 Installation

### Prerequisites
- Claude Pro, Team, or Enterprise account
- Access to Claude.ai web interface or Claude Desktop app

### Install Steps

1. **Download** the `due-diligence.skill` file from this repository

2. **Open Claude Settings**
   - Web: Click your profile → Settings → Skills
   - Desktop: Application menu → Settings → Skills

3. **Install the Skill**
   - Click "Install Skill" or "Add Custom Skill"
   - Upload the `due-diligence.skill` file
   - Confirm installation

4. **Verify Installation**
   - Start a new conversation
   - Type: "What due diligence skills do you have?"
   - Claude should acknowledge the due-diligence skill

## 📖 Usage

### Basic Usage

Simply describe your due diligence need in natural language:

```
"Conduct technical due diligence on our migration from Postgres to MongoDB"
```

```
"Evaluate HubSpot, Marketo, and ActiveCampaign for our marketing automation needs"
```

```
"Analyze the market opportunity for AI-powered code review tools"
```

### Advanced Usage

For more targeted analysis, provide specific details:

```
I need to evaluate three cloud providers (AWS, Azure, GCP) for our ML
infrastructure. Key requirements:
- GPU instances (A100 or equivalent)
- Managed Kubernetes with autoscaling
- MLOps tools (experiment tracking, model registry)
- Budget: $50K/month
- Must support multi-region deployment

Please create a comparison matrix with a recommendation by Friday.
```

### Specifying Output Format

Request the deliverable format you need:

```
"Create an executive presentation format (not full slides, just structured content)"
"Produce a decision memo with recommendation and top 3 supporting points"
"Build a detailed risk register with mitigation timelines"
"Generate a comparison matrix scoring the options"
```

## 💡 Example Workflows

### Technology Evaluation

**Scenario**: Evaluating database migration options

**Claude's Process**:
1. Loads technical due diligence reference framework
2. Researches both database systems (performance, scalability, ops complexity)
3. Analyzes migration risk and effort estimation
4. Compares total cost of ownership at scale
5. Identifies critical risks and mitigation strategies
6. Produces comparison report with clear recommendation

**Deliverable**: Technical assessment report with architecture implications, risk matrix, and migration roadmap

---

### Vendor Selection

**Scenario**: Selecting a CRM platform

**Claude's Process**:
1. Loads vendor evaluation framework
2. Catalogs requirements and scores weighted criteria
3. Researches vendor capabilities, pricing, and reviews
4. Conducts reference check research (G2, Capterra, case studies)
5. Analyzes contract terms and total cost of ownership
6. Creates comparison matrix with recommendation

**Deliverable**: Vendor comparison matrix with scores, qualitative analysis, and contract negotiation priorities

---

### Investment Analysis

**Scenario**: Evaluating a SaaS acquisition target

**Claude's Process**:
1. Loads financial, technical, and market analysis frameworks
2. Analyzes revenue quality, cohort retention, and unit economics
3. Assesses technology stack, technical debt, and team capability
4. Evaluates competitive positioning and market dynamics
5. Identifies key risks across all categories
6. Produces comprehensive due diligence report with valuation

**Deliverable**: Multi-dimensional DD report with executive summary, detailed findings by area, risk assessment, and transaction recommendation

---

### Compliance Readiness

**Scenario**: Preparing for SOC 2 Type II audit

**Claude's Process**:
1. Loads compliance review framework
2. Maps SOC 2 requirements to current controls
3. Identifies gaps and control deficiencies
4. Assesses evidence availability and documentation quality
5. Prioritizes remediation activities by risk and effort
6. Creates audit readiness roadmap

**Deliverable**: Gap analysis with prioritized remediation plan and timeline

## 🎨 Output Examples

### Standard Report Structure
```
# [Project] Due Diligence Report

## Executive Summary
- Clear recommendation with confidence level
- 3-5 key findings driving the decision
- Top 3 critical risks requiring attention
- Immediate action items

## Detailed Findings
[By category: Technical, Market, Financial, etc.]
- Key finding with evidence and citations
- Implications for decision
- Rating or score

## Risk Assessment
- Risk matrix (probability × impact)
- Detailed risk breakdown by priority
- Mitigation strategies with ownership

## Source Attribution
- Complete bibliography with access dates
- Source quality and reliability notes

## Recommendations
- Primary recommendation with rationale
- Conditions (if applicable)
- Next steps with timeline
```

### Comparison Matrix Format
```
| Criteria       | Weight | Vendor A | Vendor B | Vendor C |
|----------------|--------|----------|----------|----------|
| Capability     | 25%    | 4.0      | 3.5      | 4.5      |
| Track Record   | 20%    | 5.0      | 4.0      | 3.0      |
| Pricing        | 15%    | 3.0      | 4.0      | 5.0      |
| Security       | 20%    | 5.0      | 4.0      | 4.0      |
| Support        | 10%    | 4.0      | 5.0      | 3.0      |
| Innovation     | 10%    | 4.0      | 3.0      | 5.0      |
| TOTAL          | 100%   | 4.15     | 3.90     | 4.05     |
| RANKING        |        | 1st      | 3rd      | 2nd      |

✅ Vendor A: Best overall, strong security and track record
⚠️ Vendor B: Most mature but limited innovation
⚠️ Vendor C: Best pricing but execution risk
```

## 🎯 Best Practices

### 1. Be Specific About Scope
❌ **Too vague**: "Research this company"
✅ **Specific**: "Conduct financial due diligence focusing on revenue quality, unit economics, and burn rate for Company X"

### 2. Provide Context
Include details that shape the analysis:
- Your role and decision authority
- Timeline and urgency
- Budget constraints
- Must-have vs. nice-to-have criteria
- Stakeholders who will review findings
- Known constraints or limitations

### 3. State Your Hypothesis
If you have a preliminary view, share it:
- "I'm leaning toward Option A but concerned about scalability"
- "This looks promising but the pricing seems too good to be true"
- Claude can validate or challenge your hypothesis with evidence

### 4. Request Interim Updates
For complex multi-day analysis:
- "Can you share preliminary findings tomorrow?"
- "Flag any critical risks as soon as you discover them"
- Allows course-correction before final report

### 5. Challenge the Analysis
Ask tough questions:
- "What's the strongest argument against your recommendation?"
- "What could I be missing?"
- "What would change your mind?"
- "What's the bear case scenario?"

### 6. Iterate Based on Findings
After receiving initial analysis:
- Request deeper dives on specific concerning areas
- Ask for additional comparisons or scenarios
- Challenge assumptions that don't match your experience
- Request sensitivity analysis on key variables

## 🔍 Key Features

### Intelligent Source Management
- Tracks all sources with collection timestamps
- Distinguishes primary (first-hand) from secondary (analysis) sources
- Flags paywalled, restricted, or outdated content
- Notes information freshness and when updates are needed

### Critical Detail Detection
- Automatically surfaces footnotes and disclaimers
- Identifies temporal factors (expirations, pending changes)
- Flags dependencies and unstated assumptions
- Catches contradictions between sources requiring reconciliation

### Risk-Aware Analysis
- Systematic identification across 6+ risk categories
- Probability × Impact scoring methodology
- Mitigation strategy development with residual risk calculation
- Prioritization by materiality and urgency

### Audit Trail Integrity
- Every claim linked to specific source(s)
- Methodology and analytical approach documented
- Gaps and limitations explicitly acknowledged
- Quality assurance checklist applied

### Multi-Format Outputs
- Full reports (10-50 pages)
- Executive summaries (1-2 pages)
- Decision memos (2-3 pages)
- Comparison matrices (tables)
- Risk registers (structured lists)
- Presentation outlines (slide structure)

## 🛠️ Skill Architecture

```
due-diligence/
├── SKILL.md                          # Core 6-phase workflow framework
├── references/
│   ├── technical-dd.md               # Technical evaluation (43KB)
│   ├── market-analysis.md            # Market & competitive (38KB)
│   ├── financial-dd.md               # Financial assessment (51KB)
│   ├── vendor-evaluation.md          # Vendor selection (45KB)
│   └── compliance-review.md          # Regulatory compliance (41KB)
└── assets/
    └── report-template.md            # Professional report template (12KB)
```

**Total Package Size**: ~230KB
**Content**: 18,000+ words of frameworks, checklists, and best practices

## 🧪 Testing & Validation

This skill has been designed based on:
- Industry-standard due diligence frameworks (venture capital, private equity, consulting)
- ISO 27001, SOC 2, and GDPR compliance requirements
- Financial analysis methodologies (SaaS metrics, DCF, comps analysis)
- Technology evaluation best practices (code review, architecture assessment)

Recommended validation approach:
1. **Start simple**: Test with a known case where you have ground truth
2. **Compare outputs**: Validate against prior due diligence work you've done
3. **Verify sources**: Check that citations are accurate and relevant
4. **Challenge findings**: Ask questions to test the depth of analysis
5. **Iterate**: Provide feedback to improve future analyses

## 📊 Use Cases

### Investment & M&A
- Acquisition target evaluation
- Investment opportunity assessment
- Portfolio company analysis
- Competitive landscape research
- Market sizing and opportunity analysis

### Vendor Management
- Technology vendor selection
- Service provider evaluation
- Contract review and negotiation support
- Vendor risk assessment
- Multi-vendor comparison

### Strategic Planning
- Market entry feasibility
- Product-market fit validation
- Competitive positioning analysis
- Technology roadmap evaluation
- Build vs. buy decisions

### Compliance & Risk
- Regulatory compliance assessment
- Security posture evaluation
- Audit readiness review
- Risk identification and mitigation
- Policy gap analysis

### Technology Decisions
- Technology stack evaluation
- Architecture review
- Cloud provider selection
- Database migration assessment
- Tool and platform comparison

## ⚠️ Limitations

### What This Skill Does Well
✅ Structured research and synthesis across multiple sources
✅ Risk identification and assessment with mitigation strategies
✅ Comparative analysis and recommendation development
✅ Professional documentation with audit trails

### What This Skill Doesn't Do
❌ Legal advice (consult an attorney for legal matters)
❌ Financial advice (consult a licensed advisor for investment decisions)
❌ Real-time data access (knowledge cutoff applies)
❌ Proprietary databases (unless you provide the data)
❌ Replace human judgment (outputs should be reviewed and validated)

### Important Notes
- **Verification Required**: All findings should be independently verified for critical decisions
- **Professional Review**: Complex legal, financial, or technical matters require expert review
- **Current Information**: Web search results may be more current than training data
- **Context Dependent**: Quality of analysis depends on quality of inputs and available information

## 🤝 Contributing

This skill can be enhanced with:
- Company-specific due diligence checklists
- Industry-specific frameworks and metrics
- Internal data sources and templates
- Custom report formats and branding
- Specialized domain knowledge

To customize:
1. Unzip the `.skill` file (it's a zip archive)
2. Modify reference files or add new ones
3. Update SKILL.md to reference new content
4. Re-package using the packaging script
5. Reinstall the updated skill

## 📝 License

MIT License - See LICENSE file for details

## 🔗 Resources

- [Claude Skills Documentation](https://docs.claude.com/docs/skills)
- [Skill Creator Guide](https://github.com/anthropics/skill-creator)
- [Due Diligence Best Practices](https://example.com/dd-guide)

## 💬 Support

### Questions or Issues?
- Review the Quick Reference Guide included with this package
- Check existing documentation in the skill's reference files
- Test with simple cases before complex analyses
- Validate outputs against known ground truth

### Feedback
Share your experience:
- What worked well?
- What could be improved?
- What additional frameworks would be helpful?
- What use cases aren't well covered?

## 🗺️ Roadmap

Potential future enhancements:
- [ ] Additional domain frameworks (HR, operations, legal)
- [ ] Integration with data sources (CRM, financial systems)
- [ ] Industry-specific templates (SaaS, hardware, services)
- [ ] Automated report generation in multiple formats
- [ ] Benchmarking databases and comparative metrics
- [ ] Interactive questionnaires for scope definition

---

**Version**: 1.0.0
**Last Updated**: December 2025
**Maintained by**: Intelligent Investor Project

**Status**: ✅ Production Ready

---

## Quick Start Checklist

- [ ] Download `due-diligence.skill` file
- [ ] Install skill in Claude (Settings → Skills)
- [ ] Review Quick Reference Guide
- [ ] Test with a simple evaluation task
- [ ] Try a real due diligence project
- [ ] Provide feedback for improvements

**Ready to accelerate your due diligence process? Install the skill and start your first analysis today.**
