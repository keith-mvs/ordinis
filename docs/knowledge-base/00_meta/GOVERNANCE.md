# Knowledge Base Governance

> **Process for managing, updating, and maintaining the KB**

---

## Overview

This document defines the governance process for the Intelligent Investor Knowledge Base, including:
- Who can make changes
- Review and approval workflows
- Versioning and change management
- Quality standards
- Conflict resolution

---

## Roles & Responsibilities

### Knowledge Engine Team (Core Team)

**Responsibilities:**
- Review and approve all KB changes
- Maintain schemas and taxonomy
- Ensure publication quality
- Resolve conflicts
- Quarterly KB audits

**Authority:**
- Approve new publications
- Modify taxonomy
- Update schemas
- Deprecate content

### Contributors

**Responsibilities:**
- Propose new publications
- Document concepts
- Create skills/commands
- Report errors or gaps

**Requirements:**
- Follow contribution guidelines
- Cite sources properly
- Pass schema validation
- Submit for review

### Users

**Rights:**
- Access all published KB content
- Search and query via skills
- Provide feedback
- Request new content

---

## Content Approval Workflows

### Adding a New Publication

**Process:**

1. **Proposal** (Contributor)
   - Submit publication metadata (JSON)
   - Justify relevance to domains
   - Provide access information

2. **Initial Review** (Core Team Member)
   - Check domain fit
   - Verify publication legitimacy
   - Assess licensing/copyright
   - Decision within 5 business days

3. **Metadata Creation** (Contributor)
   - Fill complete publication schema
   - Write 2-4 sentence summary
   - Identify key concepts
   - Tag with domains

4. **Detailed Documentation** (Contributor, Optional)
   - Create markdown file (like lópez_de_prado_advances.md)
   - Extract key topics
   - Map to system components
   - Cite specific chapters/sections

5. **Final Review** (Core Team)
   - Schema validation
   - Quality check
   - Cross-reference verification
   - Approve or request revisions

6. **Publication** (Core Team)
   - Add to publications/index.json
   - Update domain README if needed
   - Announce in changelog
   - Status: `indexed`

**Timeline:** 1-2 weeks from proposal to publication

**Rejection Criteria:**
- Not relevant to trading/finance
- Copyright issues
- Duplicate of existing publication
- Low quality or credibility
- Domain mismatch

---

### Adding a New Concept

**Process:**

1. **Draft** (Contributor)
   - Create markdown file following concept.schema.json
   - Assign unique ID: `concept_[domain]_[name]`
   - Write clear definition
   - Include examples

2. **Review** (Domain Expert)
   - Technical accuracy
   - Clarity of explanation
   - Example quality
   - References valid

3. **Approval** (Core Team)
   - Schema compliance
   - No duplication
   - Proper domain assignment

4. **Publication**
   - Status: `draft` → `reviewed` → `published`
   - Add to domain index
   - Link related concepts

**Timeline:** 3-5 days

---

### Creating a Skill/Command

**Process:**

1. **Design** (Contributor)
   - Define purpose and scope
   - Specify input/output schemas
   - List error cases
   - Provide examples

2. **Implementation** (Contributor)
   - Write skill markdown (.claude/commands/)
   - Follow skill.schema.json
   - Document engine integration
   - Test with sample data

3. **Testing** (Contributor + Reviewer)
   - Test all examples
   - Verify error handling
   - Check edge cases
   - Performance validation

4. **Review** (Core Team)
   - Code quality (if applicable)
   - Documentation completeness
   - Security check (permissions)
   - Integration correctness

5. **Release**
   - Version: v1.0.0
   - Status: experimental → stable
   - Announce availability

**Timeline:** 1-2 weeks

---

## Change Management

### Versioning Standards

**Semantic Versioning (v MAJOR.MINOR.PATCH):**

- **MAJOR:** Breaking changes (schema changes, domain restructuring)
- **MINOR:** New content (publications, concepts, skills)
- **PATCH:** Fixes (typos, clarifications, link updates)

**Current Version:** v1.0.0 (as of 2025-01-28)

### Change Types

#### Type 1: Metadata Updates (No Review Required)

- Typo fixes
- Link updates
- Formatting improvements
- Tag additions

**Process:** Direct commit with changelog entry

#### Type 2: Content Updates (Light Review)

- Adding examples to concepts
- Updating publication summaries
- Minor clarifications

**Process:** Peer review + approval

#### Type 3: Structural Changes (Full Review)

- Schema modifications
- New domains
- Taxonomy changes
- Deprecated content

**Process:** Proposal → Discussion → Vote → Implementation

---

## Quality Standards

### Publications

**Required:**
- Valid JSON metadata (passes schema validation)
- Accurate author information
- Legitimate access information (DOI, ISBN, URL)
- Clear domain mapping (1-9)
- Summary (50-1000 chars)
- At least 3 key concepts

**Recommended:**
- Detailed markdown documentation
- Chapter-level breakdown
- System integration notes
- Example applications

**Forbidden:**
- Copyrighted content without permission
- Fabricated publications
- Misleading summaries
- Incorrect attributions

---

### Concepts

**Required:**
- Unique ID following convention
- Single-sentence definition
- Detailed description (2-5 paragraphs)
- At least 1 publication reference
- Examples or applications

**Recommended:**
- Mathematical formulation (if applicable)
- Code examples
- Common misconceptions
- Related concepts

**Forbidden:**
- Plagiarized content
- Opinions presented as facts
- Incorrect formulas
- Missing citations

---

### Skills/Commands

**Required:**
- Clear purpose statement
- Complete input/output schemas
- Error handling documented
- At least 2 usage examples
- Engine integration specified

**Recommended:**
- Performance characteristics
- Security considerations
- Testing guidelines
- Related skills

**Forbidden:**
- Undocumented side effects
- Unhandled errors
- Security vulnerabilities
- Ambiguous behavior

---

## Conflict Resolution

### Publication Conflicts

**Scenario:** Two publications cover same topic

**Resolution:**
- Include both if different perspectives
- Tag appropriately (academic vs practical)
- Cross-reference in documentation
- User can filter by preference

### Concept Overlaps

**Scenario:** Similar concepts in different domains

**Resolution:**
- Keep if genuinely different applications
- Cross-reference heavily
- Clarify differences in descriptions
- Consider merging if 90%+ overlap

### Domain Boundary Disputes

**Scenario:** Unclear which domain owns a concept

**Resolution:**
1. Check TAXONOMY.md guidelines
2. Review existing similar concepts
3. Core team discussion
4. Majority vote if no consensus
5. Document rationale

---

## Review Cycles

### Quarterly Audits (Every 3 Months)

**Scope:**
- Check all publication links (dead link detection)
- Verify schema compliance
- Update publication years if new editions
- Review usage analytics
- Identify gaps

**Deliverable:** Audit report with action items

### Annual Review (Yearly)

**Scope:**
- Taxonomy relevance
- Domain restructuring needs
- Schema updates for new patterns
- Deprecated content removal
- Strategic priorities

**Deliverable:** Annual KB roadmap

---

## Deprecation Process

### Marking Deprecated

**Criteria:**
- Publication superseded by newer edition
- Concept no longer relevant
- Better alternative exists
- Security/accuracy issues

**Process:**
1. Change status to `deprecated`
2. Add deprecation notice with reason
3. Point to replacement (if any)
4. Set removal date (6 months minimum)
5. Notify users

**Example:**
```json
{
  "id": "pub_old_trading_book",
  "status": "deprecated",
  "deprecation": {
    "date": "2025-01-28",
    "reason": "Superseded by 10th edition",
    "replacement": "pub_hull_options_futures",
    "removal_date": "2025-07-28"
  }
}
```

### Permanent Removal

**Criteria:**
- Deprecated for 6+ months
- No active references
- Replacement available

**Process:**
1. Final warning (30 days notice)
2. Archive (move to archive/ folder)
3. Remove from indices
4. Update all references
5. Changelog entry

---

## Access Control

### Content Access Levels

**Public (Default):**
- All publications metadata
- All concepts
- All skills documentation
- All integration patterns

**Internal (Team Only):**
- Embedding vectors (when implemented)
- Usage analytics
- User feedback (if sensitive)
- Strategic roadmap

**Restricted:**
- Proprietary content (if licensed)
- Pre-publication drafts
- Internal discussions

---

## Contribution Guidelines

### How to Contribute

1. **Fork** the intelligent-investor repository
2. **Create** feature branch: `kb/add-publication-smith-trading`
3. **Add** content following schemas
4. **Validate** using schema validators
5. **Test** if applicable (skills)
6. **Submit** pull request with description
7. **Respond** to review feedback
8. **Merge** after approval

### Commit Message Format

```
[KB] <type>: <description>

Type: publication | concept | skill | schema | docs
```

**Examples:**
```
[KB] publication: Add López de Prado Advances in ML
[KB] concept: Add concept_07_position_sizing
[KB] skill: Implement /kb-search command
[KB] schema: Update publication.schema.json for v1.1
[KB] docs: Fix typos in GOVERNANCE.md
```

---

## License & Copyright

### KB Metadata

**License:** MIT License
- All metadata JSON files
- All documentation markdown
- All schemas

**Reuse:** Freely reusable with attribution

### Referenced Publications

**Rights:** Remain with original publishers
- We document metadata only
- Fair use for short excerpts
- Links to legitimate sources
- No redistribution of full texts

### Contributed Content

**Ownership:** Contributors retain copyright
**License:** Contribution grants KB usage rights
**Attribution:** Contributors acknowledged in metadata

---

## Metrics & KPIs

### Health Metrics

**Publication Quality:**
- % with detailed docs: Target >50%
- % with broken links: Target <5%
- Average concepts per publication: Target >5

**Concept Quality:**
- % with code examples: Target >30%
- % with multiple references: Target >60%
- User satisfaction: Target >4.0/5.0

**Skill Quality:**
- % with tests: Target 100%
- Average response time: Target <2s
- Error rate: Target <1%

**System Health:**
- Schema compliance: Target 100%
- Dead links: Target 0
- Quarterly audit completion: Target 100%

---

## Communication

### Announcement Channels

**For Contributors:**
- GitHub Discussions (KB category)
- Monthly changelog
- Quarterly newsletter

**For Users:**
- Release notes
- Skill documentation updates
- KB home page announcements

### Feedback Channels

**Bug Reports:** GitHub Issues (label: kb-bug)
**Feature Requests:** GitHub Issues (label: kb-feature)
**General Feedback:** GitHub Discussions
**Security Issues:** security@intelligent-investor.io (private)

---

## Emergency Procedures

### Critical Issues

**Examples:**
- Copyright violation discovered
- Security vulnerability in skill
- Massively incorrect concept
- Schema breaking change needed urgently

**Process:**
1. **Immediate:** Change status to `archived` or disable skill
2. **Notify:** All users via emergency notification
3. **Fix:** Priority resolution within 24 hours
4. **Review:** Post-mortem within 1 week
5. **Prevent:** Update processes to prevent recurrence

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| v1.0.0 | 2025-01-28 | Initial governance framework | KB Team |

---

## Approval

**Approved By:** Knowledge Engine Core Team
**Effective Date:** 2025-01-28
**Next Review:** 2025-04-28 (Quarterly)

---

## Navigation

- [TAXONOMY](TAXONOMY.md) - Domain definitions
- [INTEGRATION_PATTERNS](INTEGRATION_PATTERNS.md) - Model integration
- [VERSIONING](VERSIONING.md) - Version standards
- [Knowledge Base Home](../README.md)

**Document Version:** v1.0.0
**Last Updated:** 2025-01-28
**Status:** Active
