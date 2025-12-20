# Documentation Style Guide
# Ordinis Trading System
# Version: 1.0.0
# Date: 2025-12-12

---

## Overview

This guide establishes documentation standards for the Ordinis project. Following these conventions ensures consistency, maintainability, and professional quality.

---

## File Naming Rules

### Naming Convention

Use **lowercase + kebab-case only** for all markdown files.

| Correct | Incorrect |
|---------|-----------|
| `architecture-overview.md` | `Architecture Overview.md` |
| `event-bus-contract.md` | `EVENT_BUS_CONTRACT.md` |
| `execution-path.md` | `execution_path.md` |

### Prohibited Patterns

- **No dates in filenames** (unless logs or release notes)
  - Put dates in document body or front matter
- **No version suffixes**: `final`, `v2`, `new`, `copy`
  - These belong in git history
- **No spaces**: Use hyphens instead
- **No underscores**: Use hyphens for word separation
- **No ALL_CAPS**: Use lowercase only

### Length Guidelines

- Keep filenames short but specific: **3-6 words max**
- One topic per file (split if multiple topics needed)

| Correct | Incorrect |
|---------|-----------|
| `event-bus-contract.md` | `detailed-event-bus-contract-and-event-taxonomy-spec.md` |
| `kill-switch.md` | `kill-switch-implementation-details-and-usage-guide.md` |

---

## Directory Structure

### Standard Documentation Areas

```
docs/
├── index.md                 # Documentation landing page
├── style-guide.md           # This file
│
├── architecture/            # System design and structure
│   ├── index.md
│   ├── context.md           # C4 Context diagram
│   ├── container.md         # C4 Container diagram
│   └── components-*.md      # Per-engine component docs
│
├── guides/                  # How-to guides (ordered)
│   ├── index.md
│   ├── 01-installation.md
│   ├── 02-configuration.md
│   └── 03-first-backtest.md
│
├── reference/               # API and technical reference
│   ├── index.md
│   └── *.md
│
├── operations/              # Operational procedures
│   ├── index.md
│   └── *.md
│
├── decisions/               # Architecture Decision Records (ADRs)
│   ├── index.md
│   └── adr-*.md
│
├── rfc/                     # Proposals (work in progress)
│   └── *.md
│
└── _assets/                 # Non-markdown assets
    ├── diagrams/            # Mermaid source files
    └── images/              # Exported images
```

### Index Files

Every docs folder **must** have an `index.md` as the landing page.

### Ordering Convention

Use **two-digit prefixes** only for guides requiring sequential reading:

```
docs/guides/
├── 01-installation.md
├── 02-configuration.md
├── 03-first-backtest.md
└── 04-paper-trading.md
```

Do **not** use numeric prefixes in reference/architecture docs unless there's a real sequence.

---

## Document Structure

### Standard Header Format

Every document should begin with:

```markdown
# Document Title

Brief one-sentence description of the document's purpose.

---

## Overview

[Expanded description of what this document covers]

---
```

### Standard Sections

**Architecture Documents:**
1. Overview / Purpose
2. Scope
3. Design / Structure
4. Contracts / Interfaces
5. Examples
6. References

**Guide Documents:**
1. Prerequisites
2. Steps (numbered)
3. Verification
4. Troubleshooting

**Reference Documents:**
1. Description
2. Parameters / Arguments
3. Return Values
4. Examples
5. See Also

### Document Metadata

Include metadata block at end of documents:

```markdown
---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
last_updated: "2025-12-12"
status: "draft|review|published"
```

---

**END OF DOCUMENT**
```

---

## Diagrams

### Diagram Standard

Use **Mermaid** by default (renders in GitHub + most doc stacks).

| Diagram Type | Use Case |
|-------------|----------|
| `flowchart` | High-level flows |
| `sequenceDiagram` | Message order, workflows |
| `stateDiagram-v2` | OMS/order lifecycle |
| `classDiagram` | Contracts (ports/adapters) |
| `erDiagram` | Database schemas |

### File Naming for Diagrams

Diagrams live as **source** + optional exported image:

```
docs/_assets/diagrams/<name>.<type>.mmd     # Source
docs/_assets/images/<name>.<type>.svg       # Export (SVG preferred)
```

Naming pattern: `<topic>.<context>.<diagram-type>.mmd`

Examples:
- `trading-flow.backtest.sequence.mmd`
- `execution.oms.state.mmd`
- `architecture.c4.container.mmd`

### Diagram Quality Gates

- No crossing lines if avoidable
- Every arrow has a **verb label** (`publishes`, `subscribes`, `calls`, `persists`)
- Include legend when symbols repeat
- Separate diagrams by concern:
  - C4 Context (system + external services)
  - C4 Container (major modules)
  - Component diagrams per engine
  - Sequence diagrams per workflow
  - State machine for orders/fills

### C4 Model

Adopt **C4** as the architecture backbone:

1. `architecture-context.md` - System context
2. `architecture-container.md` - Container diagram
3. `architecture-components-signalcore.md` - SignalCore components
4. `architecture-components-proofbench.md` - ProofBench components
5. (etc. per engine)

Each includes:
- Diagram
- "Contracts & Invariants" section

---

## Links and References

### Internal Links

Use relative paths for internal links:

```markdown
See [Architecture Overview](./architecture/index.md)
See [Installation Guide](../guides/01-installation.md)
```

### External Links

Use descriptive link text, not raw URLs:

```markdown
# Correct
See the [SQLite WAL documentation](https://www.sqlite.org/wal.html)

# Incorrect
See https://www.sqlite.org/wal.html
```

### Code References

When referencing code, include file path and line number:

```markdown
See `src/safety/kill_switch.py:150` for implementation details.
```

---

## Code Blocks

### Language Specification

Always specify the language:

````markdown
```python
def example():
    pass
```

```yaml
config:
  key: value
```

```bash
pip install ordinis
```
````

### Code Block Guidelines

- Keep code blocks focused (< 50 lines)
- Add comments for complex logic
- Use realistic variable names
- Include error handling in examples

---

## Writing Style

### Tone

- Technical and concise
- Professional but approachable
- Active voice preferred
- No unnecessary jargon

### Formatting

- Use **bold** for emphasis and key terms
- Use `code` for file names, commands, and code references
- Use *italics* sparingly for definitions
- Use > blockquotes for important notes or warnings

### Lists

- Use numbered lists for sequential steps
- Use bullet points for non-sequential items
- Keep list items parallel in structure

---

## Document Lifecycle

### WIP Documents

All work-in-progress docs live in `docs/rfc/` (proposal stage).

### Publishing

Once accepted:
1. Move to final home (`docs/architecture/`, `docs/reference/`, etc.)
2. Add ADR entry in `docs/decisions/` explaining *why*
3. Delete stale duplicates (git has history)

### Review Process

1. Create document in `docs/rfc/`
2. Request review
3. Address feedback
4. Move to final location
5. Update index files

---

## "Doc Done" Checklist

Before considering a document complete:

- [ ] Filename follows kebab-case convention
- [ ] Document has proper header structure
- [ ] All sections are complete
- [ ] Diagrams are included where helpful
- [ ] Links are tested and working
- [ ] Code examples are verified
- [ ] Metadata block is present
- [ ] Spelling and grammar checked
- [ ] Index files updated if needed

---

## Migration Notes

### Current State

Many existing documents use SCREAMING_SNAKE_CASE naming. These will be migrated to kebab-case incrementally.

### Migration Priority

1. Architecture docs (high visibility)
2. Guide docs (user-facing)
3. Reference docs
4. Knowledge base (lower priority)

### Backward Compatibility

During migration:
- Update links when renaming files
- Check for external references
- Update table of contents/indexes

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
source: "Markdown file naming standard (Wiki)"
```

---

**END OF DOCUMENT**
