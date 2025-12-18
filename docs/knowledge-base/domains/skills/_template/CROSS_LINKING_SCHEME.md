---
title: Cross-Linking Scheme (Skill Packages)
date: 2025-12-18
version: 1.1.0
type: standards
description: >
  Standard link patterns for SKILL.md, references/, scripts/, and assets/ within a skill package.
---

# Cross-Linking Scheme for Skill Packages

This document defines how skill packages should link across:
- the skill entrypoint (`SKILL.md`)
- reference documentation (`references/`)
- scripts (`scripts/`)
- assets (`assets/`)

Goals:
- make navigation obvious (humans can find “the next doc” quickly)
- keep deep detail out of the entrypoint via progressive disclosure
- keep links portable (relative paths, forward slashes)

## Canonical Structure (Recommended)

New skills should follow this structure:

```
<skill-name>/
  SKILL.md
  references/
    ...
  scripts/
    ...
  assets/
    ...
```

Note: some existing skills may use `<skill-name>.md` as the entrypoint instead of `SKILL.md`. When writing new references, prefer linking to `../SKILL.md` (or migrate the skill to `SKILL.md`).

## Link Patterns

### 1) From any reference back to the skill entrypoint

Put this immediately after the `#` title in `references/*.md`:

```markdown
**Parent Skill:** [<skill-name>](../SKILL.md)
```

If the skill uses a different entry file, link to that instead:

```markdown
**Parent Skill:** [<skill-name>](../<skill-name>.md)
```

### 2) Between reference docs (same directory)

Use sibling links without `./`:

```markdown
See also:
- [Position Sizing](position-sizing.md)
- [Examples](examples.md)
```

### 3) From `SKILL.md` to references (progressive disclosure)

`SKILL.md` should act as the navigation hub:

```markdown
## References
- [Core methodology](references/methodology.md)
- [Formulas](references/formulas.md)
- [Examples](references/examples.md)
```

Avoid nesting references more than 1 directory deep.

### 4) Linking to assets

From `SKILL.md`:

```markdown
- [Report template](assets/report-template.md)
```

From `references/*.md`:

```markdown
- [Sample inputs](../assets/sample_inputs.csv)
```

### 5) Linking to scripts

From `SKILL.md`:

```markdown
- [Calculator script](scripts/calculator.py)
```

From `references/*.md`:

```markdown
See [scripts/calculator.py](../scripts/calculator.py) for the implementation.
```

## Where To Place Links

### Start of file (recommended)

```markdown
# <Topic Title>

**Parent Skill:** [<skill-name>](../SKILL.md)
```

### In context (inline)

```markdown
Use the assumptions schema defined in [Inputs](inputs.md).
```

### End of file (“See also”)

Use this at the bottom of longer reference docs:

```markdown
---

## See also
- [<Related topic>](related-topic.md)
- [<Examples>](examples.md)
```

## Link Text Rules

- Prefer descriptive text: “Strike selection rules”, not “click here”.
- Keep link text stable: rename target files only when necessary, and update all inbound links.
- Use forward slashes (`references/examples.md`) even on Windows.

## Minimum Link Requirements

Every `references/*.md` file should include:
- a `Parent Skill` link to the entrypoint
- at least 2 sibling links in a “See also” section (or inline equivalents)

`SKILL.md` should include:
- a `References` section with links to every major doc in `references/`
- links to any important scripts and assets used in typical workflows

## Related Documentation

- [ASSETS_FOLDER_STANDARDS.md](ASSETS_FOLDER_STANDARDS.md) — what belongs in `assets/` and how to structure it
- [README.md](README.md) — how to copy and adapt this template
