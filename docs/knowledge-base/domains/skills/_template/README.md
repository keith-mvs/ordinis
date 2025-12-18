---
title: Skill Package Template (Knowledge Base)
date: 2025-12-18
version: 1.0.0
type: template
description: >
  Starter structure and standards for creating a new skill package under docs/knowledge-base/domains/skills/.
---

# Skill Package Template

This folder documents the **recommended structure and standards** for skill packages in:
`docs/knowledge-base/domains/skills/<skill-name>/`.

A “skill package” is a small, self-contained bundle that includes:
- an entrypoint doc (`SKILL.md`) with YAML front matter
- supporting documentation (`references/`)
- scripts (`scripts/`) when calculation/processing is needed
- assets (`assets/`) for templates and sample inputs

## How to Create a New Skill Package

1. Create a new folder: `docs/knowledge-base/domains/skills/<skill-name>/` (kebab-case).
2. Add `SKILL.md` and copy the front-matter pattern from an existing skill (examples below).
3. Add `references/` for deep explanations, formulas, edge cases, and longer examples.
4. Add `scripts/` only when you need executable helpers (Python calculators, validators, generators).
5. Add `assets/` for templates and sample inputs (not for primary documentation).
6. Wire it together with links (see `CROSS_LINKING_SCHEME.md`).

## Recommended Directory Layout

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

This template directory includes placeholders you can copy:
- `_assets/` → rename to `assets/`
- `_resources/` → rename to `references/`
- `_scripts/` → rename to `scripts/`

## `SKILL.md` Front Matter (Required)

Every skill entrypoint starts with YAML front matter:

```yaml
---
name: <skill-name>
description: <what it does>. Use when <when to use it>. Requires <optional dependencies>.
---
```

Guidelines:
- `name`: kebab-case, match the folder name
- `description`: third-person; include “what” + “when”; keep it concise

## Cross-Linking and Assets Standards

- Cross-linking rules: `CROSS_LINKING_SCHEME.md`
- `assets/` rules: `ASSETS_FOLDER_STANDARDS.md`

## Examples in This Repository

Use these as reference implementations:
- `docs/knowledge-base/domains/skills/due-diligence/due-dilligence.md`
- `docs/knowledge-base/domains/skills/financial-analysis/SKILL.md`
- `docs/knowledge-base/domains/skills/general-benchmarking/general-benchmarking.md`
- `docs/knowledge-base/domains/skills/technical-analysis/technical-indicators.md`
