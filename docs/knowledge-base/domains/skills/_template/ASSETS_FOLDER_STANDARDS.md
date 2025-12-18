---
title: Assets Folder Standards (Skill Packages)
date: 2025-12-18
version: 1.1.0
type: standards
description: >
  Defines what belongs in a skill package assets/ folder, how to structure and name assets, and how to reference them from SKILL.md, references/, and scripts/.
---

# Assets Folder Standards for Skill Packages

The `assets/` directory holds **supporting files used to produce outputs**: templates, sample inputs, and small static lookup tables.

It is intentionally distinct from:
- `references/`: documentation meant to be read and used for understanding
- `scripts/`: executable helpers (calculations, parsers, formatters)

If a human should read it to learn, put it in `references/`. If code should run it, put it in `scripts/`. If it is fed into a report/tool as an input or template, put it in `assets/`.

## What Belongs In `assets/`

### Output templates

Examples:
- report templates (`.md`, `.html`)
- email/message templates (`.txt`)
- diagram/graph templates (`.mmd`, `.drawio`) if they are used to generate outputs

Recommended locations:
- `assets/report-template.md` (few assets)
- `assets/templates/report-template.md` (many templates)

Template conventions:
- Use `[PLACEHOLDER]` markers for substitutions (e.g. `[TICKER]`, `[DATE]`)
- Include a short “Template Usage” comment block at the top describing required placeholders

### Sample inputs / example datasets

Examples:
- `sample_ohlcv.csv`
- `sample_financials.csv`
- `example_config.json`

Use sample assets to:
- demonstrate expected input schema
- cover typical and edge cases (missing values, outliers, low-volume bars)
- provide runnable examples for scripts

Recommended locations:
- `assets/sample_*.csv`
- `assets/samples/*.csv`

### Small static lookup / reference data

Examples:
- small calendars
- lookup tables
- defaults used by scripts

Recommended locations:
- `assets/reference-data/*.json`
- `assets/reference-data/*.csv`

Avoid storing large datasets in `assets/`.

## What Does *Not* Belong In `assets/`

- primary documentation → put in `references/`
- executable code → put in `scripts/`
- tests and fixtures → put in `tests/` or `tests/fixtures/` (if applicable)
- secrets / proprietary data → never commit; reference via env vars or URLs
- large datasets → store externally (or in the project’s data area) and document retrieval

## Structure

### Flat structure (preferred)

Use when there are only a few assets (roughly <5 files):

```
<skill-name>/
  assets/
    report-template.md
    sample_inputs.csv
    lookup-table.json
```

### Nested structure (when needed)

Use when the asset set grows:

```
<skill-name>/
  assets/
    templates/
      report-template.md
      memo-template.md
    samples/
      sample_inputs.csv
      example_config.json
    reference-data/
      holidays.csv
      lookup-table.json
```

Rule: keep it 1 level deep unless you have a strong reason to go deeper.

## Naming Conventions

### Directories

- lowercase kebab-case (e.g. `reference-data`, `sample-data`)

### Files

- markdown/templates: kebab-case (e.g. `report-template.md`)
- data files: prefer snake_case (e.g. `sample_financials.csv`) to match common tooling
- be specific; avoid `data.csv`, `template.md`, `final_v2.md`

## Size Guidance

- Target: <50 KB per file
- Soft limit: 500 KB (only if justified and still portable)
- If larger, document where to fetch it and how to verify integrity (hash, date, source)

## How To Reference Assets

### From the skill entrypoint

In `SKILL.md` (or the skill’s root `.md` file), link assets with relative paths:

```markdown
## Assets
- [Report template](assets/report-template.md)
- [Sample inputs](assets/sample_inputs.csv)
```

### From `references/`

Use `../assets/...`:

```markdown
See [sample inputs](../assets/sample_inputs.csv) for the expected schema.
```

### From `scripts/` (Python)

Use `Path(__file__).resolve()` to avoid brittle working-directory assumptions:

```python
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
sample_csv = SKILL_DIR / "assets" / "sample_inputs.csv"
```

## Examples In This Repository

- `docs/knowledge-base/domains/skills/due-diligence/assets/report-template.md`
- `docs/knowledge-base/domains/skills/financial-analysis/assets/financial-analysis-template.md`
- `docs/knowledge-base/domains/skills/technical-analysis/assets/sample_ohlcv.csv`

## Checklist

- [ ] Asset is used for output templating, sample input, or small static data
- [ ] Not a duplicate of content that belongs in `references/` or `scripts/`
- [ ] File name follows conventions and is specific
- [ ] Size is reasonable (or a retrieval method is documented)
- [ ] Linked from the skill entrypoint and/or referenced by a script (no orphan assets)

## Related Documentation

- [CROSS_LINKING_SCHEME.md](CROSS_LINKING_SCHEME.md) — link patterns for `SKILL.md` and `references/`
