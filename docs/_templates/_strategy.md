# Strategy Templates

## Purpose
This folder will house reusable blueprints for new strategies. Templates should capture the minimum set of sections, parameters, and checks needed to add a strategy consistently across research, backtesting, and live operation.

## Current status
- No templates are checked in yet. Use the structure below when adding the first set.

## What a template must include
- Overview: objective, instruments/universes, timeframes, and assumptions.
- Signals: inputs/features, signal logic, and decision criteria.
- Risk and sizing: position sizing rules, limits, stop/target logic.
- Data: required sources, sampling, cleaning, and feature freshness expectations.
- Backtesting plan: datasets, metrics, guardrails (aligned with backtesting-requirements.md).
- Deployment hooks: config knobs, runtime flags, rollout/rollback steps.
- Monitoring: health checks, drift/decay signals, alert thresholds, and owner.
- References: links to code (`src/`), tests (`tests/`), and related docs.

## Naming and placement
- Use kebab-case filenames, e.g., `momentum-breakout-template.md`.
- Keep template-specific assets (plots, large tables) outside this folder; reference them by relative path.

## How to add a template
1) Copy an existing template (once one exists) or start with the section list above.
2) Fill in required fields and align with strategy-formulation-framework.md and backtesting-requirements.md.
3) Link to any shared code/modules rather than duplicating logic.
4) Mark status (draft/published) and add a last-updated date.
5) Open a PR and tag reviewers from strategy/risk for sign-off.
