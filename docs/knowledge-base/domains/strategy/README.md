# Strategy Domain

## Purpose
This folder is the home for strategy design, testing, and deployment standards. Use it to: formulate new strategies, vet data, define backtests, document live runbooks, and decide when to retire or version a strategy.

## Contents
- standards/backtesting-requirements.md — What every backtest must include (data, metrics, checks).
- standards/data-evaluation-requirements.md — Data quality, completeness, timeliness, and acceptance criteria.
- standards/due-diligence-framework.md — Pre-deploy checklist (risk, stress, compliance, scenarios).
- standards/strategy-formulation-framework.md — Step-by-step process to design a new strategy.
- standards/nvidia-integration.md — How to use NVIDIA acceleration in research/backtesting.
- operations/live-trading-workflows.md — Runbooks for live operation, monitoring, and intervention.
- operations/performance-attribution.md — How to attribute returns and diagnose drivers/drag.
- operations/strategy-version-control.md — Versioning, A/B/shadow runs, and rollback guidance.
- operations/strategy-retirement.md — Criteria and process to pause/retire underperforming strategies.
- case-studies/ — Parking lot for real implementations and post-mortems (currently empty).
- strategy-templates/ — Patterns/checklists for new strategies (currently empty).

## How to use this folder
1) Designing a new strategy
- Start with standards/strategy-formulation-framework.md to frame hypothesis, features, signals, and risk.
- Validate inputs with standards/data-evaluation-requirements.md.
- Define your test plan using standards/backtesting-requirements.md.

2) Proving and hardening
- Run standards/due-diligence-framework.md before any live exposure.
- Use operations/performance-attribution.md to understand what is working and why.

3) Operating in production
- Follow operations/live-trading-workflows.md for monitoring, controls, and operator actions.
- Track versions and rollout/rollback per operations/strategy-version-control.md.
- Apply operations/strategy-retirement.md when performance or risk rules fail.

4) Sharing and learning
- Add concrete examples to case-studies/ when you have real results and lessons.
- Add reusable patterns to strategy-templates/ when you have a repeatable blueprint.

## Contribution notes
- Keep documents short, directive, and tied to code/tests where possible.
- Cross-link to data, risk, and execution domains when dependencies exist.
- Note status (draft/published) at the top of any new file and date updates.
