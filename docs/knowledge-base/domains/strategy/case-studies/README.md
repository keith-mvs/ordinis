# Case Studies

## Status
No case studies are checked in yet. This folder is reserved for real implementations and post-mortems once strategies have been run in backtest or live environments.

## What belongs here
- Completed strategy write-ups with clear outcomes (win/loss) and lessons.
- Links to supporting artifacts (reports in `reports/`, plots in `artifacts/`, or code refs in `src/`/`tests/`).
- Operator notes from live incidents or interventions.

## Expected structure for each case study
- Title and date range covered.
- Objective and hypothesis.
- Data used (source, window, cleaning steps).
- Method (signals, sizing, risk controls, execution assumptions).
- Results (key metrics, charts path, notable trades).
- Risks and regressions observed.
- Lessons learned and action items (code, config, monitoring changes).

## Naming and placement
- Use kebab-case filenames (example: `mean-reversion-gap-close-2025.md`).
- Keep bulky artifacts outside this folder (reference them by relative path).
- If multiple studies share assets, add a short index at the top of this README linking to them.

## How to contribute a study
1) Draft the case study using the structure above.
2) Store large outputs in `reports/` or `artifacts/` and link them.
3) Cross-link to relevant strategy docs (formulation, backtesting requirements, performance attribution).
4) Mark status (draft/published) and add a last-updated date at the top of the file.
