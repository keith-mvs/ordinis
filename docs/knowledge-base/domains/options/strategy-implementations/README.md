# Options Strategy Implementations

## Purpose

This folder hosts practical documentation and assets for implementing common options strategies (spreads, income, hedges, and volatility plays).

## Core Guides

- `butterfly-spreads.md` — Debit/credit butterflies, payoff, setup rules, risk controls.
- `covered-strategies.md` — Covered calls/puts, assignment handling, sizing against stock.
- `iron-condors.md` — Construction, strike selection, adjustments, risk/return profile.
- `protective-strategies.md` — Protective puts/collars, hedge cost vs protection.
- `vertical-spreads.md` — Bull/bear debit/credit spreads, margin and Greeks impact.
- `volatility-strategies.md` — Straddles/strangles/boxes, IV vs RV considerations.

## Detailed Playbooks (by subfolder)

- `bear-put-spread/`, `bull-call-spread/` — Deep dives with references, examples, and calculators.
- `iron-butterfly/`, `iron-condor/` — Structure, spread width optimization, greeks management.
- `long-call-butterfly/`, `long-straddle/`, `long-strangle/` — Long vol profiles and earnings plays.
- `covered-call/`, `protective-collar/`, `married-put/` — Income and downside protection variants.
- `options-strategies/` — General references (Greeks, sources, strategy comparisons).

Each subfolder typically contains:
- `SKILL.md` (strategy overview), `references/` (mechanics, strike selection, examples),
  `assets/` (templates, sample positions), and `scripts/` (calculators/prototypes).

## How to Navigate

- Start with the core guides to choose a strategy family.
- Dive into the corresponding subfolder for step-by-step mechanics and examples.
- Use `scripts/` as reference only; production code should live under `src/` with tests.
- Check `references/` for strike selection, spread width, greeks, and management playbooks.
