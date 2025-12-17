# Due Diligence Framework (Pre-Deploy)

## Purpose

Gate every strategy before live exposure with a consistent, auditable checklist.

## Core Checks

- **Data**: Passes data-evaluation requirements; manifests pinned; no lookahead/leakage; delistings included.
- **Backtest**: Meets backtesting requirements; OOS/WF degradation within limits; profitable after costs.
- **Risk**: Limits defined (exposure, concentration, DD, VaR/TE); position sizing consistent with live engine; kill-switch thresholds set.
- **Execution**: Slippage/commission modeled; venue/latency assumptions documented; order types supported; failure handling defined.
- **Dependencies**: Features/models/configs versioned; external services (feeds, brokers) validated; fallbacks defined.
- **Compliance**: Universes/blacklists/locate/short rules honored; audit logs enabled; PII/PIA controls if applicable.

## Go/No-Go Criteria

- Quant: Sharpe/Sortino/DD/TE meet targets; robustness tests passed; regime analysis done.
- Ops: Runbook ready (live-trading-workflows); monitoring/alerts configured; rollback plan tested in paper/shadow.
- Security/Access: RBAC/MFA enforced; secrets managed; approval workflow completed.
- Sign-offs: Strategy owner + Risk + Ops; approvals recorded with versions and dates.

## Required Artifacts (attach to ticket/record)

- Code/config/model version tags and data manifest.
- Backtest report with costs, robustness tests, and acceptance criteria outcomes.
- Risk limits and kill-switch settings.
- Deployment/rollback plan and owner contacts.
- Monitoring/alert thresholds and dashboards referenced.

## Post-Approval

- Start in shadow or canary by policy; log decisions for comparison.
- Time-bound review: set date for first live review and decay/health thresholds.
