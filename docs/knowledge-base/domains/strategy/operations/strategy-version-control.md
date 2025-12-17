# Strategy Version Control, A/B, and Rollback

## Purpose

Keep strategy changes auditable, testable, and safely reversible in live trading.

## Versioning Standards

- Semantic versions for strategies and configs: MAJOR.MINOR.PATCH (breaking, feature, fix).
- Pin dependencies (lock files, container digests); store artifacts in internal registry.
- Tag every production deploy with version, commit, data snapshot, and config hash.

## Rollout Patterns

- Shadow: run new version without trading; log decisions for comparison.
- A/B or canary: small % of flow on new version, deterministic routing by account/id.
- Blue/green: two identical stacks; switch traffic when green is healthy.

## Routing Example (deterministic split)

```python
import hashlib

def route(account_id):
    bucket = int(hashlib.md5(account_id.encode()).hexdigest(), 16) % 100
    return "new" if bucket < 10 else "current"  # 10% on new version
```

## Health Gates

- Pre-trade: limits loaded, risk checks pass, data latency within SLO.
- Post-deploy: monitor PnL, drawdown, slippage, error rate, latency vs control.
- Auto-rollbacks: trigger on drawdown/TE/slippage breaches, error bursts, or feed issues.

## Rollback Playbook

1) Freeze new orders; cancel outstanding if risk requires.
2) Flip traffic back to last good version (tagged).
3) Restore configs/artifacts; verify data sources and clocks.
4) Announce, record timestamps, and capture metrics (MTTD/MTTR).

## Data and Schema Versioning

- Version feature sets and model weights; keep manifests with checksums.
- Backward-compatible DB migrations; include downgrade scripts and tests.

## CI/CD Guardrails

- Require tests (unit/integration), lint/type checks, and sim/backtest smoke.
- Enforce approvals from strategy + risk for production tags.
- Store deployment logs and approvals for audit.

## Pitfalls to Avoid

- Version drift between code, configs, and models.
- Non-deterministic routing causing double-fills or missing orders.
- Unrehearsed rollbacks; untested downgrade scripts.

## Cadence

- Every change: tag, test, and document.
- Weekly: rehearsal of rollback in paper/shadow.
- Monthly: review routing logic, thresholds, and alert noise.
