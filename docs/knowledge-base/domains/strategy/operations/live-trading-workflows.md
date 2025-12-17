# Live Trading Workflows

## Purpose

Runbooks for operating strategies in production: monitoring, manual intervention, and controlled halts.

## Roles and Access

- Trader: pause/resume a strategy, adjust position size within limits.
- Risk: force kill, raise/relax limits, approve restarts.
- SRE: platform health, rollback, incident command.
- Auth: MFA + RBAC; audit every change.

## Readiness Checklist (before market open)

- Data feeds healthy; clocks in sync; pre-trade risk checks pass.
- Latency within SLO; alert channels (PagerDuty/Slack/email) green.
- Feature flags/versions recorded; rollback path tested.

## Active Monitoring (per strategy unless noted)

- PnL, drawdown, hit ratio, turnover, slippage vs budget.
- Risk: gross/net exposure, concentration, margin, VaR/limits.
- Market data: feed lag, gap detection, cross-source divergence.
- Infra: CPU/mem, queue depth, order gateway latency, error rate.
- Alerts: tiered (info/warn/critical) with runbook link and owner.

## Manual Intervention Protocol

Trigger when any of: unexpected PnL move, limit breach, feed divergence, gateway errors, or kill-switch trigger.
Steps:
1) Announce in ops channel; assign IC (incident commander).
2) Stabilize: reduce size or pause affected strategy only.
3) Validate data: check feed health, timestamps, price sanity.
4) Validate risk: exposure, open orders, stops in place.
5) Resume only after root cause understood and logged.

## Kill Switch

- Auto triggers: max intraday loss, drawdown, exposure/margin breach, feed outage, latency > SLO, error burst.
- Manual triggers: IC, Risk, or Trader with two-person approval for resume.
Actions: cancel open orders, freeze new orders, (optional) hedge net exposure, notify.

## Recovery and Restart

- Preconditions: data healthy, limits reset/approved, version pinned, rollback plan ready.
- Dry-run in paper/shadow if possible; small-size ramp with tighter limits.
- Log: trigger, actions, approvals, time to detect/mitigate/recover.

## Testing and Drills

- Weekly: kill-switch dry-run in paper; alert path check.
- Monthly: full incident drill (data feed loss, gateway fail).
- After every deployment: rollback simulation or shadow run.

## Metrics to Track

- MTTD/MTTR for incidents; false-positive/false-negative alerts.
- Kill-switch activations and recovery time.
- Slippage vs budget; latency p50/p95; error rate p95.

## Communication

- IC owns updates (what/impact/actions/ETA) in ops channel.
- Post-incident review within 24h; actions tracked to closure.
