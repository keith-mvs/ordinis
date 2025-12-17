# Data Evaluation Requirements

## Purpose

Ensure input data is fit for strategy research, backtesting, and live trading without bias or hidden defects.

## Standards

- Point-in-time: values as known at the time (no lookahead); include delisted/halted names.
- Timestamps: UTC; no overlaps; monotonic per symbol; align calendars across assets.
- Schema: explicit types (price floats, volume ints, tz-aware timestamps); record units and currency.
- Corporate actions: apply splits/dividends; maintain split factors and adjust volume.
- Vendor mix: document primary/secondary sources and failover rules.

## Quality Checks

- Missing/gaps: detect and quantify; fill only with documented rules; avoid forward-filling signals.
- Outliers: price/volume spikes beyond bands; cap or discard with rationale.
- Duplicates: drop or consolidate; log any corrections.
- Cross-source divergence: compare mid/close across feeds; alert on divergence thresholds.
- Survivorship bias: confirm presence of delisted symbols and their final prices/actions.

## Validation Snippet (Python)

```python
def validate_prices(df):
    issues = {}
    issues["missing"] = df.isna().mean()
    issues["time_monotone"] = (df.index.is_monotonic_increasing)
    issues["zero_volume"] = (df["volume"] == 0).mean()
    issues["dup_timestamps"] = df.index.duplicated().any()
    return issues
```

## Lineage & Manifests

- For every dataset: source, query params, date range, checksum, creation date.
- Store manifests with runs; pin manifest to backtest/report.

## Monitoring (live)

- Feed latency and gap detection; symbol coverage; stale price alarms.
- Alert on schema drift or unexpected nulls; auto-fail risky inputs.

## Governance

- Keep audit logs of corrections and overrides.
- Re-validate after vendor changes, symbol list updates, or major corporate actions.
