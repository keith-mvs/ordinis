import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "artifacts" / "broker_cost_model.csv"
OUT_PATH = ROOT / "docs" / "artifacts" / "BROKER_COSTS_SUMMARY.md"


def f(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def calc_row(r):
    trades = (
        f(r["Scenario"])
        and int(f(r.get("TradesPerMonth", 0)))
        or int(f(r.get("TradesPerMonth", 0)))
    )
    shares = f(r.get("AvgSharesPerTrade"))
    price = f(r.get("AvgPrice"))
    sell_frac = f(r.get("SellFraction"))  # 0..1

    commission_per_share = f(r.get("commission_per_share"))
    min_commission = f(r.get("min_commission_per_trade"))
    sec_rate_per_million = f(r.get("sec_rate_per_million"))  # e.g., $X per $1,000,000 of sells
    taf_per_share = f(r.get("taf_per_share"))
    cat_per_share = f(r.get("cat_per_share"))  # CAT fee per share
    exch_fee_per_share = f(r.get("exchange_fee_per_share"))

    short_util = f(r.get("short_utilization_fraction"))  # fraction of shares that are shorts
    borrow_rate = f(r.get("short_borrow_rate_annual"))  # APR, 0..1
    short_days = f(r.get("avg_short_hold_days"))

    # Commission: per trade = max(min_commission, commission_per_share * shares)
    commission_total = trades * max(min_commission, commission_per_share * shares)

    # Regulatory (sells): SEC + TAF
    # SEC fee on sells: (sell_notional / 1_000_000) * sec_rate_per_million
    sell_trades = int(round(trades * sell_frac))
    sell_notional = sell_trades * shares * price
    sec_fee_total = (sell_notional / 1_000_000.0) * sec_rate_per_million

    # TAF per share on sells (capped at $8.30 per trade for Alpaca)
    taf_per_trade = min(shares * taf_per_share, 8.30)
    taf_total = sell_trades * taf_per_trade

    # CAT fee per share on all trades (buys + sells)
    cat_total = trades * shares * cat_per_share

    # Exchange fees on all trades (buys + sells)
    exch_total = trades * shares * exch_fee_per_share

    # Short borrow cost: notional * (borrow_rate * days/365) * utilization
    # Approximate short notional = trades * shares * price * short_util * (sell_frac)
    short_notional = trades * shares * price * short_util * sell_frac
    borrow_total = short_notional * (borrow_rate * (short_days / 365.0))

    total = commission_total + sec_fee_total + taf_total + cat_total + exch_total + borrow_total

    return {
        "commission_total": commission_total,
        "sec_fee_total": sec_fee_total,
        "taf_total": taf_total,
        "cat_total": cat_total,
        "exchange_total": exch_total,
        "borrow_total": borrow_total,
        "monthly_total": total,
    }


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    summaries = {}
    for r in rows:
        scenario = r.get("Scenario", "Unknown")
        summaries[scenario] = calc_row(r)

    # Render MD
    lines = []
    lines.append("# Broker Costs Summary (from broker_cost_model.csv)")
    lines.append("")
    lines.append("| Scenario | Commission | SEC | TAF | CAT | Exchange | Borrow | Monthly Total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for scenario, s in summaries.items():
        lines.append(
            f"| {scenario} | ${s['commission_total']:.2f} | ${s['sec_fee_total']:.2f} | "
            f"${s['taf_total']:.2f} | ${s['cat_total']:.2f} | ${s['exchange_total']:.2f} | ${s['borrow_total']:.2f} | ${s['monthly_total']:.2f} |"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append(
        "- Alpaca: $0 commissions, TAF $0.000166/share (capped $8.30/trade), CAT $0.0000265/share."
    )
    lines.append("- SEC fee currently $0 per $1M (as of Oct 2025); check current rate.")
    lines.append("- Exchange fees/rebates vary by route/venue; conservative estimate used.")
    lines.append("- Borrow cost depends on utilization and borrow APR; adjust per symbol mix.")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
