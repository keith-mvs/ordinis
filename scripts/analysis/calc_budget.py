from collections import defaultdict
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "artifacts" / "budget_opex_estimate.csv"
OUT_PATH = ROOT / "docs" / "artifacts" / "BUDGET_SUMMARY.md"


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def coerce_float(val):
    if val is None:
        return 0.0
    v = str(val).strip()
    if not v:
        return 0.0
    try:
        return float(v)
    except ValueError:
        # In case of non-numeric placeholders
        return 0.0


def summarize(rows):
    totals = defaultdict(float)
    by_category = defaultdict(lambda: defaultdict(float))

    for r in rows:
        scenario = r.get("Scenario", "Unknown").strip() or "Unknown"
        m_cost = coerce_float(r.get("Monthly Cost"))
        cat = r.get("Category", "Other")
        totals[scenario] += m_cost
        by_category[scenario][cat] += m_cost

    return totals, by_category


def render_md(totals, by_category, csv_rel="docs/artifacts/budget_opex_estimate.csv"):
    lines = []
    lines.append("# Budget Summary (from budget_opex_estimate.csv)")
    lines.append("")
    lines.append(f"Source CSV: {csv_rel}")
    lines.append("")
    lines.append("## Scenario Totals")
    lines.append("")
    lines.append("| Scenario | Monthly Total | Annual Total |")
    lines.append("|---|---:|---:|")

    # Keep consistent ordering
    order = ["Lean", "Standard", "Enterprise", "Custom"]
    for scenario in order + [s for s in totals.keys() if s not in order]:
        monthly = totals.get(scenario, 0.0)
        annual = monthly * 12.0
        lines.append(f"| {scenario} | ${monthly:,.2f} | ${annual:,.2f} |")

    lines.append("")
    lines.append("## Category Breakdown (Monthly)")
    for scenario in order:
        if scenario not in by_category:
            continue
        lines.append("")
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append("| Category | Monthly |")
        lines.append("|---|---:|")
        cats = by_category[scenario]
        for cat, amt in sorted(cats.items()):
            lines.append(f"| {cat} | ${amt:,.2f} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Notes:")
    lines.append(
        "- Custom scenario rows default to $0.00 until you enter contracted rates in the CSV."
    )
    lines.append("- Totals reflect the 'Monthly Cost' column only.")
    lines.append("- Update the CSV and re-run this script to refresh totals.")
    return "\n".join(lines)


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")
    rows = load_rows(CSV_PATH)
    totals, by_category = summarize(rows)
    md = render_md(totals, by_category)
    OUT_PATH.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
