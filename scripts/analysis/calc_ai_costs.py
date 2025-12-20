import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "docs" / "artifacts" / "llm_inference_costs.csv"
OUT_PATH = ROOT / "docs" / "artifacts" / "AI_COSTS_SUMMARY.md"


def f(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def calc_row(r):
    pricing = (r.get("PricingType") or "").strip().lower()
    if pricing == "token":
        in_tok = f(r.get("input_tokens_per_month"))
        out_tok = f(r.get("output_tokens_per_month"))
        cin = f(r.get("cost_per_1k_input"))
        cout = f(r.get("cost_per_1k_output"))
        monthly = (in_tok / 1000.0) * cin + (out_tok / 1000.0) * cout
        return monthly, {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_in_per_1k": cin,
            "cost_out_per_1k": cout,
        }
    if pricing == "hour":
        rate = f(r.get("gpu_hour_rate"))
        hours = f(r.get("gpu_hours_per_month"))
        monthly = rate * hours
        return monthly, {"gpu_hour_rate": rate, "gpu_hours": hours}
    return 0.0, {}


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    lines = []
    lines.append("# AI/NVIDIA Model Costs Summary (from llm_inference_costs.csv)")
    lines.append("")
    lines.append("| Scenario | Provider | Model | Pricing | Monthly | Detail |")
    lines.append("|---|---|---|---|---:|---|")
    total = 0.0
    for r in rows:
        monthly, detail = calc_row(r)
        total += monthly
        detail_str = ", ".join(f"{k}={v}" for k, v in detail.items()) if detail else ""
        lines.append(
            f"| {r.get('Scenario')} | {r.get('Provider')} | {r.get('Model')} | {r.get('PricingType')} | ${monthly:.2f} | {detail_str} |"
        )

    lines.append("")
    lines.append(f"**Total (all rows)**: ${total:.2f} / month")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Fill token prices or GPU hourly rates based on your plan; rerun to refresh.")
    lines.append("- Token pricing is per 1K tokens; hours are billed at the GPU rate.")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
