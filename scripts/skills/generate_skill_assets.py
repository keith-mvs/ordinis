#!/usr/bin/env python3
"""Generate appropriate assets for all skill packages."""

from pathlib import Path
from typing import Dict, List

ORDINIS_ROOT = Path(__file__).parent.parent
SKILLS_DIR = ORDINIS_ROOT / ".claude" / "skills"


# Asset templates by skill category
OPTIONS_STRATEGIES = [
    "bear-put-spread",
    "bull-call-spread",
    "covered-call",
    "iron-butterfly",
    "iron-condor",
    "long-call-butterfly",
    "long-straddle",
    "long-strangle",
    "married-put",
    "protective-collar",
]

BOND_ANALYSIS = [
    "bond-benchmarking",
    "bond-pricing",
    "credit-risk",
    "duration-convexity",
    "option-adjusted-spread",
    "yield-measures",
]


def generate_sample_positions_csv(strategy: str) -> str:
    """Generate sample positions CSV for options strategies."""
    if strategy == "married-put":
        # Already has good sample data
        return """ticker,stock_price,shares,investment_size,put_strike,put_premium,days_to_expiration,iv_level,iv_pct,scenario_description
COST,52.75,200,10550,50.00,1.85,45,low,18,Large-cap defensive consumer stock with stable fundamentals
SNAP,23.80,400,9520,22.00,2.65,60,high,58,Small-cap social media with high volatility and growth potential
TDOC,38.50,300,11550,37.00,2.15,60,medium,32,Mid-cap telemedicine stock with moderate volatility
PLTR,28.40,500,14200,27.00,2.40,45,medium,38,Mid-cap data analytics with elevated volatility around earnings
MRNA,175.50,100,17550,170.00,8.25,90,high,52,Biotech large-cap with significant volatility and binary events"""

    elif strategy in ["bear-put-spread", "bull-call-spread"]:
        # Vertical spreads
        direction = "bearish" if "bear" in strategy else "bullish"
        return f"""ticker,entry_price,long_strike,short_strike,long_premium,short_premium,net_debit,expiration_days,iv_rank,scenario
SPY,450.00,450.00,445.00,6.50,3.25,3.25,45,45,moderate_{direction}_with_technical_confirmation
QQQ,380.00,375.00,370.00,7.25,3.75,3.50,60,38,conservative_{direction}_swing_trade
IWM,195.00,195.00,190.00,4.50,2.00,2.50,30,55,aggressive_{direction}_short_term
AAPL,175.00,175.00,170.00,5.75,2.50,3.25,45,42,tech_sector_{direction}_play
TSLA,250.00,245.00,235.00,12.50,5.00,7.50,60,68,high_volatility_{direction}_position"""

    elif strategy in ["iron-butterfly", "iron-condor"]:
        # Neutral multi-leg strategies
        return """ticker,entry_price,call_spread_strikes,put_spread_strikes,net_credit,expiration_days,iv_rank,expected_range,scenario
SPY,450.00,455-460,445-440,2.50,45,55,445-455,earnings_quiet_period_range_bound
QQQ,380.00,385-390,375-370,3.25,60,48,375-385,tech_consolidation_after_rally
DIA,350.00,355-360,345-340,2.75,45,42,345-355,blue_chip_low_volatility_grind
IWM,195.00,200-205,190-185,3.00,30,60,190-200,small_cap_choppy_action
XLE,85.00,88-91,82-79,1.75,45,38,82-88,energy_sector_sideways_trading"""

    elif strategy in ["long-straddle", "long-strangle"]:
        # Volatility strategies
        spread_type = "same strike" if "straddle" in strategy else "different strikes"
        return f"""ticker,entry_price,call_strike,put_strike,call_premium,put_premium,net_debit,expiration_days,iv_rank,catalyst,scenario
AAPL,175.00,175.00,{175 if 'straddle' in strategy else 170},8.50,{8.50 if 'straddle' in strategy else 6.25},{'17.00' if 'straddle' in strategy else '14.75'},10,45,earnings,pre_earnings_volatility_expansion_expected
TSLA,250.00,250.00,{250 if 'straddle' in strategy else 240},18.00,{18.00 if 'straddle' in strategy else 12.50},{'36.00' if 'straddle' in strategy else '30.50'},15,72,product_launch,binary_event_large_move_either_direction
META,320.00,320.00,{320 if 'straddle' in strategy else 310},14.25,{14.25 if 'straddle' in strategy else 9.75},{'28.50' if 'straddle' in strategy else '24.00'},10,58,earnings,tech_earnings_high_implied_move
AMZN,145.00,145.00,{145 if 'straddle' in strategy else 140},7.50,{7.50 if 'straddle' in strategy else 5.25},{'15.00' if 'straddle' in strategy else '12.75'},20,48,fed_decision,macro_event_breakout_potential"""

    elif strategy == "protective-collar":
        # Stock + options protection
        return """ticker,stock_price,shares,cost_basis,put_strike,put_premium,call_strike,call_premium,net_cost,expiration_days,scenario
AAPL,175.00,100,16800,170.00,4.50,185.00,3.25,125,90,protect_gains_in_concentrated_position
MSFT,380.00,50,18500,370.00,9.25,395.00,6.50,275,90,hedge_portfolio_core_holding
NVDA,485.00,30,14000,470.00,18.00,510.00,12.00,600,60,volatility_protection_with_upside_cap
GOOGL,140.00,100,13200,135.00,3.75,150.00,2.50,125,90,earnings_season_downside_insurance
META,320.00,50,15500,310.00,11.00,340.00,7.50,350,60,protect_unrealized_gains_tech_position"""

    elif strategy == "covered-call":
        # Income generation
        return """ticker,stock_price,shares,cost_basis,call_strike,call_premium,expiration_days,annualized_return,scenario
AAPL,175.00,100,17000,180.00,2.25,30,15.8,monthly_income_above_cost_basis
T,18.50,500,9000,19.00,0.35,45,8.5,high_yield_dividend_enhanced_with_calls
F,12.25,800,9600,13.00,0.28,30,11.2,deep_value_income_generation
INTC,44.00,200,8400,46.00,1.10,45,11.8,turnaround_story_income_while_waiting
XOM,110.00,100,10500,115.00,1.85,30,21.1,energy_covered_call_high_premium"""

    elif strategy == "long-call-butterfly":
        # Low-cost neutral strategy
        return """ticker,entry_price,lower_strike,middle_strike,upper_strike,max_profit,max_loss,expiration_days,iv_rank,scenario
SPY,450.00,445.00,450.00,455.00,3.75,1.25,45,42,pinning_at_major_resistance
QQQ,380.00,375.00,380.00,385.00,3.50,1.50,30,38,expected_consolidation_near_current_price
IWM,195.00,190.00,195.00,200.00,3.25,1.75,45,48,range_bound_small_cap_mean_reversion
DIA,350.00,345.00,350.00,355.00,3.00,2.00,60,35,low_volatility_blue_chip_stability"""

    return ""


def generate_analysis_template(strategy: str) -> str:
    """Generate analysis report template."""
    category = "Options" if strategy in OPTIONS_STRATEGIES else "Bond"

    template = f"""# {strategy.replace('-', ' ').title()} Analysis Report

**Date:** [REPORT_DATE]
**Analyst:** Claude
**Symbol:** [TICKER_SYMBOL]

---

## Position Summary

**Strategy:** {strategy.replace('-', ' ').title()}
**Entry Date:** [ENTRY_DATE]
**Current Date:** [CURRENT_DATE]
**Days in Trade:** [DAYS_HELD]

### Position Details

[POSITION_STRUCTURE]

---

## Current Status

**Underlying Price:** $[CURRENT_PRICE]
**Entry Price:** $[ENTRY_PRICE]
**Price Change:** [PRICE_CHANGE]% ([DIRECTION])

### Profit/Loss

**Current P&L:** $[CURRENT_PL] ([PL_PERCENTAGE]%)
**Maximum Profit:** $[MAX_PROFIT]
**Maximum Loss:** $[MAX_LOSS]
**% of Max Profit Captured:** [PERCENT_OF_MAX]%

---

## Risk Metrics

"""

    if category == "Options":
        template += """### Greeks

**Delta:** [POSITION_DELTA] (expected $[DELTA_IMPACT] per $1 move)
**Gamma:** [POSITION_GAMMA] (delta acceleration)
**Theta:** [POSITION_THETA] (daily time decay: $[THETA_DAILY])
**Vega:** [POSITION_VEGA] (IV sensitivity: $[VEGA_IMPACT] per 1% IV change)

### Implied Volatility

**Current IV:** [CURRENT_IV]%
**Entry IV:** [ENTRY_IV]%
**IV Change:** [IV_CHANGE] points
**IV Rank:** [IV_RANK] (percentile: [IV_PERCENTILE])

### Time Analysis

**Days to Expiration:** [DTE]
**Theta Acceleration:** [THETA_STATUS] (Normal / Accelerating / Critical)
**Time Value Remaining:** $[TIME_VALUE]

"""
    else:  # Bond analysis
        template += """### Duration and Convexity

**Macaulay Duration:** [DURATION] years
**Modified Duration:** [MOD_DURATION]
**Convexity:** [CONVEXITY]
**DV01:** $[DV01] per $1MM

### Yield Analysis

**Current Yield:** [CURRENT_YIELD]%
**Yield to Maturity:** [YTM]%
**Yield to Call:** [YTC]% (if applicable)
**Spread to Benchmark:** [SPREAD] bps

"""

    template += """---

## Technical Analysis

### Price Action

**Support Levels:** [SUPPORT_1], [SUPPORT_2], [SUPPORT_3]
**Resistance Levels:** [RESISTANCE_1], [RESISTANCE_2], [RESISTANCE_3]
**Trend:** [TREND_DIRECTION] ([TREND_STRENGTH])

### Indicators

**RSI(14):** [RSI_VALUE] ([RSI_INTERPRETATION])
**MACD:** [MACD_STATUS]
**Moving Averages:** [MA_STATUS]

---

## Recommendation

### Action

**Recommendation:** [HOLD / TAKE_PROFIT / CUT_LOSS / ROLL / ADJUST]

**Rationale:**
[RATIONALE_PARAGRAPH]

### Targets

**Profit Target:** $[PROFIT_TARGET] ([PROFIT_TARGET_PCT]% of max profit)
**Stop Loss:** $[STOP_LOSS] ([STOP_LOSS_PCT]% of max loss)
**Adjustment Trigger:** [ADJUSTMENT_CONDITION]

### Next Steps

1. [ACTION_ITEM_1]
2. [ACTION_ITEM_2]
3. [ACTION_ITEM_3]

---

## Trade Log

**Entry Details:**
- Date: [ENTRY_DATE]
- Price: $[ENTRY_PRICE]
- [ENTRY_SPECIFICS]

**Key Events:**
- [EVENT_DATE_1]: [EVENT_DESCRIPTION_1]
- [EVENT_DATE_2]: [EVENT_DESCRIPTION_2]

**Management Actions:**
- [ACTION_DATE_1]: [MANAGEMENT_ACTION_1]
- [ACTION_DATE_2]: [MANAGEMENT_ACTION_2]

---

**Report Generated:** [TIMESTAMP]
**Next Review:** [NEXT_REVIEW_DATE]

---

*This is a template. Replace [PLACEHOLDERS] with actual values.*
"""

    return template


def main():
    """Generate assets for all skills."""
    print("=" * 80)
    print("SKILL ASSETS GENERATOR")
    print("=" * 80)
    print()

    files_created = 0
    skills_processed = 0

    # Process options strategies
    print("Generating assets for OPTIONS STRATEGIES...")
    for strategy in OPTIONS_STRATEGIES:
        skill_dir = SKILLS_DIR / strategy
        if not skill_dir.exists():
            print(f"  [SKIP] {strategy} - directory not found")
            continue

        assets_dir = skill_dir / "assets"
        assets_dir.mkdir(exist_ok=True)

        # 1. Sample positions CSV
        csv_path = assets_dir / "sample_positions.csv"
        if not csv_path.exists() or strategy != "married-put":
            # married-put already has good data, don't overwrite
            csv_content = generate_sample_positions_csv(strategy)
            if csv_content:
                csv_path.write_text(csv_content, encoding="utf-8")
                files_created += 1
                print(f"  [CREATED] {strategy}/assets/sample_positions.csv")

        # 2. Analysis template
        template_path = assets_dir / "analysis-template.md"
        template_content = generate_analysis_template(strategy)
        template_path.write_text(template_content, encoding="utf-8")
        files_created += 1
        print(f"  [CREATED] {strategy}/assets/analysis-template.md")

        skills_processed += 1

    # Process bond analysis skills
    print("\nGenerating assets for BOND ANALYSIS...")
    for strategy in BOND_ANALYSIS:
        skill_dir = SKILLS_DIR / strategy
        if not skill_dir.exists():
            print(f"  [SKIP] {strategy} - directory not found")
            continue

        assets_dir = skill_dir / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Analysis template
        template_path = assets_dir / "analysis-template.md"
        template_content = generate_analysis_template(strategy)
        template_path.write_text(template_content, encoding="utf-8")
        files_created += 1
        print(f"  [CREATED] {strategy}/assets/analysis-template.md")

        # Sample bond data (common for all bond skills)
        if strategy == "bond-pricing":
            sample_bonds_csv = """cusip,issuer,coupon,maturity_date,face_value,settlement_date,current_price,yield_to_maturity,rating
912828YK0,US_Treasury,2.875,2028-05-15,100000,2025-12-12,98.50,3.125,AAA
459200JZ7,IBM_Corp,3.625,2027-02-12,100000,2025-12-12,101.25,3.385,A+
06051GJN6,Bank_of_America,4.000,2026-01-22,100000,2025-12-12,99.75,4.125,A-
037833BH5,Apple_Inc,2.400,2030-08-20,100000,2025-12-12,95.00,2.895,AA+
02079K107,Alphabet_Inc,1.900,2029-08-15,100000,2025-12-12,92.50,2.650,AA"""
            (assets_dir / "sample_bonds.csv").write_text(
                sample_bonds_csv, encoding="utf-8"
            )
            files_created += 1
            print(f"  [CREATED] {strategy}/assets/sample_bonds.csv")

        skills_processed += 1

    # Process other skills
    print("\nGenerating assets for OTHER SKILLS...")

    # Financial analysis
    if (SKILLS_DIR / "financial-analysis").exists():
        assets_dir = SKILLS_DIR / "financial-analysis" / "assets"
        assets_dir.mkdir(exist_ok=True)

        template_path = assets_dir / "financial-analysis-template.md"
        template_content = generate_analysis_template("financial-analysis")
        template_path.write_text(template_content, encoding="utf-8")
        files_created += 1
        print("  [CREATED] financial-analysis/assets/financial-analysis-template.md")

        # Sample financial data
        sample_data = """ticker,revenue,gross_profit,operating_income,net_income,total_assets,total_liabilities,shareholders_equity,free_cash_flow,shares_outstanding
AAPL,394328,169148,119437,99803,352755,290020,62735,99584,15634
MSFT,211915,146052,88523,72738,411976,205017,206959,65149,7446
GOOGL,307394,169411,84266,73795,402392,115008,287384,69495,12522
AMZN,574785,225152,12248,30425,462675,282304,180371,35574,10324
META,134902,108843,62402,39098,229623,47197,182426,42906,2556"""
        (assets_dir / "sample_financials.csv").write_text(sample_data, encoding="utf-8")
        files_created += 1
        print("  [CREATED] financial-analysis/assets/sample_financials.csv")

        skills_processed += 1

    # Technical analysis
    if (SKILLS_DIR / "technical-analysis").exists():
        assets_dir = SKILLS_DIR / "technical-analysis" / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Sample OHLCV data
        sample_ohlcv = """date,open,high,low,close,volume
2025-12-01,450.25,455.80,448.90,453.50,85234000
2025-12-02,453.75,456.20,451.30,454.80,72890000
2025-12-03,454.90,458.40,453.60,457.20,68450000
2025-12-04,457.50,460.10,455.80,456.30,75120000
2025-12-05,456.20,457.90,453.50,455.60,71230000
2025-12-06,455.80,459.30,454.20,458.90,79560000
2025-12-09,459.10,462.50,458.30,461.80,82340000
2025-12-10,461.90,463.20,459.40,460.50,70890000
2025-12-11,460.70,462.80,458.90,461.20,73450000
2025-12-12,461.30,464.50,460.80,463.90,76230000"""
        (assets_dir / "sample_ohlcv.csv").write_text(sample_ohlcv, encoding="utf-8")
        files_created += 1
        print("  [CREATED] technical-analysis/assets/sample_ohlcv.csv")

        skills_processed += 1

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Skills processed: {skills_processed}")
    print(f"Files created: {files_created}")
    print("\nAsset types generated:")
    print("  - Sample position CSV files (options strategies)")
    print("  - Sample bond data (bond analysis)")
    print("  - Sample financial data (financial analysis)")
    print("  - Sample OHLCV data (technical analysis)")
    print("  - Analysis report templates (all skills)")


if __name__ == "__main__":
    main()
