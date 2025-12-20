#!/usr/bin/env python3
"""Add YAML frontmatter to SKILL.md files that are missing it."""

from pathlib import Path

SKILLS_DIR = Path(r"C:\Users\kjfle\Workspace\ordinis\.claude\skills")

# Frontmatter templates for each skill
FRONTMATTER = {
    "bond-benchmarking": {
        "name": "bond-benchmarking",
        "description": "Compare bond performance against market benchmarks and indices. Analyzes yield spreads, duration matching, and relative value. Requires numpy>=1.24.0, pandas>=2.0.0. Use when evaluating bonds against peers, measuring portfolio attribution, or identifying value opportunities in fixed income markets.",
    },
    "bond-pricing": {
        "name": "bond-pricing",
        "description": "Price fixed income securities using present value, yield-to-maturity, and market conventions. Handles treasuries, corporates, municipals with various coupon frequencies. Requires numpy>=1.24.0, pandas>=2.0.0, scipy>=1.10.0. Use when valuing bonds, calculating accrued interest, or analyzing price sensitivity to yield changes.",
    },
    "credit-risk": {
        "name": "credit-risk",
        "description": "Assess credit risk and default probability for bonds using credit spreads, rating transitions, and recovery analysis. Requires numpy>=1.24.0, pandas>=2.0.0, scipy>=1.10.0. Use when evaluating corporate bonds, analyzing credit events, estimating default probabilities, or managing credit portfolio risk.",
    },
    "duration-convexity": {
        "name": "duration-convexity",
        "description": "Calculate bond duration, modified duration, and convexity for interest rate risk management. Implements Macaulay duration, effective duration, and dollar duration metrics. Requires numpy>=1.24.0, pandas>=2.0.0. Use when hedging interest rate exposure, immunizing portfolios, or analyzing price sensitivity to rate changes.",
    },
    "financial-analysis": {
        "name": "financial-analysis",
        "description": "Comprehensive financial statement analysis including ratio calculation, trend analysis, and peer comparison. Evaluates liquidity, profitability, efficiency, and leverage metrics. Requires numpy>=1.24.0, pandas>=2.0.0, matplotlib>=3.7.0. Use when analyzing company fundamentals, comparing financial performance, or conducting equity research.",
    },
    "option-adjusted-spread": {
        "name": "option-adjusted-spread",
        "description": "Calculate option-adjusted spread for bonds with embedded options using binomial trees and Monte Carlo simulation. Handles callable bonds, putable bonds, and MBS. Requires numpy>=1.24.0, pandas>=2.0.0, scipy>=1.10.0. Use when valuing bonds with optionality, comparing yields across callable and non-callable securities.",
    },
    "yield-measures": {
        "name": "yield-measures",
        "description": "Calculate comprehensive bond yield measures including current yield, yield-to-maturity, yield-to-call, and yield-to-worst. Handles various compounding conventions and day-count methods. Requires numpy>=1.24.0, pandas>=2.0.0, scipy>=1.10.0. Use when comparing bond returns, evaluating callable bonds, or analyzing yield curve relationships.",
    },
}


def add_frontmatter(skill_name: str) -> bool:
    """Add frontmatter to a SKILL.md file."""
    if skill_name not in FRONTMATTER:
        print(f"  [SKIP] No frontmatter template for {skill_name}")
        return False

    skill_dir = SKILLS_DIR / skill_name
    skill_file = skill_dir / "SKILL.md"

    if not skill_file.exists():
        print(f"  [ERROR] SKILL.md not found for {skill_name}")
        return False

    # Read current content
    with open(skill_file, encoding="utf-8") as f:
        content = f.read()

    # Check if already has frontmatter
    if content.startswith("---"):
        print(f"  [SKIP] {skill_name} already has frontmatter")
        return False

    # Build frontmatter
    fm = FRONTMATTER[skill_name]
    frontmatter = f"""---
name: {fm['name']}
description: {fm['description']}
---

"""

    # Prepend frontmatter to content
    new_content = frontmatter + content

    # Write back
    with open(skill_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"  [OK] Added frontmatter to {skill_name}")
    return True


def main():
    """Add frontmatter to all skills that need it."""
    skills_to_fix = [
        "bond-benchmarking",
        "bond-pricing",
        "credit-risk",
        "duration-convexity",
        "financial-analysis",
        "option-adjusted-spread",
        "yield-measures",
    ]

    print("=" * 80)
    print("ADDING FRONTMATTER TO SKILL.MD FILES")
    print("=" * 80)
    print(f"Processing {len(skills_to_fix)} skills...\n")

    success = 0
    failed = 0

    for skill_name in skills_to_fix:
        print(f"{skill_name}:")
        if add_frontmatter(skill_name):
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Success: {success}")
    print(f"Failed/Skipped: {failed}")


if __name__ == "__main__":
    main()
