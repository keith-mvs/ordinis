#!/usr/bin/env python3
"""Consolidate package dependencies from all skills to root requirements.txt."""

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple

ORDINIS_ROOT = Path(__file__).parent.parent
SKILLS_DIR = ORDINIS_ROOT / ".claude" / "skills"


def extract_deps_from_skill_md(skill_md_path: Path) -> Set[Tuple[str, str]]:
    """Extract dependencies from SKILL.md frontmatter description."""
    deps = set()

    with open(skill_md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Look for "Requires package>=version" in description
    pattern = r"Requires\s+([^.]+?)\."
    matches = re.findall(pattern, content, re.MULTILINE)

    for match in matches:
        # Parse "numpy>=1.24.0, pandas>=2.0.0, matplotlib>=3.7.0"
        parts = match.split(",")
        for part in parts:
            part = part.strip()
            if ">=" in part:
                pkg, version = part.split(">=")
                deps.add((pkg.strip(), version.strip()))

    return deps


def extract_deps_from_requirements_txt(req_file: Path) -> Set[Tuple[str, str]]:
    """Extract dependencies from requirements.txt."""
    deps = set()

    with open(req_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if ">=" in line:
                    pkg, version = line.split(">=")
                    deps.add((pkg.strip(), version.strip()))

    return deps


def main():
    """Consolidate all dependencies."""
    print("=" * 80)
    print("DEPENDENCY CONSOLIDATION")
    print("=" * 80)
    print()

    # Collect all dependencies
    all_deps = defaultdict(set)  # pkg -> set of versions

    # 1. Check SKILL.md files
    print("Scanning SKILL.md files...")
    skill_files = list(SKILLS_DIR.glob("*/SKILL.md"))
    for skill_md in skill_files:
        if skill_md.parent.name.startswith("_"):
            continue

        deps = extract_deps_from_skill_md(skill_md)
        for pkg, version in deps:
            all_deps[pkg].add(version)
            print(f"  {skill_md.parent.name}: {pkg}>={version}")

    # 2. Check requirements.txt files
    print("\nScanning requirements.txt files...")
    req_files = list(SKILLS_DIR.glob("*/requirements.txt"))
    for req_file in req_files:
        deps = extract_deps_from_requirements_txt(req_file)
        for pkg, version in deps:
            all_deps[pkg].add(version)
            print(f"  {req_file.parent.name}: {pkg}>={version}")

    # 3. Consolidate versions (use highest version for each package)
    print("\n" + "=" * 80)
    print("CONSOLIDATED REQUIREMENTS")
    print("=" * 80)
    print()

    consolidated = {}
    for pkg, versions in sorted(all_deps.items()):
        # Convert version strings to tuples for comparison
        version_tuples = []
        for v in versions:
            parts = v.split(".")
            version_tuples.append((tuple(int(p) for p in parts), v))

        # Get highest version
        highest = max(version_tuples, key=lambda x: x[0])[1]
        consolidated[pkg] = highest
        print(f"{pkg:20} >= {highest:10} (from {len(versions)} sources)")

    # 4. Write to root requirements.txt
    output_path = ORDINIS_ROOT / "requirements.txt"

    requirements_content = [
        "# Ordinis - Algorithmic Trading System",
        "# Consolidated dependencies from all skill packages",
        "# Generated automatically - DO NOT EDIT MANUALLY",
        "#",
        "# Last Updated: 2025-12-12",
        "",
        "# Core Dependencies (Required for all skills)",
        "# =============================================",
        "",
    ]

    # Core packages (used by most skills)
    core_pkgs = ["numpy", "pandas", "scipy", "matplotlib"]
    for pkg in core_pkgs:
        if pkg in consolidated:
            requirements_content.append(f"{pkg}>={consolidated[pkg]}")

    requirements_content.extend(
        [
            "",
            "# Specialized Dependencies",
            "# ========================",
            "",
        ]
    )

    # Specialized packages
    for pkg in sorted(consolidated.keys()):
        if pkg not in core_pkgs:
            requirements_content.append(f"{pkg}>={consolidated[pkg]}")

    requirements_content.extend(
        [
            "",
            "# Optional Dependencies (Recommended)",
            "# ====================================",
            "# Uncomment as needed:",
            "",
            "# Data Sources",
            "# yfinance>=0.2.0          # Real-time market data",
            "# alpaca-trade-api>=3.0.0  # Alpaca trading integration",
            "",
            "# Development Tools",
            "# jupyter>=1.0.0           # Interactive notebooks",
            "# pytest>=7.0.0            # Testing framework",
            "# pytest-cov>=4.0.0        # Coverage reporting",
            "# ruff>=0.1.0              # Linter and formatter",
            "",
            "# Excel Support",
            "# openpyxl>=3.1.0          # Excel file I/O",
            "",
            "# Advanced TA Libraries (optional)",
            "# TA-Lib>=0.4.24           # Technical analysis (requires system library)",
            "# pandas-ta>=0.3.14        # Alternative TA library",
            "",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(requirements_content))

    print(f"\n{' ' * 80}")
    print(f"Requirements written to: {output_path}")
    print(f"Total packages: {len(consolidated)}")
    print(f"Core packages: {len([p for p in consolidated if p in core_pkgs])}")
    print(
        f"Specialized packages: {len([p for p in consolidated if p not in core_pkgs])}"
    )


if __name__ == "__main__":
    main()
