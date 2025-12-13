#!/usr/bin/env python3
"""Audit all references folders in skill packages."""

from pathlib import Path

SKILLS_DIR = Path(r"C:\Users\kjfle\Workspace\ordinis\.claude\skills")


def audit_references(skill_dir: Path) -> dict:
    """Audit references folder for a skill."""
    result = {
        "name": skill_dir.name,
        "has_references_dir": False,
        "reference_count": 0,
        "reference_files": [],
        "total_size": 0,
    }

    refs_dir = skill_dir / "references"
    if not refs_dir.exists():
        return result

    result["has_references_dir"] = True

    for ref_file in refs_dir.glob("*.md"):
        size = ref_file.stat().st_size
        result["reference_files"].append(
            {"name": ref_file.name, "size": size, "size_kb": size / 1024}
        )
        result["total_size"] += size

    result["reference_count"] = len(result["reference_files"])

    return result


def categorize_skills(skills: list[Path]) -> dict[str, list[str]]:
    """Categorize skills by type."""
    categories = {
        "options_strategies": [],
        "bond_analysis": [],
        "financial": [],
        "technical": [],
        "other": [],
    }

    for skill in skills:
        name = skill.name
        if any(
            x in name
            for x in [
                "put",
                "call",
                "straddle",
                "strangle",
                "butterfly",
                "condor",
                "collar",
                "spread",
            ]
        ):
            categories["options_strategies"].append(name)
        elif (
            "bond" in name
            or "yield" in name
            or "duration" in name
            or "credit-risk" in name
            or "oas" in name
        ):
            categories["bond_analysis"].append(name)
        elif "financial" in name or "benchmarking" in name:
            categories["financial"].append(name)
        elif "technical" in name:
            categories["technical"].append(name)
        else:
            categories["other"].append(name)

    return categories


def main():
    """Audit all references folders."""
    if not SKILLS_DIR.exists():
        print(f"Error: Skills directory not found: {SKILLS_DIR}")
        return

    skills = [d for d in SKILLS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]

    print("=" * 80)
    print("REFERENCES FOLDER AUDIT")
    print("=" * 80)
    print(f"Auditing {len(skills)} skill packages...\n")

    # Categorize skills
    categories = categorize_skills(skills)

    # Audit each category
    for category, skill_names in categories.items():
        if not skill_names:
            continue

        print(f"\n{'=' * 80}")
        print(f"{category.upper().replace('_', ' ')}")
        print(f"{'=' * 80}")

        for skill_name in sorted(skill_names):
            skill_dir = SKILLS_DIR / skill_name
            audit = audit_references(skill_dir)

            print(f"\n{skill_name}")
            print("-" * 60)

            if not audit["has_references_dir"]:
                print("  [!] No references/ directory")
                continue

            if audit["reference_count"] == 0:
                print("  [!] Empty references/ directory")
                continue

            print(f"  Files: {audit['reference_count']}")
            print(f"  Total Size: {audit['total_size']/1024:.1f} KB")
            print("  Reference files:")
            for ref in sorted(audit["reference_files"], key=lambda x: x["name"]):
                print(f"    - {ref['name']} ({ref['size_kb']:.1f} KB)")

    # Summary by category
    print(f"\n{'=' * 80}")
    print("SUMMARY BY CATEGORY")
    print(f"{'=' * 80}\n")

    for category, skill_names in categories.items():
        if not skill_names:
            continue

        total_refs = 0
        skills_with_refs = 0

        for skill_name in skill_names:
            audit = audit_references(SKILLS_DIR / skill_name)
            if audit["reference_count"] > 0:
                skills_with_refs += 1
                total_refs += audit["reference_count"]

        print(f"{category.replace('_', ' ').title()}:")
        print(f"  Skills: {len(skill_names)}")
        print(f"  Skills with references: {skills_with_refs}")
        print(f"  Total reference files: {total_refs}")
        print(f"  Average per skill: {total_refs/len(skill_names):.1f}")
        print()


if __name__ == "__main__":
    main()
