#!/usr/bin/env python3
"""Audit skill packages for compliance with template structure."""

from pathlib import Path

SKILLS_DIR = Path(r"C:\Users\kjfle\Workspace\ordinis\.claude\skills")


def audit_skill(skill_dir: Path) -> dict[str, any]:
    """Audit a single skill package."""
    result = {
        "name": skill_dir.name,
        "has_skill_md": (skill_dir / "SKILL.md").exists(),
        "has_scripts_dir": (skill_dir / "scripts").is_dir(),
        "has_references_dir": (skill_dir / "references").is_dir(),
        "has_assets_dir": (skill_dir / "assets").is_dir(),
        "python_files_in_root": [],
        "md_files_in_root": [],
        "data_files_in_root": [],
        "script_files": [],
        "reference_files": [],
    }

    # Check root directory
    for item in skill_dir.iterdir():
        if item.is_file():
            if item.suffix == ".py":
                result["python_files_in_root"].append(item.name)
            elif item.suffix == ".md" and item.name != "SKILL.md":
                result["md_files_in_root"].append(item.name)
            elif item.suffix in [".csv", ".txt", ".json"]:
                result["data_files_in_root"].append(item.name)

    # Check scripts directory
    if result["has_scripts_dir"]:
        scripts_dir = skill_dir / "scripts"
        result["script_files"] = [f.name for f in scripts_dir.iterdir() if f.suffix == ".py"]

    # Check references directory
    if result["has_references_dir"]:
        refs_dir = skill_dir / "references"
        result["reference_files"] = [f.name for f in refs_dir.iterdir() if f.suffix == ".md"]

    return result


def main():
    """Audit all skill packages."""
    if not SKILLS_DIR.exists():
        print(f"Error: Skills directory not found: {SKILLS_DIR}")
        return

    skills = [d for d in SKILLS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]

    print(f"Auditing {len(skills)} skill packages...\n")
    print("=" * 80)

    needs_refactoring = []
    compliant = []

    for skill_dir in sorted(skills):
        audit = audit_skill(skill_dir)

        # Check if refactoring is needed
        needs_reorg = (
            audit["python_files_in_root"]
            or audit["md_files_in_root"]
            or audit["data_files_in_root"]
            or not audit["has_scripts_dir"]
            or not audit["has_references_dir"]
        )

        if needs_reorg:
            needs_refactoring.append(audit)
        else:
            compliant.append(audit)

        print(f"\n{audit['name']}")
        print(f"  SKILL.md: {'[Y]' if audit['has_skill_md'] else '[N]'}")
        print(
            f"  scripts/: {'[Y]' if audit['has_scripts_dir'] else '[N]'} ({len(audit['script_files'])} files)"
        )
        print(
            f"  references/: {'[Y]' if audit['has_references_dir'] else '[N]'} ({len(audit['reference_files'])} files)"
        )
        print(f"  assets/: {'[Y]' if audit['has_assets_dir'] else '[N]'}")

        if audit["python_files_in_root"]:
            print(f"  [!] Python files in root: {', '.join(audit['python_files_in_root'])}")
        if audit["md_files_in_root"]:
            print(f"  [!] MD files in root: {', '.join(audit['md_files_in_root'])}")
        if audit["data_files_in_root"]:
            print(f"  [!] Data files in root: {', '.join(audit['data_files_in_root'])}")

        if not needs_reorg:
            print("  [OK] Compliant with template")

    print("\n" + "=" * 80)
    print("\nSummary:")
    print(f"  Compliant: {len(compliant)}")
    print(f"  Needs refactoring: {len(needs_refactoring)}")

    if needs_refactoring:
        print("\n  Skills needing refactoring:")
        for audit in needs_refactoring:
            print(f"    - {audit['name']}")


if __name__ == "__main__":
    main()
