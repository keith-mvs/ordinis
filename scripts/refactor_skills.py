#!/usr/bin/env python3
"""Refactor skill packages to comply with template structure."""

import shutil
from pathlib import Path
from typing import List

SKILLS_DIR = Path(r"C:\Users\kjfle\Workspace\ordinis\.claude\skills")

# Files to keep in root directory
ROOT_ALLOWED = {"SKILL.md", "SKILL-CARD.md"}

# Documentation files to move to references/
DOC_FILES = {
    "README.md", "reference.md", "INSTALLATION.md", "QUICKSTART.md",
    "STATUS.md", "UPDATE_SUMMARY.md", "IMPLEMENTATION_SUMMARY.md",
    "CASE_STUDIES.md", "FIBONACCI.md", "MOMENTUM_INDICATORS.md",
    "VOLATILITY_VOLUME.md", "TREND_INDICATORS.md",
    "static_levels.md", "trend_following_cases.md",
    "volatility_indicators.md", "volume_indicators.md"
}

# Data files to move to assets/
DATA_FILES = {".csv", ".json", ".xlsx", ".txt"}


def refactor_skill(skill_dir: Path, dry_run: bool = True) -> dict:
    """Refactor a single skill package to match template structure."""
    result = {
        "name": skill_dir.name,
        "created_dirs": [],
        "moved_files": [],
        "errors": []
    }

    if skill_dir.name.startswith("_"):
        return result  # Skip template and special dirs

    print(f"\nRefactoring: {skill_dir.name}")
    print("=" * 60)

    # Create required directories
    dirs_to_create = ["scripts", "references", "assets"]
    for dir_name in dirs_to_create:
        dir_path = skill_dir / dir_name
        if not dir_path.exists():
            print(f"  Creating: {dir_name}/")
            if not dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
            result["created_dirs"].append(dir_name)

    # Process files in root directory
    for item in sorted(skill_dir.iterdir()):
        if not item.is_file():
            continue

        if item.name in ROOT_ALLOWED:
            continue

        # Determine destination
        destination = None

        # Python files -> scripts/
        if item.suffix == ".py":
            destination = skill_dir / "scripts" / item.name

        # Documentation files -> references/
        elif item.name in DOC_FILES or (item.suffix == ".md" and item.name not in ROOT_ALLOWED):
            destination = skill_dir / "references" / item.name

        # Data files -> assets/
        elif item.suffix in DATA_FILES:
            # Special case: requirements.txt stays in root
            if item.name == "requirements.txt":
                continue
            destination = skill_dir / "assets" / item.name

        if destination:
            print(f"  Moving: {item.name} -> {destination.parent.name}/")
            if not dry_run:
                try:
                    shutil.move(str(item), str(destination))
                    result["moved_files"].append((item.name, destination.parent.name))
                except Exception as e:
                    error_msg = f"Error moving {item.name}: {e}"
                    print(f"    ERROR: {error_msg}")
                    result["errors"].append(error_msg)
            else:
                result["moved_files"].append((item.name, destination.parent.name))

    return result


def main():
    """Refactor all skill packages."""
    import argparse

    parser = argparse.ArgumentParser(description="Refactor skill packages to template structure")
    parser.add_argument("--execute", action="store_true", help="Actually perform refactoring (default is dry-run)")
    parser.add_argument("--skill", type=str, help="Refactor only this skill (default is all skills)")
    args = parser.parse_args()

    if not SKILLS_DIR.exists():
        print(f"Error: Skills directory not found: {SKILLS_DIR}")
        return

    # Get list of skills to refactor
    if args.skill:
        skills = [SKILLS_DIR / args.skill]
        if not skills[0].exists():
            print(f"Error: Skill not found: {args.skill}")
            return
    else:
        skills = [d for d in SKILLS_DIR.iterdir() if d.is_dir()]

    mode = "EXECUTING" if args.execute else "DRY RUN"
    print(f"\n{'=' * 80}")
    print(f"SKILL PACKAGE REFACTORING - {mode}")
    print(f"{'=' * 80}")
    print(f"Processing {len(skills)} skill packages...")

    all_results = []
    for skill_dir in sorted(skills):
        result = refactor_skill(skill_dir, dry_run=not args.execute)
        if result["created_dirs"] or result["moved_files"] or result["errors"]:
            all_results.append(result)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    total_dirs = sum(len(r["created_dirs"]) for r in all_results)
    total_moves = sum(len(r["moved_files"]) for r in all_results)
    total_errors = sum(len(r["errors"]) for r in all_results)

    print(f"\nSkills processed: {len(all_results)}")
    print(f"Directories created: {total_dirs}")
    print(f"Files moved: {total_moves}")
    print(f"Errors: {total_errors}")

    if not args.execute:
        print(f"\n[!] This was a DRY RUN - no changes were made")
        print(f"Run with --execute to perform actual refactoring")

    if total_errors > 0:
        print(f"\nSkills with errors:")
        for result in all_results:
            if result["errors"]:
                print(f"  - {result['name']}: {len(result['errors'])} errors")


if __name__ == "__main__":
    main()
