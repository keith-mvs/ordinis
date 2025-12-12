#!/usr/bin/env python3
"""Audit assets folders across all skill packages."""

from pathlib import Path
from typing import Dict, List

SKILLS_DIR = Path(r"C:\Users\kjfle\Workspace\ordinis\.claude\skills")


def audit_assets_folder(skill_dir: Path) -> Dict:
    """Audit assets folder for a skill."""
    result = {
        "name": skill_dir.name,
        "has_assets_dir": False,
        "file_count": 0,
        "files": [],
        "total_size": 0,
    }

    assets_dir = skill_dir / "assets"
    if not assets_dir.exists():
        return result

    result["has_assets_dir"] = True

    for asset_file in assets_dir.rglob("*"):
        if asset_file.is_file():
            size = asset_file.stat().st_size
            result["files"].append(
                {
                    "name": asset_file.name,
                    "size": size,
                    "size_kb": size / 1024,
                    "extension": asset_file.suffix,
                    "relative_path": str(asset_file.relative_to(assets_dir)),
                }
            )
            result["total_size"] += size

    result["file_count"] = len(result["files"])

    return result


def main():
    """Audit all assets folders."""
    if not SKILLS_DIR.exists():
        print(f"Error: Skills directory not found: {SKILLS_DIR}")
        return

    skills = [
        d
        for d in SKILLS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ]

    print("=" * 80)
    print("ASSETS FOLDER AUDIT")
    print("=" * 80)
    print(f"Auditing {len(skills)} skill packages...\n")

    # Audit all skills
    audits = []
    for skill in sorted(skills, key=lambda x: x.name):
        audit = audit_assets_folder(skill)
        audits.append(audit)

        if audit["has_assets_dir"] and audit["file_count"] > 0:
            print(f"\n{audit['name']}")
            print("-" * 60)
            print(f"  Files: {audit['file_count']}")
            print(f"  Total Size: {audit['total_size']/1024:.1f} KB")
            print(f"  Asset files:")
            for f in sorted(audit["files"], key=lambda x: x["name"]):
                print(
                    f"    - {f['relative_path']} ({f['size_kb']:.1f} KB, {f['extension'] or 'no ext'})"
                )

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}\n")

    total_skills = len(audits)
    skills_with_assets = sum(1 for a in audits if a["has_assets_dir"])
    skills_with_files = sum(1 for a in audits if a["file_count"] > 0)
    total_files = sum(a["file_count"] for a in audits)
    total_size = sum(a["total_size"] for a in audits)

    print(f"Total skills: {total_skills}")
    print(f"Skills with assets/ directory: {skills_with_assets}")
    print(f"Skills with asset files: {skills_with_files}")
    print(f"Total asset files: {total_files}")
    print(f"Total size: {total_size/1024:.1f} KB")

    if skills_with_files > 0:
        print(f"\nAverage files per skill (with assets): {total_files/skills_with_files:.1f}")

    # File type breakdown
    print(f"\n{'=' * 80}")
    print("FILE TYPE BREAKDOWN")
    print(f"{'=' * 80}\n")

    extensions = {}
    for audit in audits:
        for f in audit["files"]:
            ext = f["extension"] or "no ext"
            if ext not in extensions:
                extensions[ext] = {"count": 0, "size": 0}
            extensions[ext]["count"] += 1
            extensions[ext]["size"] += f["size"]

    if extensions:
        for ext in sorted(extensions.keys()):
            print(
                f"{ext:15} {extensions[ext]['count']:3} files, {extensions[ext]['size']/1024:8.1f} KB"
            )
    else:
        print("No asset files found")


if __name__ == "__main__":
    main()
