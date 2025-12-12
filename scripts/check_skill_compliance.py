#!/usr/bin/env python3
"""Check SKILL.md files for compliance with template standards."""

import re
from pathlib import Path
from typing import Dict, List, Optional

SKILLS_DIR = Path(r"C:\Users\kjfle\Workspace\ordinis\.claude\skills")


def check_frontmatter(content: str) -> Dict[str, any]:
    """Check YAML frontmatter for required fields."""
    result = {
        "has_frontmatter": False,
        "has_name": False,
        "has_description": False,
        "name_value": None,
        "description_value": None,
        "errors": []
    }

    # Check for frontmatter delimiters
    if not content.startswith("---"):
        result["errors"].append("Missing frontmatter (should start with ---)")
        return result

    # Extract frontmatter
    parts = content.split("---", 2)
    if len(parts) < 3:
        result["errors"].append("Malformed frontmatter (missing closing ---)")
        return result

    result["has_frontmatter"] = True
    frontmatter = parts[1]

    # Check for required fields
    name_match = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
    desc_match = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)

    if name_match:
        result["has_name"] = True
        result["name_value"] = name_match.group(1).strip()

        # Validate name format (lowercase, numbers, hyphens only)
        if not re.match(r"^[a-z0-9-]+$", result["name_value"]):
            result["errors"].append(
                f"Name '{result['name_value']}' contains invalid characters "
                "(use lowercase, numbers, hyphens only)"
            )

        # Check length
        if len(result["name_value"]) > 64:
            result["errors"].append(f"Name too long ({len(result['name_value'])} > 64 chars)")
    else:
        result["errors"].append("Missing 'name:' field in frontmatter")

    if desc_match:
        result["has_description"] = True
        result["description_value"] = desc_match.group(1).strip()

        # Check length
        if len(result["description_value"]) > 1024:
            result["errors"].append(
                f"Description too long ({len(result['description_value'])} > 1024 chars)"
            )

        # Check for first-person language (should be third-person)
        first_person_words = ["I", "my", "our", "we"]
        desc_lower = result["description_value"].lower()
        for word in first_person_words:
            if re.search(r"\b" + word.lower() + r"\b", desc_lower):
                result["errors"].append(
                    f"Description uses first-person ('{word}'), should be third-person"
                )
                break
    else:
        result["errors"].append("Missing 'description:' field in frontmatter")

    return result


def check_file_size(file_path: Path) -> Dict[str, any]:
    """Check if SKILL.md file is under 500 lines."""
    result = {"line_count": 0, "compliant": True, "errors": []}

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        result["line_count"] = len(lines)

    if result["line_count"] > 500:
        result["compliant"] = False
        result["errors"].append(
            f"File exceeds 500 lines ({result['line_count']} lines)"
        )

    return result


def check_structure(content: str) -> Dict[str, any]:
    """Check for recommended sections."""
    result = {
        "has_overview": False,
        "has_workflow": False,
        "has_scripts": False,
        "has_references": False,
        "has_dependencies": False,
        "warnings": []
    }

    content_lower = content.lower()

    # Check for key sections (case-insensitive)
    result["has_overview"] = bool(re.search(r"^#+\s*overview", content, re.MULTILINE | re.IGNORECASE))
    result["has_workflow"] = bool(re.search(r"^#+\s*(workflow|usage|core workflow)", content, re.MULTILINE | re.IGNORECASE))
    result["has_scripts"] = bool(re.search(r"^#+\s*scripts", content, re.MULTILINE | re.IGNORECASE))
    result["has_references"] = bool(re.search(r"^#+\s*references", content, re.MULTILINE | re.IGNORECASE))
    result["has_dependencies"] = bool(re.search(r"^#+\s*dependencies", content, re.MULTILINE | re.IGNORECASE))

    # Generate warnings for missing recommended sections
    if not result["has_overview"]:
        result["warnings"].append("Missing recommended section: Overview")
    if not result["has_workflow"]:
        result["warnings"].append("Missing recommended section: Workflow/Usage")
    if not result["has_references"]:
        result["warnings"].append("Missing recommended section: References")

    return result


def check_skill(skill_dir: Path) -> Dict[str, any]:
    """Check a single skill package for compliance."""
    skill_file = skill_dir / "SKILL.md"

    result = {
        "name": skill_dir.name,
        "exists": skill_file.exists(),
        "compliant": True,
        "errors": [],
        "warnings": []
    }

    if not result["exists"]:
        result["compliant"] = False
        result["errors"].append("SKILL.md file not found")
        return result

    # Read file content
    with open(skill_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Run checks
    frontmatter = check_frontmatter(content)
    file_size = check_file_size(skill_file)
    structure = check_structure(content)

    # Aggregate results
    result["frontmatter"] = frontmatter
    result["file_size"] = file_size
    result["structure"] = structure

    # Collect errors
    result["errors"].extend(frontmatter.get("errors", []))
    result["errors"].extend(file_size.get("errors", []))

    # Collect warnings
    result["warnings"].extend(structure.get("warnings", []))

    # Determine overall compliance
    if result["errors"]:
        result["compliant"] = False

    return result


def main():
    """Check all skill packages."""
    if not SKILLS_DIR.exists():
        print(f"Error: Skills directory not found: {SKILLS_DIR}")
        return

    skills = [d for d in SKILLS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]

    print("=" * 80)
    print("SKILL.MD COMPLIANCE CHECK")
    print("=" * 80)
    print(f"Checking {len(skills)} skill packages...\n")

    compliant = []
    non_compliant = []

    for skill_dir in sorted(skills):
        result = check_skill(skill_dir)

        print(f"\n{result['name']}")
        print("-" * 60)

        if not result["exists"]:
            print("  [ERROR] SKILL.md not found")
            non_compliant.append(result)
            continue

        # File size
        line_count = result["file_size"]["line_count"]
        size_ok = result["file_size"]["compliant"]
        print(f"  Line count: {line_count}/500 {'[OK]' if size_ok else '[!]'}")

        # Frontmatter
        fm = result["frontmatter"]
        print(f"  Frontmatter: {'[Y]' if fm['has_frontmatter'] else '[N]'}")
        if fm["has_name"]:
            print(f"    name: {fm['name_value']}")
        if fm["has_description"]:
            desc_preview = fm["description_value"][:60] + "..." if len(fm["description_value"]) > 60 else fm["description_value"]
            print(f"    description: {desc_preview}")

        # Structure
        struct = result["structure"]
        sections = sum([
            struct["has_overview"],
            struct["has_workflow"],
            struct["has_scripts"],
            struct["has_references"],
            struct["has_dependencies"]
        ])
        print(f"  Sections: {sections}/5 recommended")

        # Errors and warnings
        if result["errors"]:
            print(f"  ERRORS ({len(result['errors'])}):")
            for error in result["errors"][:3]:  # Show first 3
                print(f"    - {error}")
            if len(result["errors"]) > 3:
                print(f"    ... and {len(result['errors']) - 3} more")

        if result["warnings"]:
            print(f"  WARNINGS ({len(result['warnings'])}):")
            for warning in result["warnings"][:2]:  # Show first 2
                print(f"    - {warning}")

        if result["compliant"]:
            print("  [OK] Compliant")
            compliant.append(result)
        else:
            non_compliant.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Compliant: {len(compliant)}")
    print(f"Non-compliant: {len(non_compliant)}")

    if non_compliant:
        print(f"\nSkills needing fixes:")
        for result in non_compliant:
            error_count = len(result.get("errors", []))
            print(f"  - {result['name']} ({error_count} errors)")


if __name__ == "__main__":
    main()
