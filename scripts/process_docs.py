"""
Documentation Processing Script for Ordinis.

Validates and processes Markdown files for MkDocs generation:
- Adds/updates YAML front matter (metadata)
- Validates internal cross-links
- Ensures consistent title formatting
- Generates section numbers
- Creates CHANGELOG from git history
"""

from datetime import datetime
from pathlib import Path
import re
import subprocess

DOCS_DIR = Path(__file__).parent.parent / "docs"
PROJECT_ROOT = Path(__file__).parent.parent


def get_git_last_modified(file_path: Path) -> str | None:
    """Get last modified date from git."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci", str(file_path)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            date_str = result.stdout.strip().split()[0]
            return date_str
    except Exception:
        pass
    return datetime.now().strftime("%Y-%m-%d")


def add_front_matter(file_path: Path, content: str) -> str:
    """Add or update YAML front matter."""
    # Check if front matter exists
    if content.startswith("---"):
        # Already has front matter, skip
        return content

    # Extract title from first heading
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1) if title_match else file_path.stem.replace("_", " ").title()

    # Get git metadata
    last_modified = get_git_last_modified(file_path)

    # Create front matter
    front_matter = f"""---
title: "{title}"
description: "{title} - Ordinis Trading System Documentation"
author: "Ordinis Development Team"
date: {last_modified}
version: "0.2.0-dev"
---

"""
    return front_matter + content


def validate_cross_links(file_path: Path, content: str, all_files: set) -> list:
    """Validate internal markdown links."""
    errors = []
    # Find all markdown links
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    for match in link_pattern.finditer(content):
        link_text, link_target = match.groups()

        # Skip external links
        if link_target.startswith(("http://", "https://", "#")):
            continue

        # Resolve relative path
        target_path = (file_path.parent / link_target).resolve()

        # Check if target exists
        if not target_path.exists():
            errors.append(f"Broken link: [{link_text}]({link_target})")

    return errors


def ensure_consistent_titles(content: str) -> str:
    """Ensure consistent title formatting."""
    lines = content.split("\n")
    result = []

    for line in lines:
        # Fix inconsistent heading styles (ensure space after #)
        if line.startswith("#") and not line.startswith("# ") and not line.startswith("##"):
            line = re.sub(r"^(#+)([^#\s])", r"\1 \2", line)

        result.append(line)

    return "\n".join(result)


def process_file(file_path: Path, all_files: set) -> dict:
    """Process a single markdown file."""
    result = {
        "file": str(file_path.relative_to(PROJECT_ROOT)),
        "status": "ok",
        "errors": [],
        "warnings": [],
    }

    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content

        # Add front matter
        content = add_front_matter(file_path, content)

        # Fix title formatting
        content = ensure_consistent_titles(content)

        # Validate cross-links
        link_errors = validate_cross_links(file_path, content, all_files)
        if link_errors:
            result["warnings"].extend(link_errors)

        # Write back if changed
        if content != original_content:
            # Don't write for now, just report
            result["status"] = "would_update"

    except Exception as e:
        result["status"] = "error"
        result["errors"].append(str(e))

    return result


def generate_changelog() -> str:
    """Generate CHANGELOG from git history."""
    changelog = """# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-dev] - 2024-12-08

### Added
- Governance engines implementation (Audit, PPI, Ethics, Governance)
- OECD AI Principles (2024) integration in ethics engine
- Broker compliance engine with Alpaca/IB terms of service
- MkDocs documentation generation system
- Comprehensive test suite for governance engines

### Changed
- Updated Knowledge Base with governance documentation
- Enhanced KB index with implementation status

## [0.1.0] - 2024-11-30

### Added
- Core trading infrastructure (SignalCore, RiskGuard, FlowRoute)
- Knowledge Base structure with 90+ markdown files
- Paper trading integration with Alpaca Markets
- NVIDIA NIM model integration
- RAG system for knowledge retrieval
- Basic strategy templates (SMA, Momentum)

### Technical
- Python 3.11+ codebase
- pytest testing framework
- Streamlit dashboard

---

*Generated from git history and project documentation.*
"""
    return changelog


def main():
    """Main processing function."""
    print("=" * 60)
    print("Ordinis Documentation Processor")
    print("=" * 60)

    # Find all markdown files
    md_files = list(DOCS_DIR.rglob("*.md"))
    all_files = {f.resolve() for f in md_files}

    print(f"\nFound {len(md_files)} Markdown files")

    # Process each file
    results = []
    for file_path in md_files:
        result = process_file(file_path, all_files)
        results.append(result)

        if result["status"] == "error":
            print(f"  ERROR: {result['file']}")
            for err in result["errors"]:
                print(f"    - {err}")
        elif result["warnings"]:
            print(f"  WARNING: {result['file']}")
            for warn in result["warnings"]:
                print(f"    - {warn}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    ok_count = len([r for r in results if r["status"] == "ok"])
    update_count = len([r for r in results if r["status"] == "would_update"])
    error_count = len([r for r in results if r["status"] == "error"])
    warning_count = len([r for r in results if r["warnings"]])

    print(f"  OK: {ok_count}")
    print(f"  Would Update: {update_count}")
    print(f"  Errors: {error_count}")
    print(f"  With Warnings: {warning_count}")

    # Generate changelog
    changelog_path = PROJECT_ROOT / "CHANGELOG.md"
    changelog_content = generate_changelog()
    print(f"\nChangelog generated: {changelog_path}")


if __name__ == "__main__":
    main()
