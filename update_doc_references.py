#!/usr/bin/env python3
"""
Update documentation cross-references after kebab-case rename
Created: 2025-12-12
"""

from pathlib import Path
import re

# Mapping of old names to new names
REPLACEMENTS = {
    "SIGNALCORE_SYSTEM.md": "signalcore-system.md",
    "EXECUTION_PATH.md": "execution-path.md",
    "SIMULATION_ENGINE.md": "simulation-engine.md",
    "MONITORING.md": "monitoring.md",
    "NVIDIA_INTEGRATION.md": "nvidia-integration.md",
    "RAG_SYSTEM.md": "rag-system.md",
    "PRODUCTION_ARCHITECTURE.md": "production-architecture.md",
    "PHASE1_API_REFERENCE.md": "phase1-api-reference.md",
    "ARCHITECTURE_REVIEW_RESPONSE.md": "architecture-review-response.md",
    "LAYERED_SYSTEM_ARCHITECTURE.md": "layered-system-architecture.md",
}

# Files to update (excluding session exports and the plan itself)
FILES_TO_UPDATE = [
    Path(r"C:\Users\kjfle\Workspace\ordinis\docs\architecture\phase1-api-reference.md"),
    Path(r"C:\Users\kjfle\Workspace\ordinis\docs\architecture\production-architecture.md"),
    Path(r"C:\Users\kjfle\Workspace\ordinis\docs\architecture\signalcore-system.md"),
    Path(r"C:\Users\kjfle\Workspace\ordinis\docs\architecture\rag-system.md"),
    Path(r"C:\Users\kjfle\Workspace\ordinis\docs\architecture\layered-system-architecture.md"),
    Path(r"C:\Users\kjfle\Workspace\ordinis\docs\project\PROJECT_STATUS_REPORT.md"),
    Path(r"C:\Users\kjfle\Workspace\ordinis\docs\project\CURRENT_STATUS_AND_NEXT_STEPS.md"),
    Path(r"C:\Users\kjfle\Workspace\ordinis\docs\guides\CLI_USAGE.md"),
    Path(r"C:\Users\kjfle\Workspace\ordinis\docs\index.md"),
    Path(r"C:\Users\kjfle\Workspace\ordinis\docs\DOCUMENTATION_UPDATE_REPORT_20251212.md"),
]


def update_file(file_path: Path) -> int:
    """Update references in a single file. Returns number of replacements made."""
    if not file_path.exists():
        print(f"  ⚠ File not found: {file_path}")
        return 0

    # Read file content
    content = file_path.read_text(encoding="utf-8")
    original_content = content

    # Apply all replacements
    replacement_count = 0
    for old_name, new_name in REPLACEMENTS.items():
        # Use word boundaries to avoid partial matches
        pattern = re.escape(old_name)
        if pattern in content:
            content = content.replace(old_name, new_name)
            replacement_count += content.count(new_name) - original_content.count(new_name)

    # Write back if changed
    if content != original_content:
        file_path.write_text(content, encoding="utf-8")
        return replacement_count

    return 0


def main():
    """Main execution function."""
    print("Documentation Cross-Reference Updater")
    print("=" * 50)
    print(f"\nUpdating {len(FILES_TO_UPDATE)} files...")
    print()

    total_replacements = 0
    files_modified = 0

    for file_path in FILES_TO_UPDATE:
        print(f"Processing: {file_path.name}")
        count = update_file(file_path)
        if count > 0:
            print(f"  ✓ Made {count} replacements")
            files_modified += 1
            total_replacements += count
        else:
            print("  - No changes needed")

    print()
    print("=" * 50)
    print("Complete!")
    print(f"  Files modified: {files_modified}/{len(FILES_TO_UPDATE)}")
    print(f"  Total replacements: {total_replacements}")


if __name__ == "__main__":
    main()
