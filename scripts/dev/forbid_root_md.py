#!/usr/bin/env python3
"""Pre-commit hook to forbid new markdown files in repository root.

Allows only README.md, AGENTS.md, DEVELOPMENT.md, and USAGE.md in root.
All other documentation should go in docs/.
"""

from pathlib import Path
import subprocess
import sys

ALLOWED_ROOT_MD = {"README.md", "AGENTS.md", "DEVELOPMENT.md", "USAGE.md"}


def main() -> int:
    """Check for unauthorized markdown files in repo root."""
    result = subprocess.run(  # noqa: S603 - git is trusted
        ["git", "diff", "--cached", "--name-only", "--diff-filter=A"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )

    added_files = result.stdout.strip().split("\n") if result.stdout.strip() else []

    violations = []
    for file in added_files:
        path = Path(file)
        if path.suffix.lower() == ".md" and path.parent == Path():
            if path.name not in ALLOWED_ROOT_MD:
                violations.append(file)

    if violations:
        print("ERROR: New markdown files in repo root are not allowed.")
        print("       Move documentation to docs/ directory instead.")
        print(f"       Violations: {', '.join(violations)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
