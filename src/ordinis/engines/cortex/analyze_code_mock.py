"""
Cortex Code Analysis CLI.

This module provides a CLI for analyzing code using the Cortex engine.
It simulates static analysis and LLM-based code review.

Usage:
    python -m ordinis.engines.cortex.analyze_code --dir src/ordinis/engines/ --type security
"""

import argparse
import asyncio
import json
from pathlib import Path


async def analyze_code(args):
    print("[INFO] Starting Cortex Analysis")

    if args.dry_run:
        print(
            f"[INFO] Dry run: Would analyze files in {args.dir or args.file} with type {args.type}"
        )
        return

    files_to_scan = []
    if args.file:
        p = Path(args.file)
        if p.exists():
            files_to_scan.append(p)
        else:
            print(f"[ERROR] File not found: {args.file}")
            return
    elif args.dir:
        p = Path(args.dir)
        if p.exists():
            files_to_scan.extend(list(p.rglob("*.py")))
        else:
            print(f"[ERROR] Directory not found: {args.dir}")
            return
    else:
        print("[ERROR] Must specify --file or --dir")
        return

    print(f"[INFO] Scanning {len(files_to_scan)} files...")
    print(f"[INFO] Analysis Type: {args.type}")

    # Mock Analysis Logic
    print("[INFO] Running LLM Review (Mock)...")
    await asyncio.sleep(1)  # Simulate processing

    issues = []

    for file_path in files_to_scan:
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    # Mock Security Checks
                    if args.type == "security":
                        if "password" in line.lower() or "secret" in line.lower():
                            issues.append(
                                {
                                    "file": str(file_path),
                                    "line": i + 1,
                                    "type": "security",
                                    "severity": "high",
                                    "message": "Potential hardcoded secret found",
                                }
                            )
                        if "eval(" in line or "exec(" in line:
                            issues.append(
                                {
                                    "file": str(file_path),
                                    "line": i + 1,
                                    "type": "security",
                                    "severity": "critical",
                                    "message": "Unsafe code execution detected",
                                }
                            )

                    # Mock Complexity Checks
                    elif args.type == "complexity":
                        if len(line) > 120:
                            issues.append(
                                {
                                    "file": str(file_path),
                                    "line": i + 1,
                                    "type": "complexity",
                                    "severity": "low",
                                    "message": "Line too long (>120 chars)",
                                }
                            )
                        # Very naive cyclomatic complexity simulation (indentation depth)
                        if len(line) - len(line.lstrip()) > 12:
                            issues.append(
                                {
                                    "file": str(file_path),
                                    "line": i + 1,
                                    "type": "complexity",
                                    "severity": "medium",
                                    "message": "Deep nesting detected",
                                }
                            )

                    # Mock General Review
                    elif "TODO" in line:
                        issues.append(
                            {
                                "file": str(file_path),
                                "line": i + 1,
                                "type": "suggestion",
                                "severity": "low",
                                "message": "Found TODO comment",
                            }
                        )

        except Exception as e:
            print(f"[WARN] Could not read {file_path}: {e}")

    # Report
    print("\n" + "=" * 40)
    print("CORTEX ANALYSIS REPORT")
    print("=" * 40)

    if not issues:
        print("No issues found. Code looks clean!")
    else:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(
                f"  [{issue['severity'].upper()}] {issue['file']}:{issue['line']} - {issue['message']}"
            )

        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more.")

    # Output to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": str(datetime.now(UTC)),
            "type": args.type,
            "files_scanned": len(files_to_scan),
            "issues": issues,
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[INFO] Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cortex Code Analysis Tool")

    parser.add_argument("--file", type=str, help="Analyse a single source file.")
    parser.add_argument(
        "--dir", type=str, help="Recursively analyse every *.py under the directory."
    )
    parser.add_argument(
        "--type",
        type=str,
        default="review",
        choices=["review", "complexity", "security"],
        help="Analysis type.",
    )
    parser.add_argument("--output", type=str, help="Write a structured JSON report.")
    parser.add_argument("--model-id", type=str, help="Override the default Helix model.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging verbosity.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show which files would be sent without invoking."
    )
    parser.add_argument("--device", type=int, default=0, help="GPU to use.")

    args = parser.parse_args()

    asyncio.run(analyze_code(args))


if __name__ == "__main__":
    from datetime import UTC, datetime

    main()
