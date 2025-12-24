"""
Cortex Code Analysis CLI.

This module provides a CLI for analyzing code using the CortexEngine.
It uses the actual LLM-based code analysis via Helix.

Usage:
    python -m ordinis.engines.cortex.cli --file src/ordinis/engines/cortex/engine.py --type review
    python -m ordinis.engines.cortex.cli --dir src/ordinis/engines/ --type security --output report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ordinis.engines.cortex.core.engine import CortexEngine


async def create_cortex_engine() -> "CortexEngine":
    """Create and initialize CortexEngine with Helix."""
    from ordinis.ai.helix.engine import Helix
    from ordinis.engines.cortex.core.config import CortexConfig
    from ordinis.engines.cortex.core.engine import CortexEngine

    # Initialize Helix
    helix = Helix()
    await helix.initialize()

    # Create CortexEngine with default config
    config = CortexConfig()
    engine = CortexEngine(helix=helix, config=config)
    await engine.initialize()

    return engine


async def analyze_single_file(
    engine: "CortexEngine",
    file_path: Path,
    analysis_type: str,
    verbose: bool = False,
) -> dict:
    """Analyze a single file using CortexEngine."""
    try:
        with open(file_path, encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        return {
            "file": str(file_path),
            "error": str(e),
            "status": "failed",
        }

    if verbose:
        print(f"  Analyzing: {file_path}")

    try:
        result = await engine.analyze_code(code=code, analysis_type=analysis_type)
        return {
            "file": str(file_path),
            "status": "success",
            "output_type": result.output_type.value,
            "content": result.content,
            "confidence": result.confidence,
            "reasoning": result.reasoning[:500] if result.reasoning else None,
            "model_used": result.model_used,
            "tokens": {
                "prompt": result.prompt_tokens,
                "completion": result.completion_tokens,
            },
        }
    except Exception as e:
        return {
            "file": str(file_path),
            "error": str(e),
            "status": "failed",
        }


async def analyze_code(args: argparse.Namespace) -> None:
    """Main analysis function."""
    print("[INFO] Starting Cortex Analysis (LLM-backed)")
    print(f"[INFO] Analysis Type: {args.type}")

    if args.dry_run:
        target = args.dir or args.file
        print(f"[INFO] Dry run: Would analyze files in {target} with type {args.type}")
        return

    # Collect files to analyze
    files_to_scan: list[Path] = []
    if args.file:
        p = Path(args.file)
        if p.exists():
            files_to_scan.append(p)
        else:
            print(f"[ERROR] File not found: {args.file}")
            sys.exit(1)
    elif args.dir:
        p = Path(args.dir)
        if p.exists():
            files_to_scan.extend(list(p.rglob("*.py")))
        else:
            print(f"[ERROR] Directory not found: {args.dir}")
            sys.exit(1)
    else:
        print("[ERROR] Must specify --file or --dir")
        sys.exit(1)

    # Filter by max files if specified
    if args.max_files and len(files_to_scan) > args.max_files:
        print(f"[WARN] Limiting to first {args.max_files} files (found {len(files_to_scan)})")
        files_to_scan = files_to_scan[: args.max_files]

    print(f"[INFO] Found {len(files_to_scan)} files to analyze")

    # Initialize CortexEngine
    print("[INFO] Initializing CortexEngine...")
    try:
        engine = await create_cortex_engine()
    except Exception as e:
        print(f"[ERROR] Failed to initialize CortexEngine: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    print("[INFO] CortexEngine initialized successfully")

    # Analyze files
    results = []
    success_count = 0
    error_count = 0

    for i, file_path in enumerate(files_to_scan, 1):
        if not args.quiet:
            print(f"[{i}/{len(files_to_scan)}] Analyzing {file_path.name}...")

        result = await analyze_single_file(
            engine=engine,
            file_path=file_path,
            analysis_type=args.type,
            verbose=args.verbose,
        )
        results.append(result)

        if result.get("status") == "success":
            success_count += 1
        else:
            error_count += 1
            if args.verbose:
                print(f"  [ERROR] {result.get('error', 'Unknown error')}")

    # Shutdown engine
    await engine.shutdown()

    # Generate report
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "analysis_type": args.type,
        "files_scanned": len(files_to_scan),
        "success_count": success_count,
        "error_count": error_count,
        "results": results,
    }

    # Print summary
    print("\n" + "=" * 50)
    print("CORTEX ANALYSIS REPORT")
    print("=" * 50)
    print(f"Files analyzed: {len(files_to_scan)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")

    # Show issues from successful analyses
    issues_found = []
    for r in results:
        if r.get("status") == "success":
            content = r.get("content", {})
            # Extract issues if present
            if "issues" in content:
                for issue in content["issues"]:
                    issues_found.append({
                        "file": r["file"],
                        **issue,
                    })
            elif "suggestions" in content:
                for suggestion in content["suggestions"]:
                    issues_found.append({
                        "file": r["file"],
                        "type": "suggestion",
                        "message": suggestion,
                    })

    if issues_found:
        print(f"\nFound {len(issues_found)} issues/suggestions:")
        for issue in issues_found[:10]:
            severity = issue.get("severity", "info").upper()
            msg = issue.get("message", str(issue))
            print(f"  [{severity}] {issue['file']}: {msg}")
        if len(issues_found) > 10:
            print(f"  ... and {len(issues_found) - 10} more")
    else:
        print("\nNo issues found. Code looks clean!")

    # Write output file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n[INFO] Report saved to {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cortex Code Analysis Tool (LLM-backed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python -m ordinis.engines.cortex.cli --file src/mycode.py --type review

  # Analyze a directory with security focus
  python -m ordinis.engines.cortex.cli --dir src/ordinis/engines/ --type security

  # Generate JSON report
  python -m ordinis.engines.cortex.cli --dir src/ --type review --output report.json

  # Dry run to see what would be analyzed
  python -m ordinis.engines.cortex.cli --dir src/ --dry-run
        """,
    )

    # Target selection
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--file", type=str, help="Analyze a single source file."
    )
    target_group.add_argument(
        "--dir", type=str, help="Recursively analyze every *.py under the directory."
    )

    # Analysis options
    parser.add_argument(
        "--type",
        type=str,
        default="review",
        choices=["review", "complexity", "security", "performance"],
        help="Analysis type (default: review).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to analyze (useful for large directories).",
    )

    # Output options
    parser.add_argument(
        "--output", "-o", type=str, help="Write structured JSON report to file."
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress per-file progress output."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output and errors."
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be analyzed without invoking LLM.",
    )

    # Model options (for future use)
    parser.add_argument(
        "--model-id",
        type=str,
        help="Override the default Helix model (not yet implemented).",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.file and not args.dir and not args.dry_run:
        parser.error("Must specify --file or --dir")

    # Check for API key
    if not os.getenv("NVIDIA_API_KEY"):
        print("[WARN] NVIDIA_API_KEY not set. LLM calls may fail.")

    # Run analysis
    try:
        asyncio.run(analyze_code(args))
    except KeyboardInterrupt:
        print("\n[INFO] Analysis interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
