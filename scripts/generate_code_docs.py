"""
Generate lightweight code documentation by summarizing source files with the
NVIDIA-backed Cortex/Helix path. Outputs Markdown to
`docs/knowledge-base/code/generated-docs.md`.

Usage (PowerShell):
    $env:NVIDIA_API_KEY="nvapi-..."
    python scripts/generate_code_docs.py

Notes:
- Uses model override from env `CORTEX_NVIDIA_MODEL` if set; defaults to
  `nvidia/llama-3.3-nemotron-super-49b-v1.5`.
- Skips test files by default.
- Chunks files longer than `MAX_CHARS` to avoid oversized prompts.
- Audit/logging: prints model_used per file to stdout; you can redirect logs.
"""

from __future__ import annotations

from datetime import datetime
import os
import pathlib
import sys

# Ensure repo root is on sys.path when run without installation
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ordinis.engines.cortex.core.engine import CortexEngine

SRC_ROOT = ROOT / "src" / "ordinis"
OUTPUT_PATH = ROOT / "docs" / "knowledge-base" / "code" / "generated-docs.md"

MAX_FILES = 100  # adjust as needed
MAX_CHARS = 8000  # per chunk; conservative to avoid token bloat
MIN_CHARS = 200  # skip trivial files


def iter_source_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for path in SRC_ROOT.rglob("*.py"):
        parts = set(path.parts)
        if "tests" in parts or "__pycache__" in parts:
            continue
        if path.name == "__init__.py":
            continue
        files.append(path)
        if len(files) >= MAX_FILES:
            break
    return files


def chunk_text(text: str, max_chars: int):
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        yield text[start:end]
        start = end


def main() -> int:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")

    model = os.getenv("CORTEX_NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-ultra-253b-v1")

    engine = CortexEngine(
        nvidia_api_key=api_key,
        usd_code_enabled=True,
    )
    engine.model = model  # set once; analyze_code does not accept model kwarg

    files = iter_source_files()
    lines: list[str] = []
    lines.append("# Generated Code Docs\n")
    lines.append(f"- Generated: {datetime.utcnow().isoformat()}Z")
    lines.append(f"- Model: {model}")
    lines.append(f"- Files: {len(files)}")
    lines.append("")

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text) < MIN_CHARS:
            print(f"[docs] skip tiny file {path}")
            continue
        # Single-call summary (truncate to avoid massive prompts)
        truncated = text[:MAX_CHARS]
        out = engine.analyze_code(
            truncated,
            "review",  # structured code analysis
        )
        summary = out.content.get("analysis") or out.content
        model_used = out.model_used
        print(f"[docs] {path}: model_used={model_used}")

        lines.append(f"## {path.relative_to(ROOT)}")
        lines.append("")
        lines.append("- model_used: " + (model_used or "unknown"))
        lines.append("")
        lines.append("```text")
        lines.append(summary if isinstance(summary, str) else str(summary))
        lines.append("```")
        lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[docs] wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
