"""Utility: summarize coverage gaps.

Writes a text report of files with the most missing lines based on the local
`.coverage` data file.

This is a dev helper (not used in production runtime).
"""

from __future__ import annotations

import os
from pathlib import Path

import xml.etree.ElementTree as ET

import coverage


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_file = root / ".coverage"
    rows: list[tuple[int, int, str]] = []
    if data_file.exists():
        cov = coverage.Coverage(data_file=str(data_file))
        cov.load()

        for filename in cov.get_data().measured_files():
            normalized = filename.replace("/", os.sep)
            if f"{os.sep}src{os.sep}" not in normalized:
                continue

            try:
                _fn, statement_lines, _excluded, missing_lines, _missing_str = cov.analysis2(filename)
            except coverage.CoverageException:
                continue

            rows.append((len(missing_lines), len(statement_lines), filename))
    else:
        # Fallback for environments where `.coverage` isn't present but
        # a Cobertura-style `coverage.xml` exists (e.g., committed artifacts).
        xml_path = root / "coverage.xml"
        if not xml_path.exists():
            raise SystemExit(
                f"No coverage data found at {data_file} and no coverage.xml at {xml_path}. "
                "Run pytest with coverage first."
            )

        tree = ET.parse(xml_path)
        root_el = tree.getroot()

        # Per class (file) entry includes individual <line hits="..."> tags.
        for class_el in root_el.findall(".//class"):
            filename_attr = class_el.get("filename")
            if not filename_attr:
                continue

            # Keep only source-under-src entries
            normalized = filename_attr.replace("/", os.sep)
            if not normalized.startswith("ordinis" + os.sep) and not normalized.startswith("ordinis/"):
                # Some entries may be top-level __init__.py etc; keep them too.
                pass

            statement_count = 0
            missing_count = 0
            for line_el in class_el.findall("./lines/line"):
                statement_count += 1
                hits = int(line_el.get("hits", "0"))
                if hits == 0:
                    missing_count += 1

            # Convert the Cobertura path into something comparable with the
            # `.coverage` path style.
            rows.append((missing_count, statement_count, str(root / "src" / filename_attr)))

    rows.sort(reverse=True)

    out = root / "artifacts" / "top_missed_files.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["Top missed files (by missing lines):"]
    for miss, stmts, filename in rows[:75]:
        pct = 0.0 if stmts == 0 else (1 - miss / stmts) * 100
        rel = os.path.relpath(filename, root)
        lines.append(f"{miss:5d} missing / {stmts:5d} stmts = {pct:6.1f}%  {rel}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
