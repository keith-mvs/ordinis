"""Minimal benchmark harness (manifest + checksum + schema validation).

Reads a benchmark pack manifest, validates required fields, checks file
checksums if provided, and performs a lightweight schema/row-count inspection.
This does not yet run the full Signal → Risk → Execution/Portfolio → Analytics
pipeline; it produces a report summarizing validation results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BenchmarkManifest:
    universe: str
    window: str
    start_date: str
    end_date: str
    regime_tags: list[str]
    asset_types: list[str]
    features: list[str] | None
    checksums: dict[str, str] | None
    notes: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkManifest:
        required = [
            "universe",
            "window",
            "start_date",
            "end_date",
            "regime_tags",
            "asset_types",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required manifest fields: {missing}")
        return cls(
            universe=data["universe"],
            window=data["window"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            regime_tags=list(data.get("regime_tags") or []),
            asset_types=list(data.get("asset_types") or []),
            features=list(data.get("features") or []),
            checksums=dict(data.get("checksums") or {}),
            notes=data.get("notes"),
        )


def load_manifest(manifest_path: Path) -> BenchmarkManifest:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return BenchmarkManifest.from_dict(data or {})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _load_table(file_path: Path):
    import pandas as pd

    if file_path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def validate_pack(pack_path: Path, manifest: BenchmarkManifest) -> dict[str, Any]:
    results: dict[str, Any] = {"checks": []}
    # Check checksums if provided
    if manifest.checksums:
        for filename, expected in manifest.checksums.items():
            fpath = pack_path / filename
            if not fpath.exists():
                results["checks"].append(
                    {"type": "checksum", "file": filename, "status": "missing"}
                )
                continue
            actual = _sha256(fpath)
            status = "ok" if actual == expected else "mismatch"
            results["checks"].append(
                {
                    "type": "checksum",
                    "file": filename,
                    "status": status,
                    "actual": actual,
                    "expected": expected,
                }
            )
    # Find a data file to inspect
    data_file = None
    for candidate in ("data.parquet", "data.csv"):
        cand_path = pack_path / candidate
        if cand_path.exists():
            data_file = cand_path
            break
    if data_file:
        df = _load_table(data_file)
        required = {"open", "high", "low", "close", "volume"}
        cols = set(
            df.columns.str.lower()
            if hasattr(df.columns, "str")
            else [c.lower() for c in df.columns]
        )
        missing = sorted(required - cols)
        results["data_file"] = str(data_file)
        results["row_count"] = len(df)
        results["columns"] = sorted(df.columns)
        results["schema_ok"] = not missing
        if missing:
            results["missing_columns"] = missing
    else:
        results["data_file"] = None
        results["schema_ok"] = False
        results["missing_columns"] = ["data.parquet or data.csv"]
    return results


def write_report(
    report_path: Path, manifest: BenchmarkManifest, validation: dict[str, Any]
) -> None:
    report = {
        "status": "ok" if validation.get("schema_ok") else "failed",
        "message": "Manifest validated; data inspected (pipeline execution not implemented).",
        "manifest": {
            "universe": manifest.universe,
            "window": manifest.window,
            "start_date": manifest.start_date,
            "end_date": manifest.end_date,
            "regime_tags": manifest.regime_tags,
            "asset_types": manifest.asset_types,
            "features": manifest.features,
        },
        "validation": validation,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark harness stub (validates manifest, optional checksums, and basic schema)."
    )
    parser.add_argument("--pack", required=True, help="Path to benchmark pack folder.")
    parser.add_argument("--report", required=True, help="Path to write report JSON.")
    args = parser.parse_args(argv)

    pack_path = Path(args.pack)
    manifest_path = pack_path / "manifest.yaml"
    report_path = Path(args.report)

    manifest = load_manifest(manifest_path)
    validation = validate_pack(pack_path, manifest)
    write_report(report_path, manifest, validation)

    print(f"[benchmark] validated manifest for {manifest.universe} ({manifest.window})")
    if validation.get("checks"):
        for check in validation["checks"]:
            print(f"[benchmark] checksum {check['file']}: {check['status']}")
    if validation.get("data_file"):
        print(
            f"[benchmark] inspected data: {validation['data_file']} rows={validation.get('row_count')}"
        )
    else:
        print("[benchmark] no data file found (expected data.parquet or data.csv)")
    print(f"[benchmark] report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
