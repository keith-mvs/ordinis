from pathlib import Path

import pandas as pd

from ordinis.benchmark.harness import BenchmarkManifest, load_manifest, validate_pack


def test_manifest_and_validation(tmp_path: Path):
    # Create sample data
    data_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        [
            {
                "timestamp": "2020-01-01",
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.5,
                "volume": 1000,
            },
            {
                "timestamp": "2020-01-02",
                "open": 1.5,
                "high": 2.5,
                "low": 1.0,
                "close": 2.0,
                "volume": 1200,
            },
        ]
    )
    df.to_csv(data_path, index=False)

    checksum = data_path.read_bytes()
    import hashlib

    h = hashlib.sha256()
    h.update(checksum)
    checksum_str = f"sha256:{h.hexdigest()}"

    # Write manifest with checksum
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(
        f"""
universe: eq_test
window: 3m
start_date: 2020-01-01
end_date: 2020-03-31
regime_tags: [bear]
asset_types: [equities]
features: [rsi_14]
checksums:
  data.csv: "{checksum_str}"
""",
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_file)
    assert isinstance(manifest, BenchmarkManifest)
    assert manifest.universe == "eq_test"
    assert manifest.window == "3m"

    validation = validate_pack(tmp_path, manifest)
    assert validation["schema_ok"] is True
    assert validation["row_count"] == 2
    checksum_results = [c for c in validation["checks"] if c.get("file") == "data.csv"]
    assert checksum_results and checksum_results[0]["status"] == "ok"
