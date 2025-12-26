#!/usr/bin/env python3

download_pdfs.py
- Downloads PDFs listed in pdf_manifest.json into ./pdfs
- Builds a zip containing PDFs + README.md

import argparse, json, os, re, sys, time
from pathlib import Path

import requests

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

def safe_filename(name: str) -> str:
    name = name.strip().replace("\", "_").replace("/", "_")
    name = re.sub(r"[\s]+", "_", name)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def download(url: str, out_path: Path, timeout=60) -> bool:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
            r.raise_for_status()
            # Some sites return HTML wrappers; we still save, but flag it.
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "pdf" not in ctype and not url.lower().endswith(".pdf"):
                print(f"[warn] content-type={ctype} for {url}", file=sys.stderr)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"[fail] {url} -> {out_path.name}: {e}", file=sys.stderr)
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="pdf_manifest.json", help="Path to manifest JSON")
    ap.add_argument("--out", default="network_parity_pdfs.zip", help="Output zip file")
    ap.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between downloads (politeness)")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    manifest_path = base / args.manifest
    pdf_dir = base / "pdfs"
    readme_path = base / "README.md"

    items = json.loads(manifest_path.read_text(encoding="utf-8"))

    ok, bad = [], []
    for item in items:
        fn = safe_filename(item["file_name"])
        url = item["url"]
        out = pdf_dir / fn
        if out.exists() and out.stat().st_size > 10_000:
            ok.append(fn)
            continue
        if download(url, out):
            ok.append(fn)
        else:
            bad.append({"file_name": fn, "url": url})
        time.sleep(max(args.sleep, 0))

    # Build zip
    out_zip = base / args.out
    import zipfile
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(readme_path, arcname="README.md")
        z.write(manifest_path, arcname="pdf_manifest.json")
        (base / "pdf_manifest.csv").exists() and z.write(base / "pdf_manifest.csv", arcname="pdf_manifest.csv")
        # PDFs
        if pdf_dir.exists():
            for p in sorted(pdf_dir.glob("*.pdf")):
                z.write(p, arcname=f"pdfs/{p.name}")

        # Add failure log
        if bad:
            import json as _json
            z.writestr("download_failures.json", _json.dumps(bad, indent=2))

    print(f"Created: {out_zip}")
    print(f"Downloaded PDFs: {len(ok)}/{len(items)}")
    if bad:
        print("Some downloads failed. See download_failures.json inside the zip.", file=sys.stderr)

if __name__ == "__main__":
    main()
