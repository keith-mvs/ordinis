import datetime as dt
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tarfile
import zipfile
from pathlib import Path

ROOT = Path.cwd()
DATA_NEW = ROOT / "data-new"
ARTIFACTS = ROOT / "docs" / "artifacts"

DATA_NEW.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

DATA_EXTS = {
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".orc",
    ".avro",
    ".xls",
    ".xlsx",
    ".json",
    ".jsonl",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".hdf5",
    ".h5",
}
COMPRESS_EXTS = {".gz", ".bz2", ".xz"}
ARCHIVE_EXTS = {".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz", ".7z"}

TICKER_RE = re.compile(r"\b(ticker|symbol|ric|isin|cusip|sedol|permno)\b", re.IGNORECASE)
PRICE_FIELDS = {"open", "high", "low", "close", "volume", "vwap", "adj_close", "adjclose", "last", "price", "bid", "ask"}
FUND_FIELDS = {"revenue", "eps", "ebitda", "assets", "liabilities", "equity", "cashflow", "income"}
DERIV_FIELDS = {"option", "strike", "expiry", "expiration", "greek", "delta", "gamma", "theta", "vega", "implied_vol"}
CORP_FIELDS = {"dividend", "split", "spinoff", "merger", "distribution"}
MACRO_FIELDS = {"cpi", "inflation", "gdp", "rate", "yield", "fx", "usd", "eur", "jpy"}
ALT_FIELDS = {"sentiment", "news", "headline", "event", "esg"}

BINS = {
    "reference",
    "prices",
    "corporate-actions",
    "fundamentals",
    "derivatives",
    "macro",
    "alt-data",
    "synthetic",
    "raw",
    "processed",
}

STOP_TOKENS = {"CSV", "TSV", "JSON", "DATA", "RAW", "REPORT", "TRADES", "SIGNALS", "EQUITY", "CURVE", "BACKTEST", "NA", "US", "EU", "UK", "ETF", "FX"}
PROVIDERS = {"alpaca", "polygon", "yfinance", "tiingo", "finnhub", "alphavantage", "iex", "ibkr", "fred", "quandl", "binance", "kraken", "coinbase", "bitfinex"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules", "htmlcov", "data-new"}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def comp_ext(p: Path) -> str:
    s = [x.lower() for x in p.suffixes]
    if not s:
        return ""
    if s[-1] == ".tgz":
        return ".tar.gz"
    if s[-1] in COMPRESS_EXTS and len(s) >= 2:
        return "".join(s[-2:])
    return s[-1]


def is_data_ext(ext: str) -> bool:
    if ext in DATA_EXTS or ext in ARCHIVE_EXTS:
        return True
    if any(ext.endswith(c) for c in COMPRESS_EXTS):
        base = ext
        for c in COMPRESS_EXTS:
            if base.endswith(c):
                base = base[: -len(c)]
        return base in DATA_EXTS
    return False


def norm(token: str) -> str:
    if not token:
        return "na"
    token = re.sub(r"[^a-z0-9]+", "-", token.lower()).strip("-")
    return token or "na"


def date_range(text: str) -> str:
    dates = []
    for y, m, d in re.findall(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", text):
        try:
            dates.append(dt.date(int(y), int(m), int(d)))
        except ValueError:
            continue
    if not dates:
        return "na"
    return f"{min(dates):%Y%m%d}-{max(dates):%Y%m%d}"


def freq(text: str) -> str:
    t = text.lower()
    if "tick" in t:
        return "tick"
    for key, val in [("1min", "1m"), ("1m", "1m"), ("5min", "5m"), ("5m", "5m"), ("15min", "15m"), ("15m", "15m"), ("30min", "30m"), ("30m", "30m"), ("1h", "1h"), ("60min", "1h"), ("hour", "1h")]:
        if key in t:
            return val
    if "daily" in t or "1d" in t:
        return "daily"
    if "weekly" in t:
        return "weekly"
    if "monthly" in t:
        return "monthly"
    return "na"


def region(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["nyse", "nasdaq", "amex", "us", "usa"]):
        return "us"
    if "eu" in t or "europe" in t or "emea" in t:
        return "eu"
    if "global" in t or "world" in t:
        return "global"
    return "na"


def instrument(text: str) -> str:
    t = text.lower()
    for key, val in [("crypto", "crypto"), ("forex", "fx"), ("fx", "fx"), ("futures", "futures"), ("options", "options"), ("option", "options"), ("etf", "etf"), ("index", "index"), ("equity", "equity"), ("stock", "equity")]:
        if key in t:
            return val
    return "na"


def source(text: str) -> str:
    t = text.lower()
    for p in PROVIDERS:
        if p in t:
            return p
    if any(k in t for k in ["backtest", "signals", "trades", "simulation", "simulated", "paper"]):
        return "internal"
    return "na"


def synthetic(text: str) -> str:
    t = text.lower()
    return "yes" if any(k in t for k in ["synthetic", "simulated", "simulation", "mock", "generated", "backtest", "paper"]) else "no"


def parse_header(line: str, delim: str) -> list:
    if not line:
        return []
    line = line.lstrip("\ufeff").strip()
    return [p.strip().strip("\"'") for p in line.split(delim) if p.strip()]


def read_fields(path: Path, ext: str) -> list:
    try:
        if ext in {".csv", ".tsv"}:
            delim = "\t" if ext == ".tsv" else ","
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                return parse_header(f.readline(), delim)
        if ext in {".csv.gz", ".tsv.gz"}:
            delim = "\t" if ext == ".tsv.gz" else ","
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                return parse_header(f.readline(), delim)
        if ext in {".json", ".json.gz", ".jsonl", ".jsonl.gz"}:
            opener = gzip.open if ext.endswith(".gz") else path.open
            with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
                snippet = f.read(200000)
            keys = re.findall(r"\"([A-Za-z0-9_\\-]+)\"\\s*:", snippet)
            if keys:
                return keys
        if ext in {".sqlite", ".sqlite3", ".db"}:
            cols = []
            conn = sqlite3.connect(path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            for table in tables[:50]:
                cur.execute(f"PRAGMA table_info('{table}')")
                cols.extend([r[1] for r in cur.fetchall()])
            conn.close()
            return cols
    except Exception:
        return []
    return []


def path_tokens(path: Path) -> list:
    tokens = []
    parts = [path.name, path.parent.name]
    if path.parent.parent != path.parent:
        parts.append(path.parent.parent.name)
    for part in parts:
        if ":" in part:
            continue
        for token in re.split(r"[^A-Za-z0-9]+", part):
            if token and token.isupper() and 1 <= len(token) <= 5 and token not in STOP_TOKENS:
                tokens.append(token)
    return sorted(set(tokens))


def infer_domain(text: str, fields: list) -> str:
    t = text.lower()
    lf = {f.lower() for f in fields}
    if any(k in t for k in ["reference", "mapping", "ticker", "exchange"]):
        return "reference"
    if any(k in lf for k in CORP_FIELDS) or any(k in t for k in ["dividend", "split", "spinoff", "merger"]):
        return "corporate-actions"
    if any(k in lf for k in FUND_FIELDS) or any(k in t for k in ["fundamental", "balance", "income", "cashflow"]):
        return "fundamentals"
    if any(k in lf for k in DERIV_FIELDS) or any(k in t for k in ["options", "futures", "derivatives", "greek"]):
        return "derivatives"
    if any(k in lf for k in MACRO_FIELDS) or any(k in t for k in ["macro", "cpi", "gdp", "rates", "yield", "fx"]):
        return "macro"
    if any(k in lf for k in ALT_FIELDS) or any(k in t for k in ["sentiment", "news", "esg", "events"]):
        return "alt-data"
    if any(k in t for k in ["backtest", "signals", "trades", "equity_curve", "report", "results", "processed"]):
        return "processed"
    if len(PRICE_FIELDS & lf) >= 2 or any(k in t for k in ["ohlcv", "bars", "quotes", "prices"]):
        return "prices"
    return "na"


def ticker_evidence(path: Path, fields: list) -> tuple:
    tf = [f for f in fields if TICKER_RE.search(f)]
    pf = len(PRICE_FIELDS & {f.lower() for f in fields})
    tokens = path_tokens(path)
    under_data = "data" in [p.lower() for p in path.parts]
    under_backtest = "backtest" in str(path).lower()
    if tf:
        return True, tf, tokens, "high", "explicit ticker fields"
    if tokens and pf >= 2 and (under_data or under_backtest):
        return True, tf, tokens, "medium", "ticker inferred from path tokens + price fields"
    if tokens and (under_data or under_backtest):
        return True, tf, tokens, "low", "ticker inferred from path tokens"
    if pf >= 2 and (under_data or under_backtest):
        return True, tf, tokens, "low", "ticker implied by price fields and data path"
    return False, tf, tokens, "low", "no ticker evidence"


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_versions(path: Path, version_map: dict) -> None:
    ext = "".join(path.suffixes)
    stem = path.name[:-len(ext)] if ext else path.name
    parts = stem.split("__")
    if len(parts) < 2 or not re.fullmatch(r"v\d{3}", parts[-1]):
        return
    base = "__".join(parts[:-1])
    version_map[base] = max(version_map.get(base, 0), int(parts[-1][1:]))


def safe_name(domain: str, inst: str, src: str, reg: str, frq: str, dr: str, version: str, ext: str) -> tuple:
    base = "__".join([domain, inst, src, reg, frq, dr, version])
    name = f"{base}{ext}"
    intended = name
    if len(name) > 200:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        base = "__".join([domain, inst, src, reg, frq, digest, version])
        name = f"{base}{ext}"
    return name, intended


def target_path(domain: str, src: str, inst: str, reg: str, frq: str, name: str) -> Path:
    return DATA_NEW / domain / "source" / src / "instrument" / inst / "region" / reg / "frequency" / frq / name


existing_hashes = {}
version_map = {}
if DATA_NEW.exists():
    for p in DATA_NEW.rglob("*"):
        if p.is_file():
            parse_versions(p, version_map)
            try:
                existing_hashes[sha256_path(p)] = p
            except Exception:
                pass

inventory = []
manifest = {}
exceptions = []
hash_to_path = dict(existing_hashes)


def record(row: dict, extra: dict | None = None) -> None:
    inventory.append(row)
    merged = row.copy()
    if extra:
        merged.update(extra)
    manifest[row["original_path"]] = merged


def record_exception(path_id: str, reason: str) -> None:
    exceptions.append({"path": path_id, "reason": reason})


def handle_file(path: Path, original: str | None = None) -> None:
    ext = comp_ext(path)
    fields = read_fields(path, ext)
    ok, tf, tokens, conf, note = ticker_evidence(path, fields)
    if not ok:
        return
    syn = synthetic(str(path))
    inferred = infer_domain(str(path), fields)
    domain = "synthetic" if syn == "yes" else inferred
    if domain not in BINS:
        domain = "processed"
    inst = instrument(str(path))
    src = source(str(path))
    reg = region(str(path))
    frq = freq(str(path))
    dr = date_range(str(path))
    detected = ",".join(sorted({t.lower() for t in tf})) if tf else ("path_tokens:" + ",".join(tokens) if tokens else "na")
    notes = [f"confidence={conf}", note]
    if domain == "processed" and inferred == "na":
        notes.append("domain inferred as na")
    if syn == "yes" and inferred not in {"na", "synthetic"}:
        notes.append(f"underlying_domain={inferred}")
    for label, val in [("instrument_type", inst), ("source", src), ("market_or_region", reg), ("frequency", frq), ("date_range", dr)]:
        if val == "na":
            notes.append(f"{label}=na")
    dn, ins, srcn, regn, frqn, drn = map(norm, [domain, inst, src, reg, frq, dr])
    base = "__".join([dn, ins, srcn, regn, frqn, drn])
    checksum = sha256_path(path)
    if checksum in hash_to_path:
        tgt = hash_to_path[checksum]
        notes.append("duplicate_content")
    else:
        version = f"v{version_map.get(base, 0) + 1:03d}"
        version_map[base] = version_map.get(base, 0) + 1
        name, intended = safe_name(dn, ins, srcn, regn, frqn, drn, version, ext)
        tgt = target_path(dn, srcn, ins, regn, frqn, name)
        if tgt.exists():
            try:
                if sha256_path(tgt) == checksum:
                    hash_to_path[checksum] = tgt
                else:
                    version = f"v{version_map.get(base, 0) + 1:03d}"
                    version_map[base] = version_map.get(base, 0) + 1
                    name, intended = safe_name(dn, ins, srcn, regn, frqn, drn, version, ext)
                    tgt = target_path(dn, srcn, ins, regn, frqn, name)
            except Exception:
                pass
        if checksum not in hash_to_path:
            tgt.parent.mkdir(parents=True, exist_ok=True)
            if tgt.resolve() != path.resolve():
                shutil.copyfile(path, tgt)
            hash_to_path[checksum] = tgt
    original_path = original or rel(path)
    row = {
        "original_path": original_path,
        "new_path": rel(tgt),
        "size": path.stat().st_size,
        "file_type": ext.lstrip(".") if ext else "na",
        "bin(domain)": dn,
        "synthetic_flag": syn,
        "detected_ticker_fields": detected,
        "instrument_type": ins,
        "source": srcn,
        "market_or_region": regn,
        "frequency": frqn,
        "date_range": drn,
        "checksum": checksum,
        "notes/confidence": "; ".join(notes),
    }
    extra = None
    if any(v == "na" for v in [ins, srcn, regn, frqn, drn]) or (dn == "processed" and inferred == "na"):
        record_exception(original_path, "missing metadata fields")
    record(row, extra)


def handle_archive(path: Path) -> None:
    ext = comp_ext(path)
    relp = rel(path)
    if ext == ".7z":
        record_exception(relp, "unsupported archive format (.7z)")
        return
    if ext == ".gz" and not path.name.lower().endswith(".tar.gz"):
        record_exception(relp, "unsupported compressed file (.gz without data extension)")
        return
    members = []
    try:
        if ext == ".zip":
            with zipfile.ZipFile(path) as zf:
                members = zf.infolist()
        else:
            with tarfile.open(path, "r:*") as tf:
                members = tf.getmembers()
    except Exception as exc:
        record_exception(relp, f"archive read error: {exc}")
        return
    extracted = False
    for mem in members:
        if hasattr(mem, "isreg") and not mem.isreg():
            continue
        name = mem.filename if hasattr(mem, "filename") else mem.name
        mpath = Path(name)
        mext = comp_ext(mpath)
        if mext in ARCHIVE_EXTS:
            record_exception(f"{relp}::/{name}", "nested archive not processed")
            continue
        if not is_data_ext(mext):
            continue
        try:
            if ext == ".zip":
                with zipfile.ZipFile(path) as zf:
                    data = zf.read(name)
            else:
                with tarfile.open(path, "r:*") as tf:
                    mf = tf.extractfile(mem)
                    if mf is None:
                        continue
                    data = mf.read()
        except Exception as exc:
            record_exception(f"{relp}::/{name}", f"member read error: {exc}")
            continue
        try:
            fields = []
            if mext in {".csv", ".tsv"}:
                delim = "\t" if mext == ".tsv" else ","
                head = data.decode("utf-8", errors="ignore").splitlines()[0] if data else ""
                fields = parse_header(head, delim)
            elif mext in {".json", ".jsonl"}:
                keys = re.findall(r"\"([A-Za-z0-9_\\-]+)\"\\s*:", data.decode("utf-8", errors="ignore"))
                fields = keys
        except Exception:
            fields = []
        ok, tf, tokens, conf, note = ticker_evidence(mpath, fields)
        if not ok:
            continue
        extracted = True
        syn = synthetic(name)
        inferred = infer_domain(name, fields)
        domain = "synthetic" if syn == "yes" else inferred
        if domain not in BINS:
            domain = "processed"
        inst = instrument(name)
        src = source(name)
        reg = region(name)
        frq = freq(name)
        dr = date_range(name)
        dn, ins, srcn, regn, frqn, drn = map(norm, [domain, inst, src, reg, frq, dr])
        base = "__".join([dn, ins, srcn, regn, frqn, drn])
        checksum = sha256_bytes(data)
        if checksum in hash_to_path:
            tgt = hash_to_path[checksum]
            notes = f"confidence={conf}; {note}; duplicate_content"
        else:
            version = f"v{version_map.get(base, 0) + 1:03d}"
            version_map[base] = version_map.get(base, 0) + 1
            name_out, intended = safe_name(dn, ins, srcn, regn, frqn, drn, version, mext)
            tgt = target_path(dn, srcn, ins, regn, frqn, name_out)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_bytes(data)
            hash_to_path[checksum] = tgt
            notes = f"confidence={conf}; {note}"
        row = {
            "original_path": f"{relp}::/{name}",
            "new_path": rel(tgt),
            "size": len(data),
            "file_type": mext.lstrip("."),
            "bin(domain)": dn,
            "synthetic_flag": syn,
            "detected_ticker_fields": ",".join(sorted({t.lower() for t in tf})) if tf else ("path_tokens:" + ",".join(tokens) if tokens else "na"),
            "instrument_type": ins,
            "source": srcn,
            "market_or_region": regn,
            "frequency": frqn,
            "date_range": drn,
            "checksum": checksum,
            "notes/confidence": notes,
        }
        record(row, {"archive_parent": relp})
        if any(v == "na" for v in [ins, srcn, regn, frqn, drn]):
            record_exception(row["original_path"], "missing metadata fields")
    if extracted:
        raw_ext = comp_ext(path)
        raw_src = norm(source(str(path)))
        raw_dr = norm(date_range(str(path)))
        base = "__".join(["raw", "na", raw_src, "na", "na", raw_dr])
        version = f"v{version_map.get(base, 0) + 1:03d}"
        version_map[base] = version_map.get(base, 0) + 1
        name_out, intended = safe_name("raw", "na", raw_src, "na", "na", raw_dr, version, raw_ext)
        tgt = target_path("raw", raw_src, "na", "na", "na", name_out)
        checksum = sha256_path(path)
        if checksum not in hash_to_path:
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, tgt)
            hash_to_path[checksum] = tgt
        row = {
            "original_path": relp,
            "new_path": rel(hash_to_path[checksum]),
            "size": path.stat().st_size,
            "file_type": raw_ext.lstrip(".") if raw_ext else "na",
            "bin(domain)": "raw",
            "synthetic_flag": "na",
            "detected_ticker_fields": "na",
            "instrument_type": "na",
            "source": raw_src,
            "market_or_region": "na",
            "frequency": "na",
            "date_range": raw_dr,
            "checksum": checksum,
            "notes/confidence": "raw archive copy",
        }
        record(row, {"archive_contains": True})


seen = set()
scan_steps = [
    (ROOT / "data", set()),
    (ROOT / "docs", {ROOT / "docs" / "knowledge-base"}),
    (ROOT / "docs" / "knowledge-base", set()),
    (ROOT, {ROOT / "data", ROOT / "docs"}),
]

paths = []
for base, skip in scan_steps:
    if not base.exists():
        continue
    for root, dirs, files in os.walk(base):
        rp = Path(root)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and (rp / d) not in skip]
        for name in files:
            p = rp / name
            if p in seen:
                continue
            seen.add(p)
            if is_data_ext(comp_ext(p)):
                paths.append(p)

for p in sorted(paths):
    ext = comp_ext(p)
    if ext in ARCHIVE_EXTS:
        handle_archive(p)
    else:
        handle_file(p)


counts_bin = {}
counts_source = {}
counts_type = {}
for row in inventory:
    counts_bin[row["bin(domain)"]] = counts_bin.get(row["bin(domain)"], 0) + 1
    counts_source[row["source"]] = counts_source.get(row["source"], 0) + 1
    counts_type[row["file_type"]] = counts_type.get(row["file_type"], 0) + 1

inventory_path = ARTIFACTS / "market_data_refactor_inventory.md"
with inventory_path.open("w", encoding="utf-8", newline="\n") as f:
    f.write("# Market Data Refactor Inventory\n\n")
    f.write("## Inventory\n\n")
    headers = [
        "original_path",
        "new_path",
        "size",
        "file_type",
        "bin(domain)",
        "synthetic_flag",
        "detected_ticker_fields",
        "instrument_type",
        "source",
        "market_or_region",
        "frequency",
        "date_range",
        "checksum",
        "notes/confidence",
    ]
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for row in inventory:
        values = [str(row.get(h, "")) for h in headers]
        f.write("| " + " | ".join(v.replace("|", "\\|") for v in values) + " |\n")
    f.write("\n## Summary\n\n")
    f.write(f"- total_inventory_rows: {len(inventory)}\n")
    f.write(f"- unique_checksums: {len({r['checksum'] for r in inventory})}\n")
    f.write("- counts_by_bin:\n")
    for k in sorted(counts_bin):
        f.write(f"  - {k}: {counts_bin[k]}\n")
    f.write("- counts_by_source:\n")
    for k in sorted(counts_source):
        f.write(f"  - {k}: {counts_source[k]}\n")
    f.write("- counts_by_file_type:\n")
    for k in sorted(counts_type):
        f.write(f"  - {k}: {counts_type[k]}\n")
    f.write("\n## Exceptions\n\n")
    if not exceptions:
        f.write("- none\n")
    else:
        for exc in exceptions:
            f.write(f"- {exc['path']}: {exc['reason']}\n")

manifest_path = ARTIFACTS / "market_data_refactor_manifest.json"
with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
    json.dump({"generated_at": dt.datetime.utcnow().isoformat() + "Z", "entries": manifest}, f, indent=2)

print(f"Inventory written: {inventory_path}")
print(f"Manifest written: {manifest_path}")
print(f"Inventory rows: {len(inventory)}")
print(f"Exceptions: {len(exceptions)}")
