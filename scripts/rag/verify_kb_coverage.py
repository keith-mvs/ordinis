#!/usr/bin/env python
"""
Verify KB indexing coverage and identify gaps.

Compares source files against ChromaDB indexed documents to identify:
- Missing files (not indexed)
- Partial coverage (some chunks indexed)
- Index statistics and health
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import chromadb
from chromadb.config import Settings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def get_source_files():
    """Get all source files from knowledge-base and publications."""
    kb_path = PROJECT_ROOT / "docs" / "knowledge-base"
    pub_path = PROJECT_ROOT / "docs" / "publications"

    sources = {
        "markdown": [],
        "python": [],
        "pdf_kb": [],
        "pdf_publications": [],
    }

    # Knowledge base files
    if kb_path.exists():
        sources["markdown"] = list(kb_path.rglob("*.md"))
        sources["python"] = list(kb_path.rglob("*.py"))
        sources["pdf_kb"] = list(kb_path.rglob("*.pdf"))

    # Publications
    if pub_path.exists():
        sources["pdf_publications"] = list(pub_path.glob("*.pdf"))

    return sources


def inspect_collections():
    """Inspect all ChromaDB collections and their contents."""
    chroma_path = PROJECT_ROOT / "data" / "chromadb"

    if not chroma_path.exists():
        return {"error": "ChromaDB path does not exist"}

    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )

    collections = {}
    for col in client.list_collections():
        count = col.count()
        collections[col.name] = {
            "count": count,
            "metadata": col.metadata,
            "sources": set(),
            "sample_ids": [],
            "source_files": defaultdict(int),
        }

        if count > 0:
            # Get all documents to analyze sources
            all_data = col.get(
                include=["metadatas"],
                limit=min(count, 10000),
            )

            for meta in all_data["metadatas"]:
                if meta:
                    # Try different source field names
                    source = (
                        meta.get("source")
                        or meta.get("source_file")
                        or meta.get("file_path")
                        or "unknown"
                    )
                    collections[col.name]["sources"].add(source)
                    collections[col.name]["source_files"][source] += 1

            # Get sample IDs
            collections[col.name]["sample_ids"] = all_data["ids"][:5]
            collections[col.name]["sources"] = list(collections[col.name]["sources"])[:50]
            collections[col.name]["source_files"] = dict(collections[col.name]["source_files"])

    return collections


def verify_publications_coverage(collections):
    """Verify all publications PDFs are indexed."""
    pub_path = PROJECT_ROOT / "docs" / "publications"
    pdf_files = list(pub_path.glob("*.pdf"))

    if "publications" not in collections:
        return {
            "indexed": False,
            "expected_files": len(pdf_files),
            "indexed_files": 0,
            "missing_files": [f.name for f in pdf_files],
        }

    indexed_sources = set(collections["publications"]["source_files"].keys())

    # Map PDF names to indexed sources
    indexed_count = 0
    missing = []
    partial = []
    complete = []

    for pdf in pdf_files:
        pdf_name = pdf.name
        if pdf_name in indexed_sources:
            chunk_count = collections["publications"]["source_files"][pdf_name]
            if chunk_count >= 3:  # Minimum expected chunks
                complete.append({"file": pdf_name, "chunks": chunk_count})
                indexed_count += 1
            else:
                partial.append({"file": pdf_name, "chunks": chunk_count})
        else:
            missing.append(pdf_name)

    return {
        "expected_files": len(pdf_files),
        "indexed_files": indexed_count,
        "partial_files": len(partial),
        "missing_files": len(missing),
        "complete": complete,
        "partial": partial,
        "missing": missing,
    }


def verify_kb_coverage(collections, sources):
    """Verify knowledge base files are indexed."""
    if "kb_text" not in collections:
        return {
            "indexed": False,
            "expected_files": len(sources["markdown"]),
            "indexed_files": 0,
        }

    indexed_sources = set(collections["kb_text"]["source_files"].keys())

    # Expected vs actual
    expected_md = {
        str(f.relative_to(PROJECT_ROOT / "docs" / "knowledge-base")) for f in sources["markdown"]
    }

    # Find coverage
    indexed = expected_md & indexed_sources
    missing = expected_md - indexed_sources

    return {
        "expected_files": len(sources["markdown"]),
        "indexed_files": len(indexed),
        "indexed_chunks": collections["kb_text"]["count"],
        "missing_count": len(missing),
        "sample_missing": list(missing)[:20],
        "indexed_sources": list(indexed_sources)[:20],
    }


def main():
    """Run verification."""
    print("=" * 70)
    print("KB INDEXING COVERAGE VERIFICATION REPORT")
    print("=" * 70)

    # Get source files
    print("\n[1/4] Scanning source files...")
    sources = get_source_files()
    print(f"  Markdown files: {len(sources['markdown'])}")
    print(f"  Python files: {len(sources['python'])}")
    print(f"  PDFs in KB: {len(sources['pdf_kb'])}")
    print(f"  PDFs in Publications: {len(sources['pdf_publications'])}")

    # Inspect ChromaDB
    print("\n[2/4] Inspecting ChromaDB collections...")
    collections = inspect_collections()

    if "error" in collections:
        print(f"  ERROR: {collections['error']}")
        return None

    for name, info in collections.items():
        print(f"\n  Collection: {name}")
        print(f"    Documents: {info['count']}")
        print(f"    Unique sources: {len(info['source_files'])}")
        if info["sample_ids"]:
            print(f"    Sample IDs: {info['sample_ids'][:3]}")

    # Verify publications
    print("\n[3/4] Verifying publications coverage...")
    pub_coverage = verify_publications_coverage(collections)
    print(f"  Expected PDFs: {pub_coverage['expected_files']}")
    print(f"  Fully indexed: {pub_coverage['indexed_files']}")
    print(f"  Partial: {pub_coverage.get('partial_files', 0)}")
    print(f"  Missing: {pub_coverage.get('missing_files', 0)}")

    if pub_coverage.get("missing"):
        print(f"\n  MISSING FILES ({len(pub_coverage['missing'])}):")
        for f in pub_coverage["missing"][:10]:
            print(f"    - {f}")

    if pub_coverage.get("partial"):
        print(f"\n  PARTIAL FILES ({len(pub_coverage['partial'])}):")
        for p in pub_coverage["partial"][:10]:
            print(f"    - {p['file']}: {p['chunks']} chunks")

    # Verify KB coverage
    print("\n[4/4] Verifying knowledge-base coverage...")
    kb_coverage = verify_kb_coverage(collections, sources)
    print(f"  Expected MD files: {kb_coverage['expected_files']}")
    print(f"  Indexed files: {kb_coverage['indexed_files']}")
    print(f"  Indexed chunks: {kb_coverage['indexed_chunks']}")
    print(f"  Missing files: {kb_coverage['missing_count']}")

    if kb_coverage.get("sample_missing"):
        print(f"\n  SAMPLE MISSING FILES:")
        for f in kb_coverage["sample_missing"][:10]:
            print(f"    - {f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    kb_pct = (
        (kb_coverage["indexed_files"] / kb_coverage["expected_files"] * 100)
        if kb_coverage["expected_files"] > 0
        else 0
    )
    pub_pct = (
        (pub_coverage["indexed_files"] / pub_coverage["expected_files"] * 100)
        if pub_coverage["expected_files"] > 0
        else 0
    )

    print(
        f"\nKnowledge Base Coverage: {kb_pct:.1f}% ({kb_coverage['indexed_files']}/{kb_coverage['expected_files']})"
    )
    print(
        f"Publications Coverage: {pub_pct:.1f}% ({pub_coverage['indexed_files']}/{pub_coverage['expected_files']})"
    )

    # Status
    if kb_pct < 50:
        print("\n[CRITICAL] Knowledge base indexing incomplete or failed!")
    if pub_pct < 80:
        print("\n[WARNING] Publications indexing incomplete!")

    if collections.get("codebase", {}).get("count", 0) == 0:
        print("\n[WARNING] Codebase collection is empty!")

    # Return data for further processing
    return {
        "sources": {k: len(v) for k, v in sources.items()},
        "collections": {
            k: {"count": v["count"], "sources": len(v["source_files"])}
            for k, v in collections.items()
        },
        "kb_coverage": kb_coverage,
        "pub_coverage": pub_coverage,
    }


if __name__ == "__main__":
    result = main()

    # Save report
    report_path = PROJECT_ROOT / "artifacts" / "reports" / "kb_verification_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        # Convert sets to lists for JSON serialization
        json.dump(result, f, indent=2, default=list)

    print(f"\nReport saved to: {report_path}")
