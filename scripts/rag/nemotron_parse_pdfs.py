"""
Parse PDFs from docs/publications using NVIDIA Nemotron-Parse.

Extracts structured text from research PDFs and indexes into ChromaDB
for RAG retrieval via Synapse engine.

Uses PyMuPDF to convert PDF pages to images, then sends to NVIDIA API.

Usage:
    python scripts/rag/nemotron_parse_pdfs.py
    python scripts/rag/nemotron_parse_pdfs.py --dry-run
    python scripts/rag/nemotron_parse_pdfs.py --file arxiv-1907.00212.pdf
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import fitz  # PyMuPDF
import httpx
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ParseResult:
    """Result of parsing a single PDF."""

    file_path: Path
    success: bool
    markdown_content: str = ""
    pages_parsed: int = 0
    error: str | None = None
    parse_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexResult:
    """Result of indexing parsed content."""

    file_path: Path
    chunks_created: int = 0
    tokens_indexed: int = 0
    success: bool = True
    error: str | None = None


class NemotronPDFParser:
    """
    Parse PDFs using NVIDIA Nemotron-Parse API.

    API Reference: https://build.nvidia.com/nvidia/nemotron-parse
    """

    NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    MODEL_ID = "nvidia/nemotron-parse"
    MAX_PAGES_PER_REQUEST = 10  # API limit

    def __init__(self, api_key: str | None = None):
        """
        Initialize parser.

        Args:
            api_key: NVIDIA API key. Falls back to NVIDIA_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found. Set environment variable or pass api_key.")

        self.client = httpx.AsyncClient(timeout=180.0)
        logger.info("NemotronPDFParser initialized")

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    def _pdf_to_images(self, pdf_path: Path, dpi: int = 150) -> list[bytes]:
        """
        Convert PDF pages to PNG images using PyMuPDF.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering (150 is good balance of quality/size)

        Returns:
            List of PNG image bytes for each page
        """
        images = []
        doc = fitz.open(pdf_path)

        try:
            zoom = dpi / 72  # 72 is default PDF DPI
            matrix = fitz.Matrix(zoom, zoom)

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=matrix)

                # Convert to PNG bytes
                png_bytes = pix.tobytes("png")
                images.append(png_bytes)

                logger.debug(f"Rendered page {page_num + 1}/{len(doc)}")

        finally:
            doc.close()

        return images

    async def _parse_page_image(
        self,
        image_bytes: bytes,
        page_num: int,
        total_pages: int,
    ) -> str:
        """
        Parse a single page image via Nemotron-Parse API.

        Args:
            image_bytes: PNG image bytes
            page_num: Current page number (1-indexed)
            total_pages: Total number of pages

        Returns:
            Extracted markdown text for this page
        """
        image_base64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": self.MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "markdown_no_bbox",
                        "description": "Extract document content as markdown without bounding boxes",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "markdown_no_bbox"},
            },
            "max_tokens": 8192,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        response = await self.client.post(
            self.NVIDIA_API_URL,
            headers=headers,
            json=payload,
        )

        if response.status_code == 429:
            raise Exception("Rate limited")

        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text[:200]}")

        result = response.json()
        return self._extract_markdown(result)

    async def parse_pdf(self, pdf_path: Path, max_pages: int = 50) -> ParseResult:
        """
        Parse a single PDF file by converting pages to images.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to parse (limit for large docs)

        Returns:
            ParseResult with extracted markdown content
        """
        start_time = time.perf_counter()

        if not pdf_path.exists():
            return ParseResult(
                file_path=pdf_path,
                success=False,
                error=f"File not found: {pdf_path}",
            )

        try:
            # Convert PDF to images
            logger.info(f"Converting PDF to images: {pdf_path.name}")
            page_images = self._pdf_to_images(pdf_path)

            if not page_images:
                return ParseResult(
                    file_path=pdf_path,
                    success=False,
                    error="Failed to extract pages from PDF",
                )

            total_pages = len(page_images)
            pages_to_parse = min(total_pages, max_pages)

            logger.info(
                f"Parsing {pdf_path.name}: {pages_to_parse}/{total_pages} pages "
                f"({sum(len(img) for img in page_images[:pages_to_parse]) / 1024:.0f} KB)"
            )

            # Parse each page
            all_markdown = []
            for i, image_bytes in enumerate(page_images[:pages_to_parse]):
                try:
                    logger.debug(f"Parsing page {i + 1}/{pages_to_parse}")
                    page_markdown = await self._parse_page_image(image_bytes, i + 1, pages_to_parse)
                    if page_markdown:
                        all_markdown.append(f"<!-- Page {i + 1} -->\n{page_markdown}")

                    # Rate limiting between pages
                    if i < pages_to_parse - 1:
                        await asyncio.sleep(0.5)

                except Exception as e:
                    logger.warning(f"Failed to parse page {i + 1}: {e}")
                    # Continue with other pages

            if not all_markdown:
                return ParseResult(
                    file_path=pdf_path,
                    success=False,
                    error="No content extracted from any page",
                )

            markdown_content = "\n\n".join(all_markdown)
            parse_time = (time.perf_counter() - start_time) * 1000

            # Extract metadata from filename
            metadata = self._extract_metadata(pdf_path)
            metadata["total_pages"] = total_pages
            metadata["pages_parsed"] = len(all_markdown)

            logger.success(
                f"Parsed {pdf_path.name}: {len(markdown_content)} chars, "
                f"{len(all_markdown)} pages in {parse_time:.0f}ms"
            )

            return ParseResult(
                file_path=pdf_path,
                success=True,
                markdown_content=markdown_content,
                pages_parsed=len(all_markdown),
                parse_time_ms=parse_time,
                metadata=metadata,
            )

        except httpx.TimeoutException:
            return ParseResult(
                file_path=pdf_path,
                success=False,
                error="Request timed out (180s limit)",
            )
        except Exception as e:
            logger.exception(f"Parse error for {pdf_path.name}")
            return ParseResult(
                file_path=pdf_path,
                success=False,
                error=str(e),
            )

    def _extract_markdown(self, api_response: dict) -> str:
        """Extract markdown content from API response."""
        try:
            choices = api_response.get("choices", [])
            if not choices:
                logger.debug(f"No choices in response: {api_response}")
                return ""

            message = choices[0].get("message", {})

            # Check for tool calls (structured output)
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                for call in tool_calls:
                    func = call.get("function", {})
                    if func.get("name") == "markdown_no_bbox":
                        args = func.get("arguments", "{}")
                        # Parse JSON string if needed
                        if isinstance(args, str):
                            parsed_args = json.loads(args)
                        else:
                            parsed_args = args

                        # Handle different response formats
                        if isinstance(parsed_args, dict):
                            # Try common keys for markdown content
                            for key in ["text", "markdown", "content", "output"]:
                                if key in parsed_args:
                                    val = parsed_args[key]
                                    # Handle nested structures
                                    if isinstance(val, str):
                                        return val
                                    if isinstance(val, list):
                                        return "\n".join(str(v) for v in val)
                                    return str(val)
                            # If no known key, try to extract all text-like values
                            text_parts = []
                            for k, v in parsed_args.items():
                                if isinstance(v, str) and len(v) > 20:
                                    text_parts.append(v)
                            if text_parts:
                                return "\n\n".join(text_parts)
                            # Last resort - stringify
                            return json.dumps(parsed_args)
                        if isinstance(parsed_args, str):
                            return parsed_args
                        if isinstance(parsed_args, list):
                            # Join list elements
                            return "\n".join(str(item) for item in parsed_args)
                        return str(parsed_args)

            # Fallback to direct content
            content = message.get("content")
            if content:
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    # Handle content array format
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            text_parts.append(item.get("text", str(item)))
                        else:
                            text_parts.append(str(item))
                    return "\n".join(text_parts)
                return str(content)

            return ""

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in markdown extraction: {e}")
            # Return raw arguments string if JSON parsing fails
            try:
                tool_calls = (
                    api_response.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                )
                if tool_calls:
                    return tool_calls[0].get("function", {}).get("arguments", "")
            except Exception:
                pass
            return ""
        except Exception as e:
            logger.warning(f"Failed to extract markdown: {e}")
            logger.debug(f"Response structure: {json.dumps(api_response, indent=2)[:1000]}")
            return ""

    def _extract_metadata(self, pdf_path: Path) -> dict[str, Any]:
        """Extract metadata from PDF filename."""
        filename = pdf_path.stem.lower()
        metadata = {
            "source_file": pdf_path.name,
            "domain": "research",
            "indexed_at": datetime.utcnow().isoformat(),
        }

        # Detect source type from filename
        if "arxiv" in filename:
            metadata["source_type"] = "arxiv"
            # Extract arxiv ID (e.g., arxiv-2206.12282)
            parts = filename.split("-")
            if len(parts) >= 2:
                metadata["arxiv_id"] = parts[1]
        elif "ssrn" in filename:
            metadata["source_type"] = "ssrn"
            parts = filename.split("-")
            if len(parts) >= 2:
                metadata["ssrn_id"] = parts[1]
        elif "nber" in filename:
            metadata["source_type"] = "nber"
        elif "aqr" in filename:
            metadata["source_type"] = "aqr"
        else:
            metadata["source_type"] = "paper"

        # Detect topic from filename keywords
        topic_keywords = {
            "technical-analysis": ["technical", "indicator", "chart"],
            "machine-learning": ["ml", "learning", "neural", "lstm", "gnn", "transformer"],
            "risk-management": ["risk", "var", "cvar", "volatility", "hedge"],
            "execution": ["execution", "vwap", "almgren", "optimal"],
            "factor-investing": ["factor", "momentum", "value", "quality"],
            "backtesting": ["backtest", "overfitting", "sharpe"],
            "crypto": ["crypto", "bitcoin", "currency"],
            "options": ["option", "derivative", "hedge"],
            "regime": ["regime", "hmm", "switching"],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in filename for kw in keywords):
                metadata["topic"] = topic
                break

        return metadata


class PublicationsIndexer:
    """Index parsed publications into ChromaDB."""

    COLLECTION_NAME = "publications"

    def __init__(self):
        """Initialize indexer with RAG components."""
        from ordinis.rag.config import get_config
        from ordinis.rag.embedders.text_embedder import TextEmbedder
        from ordinis.rag.vectordb.chroma_client import ChromaClient

        self.config = get_config()
        self.chroma_client = ChromaClient()
        self.text_embedder = TextEmbedder()

        # Ensure publications collection exists
        if not self.chroma_client.check_collection_exists(self.COLLECTION_NAME):
            self.chroma_client.create_collection(
                self.COLLECTION_NAME,
                metadata={"description": "Research publications from docs/publications"},
            )
            logger.info(f"Created collection: {self.COLLECTION_NAME}")

        import tiktoken

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def index_parsed_content(
        self,
        parse_result: ParseResult,
        batch_size: int = 16,
    ) -> IndexResult:
        """
        Index parsed PDF content into ChromaDB.

        Args:
            parse_result: Parsed PDF content
            batch_size: Embedding batch size

        Returns:
            IndexResult with statistics
        """
        if not parse_result.success or not parse_result.markdown_content:
            return IndexResult(
                file_path=parse_result.file_path,
                success=False,
                error=parse_result.error or "No content to index",
            )

        try:
            # Chunk the content
            chunks = self._chunk_text(parse_result.markdown_content)

            if not chunks:
                return IndexResult(
                    file_path=parse_result.file_path,
                    success=False,
                    error="No chunks created from content",
                )

            # Create metadata for each chunk
            base_metadata = parse_result.metadata.copy()
            chunk_metadata = []
            chunk_ids = []

            file_id = parse_result.file_path.stem.replace(" ", "_").replace("-", "_")

            for i, chunk in enumerate(chunks):
                meta = base_metadata.copy()
                meta["chunk_index"] = i
                meta["total_chunks"] = len(chunks)
                chunk_metadata.append(meta)
                chunk_ids.append(f"pub_{file_id}_chunk{i}")

            # Embed in batches
            all_embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                embeddings = self.text_embedder.embed(batch)
                all_embeddings.extend(embeddings.tolist())

            # Store in ChromaDB
            collection = self.chroma_client.client.get_or_create_collection(self.COLLECTION_NAME)

            # Upsert to handle re-indexing
            collection.upsert(
                documents=chunks,
                embeddings=all_embeddings,
                metadatas=chunk_metadata,
                ids=chunk_ids,
            )

            total_tokens = sum(len(self.tokenizer.encode(c)) for c in chunks)

            logger.success(
                f"Indexed {parse_result.file_path.name}: "
                f"{len(chunks)} chunks, {total_tokens} tokens"
            )

            return IndexResult(
                file_path=parse_result.file_path,
                chunks_created=len(chunks),
                tokens_indexed=total_tokens,
                success=True,
            )

        except Exception as e:
            logger.exception(f"Index error for {parse_result.file_path.name}")
            return IndexResult(
                file_path=parse_result.file_path,
                success=False,
                error=str(e),
            )

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[str]:
        """
        Chunk text into smaller pieces.

        Args:
            text: Text to chunk
            chunk_size: Target tokens per chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Skip very short chunks
            if len(chunk_text.strip()) > 50:
                chunks.append(chunk_text.strip())

            if end >= len(tokens):
                break

            start = max(end - chunk_overlap, start + 1)

        return chunks


async def parse_and_index_publications(
    publications_dir: Path,
    dry_run: bool = False,
    single_file: str | None = None,
    max_files: int | None = None,
) -> dict[str, Any]:
    """
    Parse all PDFs in publications directory and index to RAG.

    Args:
        publications_dir: Path to publications directory
        dry_run: If True, only parse without indexing
        single_file: Process only this specific file
        max_files: Maximum number of files to process

    Returns:
        Summary statistics
    """
    # Find PDFs
    if single_file:
        pdf_files = [publications_dir / single_file]
        if not pdf_files[0].exists():
            logger.error(f"File not found: {single_file}")
            return {"error": f"File not found: {single_file}"}
    else:
        pdf_files = sorted(publications_dir.glob("*.pdf"))

    if max_files:
        pdf_files = pdf_files[:max_files]

    if not pdf_files:
        logger.warning("No PDF files found")
        return {"error": "No PDF files found"}

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Initialize parser
    parser = NemotronPDFParser()
    indexer = None if dry_run else PublicationsIndexer()

    # Track results
    results = {
        "total_files": len(pdf_files),
        "parsed_success": 0,
        "parsed_failed": 0,
        "indexed_success": 0,
        "indexed_failed": 0,
        "total_chunks": 0,
        "total_tokens": 0,
        "errors": [],
    }

    try:
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

            # Parse PDF
            parse_result = await parser.parse_pdf(pdf_path)

            if parse_result.success:
                results["parsed_success"] += 1

                if not dry_run and indexer:
                    # Index to ChromaDB
                    index_result = indexer.index_parsed_content(parse_result)

                    if index_result.success:
                        results["indexed_success"] += 1
                        results["total_chunks"] += index_result.chunks_created
                        results["total_tokens"] += index_result.tokens_indexed
                    else:
                        results["indexed_failed"] += 1
                        results["errors"].append(
                            {
                                "file": pdf_path.name,
                                "stage": "index",
                                "error": index_result.error,
                            }
                        )
                else:
                    # Dry run - just show preview
                    preview = parse_result.markdown_content[:500]
                    logger.info(f"Preview:\n{preview}...")
            else:
                results["parsed_failed"] += 1
                results["errors"].append(
                    {
                        "file": pdf_path.name,
                        "stage": "parse",
                        "error": parse_result.error,
                    }
                )

            # Rate limiting - small delay between requests
            if i < len(pdf_files):
                await asyncio.sleep(1.0)

    finally:
        await parser.close()

    # Summary
    logger.info("=" * 60)
    logger.info("PARSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files: {results['total_files']}")
    logger.info(f"Parsed successfully: {results['parsed_success']}")
    logger.info(f"Parse failures: {results['parsed_failed']}")

    if not dry_run:
        logger.info(f"Indexed successfully: {results['indexed_success']}")
        logger.info(f"Index failures: {results['indexed_failed']}")
        logger.info(f"Total chunks created: {results['total_chunks']}")
        logger.info(f"Total tokens indexed: {results['total_tokens']}")

    if results["errors"]:
        logger.warning(f"Errors encountered: {len(results['errors'])}")
        for err in results["errors"][:5]:
            logger.warning(f"  - {err['file']} ({err['stage']}): {err['error']}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parse PDFs with Nemotron-Parse and index to RAG database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse only, don't index to database",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Process single file instead of all",
    )
    parser.add_argument(
        "--max",
        type=int,
        help="Maximum number of files to process",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "publications"),
        help="Publications directory path",
    )

    args = parser.parse_args()

    publications_dir = Path(args.dir)
    if not publications_dir.exists():
        logger.error(f"Publications directory not found: {publications_dir}")
        sys.exit(1)

    # Run async parsing
    results = asyncio.run(
        parse_and_index_publications(
            publications_dir=publications_dir,
            dry_run=args.dry_run,
            single_file=args.file,
            max_files=args.max,
        )
    )

    # Exit code based on success
    if results.get("error") or results.get("parsed_failed", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
