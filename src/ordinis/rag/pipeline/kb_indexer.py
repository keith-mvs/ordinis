"""Knowledge base indexing pipeline."""

from pathlib import Path
import re

from loguru import logger
import tiktoken

from ordinis.rag.config import get_config
from ordinis.rag.embedders.text_embedder import TextEmbedder
from ordinis.rag.vectordb.chroma_client import ChromaClient
from ordinis.rag.vectordb.schema import TextChunkMetadata


class KBIndexer:
    """Index knowledge base markdown documents."""

    def __init__(
        self,
        chroma_client: ChromaClient | None = None,
        text_embedder: TextEmbedder | None = None,
    ):
        """Initialize KB indexer.

        Args:
            chroma_client: ChromaDB client
            text_embedder: Text embedder
        """
        self.config = get_config()
        self.chroma_client = chroma_client or ChromaClient()
        self.text_embedder = text_embedder or TextEmbedder()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

        logger.info("KB indexer initialized")

    def index_directory(
        self,
        kb_path: Path | None = None,
        batch_size: int = 32,
    ) -> dict:
        """Index all markdown files in KB directory.

        Args:
            kb_path: Path to knowledge base directory (uses config default if None)
            batch_size: Batch size for embedding

        Returns:
            Dictionary with indexing statistics
        """
        kb_path = kb_path or self.config.kb_base_path

        if not kb_path.exists():
            msg = f"KB path does not exist: {kb_path}"
            raise FileNotFoundError(msg)

        logger.info(f"Indexing KB from: {kb_path}")

        # Find all markdown files
        md_files = list(kb_path.rglob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files")

        total_chunks = 0
        total_tokens = 0

        # Process files in batches
        all_texts = []
        all_metadata = []
        all_ids = []

        for md_file in md_files:
            try:
                chunks, metadata = self._process_file(md_file, kb_path)
                all_texts.extend(chunks)
                all_metadata.extend(metadata)

                # Generate IDs
                base_id = str(md_file.relative_to(kb_path)).replace("\\", "/").replace("/", "_")
                chunk_ids = [f"{base_id}_chunk{i}" for i in range(len(chunks))]
                all_ids.extend(chunk_ids)

                total_chunks += len(chunks)
                total_tokens += sum(len(self.tokenizer.encode(chunk)) for chunk in chunks)

                logger.debug(f"Processed {md_file.name}: {len(chunks)} chunks")

            except Exception as e:
                logger.warning(f"Failed to process {md_file}: {e}")

        # Embed and store in batches
        logger.info(f"Embedding {len(all_texts)} chunks in batches of {batch_size}")

        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i : i + batch_size]
            batch_metadata = all_metadata[i : i + batch_size]
            batch_ids = all_ids[i : i + batch_size]

            # Embed batch
            embeddings = self.text_embedder.embed(batch_texts)

            # Store in ChromaDB
            self.chroma_client.add_texts(
                texts=batch_texts,
                embeddings=embeddings,
                metadata=[m.model_dump() for m in batch_metadata],
                ids=batch_ids,
            )

            logger.info(
                f"Indexed batch {i // batch_size + 1}/{(len(all_texts) - 1) // batch_size + 1}"
            )

        stats = {
            "files_processed": len(md_files),
            "chunks_created": total_chunks,
            "total_tokens": total_tokens,
            "average_chunk_size": total_tokens // total_chunks if total_chunks > 0 else 0,
        }

        logger.success(f"KB indexing complete: {stats}")
        return stats

    def _process_file(
        self,
        file_path: Path,
        kb_base: Path,
    ) -> tuple[list[str], list[TextChunkMetadata]]:
        """Process a single markdown file into chunks.

        Args:
            file_path: Path to markdown file
            kb_base: Base KB directory

        Returns:
            Tuple of (chunk texts, chunk metadata)
        """
        # Read file
        content = file_path.read_text(encoding="utf-8")

        # Extract domain from path if present (e.g., docs/knowledge-base/02-technical-analysis/)
        relative_path = file_path.relative_to(kb_base)
        path_parts = relative_path.parts

        domain = None
        if len(path_parts) > 0:
            # Try to extract domain number from directory name (e.g., "02-technical-analysis")
            match = re.match(r"(\d{2})", path_parts[0])
            if match:
                domain = int(match.group(1))

        # Determine source identifier
        source = str(relative_path).replace("\\", "/")

        # Chunk the content
        chunks = self._chunk_text(content)

        # Create metadata for each chunk
        metadata = [
            TextChunkMetadata(
                domain=domain,
                source=source,
                chunk_index=i,
                section=None,  # Could extract from headers
            )
            for i in range(len(chunks))
        ]

        return chunks, metadata

    def _chunk_text(self, text: str) -> list[str]:
        """Chunk text into smaller pieces.

        Uses token-based chunking with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunk_size = self.config.text_chunk_size
        chunk_overlap = self.config.text_chunk_overlap

        # Tokenize
        tokens = self.tokenizer.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            # Get chunk
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move to next chunk with overlap
            start = end - chunk_overlap

            # Prevent infinite loop
            if start >= len(tokens):
                break

        return chunks
