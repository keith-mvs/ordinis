"""Code indexing pipeline."""

import ast
from pathlib import Path

from loguru import logger
import tiktoken

from ordinis.rag.config import get_config
from ordinis.rag.embedders.code_embedder import CodeEmbedder
from ordinis.rag.vectordb.chroma_client import ChromaClient
from ordinis.rag.vectordb.schema import CodeChunkMetadata


class CodeIndexer:
    """Index Python codebase using AST parsing."""

    def __init__(
        self,
        chroma_client: ChromaClient | None = None,
        code_embedder: CodeEmbedder | None = None,
    ):
        """Initialize code indexer.

        Args:
            chroma_client: ChromaDB client
            code_embedder: Code embedder
        """
        self.config = get_config()
        self.chroma_client = chroma_client or ChromaClient()
        self.code_embedder = code_embedder or CodeEmbedder()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info("Code indexer initialized")

    def index_directory(
        self,
        code_paths: Path | list[Path] | None = None,
        batch_size: int = 16,
    ) -> dict:
        """Index all Python files in directories.

        Args:
            code_paths: Single path or list of paths to code directories (uses config defaults if None)
            batch_size: Batch size for embedding

        Returns:
            Dictionary with indexing statistics
        """
        # Handle default paths if not provided
        if code_paths is None:
            code_paths = [
                self.config.code_base_path,  # src/
                self.config.project_root / "scripts",  # scripts/
                self.config.project_root / "docs" / "knowledge-base" / "code",  # docs code examples
            ]
        elif isinstance(code_paths, Path):
            code_paths = [code_paths]

        # Filter to paths that exist
        code_paths = [p for p in code_paths if p.exists()]

        if not code_paths:
            msg = "No valid code paths found"
            raise FileNotFoundError(msg)

        logger.info(f"Indexing code from: {[str(p) for p in code_paths]}")

        # Find all Python files from all paths
        py_files = []
        for code_path in code_paths:
            py_files.extend(code_path.rglob("*.py"))

        # Filter out test files and __pycache__
        py_files = [
            f
            for f in py_files
            if "__pycache__" not in str(f)
            and "test_" not in f.name
            and not f.name.startswith("test")
        ]

        logger.info(f"Found {len(py_files)} Python files")

        total_chunks = 0

        all_code = []
        all_metadata = []
        all_ids = []

        for py_file in py_files:
            try:
                chunks, metadata = self._process_file(
                    py_file, code_paths[0] if len(code_paths) == 1 else py_file.parent
                )
                all_code.extend(chunks)
                all_metadata.extend(metadata)

                # Generate IDs - use project root as base for relative paths
                try:
                    rel_path = py_file.relative_to(self.config.project_root)
                except ValueError:
                    rel_path = py_file
                base_id = str(rel_path).replace("\\", "/").replace("/", "_").replace(".py", "")
                chunk_ids = [f"{base_id}_chunk{i}" for i in range(len(chunks))]
                all_ids.extend(chunk_ids)

                total_chunks += len(chunks)

                logger.debug(f"Processed {py_file.name}: {len(chunks)} chunks")

            except Exception as e:
                logger.warning(f"Failed to process {py_file}: {e}")

        # Embed and store in batches
        logger.info(f"Embedding {len(all_code)} code chunks in batches of {batch_size}")

        for i in range(0, len(all_code), batch_size):
            batch_code = all_code[i : i + batch_size]
            batch_metadata = all_metadata[i : i + batch_size]
            batch_ids = all_ids[i : i + batch_size]

            # Embed batch
            embeddings = self.code_embedder.embed(batch_code)

            # Store in ChromaDB
            self.chroma_client.add_code(
                code=batch_code,
                embeddings=embeddings,
                metadata=[m.model_dump(exclude_none=True) for m in batch_metadata],
                ids=batch_ids,
            )

            logger.info(
                f"Indexed batch {i // batch_size + 1}/{(len(all_code) - 1) // batch_size + 1}"
            )

        stats = {
            "files_processed": len(py_files),
            "chunks_created": total_chunks,
        }

        logger.success(f"Code indexing complete: {stats}")
        return stats

    def _process_file(
        self,
        file_path: Path,
        code_base: Path,
    ) -> tuple[list[str], list[CodeChunkMetadata]]:
        """Process a single Python file using AST.

        Args:
            file_path: Path to Python file
            code_base: Base code directory

        Returns:
            Tuple of (code chunks, chunk metadata)
        """
        # Read file
        source_code = file_path.read_text(encoding="utf-8")

        # Parse AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return [], []

        chunks = []
        metadata = []

        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                code_chunk = ast.get_source_segment(source_code, node)
                if code_chunk and len(code_chunk.strip()) > 0:
                    # Determine class name if method
                    class_name = None
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef):
                            if node in parent.body:
                                class_name = parent.name
                                break

                    # Extract engine from path
                    engine = self._extract_engine(file_path, code_base)

                    chunks.append(code_chunk)
                    metadata.append(
                        CodeChunkMetadata(
                            file_path=str(file_path.relative_to(code_base)).replace("\\", "/"),
                            function_name=node.name,
                            class_name=class_name,
                            engine=engine,
                            line_start=node.lineno,
                            line_end=node.end_lineno,
                        )
                    )

            elif isinstance(node, ast.ClassDef):
                # Index entire class
                code_chunk = ast.get_source_segment(source_code, node)
                if code_chunk and len(code_chunk.strip()) > 0:
                    engine = self._extract_engine(file_path, code_base)

                    chunks.append(code_chunk)
                    metadata.append(
                        CodeChunkMetadata(
                            file_path=str(file_path.relative_to(code_base)).replace("\\", "/"),
                            function_name=None,
                            class_name=node.name,
                            engine=engine,
                            line_start=node.lineno,
                            line_end=node.end_lineno,
                        )
                    )

        return chunks, metadata

    def _extract_engine(self, file_path: Path, code_base: Path) -> str | None:
        """Extract engine name from file path.

        Args:
            file_path: Path to Python file
            code_base: Base code directory

        Returns:
            Engine name or None
        """
        relative_path = file_path.relative_to(code_base)
        parts = relative_path.parts

        # Check if path contains an engine directory
        engine_names = ["cortex", "signalcore", "riskguard", "proofbench", "flowroute"]
        for part in parts:
            if part in engine_names:
                return part

        return None
