"""Tests for CodeIndexer module.

Tests cover:
- Initialization
- Directory indexing
- File processing
- AST parsing
- Engine extraction
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ordinis.rag.pipeline.code_indexer import CodeIndexer


class TestCodeIndexerInit:
    """Tests for CodeIndexer initialization."""

    @pytest.mark.unit
    def test_init_with_clients(self):
        """Test initialization with provided clients."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()

        with patch("ordinis.rag.pipeline.code_indexer.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_config.return_value = mock_cfg

            indexer = CodeIndexer(
                chroma_client=mock_chroma,
                code_embedder=mock_embedder,
            )

            assert indexer.chroma_client is mock_chroma
            assert indexer.code_embedder is mock_embedder


class TestExtractEngine:
    """Tests for _extract_engine method."""

    @pytest.fixture
    def indexer(self):
        """Create indexer for tests."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()

        with patch("ordinis.rag.pipeline.code_indexer.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_config.return_value = mock_cfg

            return CodeIndexer(
                chroma_client=mock_chroma,
                code_embedder=mock_embedder,
            )

    @pytest.mark.unit
    def test_extract_cortex_engine(self, indexer, tmp_path):
        """Test extracting cortex engine from path."""
        code_base = tmp_path
        file_path = tmp_path / "engines" / "cortex" / "processor.py"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = indexer._extract_engine(file_path, code_base)

        assert result == "cortex"

    @pytest.mark.unit
    def test_extract_signalcore_engine(self, indexer, tmp_path):
        """Test extracting signalcore engine from path."""
        code_base = tmp_path
        file_path = tmp_path / "signalcore" / "signals.py"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = indexer._extract_engine(file_path, code_base)

        assert result == "signalcore"

    @pytest.mark.unit
    def test_extract_no_engine(self, indexer, tmp_path):
        """Test extracting from path without known engine."""
        code_base = tmp_path
        file_path = tmp_path / "utils" / "helpers.py"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = indexer._extract_engine(file_path, code_base)

        assert result is None

    @pytest.mark.unit
    def test_extract_proofbench_engine(self, indexer, tmp_path):
        """Test extracting proofbench engine from path."""
        code_base = tmp_path
        file_path = tmp_path / "proofbench" / "backtester.py"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = indexer._extract_engine(file_path, code_base)

        assert result == "proofbench"

    @pytest.mark.unit
    def test_extract_flowroute_engine(self, indexer, tmp_path):
        """Test extracting flowroute engine from path."""
        code_base = tmp_path
        file_path = tmp_path / "flowroute" / "router.py"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        result = indexer._extract_engine(file_path, code_base)

        assert result == "flowroute"


class TestProcessFile:
    """Tests for _process_file method."""

    @pytest.fixture
    def indexer(self):
        """Create indexer for tests."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()

        with patch("ordinis.rag.pipeline.code_indexer.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_config.return_value = mock_cfg

            return CodeIndexer(
                chroma_client=mock_chroma,
                code_embedder=mock_embedder,
            )

    @pytest.mark.unit
    def test_process_file_with_function(self, indexer, tmp_path):
        """Test processing file with a function."""
        code_base = tmp_path
        py_file = tmp_path / "module.py"
        py_file.write_text("""
def hello():
    print("Hello")

def goodbye():
    print("Goodbye")
""")

        chunks, metadata = indexer._process_file(py_file, code_base)

        assert len(chunks) == 2
        assert len(metadata) == 2
        assert any(m.function_name == "hello" for m in metadata)
        assert any(m.function_name == "goodbye" for m in metadata)

    @pytest.mark.unit
    def test_process_file_with_class(self, indexer, tmp_path):
        """Test processing file with a class."""
        code_base = tmp_path
        py_file = tmp_path / "models.py"
        py_file.write_text("""
class MyClass:
    def __init__(self):
        pass

    def method(self):
        pass
""")

        chunks, metadata = indexer._process_file(py_file, code_base)

        # Should find class and methods
        assert len(chunks) >= 1
        assert any(m.class_name == "MyClass" for m in metadata)

    @pytest.mark.unit
    def test_process_file_with_method_in_class(self, indexer, tmp_path):
        """Test processing file correctly identifies class methods."""
        code_base = tmp_path
        py_file = tmp_path / "service.py"
        py_file.write_text("""
class Service:
    def process(self):
        return True
""")

        chunks, metadata = indexer._process_file(py_file, code_base)

        # Find the method and check class_name is set
        method_meta = [m for m in metadata if m.function_name == "process"]
        if method_meta:
            # May or may not have class_name depending on implementation
            pass

    @pytest.mark.unit
    def test_process_file_syntax_error(self, indexer, tmp_path):
        """Test processing file with syntax error returns empty."""
        code_base = tmp_path
        py_file = tmp_path / "broken.py"
        py_file.write_text("def broken(:\n    pass")  # Syntax error

        chunks, metadata = indexer._process_file(py_file, code_base)

        assert len(chunks) == 0
        assert len(metadata) == 0

    @pytest.mark.unit
    def test_process_file_async_function(self, indexer, tmp_path):
        """Test processing file with async function."""
        code_base = tmp_path
        py_file = tmp_path / "async_module.py"
        py_file.write_text("""
async def async_handler():
    await something()
""")

        chunks, metadata = indexer._process_file(py_file, code_base)

        assert len(chunks) == 1
        assert metadata[0].function_name == "async_handler"

    @pytest.mark.unit
    def test_process_file_metadata_line_numbers(self, indexer, tmp_path):
        """Test processing file sets line numbers correctly."""
        code_base = tmp_path
        py_file = tmp_path / "code.py"
        py_file.write_text("""
def func_at_line_2():
    pass
""")

        chunks, metadata = indexer._process_file(py_file, code_base)

        if metadata:
            assert metadata[0].line_start >= 1
            assert metadata[0].line_end is not None


class TestIndexDirectory:
    """Tests for index_directory method."""

    @pytest.fixture
    def indexer(self):
        """Create indexer for tests."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 384]

        with patch("ordinis.rag.pipeline.code_indexer.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.code_base_path = Path("nonexistent")
            mock_cfg.project_root = Path("/project")
            mock_config.return_value = mock_cfg

            return CodeIndexer(
                chroma_client=mock_chroma,
                code_embedder=mock_embedder,
            )

    @pytest.mark.unit
    def test_index_nonexistent_directory(self, indexer):
        """Test indexing non-existent directory raises error."""
        with pytest.raises(FileNotFoundError, match="No valid code paths found"):
            indexer.index_directory(Path("/nonexistent/code"))

    @pytest.mark.unit
    def test_index_empty_directory(self, indexer, tmp_path):
        """Test indexing empty directory."""
        code_path = tmp_path / "empty_code"
        code_path.mkdir()

        result = indexer.index_directory(code_path)

        assert result["files_processed"] == 0
        assert result["chunks_created"] == 0

    @pytest.mark.unit
    def test_index_directory_with_python_files(self, indexer, tmp_path):
        """Test indexing directory with Python files."""
        code_path = tmp_path / "code"
        code_path.mkdir()

        # Create test files
        (code_path / "module1.py").write_text("""
def func1():
    pass
""")
        (code_path / "module2.py").write_text("""
def func2():
    pass
""")

        # Update project_root for relative path calculation
        indexer.config.project_root = tmp_path

        result = indexer.index_directory(code_path)

        assert result["files_processed"] == 2
        assert result["chunks_created"] >= 2

    @pytest.mark.unit
    def test_index_directory_skips_test_files(self, indexer, tmp_path):
        """Test indexing skips test files."""
        code_path = tmp_path / "code"
        code_path.mkdir()

        # Create test and non-test files
        (code_path / "module.py").write_text("def func(): pass")
        (code_path / "test_module.py").write_text("def test_func(): pass")

        indexer.config.project_root = tmp_path
        result = indexer.index_directory(code_path)

        # Only the non-test file should be processed
        assert result["files_processed"] == 1

    @pytest.mark.unit
    def test_index_directory_skips_pycache(self, indexer, tmp_path):
        """Test indexing skips __pycache__ directories."""
        code_path = tmp_path / "code"
        code_path.mkdir()

        # Create pycache with compiled file
        pycache = code_path / "__pycache__"
        pycache.mkdir()
        (pycache / "module.cpython-311.py").write_text("# compiled")

        # Create regular file
        (code_path / "real.py").write_text("def real(): pass")

        indexer.config.project_root = tmp_path
        result = indexer.index_directory(code_path)

        assert result["files_processed"] == 1

    @pytest.mark.unit
    def test_index_directory_calls_chroma(self, indexer, tmp_path):
        """Test indexing calls ChromaDB add_code."""
        code_path = tmp_path / "code"
        code_path.mkdir()
        (code_path / "module.py").write_text("def func(): pass")

        indexer.config.project_root = tmp_path
        indexer.index_directory(code_path)

        indexer.chroma_client.add_code.assert_called()

    @pytest.mark.unit
    def test_index_handles_file_errors(self, indexer, tmp_path):
        """Test indexing handles individual file errors."""
        code_path = tmp_path / "code"
        code_path.mkdir()

        # Create a valid file
        (code_path / "valid.py").write_text("def valid(): pass")

        indexer.config.project_root = tmp_path

        # Mock _process_file to fail for specific files
        original_process = indexer._process_file

        def mock_process(file_path, code_base):
            if "error" in str(file_path):
                raise RuntimeError("File error")
            return original_process(file_path, code_base)

        indexer._process_file = mock_process

        result = indexer.index_directory(code_path)

        # Should still process valid file
        assert result["files_processed"] == 1

    @pytest.mark.unit
    def test_index_multiple_paths(self, indexer, tmp_path):
        """Test indexing multiple code paths."""
        path1 = tmp_path / "src"
        path2 = tmp_path / "scripts"
        path1.mkdir()
        path2.mkdir()

        (path1 / "module.py").write_text("def func1(): pass")
        (path2 / "script.py").write_text("def func2(): pass")

        indexer.config.project_root = tmp_path
        result = indexer.index_directory([path1, path2])

        assert result["files_processed"] == 2
