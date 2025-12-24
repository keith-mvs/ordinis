"""Tests for Cortex CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ordinis.engines.cortex.cli import analyze_single_file, create_cortex_engine, main


class TestCortexCLI:
    """Test suite for Cortex CLI."""

    @pytest.mark.asyncio
    async def test_analyze_single_file_success(self, tmp_path: Path) -> None:
        """Test analyzing a single file successfully."""
        # Create test file
        test_file = tmp_path / "test_code.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        # Mock engine
        mock_engine = MagicMock()
        mock_output = MagicMock()
        mock_output.output_type.value = "code_analysis"
        mock_output.content = {"quality": "good", "issues": []}
        mock_output.confidence = 0.9
        mock_output.reasoning = "Code looks clean"
        mock_output.model_used = "test-model"
        mock_output.prompt_tokens = 100
        mock_output.completion_tokens = 50
        mock_engine.analyze_code = AsyncMock(return_value=mock_output)

        result = await analyze_single_file(
            engine=mock_engine,
            file_path=test_file,
            analysis_type="review",
        )

        assert result["status"] == "success"
        assert result["file"] == str(test_file)
        assert result["confidence"] == 0.9
        assert result["tokens"]["prompt"] == 100

    @pytest.mark.asyncio
    async def test_analyze_single_file_not_found(self, tmp_path: Path) -> None:
        """Test analyzing a non-existent file."""
        mock_engine = MagicMock()
        nonexistent = tmp_path / "does_not_exist.py"

        result = await analyze_single_file(
            engine=mock_engine,
            file_path=nonexistent,
            analysis_type="review",
        )

        assert result["status"] == "failed"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_analyze_single_file_engine_error(self, tmp_path: Path) -> None:
        """Test handling engine errors gracefully."""
        test_file = tmp_path / "test_code.py"
        test_file.write_text("def hello(): pass\n")

        mock_engine = MagicMock()
        mock_engine.analyze_code = AsyncMock(side_effect=RuntimeError("LLM error"))

        result = await analyze_single_file(
            engine=mock_engine,
            file_path=test_file,
            analysis_type="review",
        )

        assert result["status"] == "failed"
        assert "LLM error" in result["error"]

    def test_cli_dry_run(self, capsys, tmp_path: Path) -> None:
        """Test dry run mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")

        with patch("sys.argv", ["cortex-cli", "--file", str(test_file), "--dry-run"]):
            main()

        captured = capsys.readouterr()
        assert "Dry run" in captured.out

    def test_cli_no_args_error(self, capsys) -> None:
        """Test CLI with no arguments shows error."""
        with patch("sys.argv", ["cortex-cli"]):
            with pytest.raises(SystemExit):
                main()

    @pytest.mark.asyncio
    async def test_create_cortex_engine_function_exists(self) -> None:
        """Test that engine creation function exists and is callable."""
        # The function imports dependencies inside, so we just verify it exists
        # Full integration tests would require actual Helix/NVIDIA setup
        assert callable(create_cortex_engine)
        # Verify it's an async function
        import inspect
        assert inspect.iscoroutinefunction(create_cortex_engine)


class TestCLIOutputFormat:
    """Test CLI output formatting."""

    @pytest.mark.asyncio
    async def test_json_output_structure(self, tmp_path: Path) -> None:
        """Test that output JSON has expected structure."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass\n")

        mock_engine = MagicMock()
        mock_output = MagicMock()
        mock_output.output_type.value = "code_analysis"
        mock_output.content = {"issues": [{"severity": "low", "message": "test"}]}
        mock_output.confidence = 0.85
        mock_output.reasoning = "Test reasoning"
        mock_output.model_used = "test-model"
        mock_output.prompt_tokens = 50
        mock_output.completion_tokens = 25
        mock_engine.analyze_code = AsyncMock(return_value=mock_output)

        result = await analyze_single_file(
            engine=mock_engine,
            file_path=test_file,
            analysis_type="security",
        )

        # Verify structure
        assert "file" in result
        assert "status" in result
        assert "content" in result
        assert "confidence" in result
        assert "tokens" in result
        assert "prompt" in result["tokens"]
        assert "completion" in result["tokens"]

        # Verify serializable
        json_str = json.dumps(result)
        assert json_str  # No serialization errors
