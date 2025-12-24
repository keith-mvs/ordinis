"""
Tests for the Synapse retrieval CLI module.

Tests cover:
- run_retrieval async function
- main entry point
"""

from unittest.mock import patch

import pytest

from ordinis.rag.synapse.retrieve import main, run_retrieval


class TestRunRetrieval:
    """Tests for run_retrieval function."""

    @pytest.mark.asyncio
    async def test_run_retrieval_prints_query(self, capsys):
        """Test run_retrieval prints the query."""
        await run_retrieval("test query")

        captured = capsys.readouterr()
        assert "test query" in captured.out

    @pytest.mark.asyncio
    async def test_run_retrieval_shows_embedding_step(self, capsys):
        """Test run_retrieval shows embedding step."""
        await run_retrieval("test query")

        captured = capsys.readouterr()
        assert "Embedding query" in captured.out

    @pytest.mark.asyncio
    async def test_run_retrieval_shows_search_step(self, capsys):
        """Test run_retrieval shows search step."""
        await run_retrieval("test query")

        captured = capsys.readouterr()
        assert "Searching vector database" in captured.out

    @pytest.mark.asyncio
    async def test_run_retrieval_returns_results(self, capsys):
        """Test run_retrieval returns mock results."""
        await run_retrieval("How does the risk engine work?")

        captured = capsys.readouterr()
        assert "RETRIEVAL RESULTS" in captured.out
        assert "Result 1" in captured.out
        assert "RiskGuard" in captured.out
        assert "Score:" in captured.out
        assert "Source:" in captured.out


class TestMain:
    """Tests for main entry point."""

    @pytest.mark.unit
    def test_main_with_query_arg(self, capsys):
        """Test main with --query argument."""
        with patch("sys.argv", ["retrieve", "--query", "test query"]):
            main()

        captured = capsys.readouterr()
        assert "test query" in captured.out

    @pytest.mark.unit
    def test_main_missing_query_exits(self):
        """Test main exits when --query is missing."""
        with patch("sys.argv", ["retrieve"]):
            with pytest.raises(SystemExit):
                main()
