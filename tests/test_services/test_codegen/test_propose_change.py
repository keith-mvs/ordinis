"""
Tests for CodeGen propose_change module.

Tests cover:
- propose_change async function
- main entry point
"""

from unittest.mock import patch

import pytest

from ordinis.services.codegen.propose_change import main, propose_change


class TestProposeChange:
    """Tests for propose_change function."""

    @pytest.mark.asyncio
    async def test_propose_change_prints_description(self, capsys):
        """Test propose_change prints the description."""
        await propose_change("Add new strategy")

        captured = capsys.readouterr()
        assert "Add new strategy" in captured.out

    @pytest.mark.asyncio
    async def test_propose_change_shows_analyzing_step(self, capsys):
        """Test propose_change shows analyzing step."""
        await propose_change("Add new strategy")

        captured = capsys.readouterr()
        assert "Analyzing codebase" in captured.out

    @pytest.mark.asyncio
    async def test_propose_change_shows_generating_step(self, capsys):
        """Test propose_change shows generating step."""
        await propose_change("Add new strategy")

        captured = capsys.readouterr()
        assert "Generating solution" in captured.out

    @pytest.mark.asyncio
    async def test_propose_change_shows_proposal(self, capsys):
        """Test propose_change shows the proposal."""
        await propose_change("Add new strategy")

        captured = capsys.readouterr()
        assert "PROPOSED CHANGE" in captured.out
        assert "NewStrategyModel" in captured.out
        assert "File:" in captured.out

    @pytest.mark.asyncio
    async def test_propose_change_success_message(self, capsys):
        """Test propose_change shows success message."""
        await propose_change("Add new strategy")

        captured = capsys.readouterr()
        assert "Proposal generated successfully" in captured.out


class TestMain:
    """Tests for main entry point."""

    @pytest.mark.unit
    def test_main_with_desc_arg(self, capsys):
        """Test main with --desc argument."""
        with patch("sys.argv", ["propose_change", "--desc", "new feature"]):
            main()

        captured = capsys.readouterr()
        assert "new feature" in captured.out

    @pytest.mark.unit
    def test_main_missing_desc_exits(self):
        """Test main exits when --desc is missing."""
        with patch("sys.argv", ["propose_change"]):
            with pytest.raises(SystemExit):
                main()
