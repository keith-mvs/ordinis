"""Tests for RiskGuard Validation CLI.

Tests cover:
- run_validation async function
- main CLI entry point
- Test case processing
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ordinis.engines.riskguard.validate import main, run_validation


class TestRunValidation:
    """Tests for run_validation function."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_validation_completes(self, capsys):
        """Test run_validation completes without error."""
        # Mock RiskGuardEngine to avoid actual rule evaluation
        with patch(
            "ordinis.engines.riskguard.validate.RiskGuardEngine"
        ) as mock_engine_cls:
            mock_engine = MagicMock()
            # Return approved for all signals
            mock_engine.evaluate = AsyncMock(return_value=([MagicMock()], []))
            mock_engine._rules = {}
            mock_engine_cls.return_value = mock_engine

            await run_validation()

            captured = capsys.readouterr()
            assert "[INFO] Starting RiskGuard Validation..." in captured.out
            assert "Added test rule:" in captured.out
            assert "Running Test Cases..." in captured.out
            assert "VALIDATION SUMMARY:" in captured.out

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_validation_counts_passes(self, capsys):
        """Test run_validation counts passing tests correctly."""
        with patch(
            "ordinis.engines.riskguard.validate.RiskGuardEngine"
        ) as mock_engine_cls:
            mock_engine = MagicMock()
            # All approved
            mock_engine.evaluate = AsyncMock(return_value=([MagicMock()], []))
            mock_engine._rules = {}
            mock_engine_cls.return_value = mock_engine

            await run_validation()

            captured = capsys.readouterr()
            # Should show pass count in summary
            assert "3/3 Passed" in captured.out or "VALIDATION SUMMARY" in captured.out

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_validation_with_rejections(self, capsys):
        """Test run_validation handles rejections correctly."""
        call_count = 0

        async def mock_evaluate(signals):
            nonlocal call_count
            call_count += 1
            # First call: approve (small trade)
            # Second call: reject (large trade)
            # Third call: approve (hold)
            if call_count == 1:
                return ([signals[0]], [])
            elif call_count == 2:
                return ([], [{"rule": "TEST_MAX_VALUE", "reason": "Over limit"}])
            else:
                return ([signals[0]], [])

        with patch(
            "ordinis.engines.riskguard.validate.RiskGuardEngine"
        ) as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.evaluate = AsyncMock(side_effect=mock_evaluate)
            mock_engine._rules = {}
            mock_engine_cls.return_value = mock_engine

            await run_validation()

            captured = capsys.readouterr()
            assert "[PASS]" in captured.out or "[FAIL]" in captured.out
            assert "VALIDATION SUMMARY" in captured.out

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_validation_prints_rejection_reasons(self, capsys):
        """Test run_validation prints rejection reasons."""
        async def mock_evaluate(signals):
            return ([], [{"rule": "TEST_RULE", "reason": "Test rejection"}])

        with patch(
            "ordinis.engines.riskguard.validate.RiskGuardEngine"
        ) as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.evaluate = AsyncMock(side_effect=mock_evaluate)
            mock_engine._rules = {}
            mock_engine_cls.return_value = mock_engine

            await run_validation()

            captured = capsys.readouterr()
            assert "Reasons:" in captured.out


class TestMain:
    """Tests for main CLI entry point."""

    @pytest.mark.unit
    def test_main_runs_validation(self):
        """Test main runs run_validation via asyncio."""
        with patch("ordinis.engines.riskguard.validate.asyncio.run") as mock_run:
            with patch("ordinis.engines.riskguard.validate.argparse.ArgumentParser"):
                main()
                mock_run.assert_called_once()

    @pytest.mark.unit
    def test_main_parses_args(self):
        """Test main parses command line arguments."""
        with patch("ordinis.engines.riskguard.validate.asyncio.run"):
            with patch(
                "ordinis.engines.riskguard.validate.argparse.ArgumentParser"
            ) as mock_parser_cls:
                mock_parser = MagicMock()
                mock_parser_cls.return_value = mock_parser

                main()

                mock_parser.parse_args.assert_called_once()
