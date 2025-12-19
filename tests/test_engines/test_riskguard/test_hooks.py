"""
Tests for RiskGuard governance hooks.

Tests DrawdownHook and PositionLimitHook.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ordinis.engines.base.hooks import Decision, PreflightContext
from ordinis.engines.riskguard.hooks.drawdown import (
    DEFAULT_THRESHOLDS,
    DrawdownHook,
    DrawdownThreshold,
)
from ordinis.engines.riskguard.hooks.limits import PositionLimitHook


class TestDrawdownHook:
    """Tests for DrawdownHook."""

    @pytest.fixture
    def mock_kill_switch(self):
        """Create mock KillSwitch."""
        ks = MagicMock()
        ks.is_active = False
        ks.state.message = ""
        ks.get_status.return_value = {
            "active": False,
            "limits": {"max_drawdown_pct": 15.0},
        }
        ks._peak_equity = 100000.0
        ks._current_equity = 100000.0
        return ks

    @pytest.fixture
    def hook(self, mock_kill_switch):
        """Create DrawdownHook with mock."""
        return DrawdownHook(kill_switch=mock_kill_switch)

    def test_default_thresholds(self):
        """Test default thresholds are set correctly."""
        assert len(DEFAULT_THRESHOLDS) == 3
        assert DEFAULT_THRESHOLDS[0].drawdown_pct == 5.0
        assert DEFAULT_THRESHOLDS[1].drawdown_pct == 10.0
        assert DEFAULT_THRESHOLDS[2].drawdown_pct == 15.0

    def test_threshold_validation(self):
        """Test DrawdownThreshold validates inputs."""
        with pytest.raises(ValueError, match="drawdown_pct"):
            DrawdownThreshold(150.0, 0.5, "Invalid")
        
        with pytest.raises(ValueError, match="exposure_factor"):
            DrawdownThreshold(10.0, 1.5, "Invalid")

    @pytest.mark.asyncio
    async def test_allows_non_tracked_actions(self, hook):
        """Test hook allows non-tracked actions."""
        context = PreflightContext(
            engine="riskguard",
            action="get_status",
            inputs={},
        )
        
        result = await hook.preflight(context)
        assert result.decision == Decision.ALLOW

    @pytest.mark.asyncio
    async def test_denies_when_kill_switch_active(self, hook, mock_kill_switch):
        """Test hook denies when kill switch is active."""
        mock_kill_switch.is_active = True
        mock_kill_switch.state.message = "Daily loss limit"
        
        context = PreflightContext(
            engine="riskguard",
            action="calculate_position_size",
            inputs={},
        )
        
        result = await hook.preflight(context)
        assert result.decision == Decision.DENY
        assert "Kill switch active" in result.reason

    @pytest.mark.asyncio
    async def test_allows_low_drawdown(self, hook, mock_kill_switch):
        """Test hook allows operations with low drawdown."""
        mock_kill_switch._current_equity = 98000.0  # 2% drawdown
        
        context = PreflightContext(
            engine="riskguard",
            action="calculate_position_size",
            inputs={},
        )
        
        result = await hook.preflight(context)
        assert result.decision == Decision.ALLOW
        assert "within acceptable limits" in result.reason

    @pytest.mark.asyncio
    async def test_warns_on_moderate_drawdown(self, hook, mock_kill_switch):
        """Test hook warns and reduces exposure on moderate drawdown."""
        mock_kill_switch._current_equity = 93000.0  # 7% drawdown
        
        context = PreflightContext(
            engine="riskguard",
            action="calculate_position_size",
            inputs={},
        )
        
        result = await hook.preflight(context)
        assert result.decision == Decision.WARN
        assert "exposure_factor" in result.adjustments
        assert result.adjustments["exposure_factor"] == 0.75

    @pytest.mark.asyncio
    async def test_denies_on_severe_drawdown(self, hook, mock_kill_switch):
        """Test hook denies on severe drawdown."""
        mock_kill_switch._current_equity = 84000.0  # 16% drawdown
        
        context = PreflightContext(
            engine="riskguard",
            action="calculate_position_size",
            inputs={},
        )
        
        result = await hook.preflight(context)
        assert result.decision == Decision.DENY
        assert "halt threshold" in result.reason

    def test_exposure_factor_property(self, hook):
        """Test exposure_factor property."""
        assert hook.exposure_factor == 1.0


class TestPositionLimitHook:
    """Tests for PositionLimitHook."""

    @pytest.fixture
    def hook(self):
        """Create PositionLimitHook."""
        return PositionLimitHook(
            max_position_pct=0.10,
            max_sector_pct=0.30,
            max_concurrent_positions=5,
            portfolio_value=100000.0,
        )

    @pytest.mark.asyncio
    async def test_allows_valid_position(self, hook):
        """Test hook allows positions within limits."""
        context = PreflightContext(
            engine="riskguard",
            action="calculate_position_size",
            inputs={
                "symbol": "AAPL",
                "position_value": 5000.0,  # 5%, under 10% limit
                "sector": "technology",
            },
        )
        
        result = await hook.preflight(context)
        assert result.decision == Decision.ALLOW

    @pytest.mark.asyncio
    async def test_denies_oversized_position(self, hook):
        """Test hook denies positions exceeding max size."""
        context = PreflightContext(
            engine="riskguard",
            action="calculate_position_size",
            inputs={
                "symbol": "AAPL",
                "position_value": 15000.0,  # 15%, over 10% limit
                "sector": "technology",
            },
        )
        
        result = await hook.preflight(context)
        assert result.decision == Decision.DENY
        assert "exceeds" in result.reason

    @pytest.mark.asyncio
    async def test_denies_too_many_positions(self, hook):
        """Test hook denies when at max concurrent positions."""
        # Set 5 existing positions
        hook._current_positions = {
            f"SYM{i}": {"market_value": 5000.0} for i in range(5)
        }
        
        context = PreflightContext(
            engine="riskguard",
            action="calculate_position_size",
            inputs={
                "symbol": "NEW",  # New position
                "position_value": 5000.0,
                "sector": "technology",
            },
        )
        
        result = await hook.preflight(context)
        assert result.decision == Decision.DENY
        assert "max 5 positions" in result.reason

    @pytest.mark.asyncio
    async def test_warns_on_sector_concentration(self, hook):
        """Test hook warns on high sector concentration."""
        hook._current_positions = {
            "AAPL": {"market_value": 10000.0, "sector": "technology"},
            "MSFT": {"market_value": 15000.0, "sector": "technology"},
        }
        
        context = PreflightContext(
            engine="riskguard",
            action="calculate_position_size",
            inputs={
                "symbol": "NVDA",
                "position_value": 10000.0,  # Would make tech 35%
                "sector": "technology",
            },
        )
        
        result = await hook.preflight(context)
        assert result.decision == Decision.WARN
        assert "concentration" in result.reason

    def test_update_portfolio_value(self, hook):
        """Test updating portfolio value."""
        hook.update_portfolio_value(150000.0)
        assert hook._portfolio_value == 150000.0

    def test_update_positions(self, hook):
        """Test updating current positions."""
        positions = {"AAPL": {"market_value": 5000.0}}
        hook.update_positions(positions)
        assert hook._current_positions == positions
