"""
Tests for MCP server Phase 1 tools.

Tests strategy configuration tools, market context injection,
performance metrics, and capital preservation controls.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


class TestStrategyConfigTools:
    """Tests for strategy configuration tools."""

    @pytest.fixture
    def mock_state(self):
        """Create mock OrdinisState."""
        from ordinis.mcp.server import OrdinisState

        state = OrdinisState()
        state.strategies_config = {
            "atr_optimized_rsi": {
                "strategy": {"name": "ATR-Optimized RSI", "version": "1.0"},
                "global_params": {
                    "rsi_period": 14,
                    "default_rsi_oversold": 30,
                },
                "risk_management": {
                    "max_position_size_pct": 3.0,
                },
                "symbols": {
                    "AAPL": {"rsi_oversold": 28, "atr_stop_mult": 1.5},
                    "AMD": {"rsi_oversold": 30, "atr_stop_mult": 2.0},
                },
            },
        }
        return state

    @pytest.mark.asyncio
    async def test_get_strategy_config_returns_strategy(self, mock_state):
        """Test get_strategy_config returns correct data."""
        from ordinis.mcp.server import _state, get_strategy_config

        # Patch global state
        original_config = _state.strategies_config
        _state.strategies_config = mock_state.strategies_config

        try:
            result = await get_strategy_config("atr_optimized_rsi")
            data = json.loads(result)

            assert data["strategy"] == "atr_optimized_rsi"
            assert data["name"] == "ATR-Optimized RSI"
            assert "global_params" in data
            assert data["symbol_count"] == 2
        finally:
            _state.strategies_config = original_config

    @pytest.mark.asyncio
    async def test_get_strategy_config_symbol_specific(self, mock_state):
        """Test get_strategy_config with symbol parameter."""
        from ordinis.mcp.server import _state, get_strategy_config

        original_config = _state.strategies_config
        _state.strategies_config = mock_state.strategies_config

        try:
            result = await get_strategy_config("atr_optimized_rsi", symbol="AAPL")
            data = json.loads(result)

            assert data["symbol"] == "AAPL"
            assert data["symbol_config"]["rsi_oversold"] == 28
        finally:
            _state.strategies_config = original_config

    @pytest.mark.asyncio
    async def test_get_strategy_config_not_found(self, mock_state):
        """Test get_strategy_config with unknown strategy."""
        from ordinis.mcp.server import _state, get_strategy_config

        original_config = _state.strategies_config
        _state.strategies_config = mock_state.strategies_config

        try:
            result = await get_strategy_config("unknown_strategy")
            data = json.loads(result)

            assert "error" in data
            assert "available" in data
        finally:
            _state.strategies_config = original_config

    @pytest.mark.asyncio
    async def test_update_strategy_config_validates_bounds(self, mock_state):
        """Test update_strategy_config rejects out-of-bounds values."""
        from ordinis.mcp.server import _state, update_strategy_config

        original_config = _state.strategies_config
        _state.strategies_config = mock_state.strategies_config

        try:
            # RSI oversold must be 10-40
            result = await update_strategy_config("atr_optimized_rsi", "rsi_oversold", 50)
            data = json.loads(result)

            assert "error" in data
            assert data["rejected"] is True
            assert "bounds" in data
        finally:
            _state.strategies_config = original_config

    @pytest.mark.asyncio
    async def test_update_strategy_config_success(self, mock_state):
        """Test update_strategy_config applies valid changes."""
        from ordinis.mcp.server import _state, update_strategy_config

        original_config = _state.strategies_config
        _state.strategies_config = mock_state.strategies_config

        try:
            with patch.object(_state, "_save_strategy", return_value=True):
                result = await update_strategy_config(
                    "atr_optimized_rsi", "rsi_oversold", 25, symbol="AAPL"
                )
                data = json.loads(result)

                assert data["success"] is True
                assert data["old_value"] == 28
                assert data["new_value"] == 25
        finally:
            _state.strategies_config = original_config


class TestMarketContextTools:
    """Tests for market context injection."""

    @pytest.mark.asyncio
    async def test_inject_market_context(self):
        """Test inject_market_context stores context."""
        from ordinis.mcp.server import _state, inject_market_context

        result = await inject_market_context("Fed announced rate hold, risk-on sentiment")
        data = json.loads(result)

        assert data["success"] is True
        assert "Fed announced" in data["context"]
        assert _state._market_context == "Fed announced rate hold, risk-on sentiment"
        assert _state._market_context_timestamp is not None

    @pytest.mark.asyncio
    async def test_get_market_context(self):
        """Test get_market_context returns current context."""
        from ordinis.mcp.server import get_market_context, inject_market_context

        await inject_market_context("Test context")
        result = await get_market_context()
        data = json.loads(result)

        assert data["context"] == "Test context"
        assert data["timestamp"] is not None
        assert data["age_minutes"] is not None


class TestCapitalPreservationTools:
    """Tests for capital preservation controls."""

    @pytest.mark.asyncio
    async def test_get_drawdown_status_simulated(self):
        """Test get_drawdown_status without live KillSwitch."""
        from ordinis.mcp.server import _state, get_drawdown_status

        # Ensure no live kill switch
        _state._kill_switch = None

        result = await get_drawdown_status()
        data = json.loads(result)

        assert "drawdown_pct" in data
        assert "daily_pnl" in data
        assert "consecutive_losses" in data
        assert "can_trade" in data
        assert data["mode"] == "simulated"

    @pytest.mark.asyncio
    async def test_reduce_exposure_validates_factor(self):
        """Test reduce_exposure rejects invalid factors."""
        from ordinis.mcp.server import reduce_exposure

        result = await reduce_exposure(1.5)  # > 1.0 is invalid
        data = json.loads(result)

        assert "error" in data
        assert data["rejected"] is True

    @pytest.mark.asyncio
    async def test_reduce_exposure_applies_factor(self):
        """Test reduce_exposure applies to all strategies."""
        from ordinis.mcp.server import _state, reduce_exposure

        # Set up test strategy
        _state.strategies_config = {
            "test_strategy": {
                "risk_management": {"max_position_size_pct": 3.0},
            }
        }

        result = await reduce_exposure(0.75)
        data = json.loads(result)

        assert data["success"] is True
        assert data["exposure_factor"] == 0.75
        assert "test_strategy" in data["strategies_affected"]

    @pytest.mark.asyncio
    async def test_analyze_open_positions(self):
        """Test analyze_open_positions returns position analysis."""
        from ordinis.mcp.server import analyze_open_positions

        result = await analyze_open_positions()
        data = json.loads(result)

        assert "position_count" in data
        assert "total_unrealized_pnl" in data
        assert "positions" in data
        assert "summary" in data
        assert len(data["positions"]) > 0

        # Check position has required fields
        position = data["positions"][0]
        assert "symbol" in position
        assert "unrealized_pnl" in position
        assert "recommendation" in position


class TestPerformanceMetricsResources:
    """Tests for performance metrics resources."""

    @pytest.mark.asyncio
    async def test_pnl_metrics_resource(self):
        """Test ordinis://metrics/pnl resource."""
        from ordinis.mcp.server import get_pnl_metrics

        result = await get_pnl_metrics()
        data = json.loads(result)

        assert "daily" in data
        assert "performance" in data
        assert "risk_adjusted" in data
        assert "drawdown" in data
        assert "sharpe_estimate" in data["risk_adjusted"]

    @pytest.mark.asyncio
    async def test_signal_metrics_resource(self):
        """Test ordinis://metrics/signals resource."""
        from ordinis.mcp.server import get_signal_metrics

        result = await get_signal_metrics()
        data = json.loads(result)

        assert "strategies" in data
        assert "total_signals" in data
        assert len(data["strategies"]) > 0

        strategy = data["strategies"][0]
        assert "strategy" in strategy
        assert "hit_rate_pct" in strategy

    @pytest.mark.asyncio
    async def test_position_metrics_resource(self):
        """Test ordinis://metrics/positions resource."""
        from ordinis.mcp.server import get_position_metrics

        result = await get_position_metrics()
        data = json.loads(result)

        assert "position_count" in data
        assert "total_market_value" in data
        assert "sector_allocation" in data
        assert "positions" in data
