"""
Integration tests for Sprint 1 & 2 implementations.

Tests the integration between:
- MCP Server and PortfolioEngine
- Agent Runner and MCP Tools
- MassivePlugin and NewsContextHook
- RegimeHook and SignalCore
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import pytest

# =============================================================================
# Test: MassivePlugin Integration
# =============================================================================


class TestMassivePlugin:
    """Tests for the Massive news plugin."""

    @pytest.fixture
    def plugin(self):
        """Create a configured MassivePlugin."""
        from ordinis.plugins.massive import MassivePlugin, MassivePluginConfig

        config = MassivePluginConfig(name="massive", use_mock=True)
        return MassivePlugin(config)

    @pytest.mark.asyncio
    async def test_plugin_initialization(self, plugin):
        """Test plugin initializes successfully."""
        result = await plugin.initialize()
        assert result is True
        assert plugin.status.value == "ready"
        await plugin.shutdown()

    @pytest.mark.asyncio
    async def test_get_news_for_symbol(self, plugin):
        """Test fetching news for a specific symbol."""
        await plugin.initialize()

        news = await plugin.get_news(symbols=["AAPL"], limit=5)

        assert isinstance(news, list)
        assert len(news) <= 5
        for article in news:
            assert "headline" in article
            assert "sentiment" in article
            assert "timestamp" in article

        await plugin.shutdown()

    @pytest.mark.asyncio
    async def test_get_market_news(self, plugin):
        """Test fetching broad market news."""
        await plugin.initialize()

        news = await plugin.get_market_news(limit=10)

        assert isinstance(news, list)
        assert len(news) <= 10

        await plugin.shutdown()

    @pytest.mark.asyncio
    async def test_get_sentiment(self, plugin):
        """Test sentiment analysis for a symbol."""
        await plugin.initialize()

        sentiment = await plugin.get_sentiment("AAPL")

        assert "symbol" in sentiment
        assert "sentiment" in sentiment
        assert "score" in sentiment
        assert -1.0 <= sentiment["score"] <= 1.0

        await plugin.shutdown()

    @pytest.mark.asyncio
    async def test_get_earnings_calendar(self, plugin):
        """Test earnings calendar retrieval."""
        await plugin.initialize()

        earnings = await plugin.get_earnings_calendar(days_ahead=7)

        assert isinstance(earnings, list)
        for entry in earnings:
            assert "symbol" in entry
            assert "earnings_date" in entry

        await plugin.shutdown()

    @pytest.mark.asyncio
    async def test_news_caching(self, plugin):
        """Test that news is cached properly."""
        await plugin.initialize()

        # First call
        news1 = await plugin.get_news(symbols=["AAPL"], limit=5)
        # Second call should hit cache
        news2 = await plugin.get_news(symbols=["AAPL"], limit=5)

        # Same number of articles (from cache)
        assert len(news1) == len(news2)

        await plugin.shutdown()


# =============================================================================
# Test: RegimeHook Integration
# =============================================================================


class TestRegimeHook:
    """Tests for the market regime hook."""

    @pytest.fixture
    def hook(self):
        """Create a configured RegimeHook."""
        from ordinis.engines.signalcore.hooks.regime import MarketRegime, RegimeHook

        return RegimeHook(initial_regime=MarketRegime.UNKNOWN)

    @pytest.mark.asyncio
    async def test_regime_detection_trending_up(self, hook):
        """Test detection of trending up regime."""
        from ordinis.engines.signalcore.hooks.regime import MarketRegime

        regime = hook.detect_from_indicators(vix=15.0, spy_return_5d=3.5)

        assert regime == MarketRegime.TRENDING_UP

    @pytest.mark.asyncio
    async def test_regime_detection_high_volatility(self, hook):
        """Test detection of high volatility regime."""
        from ordinis.engines.signalcore.hooks.regime import MarketRegime

        regime = hook.detect_from_indicators(vix=35.0)

        assert regime == MarketRegime.HIGH_VOLATILITY

    @pytest.mark.asyncio
    async def test_regime_detection_ranging(self, hook):
        """Test detection of ranging regime."""
        from ordinis.engines.signalcore.hooks.regime import MarketRegime

        regime = hook.detect_from_indicators(spy_return_5d=0.5)

        assert regime == MarketRegime.RANGING

    @pytest.mark.asyncio
    async def test_preflight_allows_normal_operations(self, hook):
        """Test preflight allows operations in normal regime."""
        from ordinis.engines.base.hooks import Decision, PreflightContext
        from ordinis.engines.signalcore.hooks.regime import MarketRegime

        hook.set_regime(MarketRegime.TRENDING_UP, confidence=0.8)

        context = PreflightContext(
            operation="generate_signal",
            parameters={"symbol": "AAPL"},
            timestamp=datetime.now(UTC),
            trace_id="test-1",
        )
        context.action = "generate_signal"
        context.inputs = {"symbol": "AAPL", "direction": "long"}

        result = await hook.preflight(context)

        assert result.decision in (Decision.ALLOW, Decision.WARN)
        assert "regime" in result.adjustments

    @pytest.mark.asyncio
    async def test_preflight_blocks_in_risk_off(self, hook):
        """Test preflight blocks new entries in risk-off regime."""
        from ordinis.engines.base.hooks import Decision, PreflightContext
        from ordinis.engines.signalcore.hooks.regime import MarketRegime

        hook.set_regime(MarketRegime.RISK_OFF, confidence=0.9)

        context = PreflightContext(
            operation="generate_signal",
            parameters={"symbol": "AAPL"},
            timestamp=datetime.now(UTC),
            trace_id="test-2",
        )
        context.action = "generate_signal"
        context.inputs = {"symbol": "AAPL"}

        result = await hook.preflight(context)

        assert result.decision == Decision.DENY

    @pytest.mark.asyncio
    async def test_position_size_modifier(self, hook):
        """Test position size modifier is applied correctly."""
        from ordinis.engines.signalcore.hooks.regime import MarketRegime

        hook.set_regime(MarketRegime.HIGH_VOLATILITY, confidence=0.8)

        assert hook.regime_state.position_size_modifier == 0.5

    def test_strategy_adjustments(self, hook):
        """Test strategy-specific adjustments are returned."""
        from ordinis.engines.signalcore.hooks.regime import MarketRegime

        hook.set_regime(MarketRegime.TRENDING_UP, confidence=0.8)

        adjustments = hook.get_strategy_adjustments("atr_optimized_rsi")

        assert "position_size_modifier" in adjustments
        assert "rsi_oversold" in adjustments or "atr_stop_mult" in adjustments


# =============================================================================
# Test: Agent Runner Integration
# =============================================================================


class TestAgentRunner:
    """Tests for the agent runner."""

    @pytest.fixture
    def runner(self, tmp_path):
        """Create an AgentRunner with test config."""
        from ordinis.agents.runner import AgentRunner

        # Create test agent config
        agent_config = {
            "name": "test_agent",
            "description": "Test agent for integration tests",
            "schedule": "*/5 * * * *",
            "model": "test-model",
            "timeout_seconds": 60,
            "system_prompt": "You are a test agent.",
            "tools": ["test_tool"],
            "resources": ["test://resource"],
            "workflow": [
                {"step": "read_data", "action": "read_resource", "resource": "test://resource"},
                {"step": "log_result", "action": "log", "level": "info", "message": "Test passed"},
            ],
        }

        config_path = tmp_path / "test_agent.yaml"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(agent_config, f)

        return AgentRunner(agents_dir=tmp_path)

    def test_list_agents(self, runner):
        """Test listing available agents."""
        agents = runner.list_agents()

        assert "test_agent" in agents

    def test_load_agent(self, runner):
        """Test loading agent configuration."""
        config = runner.load_agent("test_agent")

        assert config.name == "test_agent"
        assert len(config.workflow) == 2
        assert config.tools == ["test_tool"]

    @pytest.mark.asyncio
    async def test_run_agent_basic(self, runner):
        """Test running a basic agent workflow."""
        # Register mock resource
        runner.register_resource("test://resource", lambda: {"data": "test_value"})

        result = await runner.run_agent("test_agent")

        assert result.agent_name == "test_agent"
        assert result.steps_executed == 2
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_agent_with_tool(self, runner, tmp_path):
        """Test running agent with tool calls."""
        from ordinis.agents.runner import AgentRunner
        import yaml

        # Create agent with tool
        agent_config = {
            "name": "tool_agent",
            "model": "test-model",
            "timeout_seconds": 60,
            "system_prompt": "Test",
            "tools": ["get_value"],
            "resources": [],
            "workflow": [
                {"step": "call_tool", "action": "call_tool", "tool": "get_value", "params": {}},
            ],
        }

        config_path = tmp_path / "tool_agent.yaml"
        with open(config_path, "w") as f:
            yaml.dump(agent_config, f)

        runner = AgentRunner(agents_dir=tmp_path)
        runner.register_tool("get_value", lambda: {"value": 42})

        result = await runner.run_agent("tool_agent")

        assert result.success is True
        assert result.step_results["call_tool"]["output"]["value"] == 42

    @pytest.mark.asyncio
    async def test_run_agent_with_condition(self, runner, tmp_path):
        """Test conditional step execution."""
        from ordinis.agents.runner import AgentRunner
        import yaml

        agent_config = {
            "name": "conditional_agent",
            "model": "test-model",
            "timeout_seconds": 60,
            "system_prompt": "Test",
            "tools": [],
            "resources": [],
            "workflow": [
                {"step": "check", "action": "read_resource", "resource": "test://check"},
                {
                    "step": "conditional_log",
                    "action": "log",
                    "level": "info",
                    "message": "Condition met",
                    "condition": "{{ check.output.enabled }}",
                },
            ],
        }

        config_path = tmp_path / "conditional_agent.yaml"
        with open(config_path, "w") as f:
            yaml.dump(agent_config, f)

        runner = AgentRunner(agents_dir=tmp_path)
        runner.register_resource("test://check", lambda: {"enabled": True})

        result = await runner.run_agent("conditional_agent")

        assert result.success is True
        assert result.steps_executed == 2


# =============================================================================
# Test: DrawdownHook Integration
# =============================================================================


class TestDrawdownHook:
    """Tests for the drawdown governance hook."""

    @pytest.fixture
    def hook(self):
        """Create a DrawdownHook with mock KillSwitch."""
        from ordinis.engines.riskguard.hooks.drawdown import DrawdownHook

        return DrawdownHook(kill_switch=None)

    @pytest.mark.asyncio
    async def test_preflight_allows_normal_drawdown(self, hook):
        """Test preflight allows operations at low drawdown."""
        from ordinis.engines.base.hooks import Decision, PreflightContext

        context = PreflightContext(
            operation="calculate_position_size",
            parameters={},
            timestamp=datetime.now(UTC),
            trace_id="test-1",
        )
        context.action = "calculate_position_size"

        result = await hook.preflight(context)

        # At 0% drawdown, should allow
        assert result.decision == Decision.ALLOW

    @pytest.mark.asyncio
    async def test_exposure_factor_tracking(self, hook):
        """Test exposure factor is tracked correctly."""
        # Initial factor should be 1.0
        assert hook.exposure_factor == 1.0


# =============================================================================
# Test: MCP Tools Integration
# =============================================================================


class TestMCPToolsIntegration:
    """Integration tests for MCP tools."""

    @pytest.mark.asyncio
    async def test_mcp_server_initialization(self):
        """Test MCP server state initialization."""
        from ordinis.mcp.server import _state

        result = await _state.initialize()
        assert result is True
        assert _state.is_initialized is True

    @pytest.mark.asyncio
    async def test_get_market_news_tool(self):
        """Test get_market_news MCP tool."""
        from ordinis.mcp.server import _state, get_market_news

        await _state.initialize()

        result = await get_market_news(symbols=["AAPL"], limit=5)
        data = json.loads(result)

        assert "articles" in data
        assert "count" in data
        assert data["count"] <= 5

    @pytest.mark.asyncio
    async def test_get_earnings_calendar_tool(self):
        """Test get_earnings_calendar MCP tool."""
        from ordinis.mcp.server import _state, get_earnings_calendar

        await _state.initialize()

        result = await get_earnings_calendar(days_ahead=5)
        data = json.loads(result)

        assert "earnings" in data
        assert "count" in data

    @pytest.mark.asyncio
    async def test_get_sentiment_tool(self):
        """Test get_sentiment MCP tool."""
        from ordinis.mcp.server import _state, get_sentiment

        await _state.initialize()

        result = await get_sentiment("AAPL")
        data = json.loads(result)

        assert "symbol" in data
        assert "sentiment" in data
        assert "score" in data

    @pytest.mark.asyncio
    async def test_analyze_open_positions_tool(self):
        """Test analyze_open_positions MCP tool."""
        from ordinis.mcp.server import _state, analyze_open_positions

        await _state.initialize()

        result = await analyze_open_positions()
        data = json.loads(result)

        assert "position_count" in data
        assert "positions" in data
        assert "summary" in data

    @pytest.mark.asyncio
    async def test_inject_market_context_tool(self):
        """Test inject_market_context MCP tool."""
        from ordinis.mcp.server import _state, inject_market_context

        await _state.initialize()

        result = await inject_market_context("Test market context for integration test")
        data = json.loads(result)

        assert data["success"] is True
        assert "Test market context" in data["context"]

    @pytest.mark.asyncio
    async def test_get_drawdown_status_tool(self):
        """Test get_drawdown_status MCP tool."""
        from ordinis.mcp.server import _state, get_drawdown_status

        await _state.initialize()

        result = await get_drawdown_status()
        data = json.loads(result)

        assert "drawdown_pct" in data
        assert "can_trade" in data
        assert "limits" in data


# =============================================================================
# Test: End-to-End Agent Flow
# =============================================================================


class TestEndToEndAgentFlow:
    """End-to-end tests for agent workflow execution."""

    @pytest.mark.asyncio
    async def test_strategy_optimizer_mock_flow(self, tmp_path):
        """Test strategy optimizer agent with mocked tools."""
        from ordinis.agents.runner import AgentRunner
        import yaml

        # Simplified strategy optimizer workflow
        agent_config = {
            "name": "strategy_optimizer_test",
            "model": "test-model",
            "timeout_seconds": 60,
            "system_prompt": "Optimize strategies based on performance.",
            "tools": ["get_strategy_config", "get_pnl_metrics"],
            "resources": [],
            "workflow": [
                {"step": "get_config", "action": "call_tool", "tool": "get_strategy_config"},
                {"step": "get_metrics", "action": "call_tool", "tool": "get_pnl_metrics"},
                {
                    "step": "analyze",
                    "action": "llm_analyze",
                    "input": "Config: {{ get_config.output }}, Metrics: {{ get_metrics.output }}",
                },
            ],
        }

        config_path = tmp_path / "strategy_optimizer_test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(agent_config, f)

        runner = AgentRunner(agents_dir=tmp_path)

        # Register mock tools
        runner.register_tool(
            "get_strategy_config",
            lambda: {
                "strategy": "atr_optimized_rsi",
                "rsi_oversold": 30,
                "atr_stop_mult": 1.5,
            },
        )
        runner.register_tool(
            "get_pnl_metrics",
            lambda: {
                "win_rate": 55.0,
                "sharpe": 1.8,
                "daily_pnl": 250.0,
            },
        )

        result = await runner.run_agent("strategy_optimizer_test")

        assert result.success is True
        assert result.steps_executed == 3
        assert "get_config" in result.step_results
        assert "get_metrics" in result.step_results

    @pytest.mark.asyncio
    async def test_drawdown_response_mock_flow(self, tmp_path):
        """Test drawdown response agent with mocked tools."""
        from ordinis.agents.runner import AgentRunner
        import yaml

        agent_config = {
            "name": "drawdown_response_test",
            "model": "test-model",
            "timeout_seconds": 60,
            "system_prompt": "Monitor drawdown and respond appropriately.",
            "tools": ["get_drawdown_status"],
            "resources": [],
            "workflow": [
                {"step": "check_drawdown", "action": "call_tool", "tool": "get_drawdown_status"},
                {
                    "step": "log_status",
                    "action": "log",
                    "level": "info",
                    "message": "Drawdown: {{ check_drawdown.output.drawdown_pct }}%",
                },
            ],
        }

        config_path = tmp_path / "drawdown_response_test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(agent_config, f)

        runner = AgentRunner(agents_dir=tmp_path)
        runner.register_tool(
            "get_drawdown_status",
            lambda: {
                "drawdown_pct": 3.5,
                "daily_pnl": -150.0,
                "can_trade": True,
            },
        )

        result = await runner.run_agent("drawdown_response_test")

        assert result.success is True
        assert result.steps_executed == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
