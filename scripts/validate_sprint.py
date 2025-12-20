"""Quick validation script for Sprint 1 & 2 implementations."""

import asyncio
import sys

def test_imports():
    """Test all Sprint 1 & 2 imports."""
    results = []
    
    try:
        from ordinis.plugins.massive import MassivePlugin, MassivePluginConfig
        results.append(("MassivePlugin", True, None))
    except Exception as e:
        results.append(("MassivePlugin", False, str(e)))
    
    try:
        from ordinis.agents.runner import AgentRunner, AgentConfig, AgentResult
        results.append(("AgentRunner", True, None))
    except Exception as e:
        results.append(("AgentRunner", False, str(e)))
    
    try:
        from ordinis.engines.signalcore.hooks.regime import RegimeHook, MarketRegime
        results.append(("RegimeHook", True, None))
    except Exception as e:
        results.append(("RegimeHook", False, str(e)))
    
    try:
        from ordinis.engines.riskguard.hooks.drawdown import DrawdownHook
        results.append(("DrawdownHook", True, None))
    except Exception as e:
        results.append(("DrawdownHook", False, str(e)))
    
    try:
        from ordinis.mcp.server import _state
        results.append(("MCP Server State", True, None))
    except Exception as e:
        results.append(("MCP Server State", False, str(e)))
    
    return results


async def test_massive_plugin():
    """Test MassivePlugin functionality."""
    from ordinis.plugins.massive import MassivePlugin, MassivePluginConfig
    
    config = MassivePluginConfig(name="massive", use_mock=True)
    plugin = MassivePlugin(config)
    
    # Initialize
    result = await plugin.initialize()
    assert result is True, "Plugin initialization failed"
    print("  ✓ Plugin initialized")
    
    # Get news
    news = await plugin.get_news(symbols=["AAPL"], limit=3)
    assert len(news) > 0, "No news returned"
    print(f"  ✓ Got {len(news)} news articles")
    
    # Get sentiment
    sentiment = await plugin.get_sentiment("AAPL")
    assert "score" in sentiment, "Sentiment missing score"
    print(f"  ✓ Sentiment score: {sentiment['score']:.3f}")
    
    # Get earnings
    earnings = await plugin.get_earnings_calendar(days_ahead=5)
    assert isinstance(earnings, list), "Earnings not a list"
    print(f"  ✓ Got {len(earnings)} upcoming earnings")
    
    await plugin.shutdown()
    print("  ✓ Plugin shutdown")


async def test_regime_hook():
    """Test RegimeHook functionality."""
    from ordinis.engines.signalcore.hooks.regime import RegimeHook, MarketRegime
    
    hook = RegimeHook(initial_regime=MarketRegime.UNKNOWN)
    
    # Test regime detection
    regime = hook.detect_from_indicators(vix=15.0, spy_return_5d=3.5)
    assert regime == MarketRegime.TRENDING_UP
    print(f"  ✓ Detected regime: {regime.name}")
    
    # Test high volatility detection
    regime = hook.detect_from_indicators(vix=35.0)
    assert regime == MarketRegime.HIGH_VOLATILITY
    print(f"  ✓ High VIX regime: {regime.name}")
    
    # Test position size modifier
    hook.set_regime(MarketRegime.HIGH_VOLATILITY, confidence=0.8)
    assert hook.regime_state.position_size_modifier == 0.5
    print(f"  ✓ Position size modifier: {hook.regime_state.position_size_modifier}")


async def test_agent_runner():
    """Test AgentRunner functionality."""
    import tempfile
    import yaml
    from pathlib import Path
    from ordinis.agents.runner import AgentRunner
    
    # Create temp agent config
    agent_config = {
        "name": "test_agent",
        "model": "test-model",
        "timeout_seconds": 60,
        "system_prompt": "Test agent",
        "tools": ["test_tool"],
        "resources": [],
        "workflow": [
            {"step": "call_tool", "action": "call_tool", "tool": "test_tool"},
            {"step": "log_result", "action": "log", "level": "info", "message": "Test done"},
        ],
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_agent.yaml"
        with open(config_path, "w") as f:
            yaml.dump(agent_config, f)
        
        runner = AgentRunner(agents_dir=Path(tmpdir))
        runner.register_tool("test_tool", lambda: {"value": 42})
        
        # List agents
        agents = runner.list_agents()
        assert "test_agent" in agents
        print(f"  ✓ Listed agents: {agents}")
        
        # Run agent
        result = await runner.run_agent("test_agent")
        assert result.success is True
        assert result.steps_executed == 2
        print(f"  ✓ Agent ran {result.steps_executed} steps in {result.duration_seconds:.3f}s")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Sprint 1 & 2 Validation")
    print("=" * 60)
    
    # Test imports
    print("\n1. Testing Imports...")
    results = test_imports()
    all_passed = True
    for name, passed, error in results:
        if passed:
            print(f"   ✓ {name}")
        else:
            print(f"   ✗ {name}: {error}")
            all_passed = False
    
    if not all_passed:
        print("\n❌ Import tests failed!")
        return 1
    
    # Test MassivePlugin
    print("\n2. Testing MassivePlugin...")
    try:
        await test_massive_plugin()
        print("   ✓ MassivePlugin tests passed")
    except Exception as e:
        print(f"   ✗ MassivePlugin tests failed: {e}")
        return 1
    
    # Test RegimeHook
    print("\n3. Testing RegimeHook...")
    try:
        await test_regime_hook()
        print("   ✓ RegimeHook tests passed")
    except Exception as e:
        print(f"   ✗ RegimeHook tests failed: {e}")
        return 1
    
    # Test AgentRunner
    print("\n4. Testing AgentRunner...")
    try:
        await test_agent_runner()
        print("   ✓ AgentRunner tests passed")
    except Exception as e:
        print(f"   ✗ AgentRunner tests failed: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("✓ ALL SPRINT 1 & 2 TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
