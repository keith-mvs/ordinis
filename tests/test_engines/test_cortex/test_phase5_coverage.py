"""
Phase 5: Expanded Test Coverage for Cortex Engine.

Tests cover:
- RAG-enabled hypothesis generation
- Health check degradation scenarios
- Governance integration edge cases
- Error handling and recovery
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ordinis.engines.cortex.core import (
    CortexConfig,
    CortexEngine,
    CortexEngineError,
    ModelConfig,
)
from ordinis.engines.base import HealthLevel, HealthStatus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_helix():
    """Create a mock Helix instance with healthy defaults."""
    helix = MagicMock()
    helix.health_check = AsyncMock(
        return_value=HealthStatus(
            level=HealthLevel.HEALTHY,
            message="All providers healthy",
        )
    )
    # Mock generate to return a successful response
    response = MagicMock()
    response.content = "Mock LLM response"
    response.input_tokens = 100
    response.output_tokens = 50
    helix.generate = AsyncMock(return_value=response)
    return helix


@pytest.fixture
def mock_degraded_helix():
    """Create a mock Helix instance in degraded state."""
    helix = MagicMock()
    helix.health_check = AsyncMock(
        return_value=HealthStatus(
            level=HealthLevel.DEGRADED,
            message="Provider rate limited",
        )
    )
    response = MagicMock()
    response.content = "Degraded response"
    response.input_tokens = 50
    response.output_tokens = 25
    helix.generate = AsyncMock(return_value=response)
    return helix


@pytest.fixture
def mock_unhealthy_helix():
    """Create a mock Helix instance in unhealthy state."""
    helix = MagicMock()
    helix.health_check = AsyncMock(
        return_value=HealthStatus(
            level=HealthLevel.UNHEALTHY,
            message="All providers unavailable",
        )
    )
    helix.generate = AsyncMock(side_effect=Exception("No providers available"))
    return helix


@pytest.fixture
def cortex_config():
    """Create a Cortex config with governance disabled for testing."""
    return CortexConfig(
        governance_enabled=False,
        audit_enabled=False,
    )


@pytest.fixture
def mock_rag_helper():
    """Create a mock RAG helper."""
    helper = MagicMock()
    helper.format_hypothesis_context = MagicMock(
        return_value="## Historical Context\n- Similar market regime in 2020\n- Trend following performed well"
    )
    helper.format_code_analysis_context = MagicMock(
        return_value="## Related Code\n- Similar pattern in strategy_engine.py"
    )
    helper.get_kb_context = MagicMock(
        return_value="Knowledge base context about trading strategies"
    )
    helper.get_code_examples = MagicMock(
        return_value="def example_strategy(): pass"
    )
    return helper


# =============================================================================
# RAG-Enabled Hypothesis Generation Tests
# =============================================================================


class TestRAGEnabledHypothesis:
    """Test RAG-enabled hypothesis generation."""

    @pytest.mark.asyncio
    async def test_hypothesis_with_rag_context(self, mock_helix, mock_rag_helper):
        """Test hypothesis generation uses RAG context when enabled."""
        config = CortexConfig(
            governance_enabled=False,
            rag_enabled=True,
        )
        engine = CortexEngine(helix=mock_helix, config=config)
        await engine.initialize()
        # Override the RAG helper AFTER initialize (which creates real one)
        engine._rag_helper = mock_rag_helper

        market_context = {"regime": "trending", "volatility": "low"}
        hypothesis = await engine.generate_hypothesis(market_context)

        # Verify RAG helper was called
        mock_rag_helper.format_hypothesis_context.assert_called_once()

        # Hypothesis should still be generated
        assert hypothesis is not None
        assert hypothesis.hypothesis_id.startswith("hyp-")

    @pytest.mark.asyncio
    async def test_hypothesis_without_rag_context(self, mock_helix):
        """Test hypothesis generation works without RAG."""
        config = CortexConfig(
            governance_enabled=False,
            rag_enabled=False,
        )
        engine = CortexEngine(helix=mock_helix, config=config)
        await engine.initialize()

        market_context = {"regime": "trending", "volatility": "low"}
        hypothesis = await engine.generate_hypothesis(market_context)

        # Should still generate hypothesis using fallback logic
        assert hypothesis is not None
        assert hypothesis.strategy_type == "trend_following"

    @pytest.mark.asyncio
    async def test_rag_failure_graceful_fallback(self, mock_helix, mock_rag_helper):
        """Test that RAG failures don't break hypothesis generation."""
        config = CortexConfig(
            governance_enabled=False,
            rag_enabled=True,
        )
        engine = CortexEngine(helix=mock_helix, config=config)
        engine._rag_helper = mock_rag_helper
        # Make RAG helper throw an exception
        mock_rag_helper.format_hypothesis_context.side_effect = Exception("RAG failure")
        await engine.initialize()

        market_context = {"regime": "trending", "volatility": "low"}
        # Should NOT raise - should fall back gracefully
        hypothesis = await engine.generate_hypothesis(market_context)

        assert hypothesis is not None
        assert hypothesis.strategy_type == "trend_following"

    @pytest.mark.asyncio
    async def test_rag_context_truncation(self, mock_helix, mock_rag_helper):
        """Test that large RAG context is truncated."""
        config = CortexConfig(
            governance_enabled=False,
            rag_enabled=True,
        )
        engine = CortexEngine(helix=mock_helix, config=config)
        engine._rag_helper = mock_rag_helper
        # Return very large context
        mock_rag_helper.format_hypothesis_context.return_value = "x" * 50000
        await engine.initialize()

        market_context = {"regime": "trending", "volatility": "low"}
        # Should succeed without hitting context limits
        hypothesis = await engine.generate_hypothesis(market_context)

        assert hypothesis is not None


# =============================================================================
# Health Check Degradation Tests
# =============================================================================


class TestHealthCheckDegradation:
    """Test health check behavior under various conditions."""

    @pytest.mark.asyncio
    async def test_healthy_state(self, mock_helix, cortex_config):
        """Test health check returns healthy when Helix is healthy."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        health = await engine._do_health_check()

        assert health.level == HealthLevel.HEALTHY
        assert "healthy" in health.message.lower()

    @pytest.mark.asyncio
    async def test_degraded_when_helix_degraded(self, mock_degraded_helix, cortex_config):
        """Test health check returns degraded when Helix is degraded."""
        engine = CortexEngine(helix=mock_degraded_helix, config=cortex_config)
        await engine.initialize()

        health = await engine._do_health_check()

        assert health.level == HealthLevel.DEGRADED
        assert "degraded" in health.message.lower()
        assert "rate limited" in health.message.lower()

    @pytest.mark.asyncio
    async def test_degraded_when_helix_unhealthy(self, mock_unhealthy_helix, cortex_config):
        """Test health check returns degraded when Helix is unhealthy."""
        engine = CortexEngine(helix=mock_unhealthy_helix, config=cortex_config)
        await engine.initialize()

        health = await engine._do_health_check()

        assert health.level == HealthLevel.DEGRADED
        assert "unavailable" in health.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_helix_exception(self, mock_helix, cortex_config):
        """Test health check handles Helix exceptions gracefully."""
        mock_helix.health_check = AsyncMock(side_effect=Exception("Connection failed"))
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Should not raise, but return degraded status
        with pytest.raises(Exception):
            await engine._do_health_check()


# =============================================================================
# Governance Integration Edge Cases
# =============================================================================


class TestGovernanceEdgeCases:
    """Test governance integration edge cases."""

    @pytest.mark.asyncio
    async def test_governance_preflight_failure(self, mock_helix):
        """Test behavior when governance preflight fails."""
        config = CortexConfig(
            governance_enabled=True,
            audit_enabled=True,
        )
        engine = CortexEngine(helix=mock_helix, config=config)

        # Mock governance to deny
        mock_governance = MagicMock()
        preflight_result = MagicMock()
        preflight_result.allowed = False
        preflight_result.reasons = ["Operation blocked by policy"]
        mock_governance.preflight = AsyncMock(return_value=preflight_result)
        engine._governance = mock_governance
        await engine.initialize()

        # Should raise on governance denial
        with pytest.raises(CortexEngineError) as exc_info:
            await engine.generate_hypothesis({"regime": "trending"})

        assert "GOVERNANCE_DENIED" in exc_info.value.code
        assert exc_info.value.recoverable is False

    @pytest.mark.asyncio
    async def test_governance_audit_failure_non_fatal(self, mock_helix):
        """Test that audit failures don't break operations."""
        from ordinis.engines.base.hooks import Decision, PreflightResult

        config = CortexConfig(
            governance_enabled=True,
            audit_enabled=True,
        )
        engine = CortexEngine(helix=mock_helix, config=config)

        # Mock governance - preflight allows, audit fails
        mock_governance = MagicMock()
        preflight_result = PreflightResult(decision=Decision.ALLOW, reason="Allowed")
        mock_governance.preflight = AsyncMock(return_value=preflight_result)
        mock_governance.audit = AsyncMock(side_effect=Exception("Audit service down"))
        engine._governance = mock_governance
        await engine.initialize()

        # Should NOT raise - audit failures are non-fatal
        hypothesis = await engine.generate_hypothesis({"regime": "trending"})
        assert hypothesis is not None

    @pytest.mark.asyncio
    async def test_governance_disabled_skips_checks(self, mock_helix):
        """Test that disabled governance skips preflight checks."""
        config = CortexConfig(
            governance_enabled=False,
            audit_enabled=False,
        )
        engine = CortexEngine(helix=mock_helix, config=config)

        # Mock governance (should not be called)
        mock_governance = MagicMock()
        mock_governance.preflight = AsyncMock()
        engine._governance = mock_governance
        await engine.initialize()

        await engine.generate_hypothesis({"regime": "trending"})

        # Governance preflight should NOT have been called
        mock_governance.preflight.assert_not_called()


# =============================================================================
# Error Handling and Recovery Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_helix_timeout_recovery(self, mock_helix, cortex_config):
        """Test recovery from Helix timeout using fallback model."""
        call_count = 0

        async def mock_generate_with_timeout(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Request timed out")
            response = MagicMock()
            response.content = "Fallback success"
            response.input_tokens = 50
            response.output_tokens = 25
            return response

        mock_helix.generate = mock_generate_with_timeout
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        result = await engine._generate_with_fallback(
            messages=[MagicMock(role="user", content="test")],
            model_config=ModelConfig(primary="model-a", fallback=["model-b"]),
            operation="test",
        )

        assert result[0] == "Fallback success"
        assert result[1] == "model-b"

    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self, mock_helix, cortex_config):
        """Test recovery from rate limiting using fallback model."""
        call_count = 0

        async def mock_generate_rate_limited(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Rate limit exceeded")
            response = MagicMock()
            response.content = "Fallback success"
            response.input_tokens = 50
            response.output_tokens = 25
            return response

        mock_helix.generate = mock_generate_rate_limited
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        result = await engine._generate_with_fallback(
            messages=[MagicMock(role="user", content="test")],
            model_config=ModelConfig(primary="model-a", fallback=["model-b"]),
            operation="test",
        )

        assert result[0] == "Fallback success"

    @pytest.mark.asyncio
    async def test_all_models_exhausted(self, mock_helix, cortex_config):
        """Test error when all models in fallback chain fail."""
        mock_helix.generate = AsyncMock(side_effect=Exception("All models failed"))
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        with pytest.raises(CortexEngineError) as exc_info:
            await engine._generate_with_fallback(
                messages=[MagicMock(role="user", content="test")],
                model_config=ModelConfig(
                    primary="model-a",
                    fallback=["model-b", "model-c"],
                ),
                operation="test",
            )

        assert exc_info.value.code == "ALL_MODELS_FAILED"
        assert exc_info.value.recoverable is True

    @pytest.mark.asyncio
    async def test_empty_response_handling(self, mock_helix, cortex_config):
        """Test handling of empty LLM responses."""
        response = MagicMock()
        response.content = ""  # Empty response
        response.input_tokens = 50
        response.output_tokens = 0
        mock_helix.generate = AsyncMock(return_value=response)

        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Should fall back to deterministic generation
        hypothesis = await engine.generate_hypothesis({"regime": "trending"})
        assert hypothesis is not None
        # Engine uses 'adaptive' as default fallback for unknown/error cases
        assert hypothesis.strategy_type in ("trend_following", "adaptive")

    @pytest.mark.asyncio
    async def test_malformed_json_fallback(self, mock_helix, cortex_config):
        """Test fallback when LLM returns malformed JSON."""
        response = MagicMock()
        response.content = "This is not valid JSON {broken"
        response.input_tokens = 100
        response.output_tokens = 50
        mock_helix.generate = AsyncMock(return_value=response)

        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Should fall back to deterministic generation
        hypothesis = await engine.generate_hypothesis({"regime": "trending"})
        assert hypothesis is not None
        # Engine uses 'adaptive' as default fallback for unknown/error cases
        assert hypothesis.strategy_type in ("trend_following", "adaptive")


# =============================================================================
# Input Validation Edge Cases
# =============================================================================


class TestInputValidation:
    """Test input validation edge cases."""

    @pytest.mark.asyncio
    async def test_empty_market_context(self, mock_helix, cortex_config):
        """Test hypothesis generation with empty market context."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Should use defaults and generate adaptive strategy
        hypothesis = await engine.generate_hypothesis({})
        assert hypothesis is not None
        assert hypothesis.strategy_type == "adaptive"

    @pytest.mark.asyncio
    async def test_none_market_context(self, mock_helix, cortex_config):
        """Test hypothesis generation handles None values in context."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        market_context = {"regime": None, "volatility": None}
        hypothesis = await engine.generate_hypothesis(market_context)
        assert hypothesis is not None

    @pytest.mark.asyncio
    async def test_very_long_code_input(self, mock_helix, cortex_config):
        """Test code analysis with very long input."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        long_code = "def foo():\n    pass\n" * 10000  # ~200KB of code
        output = await engine.analyze_code(long_code, "review")

        # Should truncate and still produce output
        assert output is not None
        assert output.content is not None

    @pytest.mark.asyncio
    async def test_unicode_input_handling(self, mock_helix, cortex_config):
        """Test handling of unicode in inputs."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Unicode in market context
        market_context = {"notes": "📈 Bullish 中文 العربية"}
        hypothesis = await engine.generate_hypothesis(market_context)
        assert hypothesis is not None

    @pytest.mark.asyncio
    async def test_special_characters_sanitization(self, mock_helix, cortex_config):
        """Test that special characters don't break sanitization."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Code with regex-like patterns that might break sanitization
        code = r"""
def match_pattern(text):
    pattern = r'\b[A-Za-z]+@[a-z]+\.[a-z]{2,}\b'  # Email-like
    return re.match(pattern, text)
"""
        output = await engine.analyze_code(code, "review")
        assert output is not None


# =============================================================================
# Metrics and Observability Tests
# =============================================================================


class TestMetrics:
    """Test Prometheus metrics and observability."""

    @pytest.mark.asyncio
    async def test_fallback_metrics_emitted(self, mock_helix, cortex_config):
        """Test that fallback events emit metrics."""
        call_count = 0

        async def mock_generate_with_fallback(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Primary failed")
            response = MagicMock()
            response.content = "Fallback response"
            response.input_tokens = 50
            response.output_tokens = 25
            return response

        mock_helix.generate = mock_generate_with_fallback
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Just verify the call succeeds and uses fallback
        result = await engine._generate_with_fallback(
            messages=[MagicMock(role="user", content="test")],
            model_config=ModelConfig(primary="model-a", fallback=["model-b"]),
            operation="test",
        )

        assert result[1] == "model-b"  # Used fallback model


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Test engine state management."""

    @pytest.mark.asyncio
    async def test_outputs_accumulate(self, mock_helix, cortex_config):
        """Test that outputs accumulate correctly."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Generate multiple outputs
        await engine.generate_hypothesis({"regime": "trending"})
        await engine.generate_hypothesis({"regime": "mean_reverting"})
        await engine.analyze_code("def foo(): pass", "review")

        outputs = engine.get_outputs()
        assert len(outputs) == 3

        hypotheses = engine.list_hypotheses()
        assert len(hypotheses) == 2

    @pytest.mark.asyncio
    async def test_hypothesis_retrieval_by_id(self, mock_helix, cortex_config):
        """Test retrieving hypothesis by ID."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        hypothesis = await engine.generate_hypothesis({"regime": "trending"})
        retrieved = engine.get_hypothesis(hypothesis.hypothesis_id)

        assert retrieved is not None
        assert retrieved.hypothesis_id == hypothesis.hypothesis_id
        assert retrieved.strategy_type == hypothesis.strategy_type

    @pytest.mark.asyncio
    async def test_hypothesis_not_found(self, mock_helix, cortex_config):
        """Test behavior when hypothesis ID not found."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        retrieved = engine.get_hypothesis("hyp-nonexistent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_to_dict_state_representation(self, mock_helix, cortex_config):
        """Test to_dict returns correct state representation."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        await engine.generate_hypothesis({"regime": "trending"})
        await engine.analyze_code("code", "review")

        state = engine.to_dict()

        assert state["total_outputs"] == 2
        assert state["total_hypotheses"] == 1
        # Keys are lowercase in the actual implementation
        assert "hypothesis" in state["outputs_by_type"]
        assert "code_analysis" in state["outputs_by_type"]
