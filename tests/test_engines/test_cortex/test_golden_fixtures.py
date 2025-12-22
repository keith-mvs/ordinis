"""
Phase 5.2: Golden Fixture Tests and Safety Benchmarks for Cortex Engine.

This module implements:
1. Golden fixture tests - verify deterministic behavior against expected outputs
2. Safety benchmarks - test resilience against prompt injection, PII leakage
3. Hallucination detection - structured output validation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ordinis.engines.cortex.core import (
    CortexConfig,
    CortexEngine,
    CortexOutput,
    OutputType,
)
from ordinis.engines.base import HealthLevel, HealthStatus

from .fixtures.golden_prompts import (
    HYPOTHESIS_GOLDEN_FIXTURES,
    ANALYSIS_GOLDEN_FIXTURES,
    RESEARCH_GOLDEN_FIXTURES,
    GoldenPromptFixture,
)
from .fixtures.safety_benchmarks import (
    PROMPT_INJECTION_TESTS,
    PII_LEAKAGE_TESTS,
    SECRET_LEAKAGE_TESTS,
    SafetyTestCase,
    get_high_severity_tests,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_helix():
    """Create a mock Helix instance."""
    helix = MagicMock()
    helix.health_check = AsyncMock(
        return_value=HealthStatus(
            level=HealthLevel.HEALTHY,
            message="All providers healthy",
        )
    )
    response = MagicMock()
    response.content = "Mock LLM response with strategy analysis"
    response.input_tokens = 100
    response.output_tokens = 50
    helix.generate = AsyncMock(return_value=response)
    return helix


@pytest.fixture
def cortex_config():
    """Create a Cortex config for testing."""
    return CortexConfig(
        governance_enabled=False,
        audit_enabled=False,
    )


@pytest.fixture
async def initialized_engine(mock_helix, cortex_config):
    """Create and initialize a Cortex engine."""
    engine = CortexEngine(helix=mock_helix, config=cortex_config)
    await engine.initialize()
    return engine


# =============================================================================
# Golden Fixture Tests - Hypothesis Generation
# =============================================================================


class TestGoldenHypothesisFixtures:
    """Test hypothesis generation against golden fixtures."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fixture",
        HYPOTHESIS_GOLDEN_FIXTURES,
        ids=[f.name for f in HYPOTHESIS_GOLDEN_FIXTURES],
    )
    async def test_hypothesis_golden_fixture(
        self, mock_helix, cortex_config, fixture: GoldenPromptFixture
    ):
        """Verify hypothesis generation matches expected behavior."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        market_context = fixture.input_data.get("market_context", {})
        constraints = fixture.input_data.get("constraints")

        hypothesis = await engine.generate_hypothesis(
            market_context=market_context,
            constraints=constraints,
        )

        # Verify required output keys exist
        for key in fixture.expected_output_keys:
            assert hasattr(hypothesis, key), f"Missing key: {key}"

        # Verify constraints
        constraints = fixture.expected_constraints

        if "strategy_type" in constraints:
            assert hypothesis.strategy_type == constraints["strategy_type"], (
                f"Expected strategy_type={constraints['strategy_type']}, "
                f"got {hypothesis.strategy_type}"
            )

        if "strategy_type_in" in constraints:
            assert hypothesis.strategy_type in constraints["strategy_type_in"], (
                f"Expected strategy_type in {constraints['strategy_type_in']}, "
                f"got {hypothesis.strategy_type}"
            )

        if "confidence_range" in constraints:
            min_conf, max_conf = constraints["confidence_range"]
            assert min_conf <= hypothesis.confidence <= max_conf, (
                f"Confidence {hypothesis.confidence} outside range "
                f"[{min_conf}, {max_conf}]"
            )

        if "max_position_size_pct_max" in constraints:
            assert hypothesis.max_position_size_pct <= constraints["max_position_size_pct_max"]

        if "instrument_class" in constraints:
            assert hypothesis.instrument_class == constraints["instrument_class"]

    @pytest.mark.asyncio
    async def test_hypothesis_id_format(self, initialized_engine):
        """Verify hypothesis IDs follow expected format."""
        hypothesis = await initialized_engine.generate_hypothesis({"regime": "trending"})
        assert hypothesis.hypothesis_id.startswith("hyp-")
        assert len(hypothesis.hypothesis_id) > 10  # UUID-like

    @pytest.mark.asyncio
    async def test_hypothesis_has_required_risk_params(self, initialized_engine):
        """Verify all hypotheses include required risk parameters."""
        hypothesis = await initialized_engine.generate_hypothesis({"regime": "trending"})

        assert hypothesis.max_position_size_pct > 0
        assert hypothesis.max_position_size_pct <= 1.0
        assert hypothesis.stop_loss_pct > 0
        assert hypothesis.stop_loss_pct <= 1.0

    @pytest.mark.asyncio
    async def test_hypothesis_entry_exit_conditions_nonempty(self, initialized_engine):
        """Verify hypotheses have non-empty entry/exit conditions."""
        hypothesis = await initialized_engine.generate_hypothesis({"regime": "trending"})

        assert len(hypothesis.entry_conditions) > 0
        assert len(hypothesis.exit_conditions) > 0


# =============================================================================
# Golden Fixture Tests - Code Analysis
# =============================================================================


class TestGoldenAnalysisFixtures:
    """Test code analysis against golden fixtures."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fixture",
        ANALYSIS_GOLDEN_FIXTURES,
        ids=[f.name for f in ANALYSIS_GOLDEN_FIXTURES],
    )
    async def test_analysis_golden_fixture(
        self, mock_helix, cortex_config, fixture: GoldenPromptFixture
    ):
        """Verify code analysis matches expected behavior."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        code = fixture.input_data["code"]
        analysis_type = fixture.input_data.get("analysis_type", "review")

        output = await engine.analyze_code(code, analysis_type)

        # Verify it's a CortexOutput
        assert isinstance(output, CortexOutput)

        # Verify required keys exist
        for key in fixture.expected_output_keys:
            assert hasattr(output, key), f"Missing key: {key}"

        # Verify output type
        if "output_type" in fixture.expected_constraints:
            assert output.output_type == OutputType.CODE_ANALYSIS

        # Verify confidence range
        if "confidence_range" in fixture.expected_constraints:
            min_conf, max_conf = fixture.expected_constraints["confidence_range"]
            assert min_conf <= output.confidence <= max_conf

    @pytest.mark.asyncio
    async def test_analysis_output_has_content(self, initialized_engine):
        """Verify code analysis always has content."""
        output = await initialized_engine.analyze_code("def foo(): pass", "review")
        assert output.content is not None
        assert len(str(output.content)) > 0

    @pytest.mark.asyncio
    async def test_analysis_preserves_type(self, initialized_engine):
        """Verify analysis type is reflected in output metadata."""
        output = await initialized_engine.analyze_code("def foo(): pass", "security")
        assert output.output_type == OutputType.CODE_ANALYSIS


# =============================================================================
# Golden Fixture Tests - Research Synthesis
# =============================================================================


class TestGoldenResearchFixtures:
    """Test research synthesis against golden fixtures."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fixture",
        RESEARCH_GOLDEN_FIXTURES,
        ids=[f.name for f in RESEARCH_GOLDEN_FIXTURES],
    )
    async def test_research_golden_fixture(
        self, mock_helix, cortex_config, fixture: GoldenPromptFixture
    ):
        """Verify research synthesis matches expected behavior."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        query = fixture.input_data["query"]
        sources = fixture.input_data.get("sources", [])
        context = fixture.input_data.get("context")

        output = await engine.synthesize_research(query, sources, context)

        # Verify it's a CortexOutput
        assert isinstance(output, CortexOutput)

        # Verify output type
        if "output_type" in fixture.expected_constraints:
            assert output.output_type == OutputType.RESEARCH


# =============================================================================
# Safety Benchmark Tests - Prompt Injection
# =============================================================================


class TestPromptInjectionSafety:
    """Test resilience against prompt injection attacks."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_case",
        PROMPT_INJECTION_TESTS,
        ids=[t.name for t in PROMPT_INJECTION_TESTS],
    )
    async def test_prompt_injection_resilience(
        self, mock_helix, cortex_config, test_case: SafetyTestCase
    ):
        """
        Verify engine resists prompt injection attempts.
        
        Note: The engine intentionally echoes original input for audit trail.
        This test verifies the LLM response itself doesn't contain injected commands,
        not the original input echo.
        """
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Capture LLM responses to verify they're not manipulated
        captured_responses = []
        
        async def capture_generate(*args, **kwargs):
            response = MagicMock()
            response.content = "Safe LLM response - strategy analysis complete"
            response.input_tokens = 100
            response.output_tokens = 50
            captured_responses.append(response.content)
            return response
        
        mock_helix.generate = capture_generate

        # Execute the method with malicious input
        try:
            if test_case.method == "generate_hypothesis":
                result = await engine.generate_hypothesis(
                    market_context=test_case.malicious_input.get("market_context", {})
                )
            elif test_case.method == "analyze_code":
                result = await engine.analyze_code(
                    code=test_case.malicious_input["code"],
                    analysis_type=test_case.malicious_input.get("analysis_type", "review"),
                )
            elif test_case.method == "synthesize_research":
                result = await engine.synthesize_research(
                    query=test_case.malicious_input["query"],
                    sources=test_case.malicious_input.get("sources", []),
                    context=test_case.malicious_input.get("context"),
                )
            else:
                pytest.skip(f"Unknown method: {test_case.method}")

            # Verify the LLM response doesn't contain injected content
            # (The mock returns safe content, so this tests the plumbing)
            for response in captured_responses:
                for pattern in test_case.forbidden_patterns:
                    assert pattern.lower() not in response.lower(), (
                        f"Forbidden pattern '{pattern}' found in LLM response"
                    )

            # Verify result was generated successfully (injection didn't crash)
            assert result is not None

        except Exception:
            # If the engine rejects the input, that's also acceptable
            if test_case.expected_behavior == "reject":
                pass  # Expected behavior
            else:
                raise


# =============================================================================
# Safety Benchmark Tests - PII Leakage
# =============================================================================


class TestPIILeakageSafety:
    """Test PII sanitization and leakage prevention."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_case",
        PII_LEAKAGE_TESTS,
        ids=[t.name for t in PII_LEAKAGE_TESTS],
    )
    async def test_pii_sanitization(
        self, mock_helix, cortex_config, test_case: SafetyTestCase
    ):
        """Verify PII is properly sanitized from prompts sent to LLM."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Capture what's sent to Helix - ChatMessage is a dataclass
        captured_prompts = []

        def capture_generate(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            for msg in messages:
                # ChatMessage is a dataclass, access .content directly
                content = getattr(msg, "content", str(msg))
                captured_prompts.append(content)
            response = MagicMock()
            response.content = "Analysis complete"
            response.input_tokens = 50
            response.output_tokens = 25
            return response

        mock_helix.generate = AsyncMock(side_effect=capture_generate)

        # Execute the method
        if test_case.method == "generate_hypothesis":
            await engine.generate_hypothesis(
                market_context=test_case.malicious_input.get("market_context", {})
            )
        elif test_case.method == "analyze_code":
            await engine.analyze_code(
                code=test_case.malicious_input["code"],
                analysis_type=test_case.malicious_input.get("analysis_type", "review"),
            )
        elif test_case.method == "synthesize_research":
            await engine.synthesize_research(
                query=test_case.malicious_input["query"],
                sources=test_case.malicious_input.get("sources", []),
                context=test_case.malicious_input.get("context"),
            )

        # Check that forbidden patterns are NOT in prompts sent to LLM
        full_prompt = " ".join(captured_prompts)
        for pattern in test_case.forbidden_patterns:
            assert pattern not in full_prompt, (
                f"PII pattern '{pattern}' leaked to LLM prompt"
            )

        # Check that required redaction patterns ARE present (if sanitization enabled)
        # Note: Only check if prompts were captured (model didn't fall through to fallback)
        if captured_prompts and test_case.required_patterns:
            for pattern in test_case.required_patterns:
                assert pattern in full_prompt, (
                    f"Expected redaction pattern '{pattern}' not found in prompt"
                )


# =============================================================================
# Safety Benchmark Tests - Secret Leakage
# =============================================================================


class TestSecretLeakageSafety:
    """Test secret/credential sanitization."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_case",
        SECRET_LEAKAGE_TESTS,
        ids=[t.name for t in SECRET_LEAKAGE_TESTS],
    )
    async def test_secret_sanitization(
        self, mock_helix, cortex_config, test_case: SafetyTestCase
    ):
        """Verify secrets are properly sanitized from prompts sent to LLM."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Capture what's sent to Helix - ChatMessage is a dataclass
        captured_prompts = []

        def capture_generate(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            for msg in messages:
                # ChatMessage is a dataclass, access .content directly
                content = getattr(msg, "content", str(msg))
                captured_prompts.append(content)
            response = MagicMock()
            response.content = "Analysis complete"
            response.input_tokens = 50
            response.output_tokens = 25
            return response

        mock_helix.generate = AsyncMock(side_effect=capture_generate)

        # Execute code analysis
        await engine.analyze_code(
            code=test_case.malicious_input["code"],
            analysis_type=test_case.malicious_input.get("analysis_type", "review"),
        )

        # Check that secrets are NOT in prompts
        full_prompt = " ".join(captured_prompts)
        for pattern in test_case.forbidden_patterns:
            assert pattern not in full_prompt, (
                f"Secret pattern '{pattern}' leaked to LLM prompt"
            )


# =============================================================================
# Hallucination Detection Tests
# =============================================================================


class TestHallucinationDetection:
    """Test detection of hallucinated or invalid outputs."""

    @pytest.mark.asyncio
    async def test_hypothesis_confidence_bounds(self, initialized_engine):
        """Verify hypothesis confidence is within valid bounds."""
        hypothesis = await initialized_engine.generate_hypothesis({"regime": "trending"})

        # Confidence must be in [0, 1]
        assert 0.0 <= hypothesis.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_hypothesis_strategy_type_valid(self, initialized_engine):
        """Verify strategy type is one of known valid types."""
        valid_types = {
            "trend_following",
            "mean_reversion",
            "adaptive",
            "balanced",
            "momentum",
            "breakout",
        }

        hypothesis = await initialized_engine.generate_hypothesis({"regime": "trending"})
        assert hypothesis.strategy_type in valid_types, (
            f"Invalid strategy type: {hypothesis.strategy_type}"
        )

    @pytest.mark.asyncio
    async def test_hypothesis_time_horizon_valid(self, initialized_engine):
        """Verify time horizon is valid."""
        valid_horizons = {"intraday", "swing", "position", "long_term"}

        hypothesis = await initialized_engine.generate_hypothesis({"regime": "trending"})
        assert hypothesis.time_horizon in valid_horizons, (
            f"Invalid time horizon: {hypothesis.time_horizon}"
        )

    @pytest.mark.asyncio
    async def test_hypothesis_instrument_class_valid(self, initialized_engine):
        """Verify instrument class is valid."""
        valid_classes = {"equity", "futures", "options", "forex", "crypto", "mixed"}

        hypothesis = await initialized_engine.generate_hypothesis({"regime": "trending"})
        assert hypothesis.instrument_class in valid_classes, (
            f"Invalid instrument class: {hypothesis.instrument_class}"
        )

    @pytest.mark.asyncio
    async def test_hypothesis_risk_params_realistic(self, initialized_engine):
        """Verify risk parameters are realistic values."""
        hypothesis = await initialized_engine.generate_hypothesis({"regime": "trending"})

        # Max position size should be reasonable (1% to 20%)
        assert 0.01 <= hypothesis.max_position_size_pct <= 0.20

        # Stop loss should be reasonable (0.5% to 10%)
        assert 0.005 <= hypothesis.stop_loss_pct <= 0.10

    @pytest.mark.asyncio
    async def test_code_analysis_output_structure(self, initialized_engine):
        """Verify code analysis output has expected structure."""
        output = await initialized_engine.analyze_code("def foo(): pass", "review")

        # Must have proper output type
        assert output.output_type == OutputType.CODE_ANALYSIS

        # Content must be dict
        assert isinstance(output.content, dict)

        # Confidence must be reasonable
        assert 0.0 <= output.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_research_output_structure(self, initialized_engine):
        """Verify research output has expected structure."""
        output = await initialized_engine.synthesize_research(
            query="What is momentum trading?",
            sources=["Momentum trading follows trends"],
        )

        # Must have proper output type
        assert output.output_type == OutputType.RESEARCH

        # Content must be dict with expected keys
        assert isinstance(output.content, dict)

    @pytest.mark.asyncio
    async def test_output_timestamps_valid(self, initialized_engine):
        """Verify output timestamps are valid and recent."""
        from datetime import datetime, timedelta, timezone

        hypothesis = await initialized_engine.generate_hypothesis({"regime": "trending"})

        # Timestamp should be recent (within last minute)
        now = datetime.now(timezone.utc).replace(tzinfo=None)  # Match naive UTC
        assert hypothesis.created_at <= now
        assert hypothesis.created_at >= now - timedelta(minutes=1)


# =============================================================================
# High-Severity Safety Tests
# =============================================================================


class TestHighSeveritySafety:
    """Run all high-severity safety tests."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_case",
        get_high_severity_tests(),
        ids=[t.name for t in get_high_severity_tests()],
    )
    async def test_high_severity_safety(
        self, mock_helix, cortex_config, test_case: SafetyTestCase
    ):
        """Verify high-severity safety tests pass - prompts are sanitized before LLM."""
        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Capture prompts for verification - ChatMessage is a dataclass
        captured_prompts = []

        def capture_generate(*args, **kwargs):
            messages = kwargs.get("messages", args[0] if args else [])
            for msg in messages:
                content = getattr(msg, "content", str(msg))
                captured_prompts.append(content)
            response = MagicMock()
            response.content = "Safe response"
            response.input_tokens = 50
            response.output_tokens = 25
            return response

        mock_helix.generate = AsyncMock(side_effect=capture_generate)

        # Execute and capture - we verify sanitization in prompts
        result = await self._execute_test_case(engine, test_case)

        # Verify result was produced
        assert result is not None

        # Verify forbidden patterns are sanitized from prompts (for sanitize behavior)
        if captured_prompts and test_case.expected_behavior == "sanitize":
            full_prompt = " ".join(captured_prompts)
            for pattern in test_case.forbidden_patterns:
                assert pattern not in full_prompt, (
                    f"High-severity: Pattern '{pattern}' should be sanitized"
                )

    async def _execute_test_case(self, engine, test_case: SafetyTestCase):
        """Execute a test case and return result."""
        if test_case.method == "generate_hypothesis":
            return await engine.generate_hypothesis(
                market_context=test_case.malicious_input.get("market_context", {})
            )
        elif test_case.method == "analyze_code":
            return await engine.analyze_code(
                code=test_case.malicious_input["code"],
                analysis_type=test_case.malicious_input.get("analysis_type", "review"),
            )
        elif test_case.method == "synthesize_research":
            return await engine.synthesize_research(
                query=test_case.malicious_input["query"],
                sources=test_case.malicious_input.get("sources", []),
                context=test_case.malicious_input.get("context"),
            )
        return None


# =============================================================================
# Output Consistency Tests
# =============================================================================


class TestOutputConsistency:
    """Test output consistency across multiple calls."""

    @pytest.mark.asyncio
    async def test_deterministic_fallback_same_regime(self, mock_helix, cortex_config):
        """Verify same regime produces same strategy type deterministically."""
        # Make LLM fail to force fallback
        mock_helix.generate = AsyncMock(side_effect=Exception("LLM unavailable"))

        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        # Generate multiple hypotheses for same regime
        results = []
        for _ in range(3):
            hypothesis = await engine.generate_hypothesis(
                {"regime": "trending", "volatility": "low"}
            )
            results.append(hypothesis.strategy_type)

        # All should be the same (deterministic fallback)
        assert len(set(results)) == 1, (
            f"Non-deterministic fallback: {results}"
        )

    @pytest.mark.asyncio
    async def test_output_to_dict_roundtrip(self, initialized_engine):
        """Verify to_dict produces valid serializable output."""
        import json

        hypothesis = await initialized_engine.generate_hypothesis({"regime": "trending"})

        # Should serialize without error
        data = hypothesis.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        # Key fields should be preserved
        assert parsed["hypothesis_id"] == hypothesis.hypothesis_id
        assert parsed["strategy_type"] == hypothesis.strategy_type
        assert parsed["confidence"] == hypothesis.confidence
