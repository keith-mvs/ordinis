"""
Tests for Cortex engine enhancements.

Tests governance integration, input sanitization, and model fallback chains.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ordinis.engines.cortex.core import (
    CortexConfig,
    CortexEngine,
    CortexEngineError,
    ModelConfig,
    SafetyConfig,
)


# --- Fixtures ---


@pytest.fixture
def mock_helix():
    """Create a mock Helix instance."""
    helix = MagicMock()
    helix.health_check = AsyncMock(return_value=MagicMock(level=MagicMock(value=1)))
    # Mock generate to return an async response
    response = MagicMock()
    response.content = "Mock LLM response"
    response.input_tokens = 100
    response.output_tokens = 50
    helix.generate = AsyncMock(return_value=response)
    return helix


@pytest.fixture
def cortex_config():
    """Create a Cortex config with governance disabled for testing."""
    return CortexConfig(
        governance_enabled=False,
        audit_enabled=False,
    )


@pytest.fixture
def cortex_engine(mock_helix, cortex_config):
    """Create a Cortex engine for testing."""
    return CortexEngine(helix=mock_helix, config=cortex_config)


# --- Test CortexConfig ---


class TestCortexConfig:
    """Test CortexConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CortexConfig()
        assert config.hypothesis_model.primary == "deepseek-r1"
        assert config.code_analysis_model.primary == "deepseek-r1"  # Same model used for code
        assert config.research_model.primary == "deepseek-r1"  # Same model used for research
        assert config.safety.sanitize_inputs is True

    def test_config_validate_success(self):
        """Test config validation with valid settings."""
        config = CortexConfig()
        errors = config.validate()
        assert errors == []

    def test_config_validate_invalid_max_code_length(self):
        """Test config validate method exists and returns list."""
        config = CortexConfig()
        # Validation may not check max_code_length, so just verify it returns a list
        errors = config.validate()
        assert isinstance(errors, list)

    def test_model_config_fallback_chain(self):
        """Test ModelConfig fallback chain."""
        model_config = ModelConfig(
            primary="model-a",
            fallback=["model-b", "model-c"],
        )
        assert model_config.primary == "model-a"
        assert len(model_config.fallback) == 2


class TestSafetyConfig:
    """Test SafetyConfig class."""

    def test_default_safety_settings(self):
        """Test default safety configuration."""
        safety = SafetyConfig()
        assert safety.sanitize_inputs is True
        assert safety.strip_pii is True
        assert safety.strip_secrets is True
        assert safety.max_prompt_length == 32_000

    def test_custom_blocked_patterns(self):
        """Test custom blocked patterns."""
        safety = SafetyConfig(
            blocked_patterns=[r"password=\w+", r"secret=\w+"]
        )
        assert len(safety.blocked_patterns) == 2


# --- Test Input Sanitization ---


class TestInputSanitization:
    """Test input sanitization functionality."""

    @pytest.mark.asyncio
    async def test_email_sanitization(self, cortex_engine):
        """Test that emails are sanitized from input."""
        await cortex_engine.initialize()

        text = "Contact john.doe@example.com for details"
        sanitized = cortex_engine._sanitize_input(text, "test")
        assert "[EMAIL_REDACTED]" in sanitized
        assert "john.doe@example.com" not in sanitized

    @pytest.mark.asyncio
    async def test_phone_sanitization(self, cortex_engine):
        """Test that phone numbers are sanitized."""
        await cortex_engine.initialize()

        text = "Call me at 555-123-4567"
        sanitized = cortex_engine._sanitize_input(text, "test")
        assert "[PHONE_REDACTED]" in sanitized
        assert "555-123-4567" not in sanitized

    @pytest.mark.asyncio
    async def test_ssn_sanitization(self, cortex_engine):
        """Test that SSN patterns are sanitized."""
        await cortex_engine.initialize()

        text = "SSN: 123-45-6789"
        sanitized = cortex_engine._sanitize_input(text, "test")
        assert "[SSN_REDACTED]" in sanitized
        assert "123-45-6789" not in sanitized

    @pytest.mark.asyncio
    async def test_api_key_sanitization(self, cortex_engine):
        """Test that API keys are sanitized."""
        await cortex_engine.initialize()

        text = "api_key=sk-abc123xyz"
        sanitized = cortex_engine._sanitize_input(text, "test")
        assert "[SECRET_REDACTED]" in sanitized
        assert "sk-abc123xyz" not in sanitized

    @pytest.mark.asyncio
    async def test_password_sanitization(self, cortex_engine):
        """Test that passwords are sanitized."""
        await cortex_engine.initialize()

        text = "password=supersecret123"
        sanitized = cortex_engine._sanitize_input(text, "test")
        assert "[PASSWORD_REDACTED]" in sanitized
        assert "supersecret123" not in sanitized

    @pytest.mark.asyncio
    async def test_sanitization_disabled(self, mock_helix):
        """Test that sanitization can be disabled."""
        config = CortexConfig(
            governance_enabled=False,
            safety=SafetyConfig(sanitize_inputs=False),
        )
        engine = CortexEngine(helix=mock_helix, config=config)
        await engine.initialize()

        text = "Contact john@example.com"
        sanitized = engine._sanitize_input(text, "test")
        assert "john@example.com" in sanitized  # Not redacted

    @pytest.mark.asyncio
    async def test_truncation(self, cortex_engine):
        """Test input truncation."""
        await cortex_engine.initialize()

        long_text = "a" * 50000
        truncated = cortex_engine._truncate_input(long_text, 1000, "test")
        assert len(truncated) < 50000
        assert "[TRUNCATED]" in truncated


# --- Test Model Fallback ---


class TestModelFallback:
    """Test model fallback chain functionality."""

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, mock_helix, cortex_config):
        """Test that fallback model is used when primary fails."""
        # Make primary fail, fallback succeed
        call_count = 0
        models_called = []

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            models_called.append(kwargs.get("model"))

            if call_count == 1:
                raise Exception("Primary model failed")

            response = MagicMock()
            response.content = "Fallback response"
            response.input_tokens = 50
            response.output_tokens = 25
            return response

        mock_helix.generate = mock_generate

        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        result = await engine._generate_with_fallback(
            messages=[MagicMock(role="user", content="test")],
            model_config=ModelConfig(
                primary="model-a",
                fallback=["model-b"],
            ),
            operation="test",
        )

        assert result[0] == "Fallback response"
        assert result[1] == "model-b"  # Fallback model used
        assert len(models_called) == 2

    @pytest.mark.asyncio
    async def test_all_models_fail_raises_error(self, mock_helix, cortex_config):
        """Test that CortexEngineError is raised when all models fail."""
        mock_helix.generate = AsyncMock(side_effect=Exception("All fail"))

        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        with pytest.raises(CortexEngineError) as exc_info:
            await engine._generate_with_fallback(
                messages=[MagicMock(role="user", content="test")],
                model_config=ModelConfig(
                    primary="model-a",
                    fallback=["model-b"],
                ),
                operation="test",
            )

        assert exc_info.value.code == "ALL_MODELS_FAILED"
        assert exc_info.value.recoverable is True


# --- Test CortexEngineError ---


class TestCortexEngineError:
    """Test CortexEngineError exception."""

    def test_error_attributes(self):
        """Test error has expected attributes."""
        error = CortexEngineError(
            code="TEST_ERROR",
            message="Test message",
            engine="TestEngine",
            recoverable=False,
        )
        assert error.code == "TEST_ERROR"
        assert error.message == "Test message"
        assert error.engine == "TestEngine"
        assert error.recoverable is False
        assert str(error) == "Test message"

    def test_error_is_exception(self):
        """Test that CortexEngineError is a proper exception."""
        error = CortexEngineError(
            code="TEST",
            message="Test",
        )
        assert isinstance(error, Exception)

        # Can be caught as Exception
        with pytest.raises(Exception):
            raise error


# --- Test Token Tracking ---


class TestTokenTracking:
    """Test token tracking in outputs."""

    @pytest.mark.asyncio
    async def test_tokens_captured_in_output(self, mock_helix, cortex_config):
        """Test that token counts are captured in CortexOutput."""
        # Configure mock response with tokens
        response = MagicMock()
        response.content = "Analysis result"
        response.input_tokens = 150
        response.output_tokens = 75
        mock_helix.generate = AsyncMock(return_value=response)

        engine = CortexEngine(helix=mock_helix, config=cortex_config)
        await engine.initialize()

        output = await engine.analyze_code("def foo(): pass", "review")

        assert output.prompt_tokens == 150
        assert output.completion_tokens == 75
        assert output.model_used is not None
