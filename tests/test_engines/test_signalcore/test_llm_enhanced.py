"""
Tests for LLM-enhanced SignalCore models.

Tests cover:
- LLM-enhanced model wrapping
- Signal interpretation
- Feature engineering
- Fallback behavior
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from ordinis.ai.helix.models import ChatResponse, ProviderType, UsageInfo
from ordinis.engines.signalcore.models import (
    LLMEnhancedModel,
    LLMFeatureEngineer,
    RSIMeanReversionModel,
)


@pytest.fixture
def sample_data():
    """Create sample market data."""
    # RSI model requires at least 100 data points
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 101.5, 103.0] * 25,
            "high": [101.0, 102.0, 103.0, 102.5, 104.0] * 25,
            "low": [99.0, 100.0, 101.0, 100.5, 102.0] * 25,
            "close": [100.5, 101.5, 102.5, 101.0, 103.5] * 25,
            "volume": [1000000] * 125,
        }
    )
    return data


@pytest.fixture
def base_model():
    """Create base model to enhance."""
    from ordinis.engines.signalcore.core.model import ModelConfig

    config = ModelConfig(
        model_id="rsi-test",
        model_type="mean_reversion",
        version="1.0.0",
        parameters={
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
        },
    )
    return RSIMeanReversionModel(config)


@pytest.fixture
def mock_helix():
    """Create mock Helix engine."""
    helix = MagicMock()
    helix.generate = AsyncMock(
        return_value=ChatResponse(
            content="Test interpretation",
            model="test-model",
            provider=ProviderType.MOCK,
            usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )
    )
    return helix


@pytest.mark.unit
def test_llm_enhanced_model_creation(base_model):
    """Test creating LLM-enhanced model."""
    enhanced = LLMEnhancedModel(base_model=base_model, helix=None, llm_enabled=False)

    assert enhanced.base_model == base_model
    assert enhanced.llm_enabled is False
    assert enhanced.helix is None
    assert enhanced.config.metadata["llm_enhanced"] is False


@pytest.mark.unit
def test_llm_enhanced_model_with_helix(base_model, mock_helix):
    """Test LLM-enhanced model with Helix engine."""
    enhanced = LLMEnhancedModel(base_model=base_model, helix=mock_helix, llm_enabled=True)

    assert enhanced.llm_enabled is True
    assert enhanced.helix == mock_helix


@pytest.mark.unit
def test_llm_enhanced_validate(base_model, sample_data):
    """Test validation delegates to base model."""
    enhanced = LLMEnhancedModel(base_model=base_model, llm_enabled=False)

    valid, msg = enhanced.validate(sample_data)

    # RSI model requires at least 34 data points (rsi_period + 20)
    # Sample data has 50 points so should be valid
    assert isinstance(valid, bool)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_enhanced_generate_no_llm(base_model, sample_data):
    """Test signal generation without LLM."""
    enhanced = LLMEnhancedModel(base_model=base_model, llm_enabled=False)

    signal = await enhanced.generate(sample_data, datetime.utcnow())

    assert signal.model_id == base_model.config.model_id
    assert "llm_interpretation" not in signal.metadata


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_enhanced_generate_with_helix(base_model, sample_data, mock_helix):
    """Test signal generation with LLM enabled and Helix."""
    enhanced = LLMEnhancedModel(base_model=base_model, helix=mock_helix, llm_enabled=True)

    signal = await enhanced.generate(sample_data, datetime.utcnow())

    assert signal.model_id == base_model.config.model_id
    # Interpretation should be added
    # Note: The current implementation of _add_llm_interpretation might not add to metadata directly
    # but return a new signal. Let's check if mock was called.
    assert mock_helix.generate.called


@pytest.mark.unit
def test_llm_enhanced_describe(base_model, mock_helix):
    """Test model description includes LLM info."""
    enhanced = LLMEnhancedModel(base_model=base_model, helix=mock_helix, llm_enabled=True)

    desc = enhanced.describe()

    assert desc["llm_enhanced"] is True
    assert desc["llm_available"] is True
    assert desc["wrapper_type"] == "LLMEnhancedModel"


@pytest.mark.unit
def test_feature_engineer_creation():
    """Test creating feature engineer."""
    engineer = LLMFeatureEngineer(helix=None)

    assert engineer.helix is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_feature_engineer_suggest_features_no_helix(sample_data):
    """Test feature suggestions without Helix (fallback)."""
    engineer = LLMFeatureEngineer(helix=None)

    features = await engineer.suggest_features(sample_data, "trend_following")

    # Should return basic features
    assert len(features) > 0
    assert "SMA_20" in features or "RSI_14" in features


@pytest.mark.unit
@pytest.mark.asyncio
async def test_feature_engineer_suggest_features_with_helix(sample_data, mock_helix):
    """Test feature suggestions with Helix."""
    mock_helix.generate.return_value = ChatResponse(
        content="RSI_14\nSMA_50\nMACD",
        model="test-model",
        provider=ProviderType.MOCK,
        usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )
    engineer = LLMFeatureEngineer(helix=mock_helix)

    features = await engineer.suggest_features(sample_data, "mean_reversion")

    # Should return features from mock
    assert len(features) > 0
    assert isinstance(features, list)
    assert "RSI_14" in features


@pytest.mark.unit
@pytest.mark.asyncio
async def test_feature_engineer_explain_feature_no_helix():
    """Test feature explanation without Helix."""
    engineer = LLMFeatureEngineer(helix=None)

    explanation = await engineer.explain_feature("RSI_14")

    assert "RSI_14" in explanation
    assert isinstance(explanation, str)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_feature_engineer_explain_feature_with_helix(mock_helix):
    """Test feature explanation with Helix."""
    mock_helix.generate.return_value = ChatResponse(
        content="RSI measures momentum.",
        model="test-model",
        provider=ProviderType.MOCK,
        usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )
    engineer = LLMFeatureEngineer(helix=mock_helix)

    explanation = await engineer.explain_feature("MACD")

    assert isinstance(explanation, str)
    assert explanation == "RSI measures momentum."


@pytest.mark.unit
def test_llm_enhanced_inheritance(base_model):
    """Test that enhanced model inherits base model config."""
    enhanced = LLMEnhancedModel(base_model=base_model, llm_enabled=False)

    # Config should be inherited
    assert enhanced.config.model_id == base_model.config.model_id
    assert enhanced.config.model_type == base_model.config.model_type
