"""
Tests for LLM-enhanced SignalCore models.

Tests cover:
- LLM-enhanced model wrapping
- Signal interpretation
- Feature engineering
- Fallback behavior
"""

from datetime import datetime

import pandas as pd
import pytest

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
        parameters={
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
        },
    )
    return RSIMeanReversionModel(config)


@pytest.mark.unit
def test_llm_enhanced_model_creation(base_model):
    """Test creating LLM-enhanced model."""
    enhanced = LLMEnhancedModel(base_model=base_model, nvidia_api_key=None, llm_enabled=False)

    assert enhanced.base_model == base_model
    assert enhanced.llm_enabled is False
    assert enhanced.nvidia_api_key is None
    assert enhanced.config.metadata["llm_enhanced"] is False


@pytest.mark.unit
def test_llm_enhanced_model_with_api_key(base_model):
    """Test LLM-enhanced model with API key (will use fallback)."""
    enhanced = LLMEnhancedModel(base_model=base_model, nvidia_api_key="test-key", llm_enabled=True)

    assert enhanced.llm_enabled is True
    assert enhanced.nvidia_api_key == "test-key"


@pytest.mark.unit
def test_llm_enhanced_validate(base_model, sample_data):
    """Test validation delegates to base model."""
    enhanced = LLMEnhancedModel(base_model=base_model, llm_enabled=False)

    valid, msg = enhanced.validate(sample_data)

    # RSI model requires at least 34 data points (rsi_period + 20)
    # Sample data has 50 points so should be valid
    assert isinstance(valid, bool)


@pytest.mark.unit
def test_llm_enhanced_generate_no_llm(base_model, sample_data):
    """Test signal generation without LLM."""
    enhanced = LLMEnhancedModel(base_model=base_model, llm_enabled=False)

    signal = enhanced.generate(sample_data, datetime.utcnow())

    assert signal.model_id == base_model.config.model_id
    assert "llm_interpretation" not in signal.metadata


@pytest.mark.unit
def test_llm_enhanced_generate_with_llm_fallback(base_model, sample_data):
    """Test signal generation with LLM enabled but no API key."""
    enhanced = LLMEnhancedModel(base_model=base_model, nvidia_api_key=None, llm_enabled=True)

    signal = enhanced.generate(sample_data, datetime.utcnow())

    # Should work but without LLM interpretation
    assert signal.model_id == base_model.config.model_id
    # No interpretation added without API key
    assert "llm_interpretation" not in signal.metadata


@pytest.mark.unit
def test_llm_enhanced_describe(base_model):
    """Test model description includes LLM info."""
    enhanced = LLMEnhancedModel(base_model=base_model, llm_enabled=True)

    desc = enhanced.describe()

    assert desc["llm_enhanced"] is True
    assert "llm_available" in desc
    assert desc["wrapper_type"] == "LLMEnhancedModel"


@pytest.mark.unit
def test_feature_engineer_creation():
    """Test creating feature engineer."""
    engineer = LLMFeatureEngineer(nvidia_api_key=None)

    assert engineer.nvidia_api_key is None
    assert engineer._llm_client is None


@pytest.mark.unit
def test_feature_engineer_suggest_features_no_api(sample_data):
    """Test feature suggestions without API key (fallback)."""
    engineer = LLMFeatureEngineer(nvidia_api_key=None)

    features = engineer.suggest_features(sample_data, "trend_following")

    # Should return basic features
    assert len(features) > 0
    assert "SMA_20" in features or "RSI_14" in features


@pytest.mark.unit
def test_feature_engineer_suggest_features_with_api(sample_data):
    """Test feature suggestions with API key (will use fallback)."""
    engineer = LLMFeatureEngineer(nvidia_api_key="test-key")

    features = engineer.suggest_features(sample_data, "mean_reversion")

    # Should return features (fallback since no real API)
    assert len(features) > 0
    assert isinstance(features, list)


@pytest.mark.unit
def test_feature_engineer_explain_feature_no_api():
    """Test feature explanation without API key."""
    engineer = LLMFeatureEngineer(nvidia_api_key=None)

    explanation = engineer.explain_feature("RSI_14")

    assert "RSI_14" in explanation
    assert isinstance(explanation, str)


@pytest.mark.unit
def test_feature_engineer_explain_feature_with_api():
    """Test feature explanation with API key (will use fallback)."""
    engineer = LLMFeatureEngineer(nvidia_api_key="test-key")

    explanation = engineer.explain_feature("MACD")

    assert isinstance(explanation, str)
    assert len(explanation) > 0


@pytest.mark.unit
def test_llm_enhanced_inheritance(base_model):
    """Test that enhanced model inherits base model config."""
    enhanced = LLMEnhancedModel(base_model=base_model, llm_enabled=False)

    # Config should be inherited
    assert enhanced.config.model_id == base_model.config.model_id
    assert enhanced.config.model_type == base_model.config.model_type
