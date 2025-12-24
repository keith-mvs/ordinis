import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from ordinis.engines.signalcore.core.engine import SignalCoreEngine, SignalCoreEngineConfig
from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Signal, SignalType, Direction, SignalBatch
from ordinis.engines.base import HealthLevel

# -------------------------------------------------------------------------
# Mocks and Fixtures
# -------------------------------------------------------------------------

class MockModel(Model):
    async def generate(self, symbol: str, data: pd.DataFrame, timestamp: datetime) -> Signal:
        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            probability=0.8,
            score=0.8,
            model_id=self.config.model_id
        )

@pytest.fixture
def mock_registry():
    with patch("ordinis.engines.signalcore.core.engine.ModelRegistry") as mock:
        registry_instance = mock.return_value
        registry_instance.list_models.return_value = []
        registry_instance.generate_all = AsyncMock()
        yield registry_instance

@pytest.fixture
def engine(mock_registry):
    config = SignalCoreEngineConfig(enable_governance=False)
    engine = SignalCoreEngine(config)
    engine._registry = mock_registry  # Inject mock registry directly
    return engine

@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.5] * 10,
        "volume": [1000] * 10
    })
    return {"AAPL": df}

# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_initialization(engine):
    await engine.initialize()
    assert engine._signals_generated == 0
    assert engine._last_generation is None

@pytest.mark.asyncio
async def test_health_check_no_models(engine, mock_registry):
    mock_registry.list_models.side_effect = lambda enabled_only=False: []
    status = await engine.check_health()
    assert status.level == HealthLevel.DEGRADED
    assert "No enabled models" in status.message

@pytest.mark.asyncio
async def test_health_check_healthy(engine, mock_registry):
    mock_registry.list_models.side_effect = lambda enabled_only=False: ["m1"] if enabled_only else ["m1"]
    status = await engine.check_health()
    assert status.level == HealthLevel.HEALTHY
    assert "operational" in status.message

def test_model_registration(engine, mock_registry):
    model = MockModel(ModelConfig(model_id="test_model", model_type="mock"))
    
    engine.register_model(model)
    mock_registry.register.assert_called_with(model)
    assert "test_model" in engine.config.registered_models

    engine.unregister_model("test_model")
    mock_registry.unregister.assert_called_with("test_model")
    assert "test_model" not in engine.config.registered_models

def test_get_model(engine, mock_registry):
    engine.get_model("m1")
    mock_registry.get.assert_called_with("m1")

def test_list_models(engine, mock_registry):
    engine.list_models(enabled_only=True)
    mock_registry.list_models.assert_called_with(enabled_only=True)

@pytest.mark.asyncio
async def test_generate_signal_success(engine, mock_registry, sample_data):
    # Setup mock model
    model = MockModel(ModelConfig(model_id="m1", model_type="mock"))
    mock_registry.get.return_value = model
    mock_registry.list_models.return_value = ["m1"]

    df = sample_data["AAPL"]
    signal = await engine.generate_signal("AAPL", df)
    
    assert signal is not None
    assert signal.symbol == "AAPL"
    assert engine._signals_generated == 1

@pytest.mark.asyncio
async def test_generate_signal_validation_failure(engine, mock_registry, sample_data):
    # Setup mock model that fails validation
    model = MagicMock(spec=Model)
    model.config = ModelConfig(model_id="m1", model_type="mock")
    model.validate.return_value = (False, "Invalid data")
    mock_registry.get.return_value = model
    mock_registry.list_models.return_value = ["m1"]

    df = sample_data["AAPL"]
    signal = await engine.generate_signal("AAPL", df)
    
    assert signal is None
    assert engine._signals_generated == 0

@pytest.mark.asyncio
async def test_generate_signal_execution_failure(engine, mock_registry, sample_data):
    # Setup mock model that raises exception
    model = MockModel(ModelConfig(model_id="m1", model_type="mock"))
    # Patch generate to raise exception
    model.generate = AsyncMock(side_effect=ValueError("Model crashed"))
    mock_registry.get.return_value = model
    mock_registry.list_models.return_value = ["m1"]

    df = sample_data["AAPL"]
    signal = await engine.generate_signal("AAPL", df)
    
    assert signal is None
    # Assuming exception handling logs error but doesn't crash engine

@pytest.mark.asyncio
async def test_generate_batch_success(engine, mock_registry, sample_data):
    # Setup mock registry response
    dt = datetime.now(timezone.utc)
    s1 = Signal(symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, direction=Direction.LONG, probability=0.8, score=0.8, model_id="m1")
    batch_response = SignalBatch(timestamp=dt, signals=[s1], universe=["AAPL"])
    mock_registry.generate_all.return_value = batch_response

    batch = await engine.generate_batch(sample_data)
    
    assert len(batch.signals) == 1
    assert batch.signals[0].symbol == "AAPL"
    assert engine._signals_generated == 1

@pytest.mark.asyncio
async def test_generate_batch_feedback(engine, mock_registry, sample_data):
    # Setup feedback mock
    engine._feedback = MagicMock()
    engine._feedback.should_allow_signals.return_value = (True, "OK")
    engine._feedback.record_signal_batch = AsyncMock()

    # Setup mock registry response
    dt = datetime.now(timezone.utc)
    s1 = Signal(symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, direction=Direction.LONG, probability=0.8, score=0.8, model_id="m1")
    batch_response = SignalBatch(timestamp=dt, signals=[s1], universe=["AAPL"])
    mock_registry.generate_all.return_value = batch_response

    await engine.generate_batch(sample_data)
    
    engine._feedback.should_allow_signals.assert_called()
    engine._feedback.record_signal_batch.assert_called()

@pytest.mark.asyncio
async def test_generate_batch_circuit_breaker(engine, sample_data):
    # Setup feedback mock to block signals
    engine._feedback = MagicMock()
    engine._feedback.should_allow_signals.return_value = (False, "Circuit open")

    batch = await engine.generate_batch(sample_data)
    
    assert len(batch.signals) == 0

def test_filter_actionable(engine):
    dt = datetime.now(timezone.utc)
    # Actionable signal
    s1 = Signal(symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, direction=Direction.LONG, probability=0.8, score=0.8)
    # Non-actionable signal (low prob)
    s2 = Signal(symbol="GOOGL", timestamp=dt, signal_type=SignalType.ENTRY, direction=Direction.LONG, probability=0.4, score=0.8)
    
    batch = SignalBatch(timestamp=dt, signals=[s1, s2], universe=["AAPL", "GOOGL"])
    
    filtered = engine.filter_actionable(batch)
    assert len(filtered) == 1
    assert filtered[0].symbol == "AAPL"

def test_get_metrics(engine, mock_registry):
    engine._signals_generated = 10
    mock_registry.list_models.side_effect = lambda enabled_only=False: ["m1"] if enabled_only else ["m1", "m2"]
    
    metrics = engine.get_metrics()
    assert metrics.custom_metrics["signals_generated"] == 10
    assert metrics.custom_metrics["models_enabled"] == 1
    assert metrics.custom_metrics["models_registered"] == 2

@pytest.mark.asyncio
async def test_generate_signals_protocol(engine, mock_registry, sample_data):
    # Test generate_signals (protocol implementation)
    dt = datetime.now(timezone.utc)
    s1 = Signal(symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, direction=Direction.LONG, probability=0.8, score=0.8, model_id="m1")
    batch_response = SignalBatch(timestamp=dt, signals=[s1], universe=["AAPL"])
    mock_registry.generate_all.return_value = batch_response

    signals = await engine.generate_signals(sample_data)
    assert len(signals) == 1
    assert signals[0] == s1

    # Invalid input
    signals = await engine.generate_signals("invalid input")
    assert signals == []
