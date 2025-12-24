import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from ordinis.engines.signalcore.core.signal import (
    Signal, 
    SignalBatch, 
    SignalType, 
    Direction
)
from ordinis.engines.signalcore.core.model import (
    Model, 
    ModelConfig, 
    ModelRegistry
)

# -------------------------------------------------------------------------
# Test Signal
# -------------------------------------------------------------------------

def test_signal_initialization_valid():
    """Test valid signal initialization."""
    dt = datetime.now(timezone.utc)
    s = Signal(
        symbol="AAPL",
        timestamp=dt,
        signal_type=SignalType.ENTRY,
        direction=Direction.LONG,
        probability=0.75,
        score=0.5
    )
    assert s.symbol == "AAPL"
    assert s.probability == 0.75
    assert s.score == 0.5
    assert s.is_actionable()

def test_signal_validation_errors():
    """Test signal validation logic in __post_init__."""
    dt = datetime.now(timezone.utc)
    
    # Invalid probability > 1.0
    with pytest.raises(ValueError, match="Probability must be in"):
        Signal(
            symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, 
            direction=Direction.LONG, probability=1.5
        )

    # Invalid probability < 0.0
    with pytest.raises(ValueError, match="Probability must be in"):
        Signal(
            symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, 
            direction=Direction.LONG, probability=-0.1
        )

    # Invalid score > 1.0
    with pytest.raises(ValueError, match="Score must be in"):
        Signal(
            symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, 
            direction=Direction.LONG, score=1.5
        )

    # Invalid confidence interval
    with pytest.raises(ValueError, match="Invalid confidence interval"):
        Signal(
            symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, 
            direction=Direction.LONG, confidence_interval=(0.5, 0.4)
        )

def test_signal_is_actionable():
    """Test is_actionable logic."""
    dt = datetime.now(timezone.utc)
    
    # Strong signal
    s1 = Signal(
        symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, 
        direction=Direction.LONG, probability=0.8, score=0.5
    )
    assert s1.is_actionable(min_probability=0.6, min_score=0.3)

    # Weak probability
    s2 = Signal(
        symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, 
        direction=Direction.LONG, probability=0.5, score=0.5
    )
    assert not s2.is_actionable(min_probability=0.6, min_score=0.3)

    # Weak score
    s3 = Signal(
        symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, 
        direction=Direction.LONG, probability=0.8, score=0.1
    )
    assert not s3.is_actionable(min_probability=0.6, min_score=0.3)

    # HOLD signal is never actionable
    s4 = Signal(
        symbol="AAPL", timestamp=dt, signal_type=SignalType.HOLD, 
        direction=Direction.NEUTRAL, probability=0.9, score=0.9
    )
    assert not s4.is_actionable()

def test_signal_to_dict():
    """Test serialization of Signal."""
    dt = datetime.now(timezone.utc)
    s = Signal(
        symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, 
        direction=Direction.LONG
    )
    d = s.to_dict()
    assert d["symbol"] == "AAPL"
    assert d["signal_type"] == "entry"
    assert d["direction"] == "long"
    assert d["timestamp"] == dt.isoformat()

# -------------------------------------------------------------------------
# Test SignalBatch
# -------------------------------------------------------------------------

def test_signal_batch_operations():
    """Test SignalBatch methods."""
    dt = datetime.now(timezone.utc)
    s1 = Signal(symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, direction=Direction.LONG, probability=0.8, score=0.5)
    s2 = Signal(symbol="GOOGL", timestamp=dt, signal_type=SignalType.EXIT, direction=Direction.SHORT, probability=0.4, score=0.2)
    s3 = Signal(symbol="MSFT", timestamp=dt, signal_type=SignalType.HOLD, direction=Direction.NEUTRAL, probability=0.9, score=0.9)
    
    batch = SignalBatch(
        timestamp=dt,
        signals=[s1, s2, s3],
        universe=["AAPL", "GOOGL", "MSFT"]
    )

    # filter_actionable
    actionable = batch.filter_actionable(min_probability=0.6, min_score=0.3)
    assert len(actionable) == 1
    assert actionable[0].symbol == "AAPL"

    # get_by_symbol
    assert batch.get_by_symbol("AAPL") == s1
    assert batch.get_by_symbol("UNKNOWN") is None

    # get_entry/exit_signals
    assert len(batch.get_entry_signals()) == 1
    assert batch.get_entry_signals()[0].symbol == "AAPL"
    assert len(batch.get_exit_signals()) == 1
    assert batch.get_exit_signals()[0].symbol == "GOOGL"

    # to_dict
    d = batch.to_dict()
    assert len(d["signals"]) == 3
    assert d["universe"] == ["AAPL", "GOOGL", "MSFT"]

# -------------------------------------------------------------------------
# Test Model and Registry
# -------------------------------------------------------------------------

class MockModel(Model):
    """Concrete mock model for testing."""
    async def generate(self, symbol: str, data: pd.DataFrame, timestamp: datetime) -> Signal:
        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            model_id=self.config.model_id
        )

def test_model_base_functionality():
    """Test Model base class methods."""
    config = ModelConfig(
        model_id="test_model",
        model_type="mock",
        min_data_points=5
    )
    model = MockModel(config)
    
    # Test validate
    df_valid = pd.DataFrame({
        "open": [1]*10, "high": [2]*10, "low": [0.5]*10, "close": [1.5]*10, "volume": [100]*10
    })
    is_valid, msg = model.validate(df_valid)
    assert is_valid
    assert msg == "Valid"

    # Test validate insufficient data
    df_short = pd.DataFrame({
        "open": [1]*2, "high": [2]*2, "low": [0.5]*2, "close": [1.5]*2, "volume": [100]*2
    })
    is_valid, msg = model.validate(df_short)
    assert not is_valid
    assert "Insufficient data" in msg

    # Test validate missing columns
    df_missing = pd.DataFrame({"open": [1]*10})
    is_valid, msg = model.validate(df_missing)
    assert not is_valid
    assert "Missing columns" in msg

    # Test describe
    desc = model.describe()
    assert desc["model_id"] == "test_model"
    assert desc["model_type"] == "mock"

def test_model_registry_operations():
    """Test ModelRegistry registration and retrieval."""
    registry = ModelRegistry()
    config = ModelConfig(model_id="m1", model_type="type1")
    model = MockModel(config)

    # Register
    registry.register(model)
    assert registry.get("m1") == model
    assert "m1" in registry.list_models()

    # Duplicate registration
    with pytest.raises(ValueError):
        registry.register(model)

    # Get by type
    models = registry.get_by_type("type1")
    assert len(models) == 1
    assert models[0] == model

    # Unregister
    registry.unregister("m1")
    with pytest.raises(KeyError):
        registry.get("m1")

@pytest.mark.asyncio
async def test_registry_generate_all():
    """Test generating signals from all registered models."""
    registry = ModelRegistry()
    
    # Setup models
    m1 = MockModel(ModelConfig(model_id="m1", model_type="t1"))
    m2 = MockModel(ModelConfig(model_id="m2", model_type="t1"))
    m3 = MockModel(ModelConfig(model_id="m3", model_type="t1", enabled=False))
    
    registry.register(m1)
    registry.register(m2)
    registry.register(m3)

    # Setup data
    df = pd.DataFrame({
        "open": [1]*100, "high": [2]*100, "low": [0.5]*100, "close": [1.5]*100, "volume": [100]*100
    })
    data = {"AAPL": df}
    timestamp = datetime.now(timezone.utc)

    # Generate
    batch = await registry.generate_all(data, timestamp)
    
    # m3 is disabled, so we expect 2 signals (m1 for AAPL, m2 for AAPL)
    assert len(batch.signals) == 2
    model_ids = {s.model_id for s in batch.signals}
    assert "m1" in model_ids
    assert "m2" in model_ids
    assert "m3" not in model_ids

def test_persistence(tmp_path):
    """Test saving and loading models."""
    registry = ModelRegistry()
    config = ModelConfig(model_id="p1", model_type="persist")
    model = MockModel(config)
    registry.register(model)

    # Save
    registry.save_all(tmp_path)
    
    # Load into new registry
    new_registry = ModelRegistry()
    loaded_ids = new_registry.load_all(tmp_path, {"MockModel": MockModel})
    
    assert "p1" in loaded_ids
    loaded_model = new_registry.get("p1")
    assert loaded_model.config.model_id == "p1"
    assert loaded_model.config.model_type == "persist"
