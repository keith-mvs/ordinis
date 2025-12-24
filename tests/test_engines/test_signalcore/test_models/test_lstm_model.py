"""Tests for LSTM model."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType
from ordinis.engines.signalcore.models.lstm_model import LSTMModel, LSTMNet


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
    np.random.seed(42)
    
    prices = 100 + np.cumsum(np.random.randn(200) * 2)
    data = pd.DataFrame(
        {
            "open": prices + np.random.randn(200) * 0.5,
            "high": prices + np.abs(np.random.randn(200) * 1.5),
            "low": prices - np.abs(np.random.randn(200) * 1.5),
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, size=200),
            "symbol": ["AAPL"] * 200,
        },
        index=dates,
    )
    return data


@pytest.fixture
def lstm_config():
    """Create LSTM model config."""
    return ModelConfig(
        model_id="lstm_test",
        model_type="ml",
        version="1.0.0",
        parameters={
            "input_dim": 5,
            "hidden_dim": 32,
            "num_layers": 1,
            "output_dim": 1,
            "sequence_length": 30,
            "epochs": 2,  # Reduced for faster tests
            "batch_size": 16,
            "learning_rate": 0.001,
        },
        min_data_points=50,
    )


class TestLSTMNet:
    """Tests for LSTMNet architecture."""

    def test_init(self):
        """Test LSTMNet initialization."""
        model = LSTMNet(input_dim=5, hidden_dim=32, num_layers=2, output_dim=1)
        
        assert model.hidden_dim == 32
        assert model.num_layers == 2
        assert isinstance(model.lstm, torch.nn.LSTM)
        assert isinstance(model.fc, torch.nn.Linear)

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        model = LSTMNet(input_dim=5, hidden_dim=32, num_layers=2, output_dim=1)
        batch_size, seq_len, input_dim = 4, 30, 5
        
        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)
        
        assert output.shape == (batch_size, 1)

    def test_forward_with_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = LSTMNet(input_dim=5, hidden_dim=32, num_layers=2, output_dim=1)
        
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 30, 5)
            output = model(x)
            assert output.shape == (batch_size, 1)


class TestLSTMModelInit:
    """Tests for LSTMModel initialization."""

    def test_init_with_config(self, lstm_config):
        """Test initialization with config."""
        model = LSTMModel(lstm_config)
        
        assert model.config == lstm_config
        assert model.model is None  # Not trained yet
        assert model.device in [torch.device("cuda"), torch.device("cpu")]

    def test_init_default_config(self):
        """Test initialization with default config."""
        model = LSTMModel()
        
        assert model.config.model_id == "lstm_v1"
        assert model.config.model_type == "ml"
        assert model.config.parameters["input_dim"] == 5
        assert model.config.parameters["hidden_dim"] == 64


class TestLSTMModelPrepareData:
    """Tests for data preparation."""

    def test_prepare_data_sufficient_data(self, lstm_config, sample_ohlcv_data):
        """Test data preparation with sufficient data."""
        model = LSTMModel(lstm_config)
        seq_len = 30
        
        X, y = model._prepare_data(sample_ohlcv_data, seq_len)
        
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert X.shape[0] == len(sample_ohlcv_data) - seq_len
        assert X.shape[1] == seq_len
        assert X.shape[2] == 5  # OHLCV features
        assert y.shape[0] == len(sample_ohlcv_data) - seq_len

    def test_prepare_data_insufficient_data(self, lstm_config):
        """Test data preparation with insufficient data."""
        model = LSTMModel(lstm_config)
        
        # Create data with only a few rows
        data = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [1000000, 1100000],
            }
        )
        
        X, y = model._prepare_data(data, sequence_length=30)
        
        assert len(X) == 0
        assert y is None

    def test_prepare_data_normalization(self, lstm_config, sample_ohlcv_data):
        """Test that data is normalized."""
        model = LSTMModel(lstm_config)
        
        X, y = model._prepare_data(sample_ohlcv_data, sequence_length=30)
        
        # Check that mean and std are stored
        assert hasattr(model, "mean")
        assert hasattr(model, "std")
        assert len(model.mean) == 5
        assert len(model.std) == 5


class TestLSTMModelTraining:
    """Tests for model training."""

    def test_train_creates_model(self, lstm_config, sample_ohlcv_data):
        """Test that training creates a model."""
        model = LSTMModel(lstm_config)
        
        assert model.model is None
        model.train(sample_ohlcv_data)
        assert model.model is not None
        assert isinstance(model.model, LSTMNet)

    def test_train_with_sufficient_data(self, lstm_config, sample_ohlcv_data):
        """Test training with sufficient data."""
        model = LSTMModel(lstm_config)
        model.train(sample_ohlcv_data)
        
        # Model should be created and in eval mode after training
        assert model.model is not None
        # Can't easily test if model is trained, but at least it didn't crash

    def test_train_with_insufficient_data(self, lstm_config, capsys):
        """Test training with insufficient data."""
        model = LSTMModel(lstm_config)
        
        data = pd.DataFrame(
            {
                "open": [100],
                "high": [102],
                "low": [99],
                "close": [101],
                "volume": [1000000],
            }
        )
        
        model.train(data)
        captured = capsys.readouterr()
        
        assert "Insufficient data" in captured.out
        assert model.model is None


@pytest.mark.asyncio
class TestLSTMModelGenerate:
    """Tests for signal generation."""

    async def test_generate_without_training(self, lstm_config, sample_ohlcv_data):
        """Test signal generation without training."""
        model = LSTMModel(lstm_config)
        
        signal = await model.generate(sample_ohlcv_data, datetime.now())
        
        assert isinstance(signal, Signal)
        assert signal.signal_type == SignalType.HOLD
        assert signal.direction == Direction.NEUTRAL
        assert signal.metadata["reason"] == "Model not trained"

    async def test_generate_insufficient_data(self, lstm_config, sample_ohlcv_data):
        """Test signal generation with insufficient data."""
        model = LSTMModel(lstm_config)
        model.train(sample_ohlcv_data)
        
        # Use only a few rows
        small_data = sample_ohlcv_data.head(10)
        signal = await model.generate(small_data, datetime.now())
        
        assert signal.signal_type == SignalType.HOLD
        assert signal.metadata["reason"] == "Insufficient data"

    async def test_generate_after_training(self, lstm_config, sample_ohlcv_data):
        """Test signal generation after training."""
        model = LSTMModel(lstm_config)
        model.train(sample_ohlcv_data)
        
        signal = await model.generate(sample_ohlcv_data, datetime.now())
        
        assert isinstance(signal, Signal)
        assert signal.symbol == "AAPL"
        assert signal.direction in [Direction.LONG, Direction.SHORT]
        assert 0.0 <= signal.score <= 1.0
        assert "probability" in signal.metadata

    async def test_generate_direction_logic(self, lstm_config, sample_ohlcv_data):
        """Test direction is based on probability threshold."""
        model = LSTMModel(lstm_config)
        model.train(sample_ohlcv_data)
        
        signal = await model.generate(sample_ohlcv_data, datetime.now())
        
        prob = signal.metadata["probability"]
        if prob > 0.5:
            assert signal.direction == Direction.LONG
        else:
            assert signal.direction == Direction.SHORT

    async def test_generate_strength_calculation(self, lstm_config, sample_ohlcv_data):
        """Test score calculation."""
        model = LSTMModel(lstm_config)
        model.train(sample_ohlcv_data)
        
        signal = await model.generate(sample_ohlcv_data, datetime.now())
        
        prob = signal.metadata["probability"]
        expected_score = abs(prob - 0.5) * 2
        assert abs(signal.score - expected_score) < 1e-6


class TestLSTMModelPersistence:
    """Tests for model save/load."""

    def test_save_creates_files(self, lstm_config, sample_ohlcv_data):
        """Test that save creates all necessary files."""
        model = LSTMModel(lstm_config)
        model.train(sample_ohlcv_data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model.save(path)
            
            assert (path / "config.json").exists()
            assert (path / "metadata.json").exists()
            assert (path / "model_state.pt").exists()
            assert (path / "scaler.pt").exists()

    def test_save_without_training(self, lstm_config):
        """Test save without training (no model weights)."""
        model = LSTMModel(lstm_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model.save(path)
            
            assert (path / "config.json").exists()
            assert (path / "metadata.json").exists()
            assert (path / "scaler.pt").exists()
            # model_state.pt should not exist since model was never trained
            assert not (path / "model_state.pt").exists()

    def test_load_trained_model(self, lstm_config, sample_ohlcv_data):
        """Test loading a trained model."""
        model = LSTMModel(lstm_config)
        model.train(sample_ohlcv_data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model.save(path)
            
            loaded_model = LSTMModel.load(path)
            
            assert loaded_model.config.model_id == lstm_config.model_id
            assert loaded_model.model is not None
            assert hasattr(loaded_model, "mean")
            assert hasattr(loaded_model, "std")
            # Check that mean and std match
            assert np.allclose(loaded_model.mean, model.mean)
            assert np.allclose(loaded_model.std, model.std)

    @pytest.mark.asyncio
    async def test_loaded_model_generates_signals(self, lstm_config, sample_ohlcv_data):
        """Test that loaded model can generate signals."""
        model = LSTMModel(lstm_config)
        model.train(sample_ohlcv_data)
        
        # Generate signal with original model
        original_signal = await model.generate(sample_ohlcv_data, datetime.now())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model.save(path)
            loaded_model = LSTMModel.load(path)
            
            # Generate signal with loaded model
            loaded_signal = await loaded_model.generate(sample_ohlcv_data, datetime.now())
            
            # Signals should be identical
            assert loaded_signal.direction == original_signal.direction
            assert abs(loaded_signal.score - original_signal.score) < 1e-4
            assert abs(loaded_signal.metadata["probability"] - original_signal.metadata["probability"]) < 1e-4
