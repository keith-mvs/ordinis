"""
LSTM-based price prediction model.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType


class LSTMNet(nn.Module):
    """LSTM Network architecture."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(Model):
    """
    LSTM model for price direction prediction.
    """

    def __init__(self, config: ModelConfig | None = None):
        if config is None:
            config = ModelConfig(
                model_id="lstm_v1",
                model_type="ml",
                version="1.0.0",
                parameters={
                    "input_dim": 5,  # OHLCV
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "output_dim": 1,
                    "sequence_length": 60,
                    "epochs": 20,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                },
                min_data_points=200,
            )
        super().__init__(config)
        self.model: LSTMNet | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_data(
        self, data: pd.DataFrame, sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare data for LSTM."""
        # Simple normalization
        features = data[["open", "high", "low", "close", "volume"]].values
        # In a real scenario, use a scaler and save it
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        features = (features - self.mean) / (self.std + 1e-8)

        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i : i + sequence_length])
            # Target: 1 if next close > current close, else 0
            if i + sequence_length < len(features):
                target = (
                    1.0
                    if features[i + sequence_length, 3] > features[i + sequence_length - 1, 3]
                    else 0.0
                )
                y.append(target)

        if not X:
            return torch.tensor([]), None

        X_tensor = torch.FloatTensor(np.array(X)).to(self.device)
        y_tensor = torch.FloatTensor(np.array(y)).unsqueeze(1).to(self.device) if y else None

        return X_tensor, y_tensor

    def train(self, data: pd.DataFrame):
        """Train the LSTM model."""
        params = self.config.parameters
        seq_len = params.get("sequence_length", 60)

        X, y = self._prepare_data(data, seq_len)
        if len(X) == 0:
            print("Insufficient data for training")
            return

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=params.get("batch_size", 32), shuffle=True)

        self.model = LSTMNet(
            input_dim=params.get("input_dim", 5),
            hidden_dim=params.get("hidden_dim", 64),
            num_layers=params.get("num_layers", 2),
            output_dim=params.get("output_dim", 1),
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.get("learning_rate", 0.001))

        epochs = params.get("epochs", 10)
        print(f"Training LSTM for {epochs} epochs...")

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """Generate signal."""
        if self.model is None:
            # In a real system, we would load weights here
            # For now, just return HOLD if not trained
            return Signal(
                symbol=data["symbol"].iloc[-1] if "symbol" in data.columns else "UNKNOWN",
                timestamp=timestamp,
                signal_type=SignalType.HOLD,
                direction=Direction.NEUTRAL,
                strength=0.0,
                model_id=self.config.model_id,
                metadata={"reason": "Model not trained"},
            )

        seq_len = self.config.parameters.get("sequence_length", 60)
        if len(data) < seq_len:
            return Signal(
                symbol=data["symbol"].iloc[-1] if "symbol" in data.columns else "UNKNOWN",
                timestamp=timestamp,
                signal_type=SignalType.HOLD,
                direction=Direction.NEUTRAL,
                strength=0.0,
                model_id=self.config.model_id,
                metadata={"reason": "Insufficient data"},
            )

        # Prepare last sequence
        # Re-use normalization logic (should be refactored to use saved scaler)
        features = data[["open", "high", "low", "close", "volume"]].tail(seq_len).values
        # Use stored mean/std if available, else compute (which is wrong for inference but ok for this demo)
        if hasattr(self, "mean"):
            features = (features - self.mean) / (self.std + 1e-8)

        X_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            prob = torch.sigmoid(output).item()

        direction = Direction.LONG if prob > 0.5 else Direction.SHORT
        strength = abs(prob - 0.5) * 2

        return Signal(
            symbol=data["symbol"].iloc[-1] if "symbol" in data.columns else "UNKNOWN",
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if strength > 0.2 else SignalType.HOLD,
            direction=direction,
            strength=strength,
            model_id=self.config.model_id,
            metadata={"probability": prob},
        )

    # -------------------------------------------------------------------------
    # Persistence Methods (G-ML-1, G-ML-2 - scaler persistence)
    # -------------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save model to disk including scaler parameters.

        Persists:
        - model_state.pt: PyTorch model weights
        - scaler.pt: Normalization parameters (mean, std)
        - config.json + metadata.json (via parent)

        Args:
            path: Directory to save model artifacts.
        """
        # Call parent to save config and metadata
        super().save(path)

        path = Path(path)

        # Save model weights if trained
        if self.model is not None:
            model_path = path / "model_state.pt"
            torch.save(self.model.state_dict(), model_path)

        # Save scaler parameters (G-ML-2: critical for training/serving parity)
        scaler_path = path / "scaler.pt"
        scaler_state = {
            "mean": getattr(self, "mean", None),
            "std": getattr(self, "std", None),
        }
        torch.save(scaler_state, scaler_path)

    @classmethod
    def load(cls, path: Path) -> "LSTMModel":
        """
        Load model from disk including scaler parameters.

        Args:
            path: Directory containing model artifacts.

        Returns:
            Loaded LSTMModel instance with weights and scaler restored.
        """
        import json

        path = Path(path)

        # Load config
        config_path = path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        config = ModelConfig(**config_dict)
        model = cls(config)

        # Load scaler parameters (G-ML-2: restore for inference parity)
        scaler_path = path / "scaler.pt"
        if scaler_path.exists():
            scaler_state = torch.load(scaler_path, weights_only=True)
            if scaler_state.get("mean") is not None:
                model.mean = scaler_state["mean"]
            if scaler_state.get("std") is not None:
                model.std = scaler_state["std"]

        # Load model weights if they exist
        model_path = path / "model_state.pt"
        if model_path.exists():
            params = config.parameters
            model.model = LSTMNet(
                input_dim=params.get("input_dim", 5),
                hidden_dim=params.get("hidden_dim", 64),
                num_layers=params.get("num_layers", 2),
                output_dim=params.get("output_dim", 1),
            ).to(model.device)
            model.model.load_state_dict(
                torch.load(model_path, map_location=model.device, weights_only=True)
            )
            model.model.eval()

        return model
