"""
Base model classes and registry for SignalCore.

All models must be testable, reproducible, and auditable.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .signal import Signal, SignalBatch


@dataclass
class ModelConfig:
    """
    Configuration for a SignalCore model.

    Attributes:
        model_id: Unique identifier
        model_type: Type of model (technical, ml, factor, etc.)
        version: Version string
        parameters: Model-specific parameters
        enabled: Whether model is active
        min_data_points: Minimum data required
        lookback_period: Historical data window needed
        update_frequency: How often to generate signals
    """

    model_id: str
    model_type: str
    version: str = "1.0.0"
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    min_data_points: int = 100
    lookback_period: int = 252  # Trading days
    update_frequency: str = "1d"
    metadata: dict[str, Any] = field(default_factory=dict)


class Model(ABC):
    """
    Abstract base class for all SignalCore models.

    All models must implement:
    - generate(): Produce signals from market data
    - validate(): Check if model can generate valid signals
    - describe(): Provide model metadata and state

    Models are stateless processors - they don't maintain internal state
    beyond their configuration.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize model with configuration.

        Args:
            config: Model configuration
        """
        self.config = config
        self._last_update: datetime | None = None

    @abstractmethod
    async def generate(self, symbol: str, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate trading signal from market data.

        Args:
            symbol: Stock ticker symbol
            data: Historical OHLCV data (indexed by timestamp)
            timestamp: Current time for signal generation

        Returns:
            Signal object with model predictions

        Raises:
            ValueError: If data is insufficient or invalid
        """

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate if model can generate signals from ordinis.data.

        Args:
            data: Historical OHLCV data

        Returns:
            (is_valid, message) tuple
        """
        if len(data) < self.config.min_data_points:
            return (
                False,
                f"Insufficient data: {len(data)} < {self.config.min_data_points}",
            )

        required_columns = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            return False, f"Missing columns: {missing}"

        if data.isnull().any().any():
            return False, "Data contains null values"

        return True, "Valid"

    def describe(self) -> dict[str, Any]:
        """
        Get model metadata and current state.

        Returns:
            Dictionary with model information
        """
        return {
            "model_id": self.config.model_id,
            "model_type": self.config.model_type,
            "version": self.config.version,
            "enabled": self.config.enabled,
            "parameters": self.config.parameters,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "min_data_points": self.config.min_data_points,
            "lookback_period": self.config.lookback_period,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"id={self.config.model_id}, "
            f"version={self.config.version}, "
            f"enabled={self.config.enabled})"
        )

    # -------------------------------------------------------------------------
    # Persistence Methods (G-ML-1, G-ML-2)
    # -------------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Subclasses with trainable state (e.g., neural networks, scalers)
        MUST override this method to persist their artifacts.

        Args:
            path: Directory to save model artifacts.
        """
        import json

        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = path / "config.json"
        config_dict = asdict(self.config)
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        # Save metadata
        metadata_path = path / "metadata.json"
        metadata = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "last_update": self._last_update.isoformat() if self._last_update else None,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Model":
        """
        Load model from disk.

        Subclasses with trainable state MUST override this method.

        Args:
            path: Directory containing model artifacts.

        Returns:
            Loaded model instance.
        """
        import json

        config_path = path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        config = ModelConfig(**config_dict)
        return cls(config)

    def get_artifact_path(self, base_dir: Path) -> Path:
        """
        Get the standard artifact path for this model.

        Args:
            base_dir: Base directory for model artifacts.

        Returns:
            Path: models/{model_id}/{version}/
        """
        return base_dir / "models" / self.config.model_id / self.config.version


class ModelRegistry:
    """
    Registry for managing SignalCore models.

    Provides model registration, retrieval, and lifecycle management.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._models: dict[str, Model] = {}
        self._configs: dict[str, ModelConfig] = {}

    def register(self, model: Model) -> None:
        """
        Register a model in the registry.

        Args:
            model: Model instance to register

        Raises:
            ValueError: If model_id already registered
        """
        model_id = model.config.model_id

        if model_id in self._models:
            raise ValueError(f"Model {model_id} already registered")

        self._models[model_id] = model
        self._configs[model_id] = model.config

    def unregister(self, model_id: str) -> None:
        """
        Remove model from registry.

        Args:
            model_id: Model identifier

        Raises:
            KeyError: If model not found
        """
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} not found")

        del self._models[model_id]
        del self._configs[model_id]

    def get(self, model_id: str) -> Model:
        """
        Get model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Model instance

        Raises:
            KeyError: If model not found
        """
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} not found")

        return self._models[model_id]

    def list_models(self, enabled_only: bool = False) -> list[str]:
        """
        List registered model IDs.

        Args:
            enabled_only: Only return enabled models

        Returns:
            List of model IDs
        """
        if not enabled_only:
            return list(self._models.keys())

        return [model_id for model_id, model in self._models.items() if model.config.enabled]

    def get_by_type(self, model_type: str) -> list[Model]:
        """
        Get all models of a specific type.

        Args:
            model_type: Model type to filter

        Returns:
            List of matching models
        """
        return [model for model in self._models.values() if model.config.model_type == model_type]

    async def generate_all(self, data: dict[str, pd.DataFrame], timestamp: datetime) -> SignalBatch:
        """
        Generate signals from all enabled models.

        Args:
            data: Dictionary of symbol -> OHLCV data
            timestamp: Current timestamp

        Returns:
            SignalBatch with signals from all models
        """
        signals = []

        for model_id in self.list_models(enabled_only=True):
            model = self._models[model_id]

            for symbol, df in data.items():
                try:
                    is_valid, _msg = model.validate(df)
                    if not is_valid:
                        continue

                    signal = await model.generate(symbol, df, timestamp)
                    if signal:
                        signals.append(signal)

                except Exception as e:
                    print(f"[ERROR] Model {model_id} failed: {e}")
                    import traceback

                    traceback.print_exc()
                    # Skip model on error, don't fail entire batch
                    # Logging intentionally omitted to avoid noise from expected failures
                    continue

        return SignalBatch(timestamp=timestamp, signals=signals, universe=list(data.keys()))

    def to_dict(self) -> dict[str, Any]:
        """Get registry state as dictionary."""
        return {
            "models": {model_id: model.describe() for model_id, model in self._models.items()},
            "total_models": len(self._models),
            "enabled_models": len(self.list_models(enabled_only=True)),
        }

    # -------------------------------------------------------------------------
    # Persistence Methods (G-ML-1)
    # -------------------------------------------------------------------------

    def save_all(self, base_dir: Path) -> dict[str, Path]:
        """
        Save all registered models to disk.

        Args:
            base_dir: Base directory for model artifacts.

        Returns:
            Dictionary mapping model_id -> save path.
        """
        import json

        saved = {}
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        for model_id, model in self._models.items():
            model_path = model.get_artifact_path(base_dir)
            model.save(model_path)
            saved[model_id] = model_path

        # Save registry index
        index_path = base_dir / "registry_index.json"
        index = {
            "models": {
                model_id: {
                    "path": str(path.relative_to(base_dir)),
                    "class": self._models[model_id].__class__.__name__,
                    "module": self._models[model_id].__class__.__module__,
                }
                for model_id, path in saved.items()
            },
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        return saved

    def load_all(self, base_dir: Path, model_classes: dict[str, type]) -> list[str]:
        """
        Load all models from disk.

        Args:
            base_dir: Base directory containing model artifacts.
            model_classes: Dictionary mapping class names to model classes.

        Returns:
            List of loaded model IDs.
        """
        import json

        base_dir = Path(base_dir)
        index_path = base_dir / "registry_index.json"

        if not index_path.exists():
            return []

        with open(index_path) as f:
            index = json.load(f)

        loaded = []
        for model_id, info in index.get("models", {}).items():
            class_name = info["class"]
            if class_name not in model_classes:
                print(f"[WARN] Unknown model class: {class_name}, skipping {model_id}")
                continue

            model_path = base_dir / info["path"]
            if not model_path.exists():
                print(f"[WARN] Model path not found: {model_path}, skipping {model_id}")
                continue

            model_cls = model_classes[class_name]
            model = model_cls.load(model_path)
            self._models[model_id] = model
            self._configs[model_id] = model.config
            loaded.append(model_id)

        return loaded


# Global registry instance
registry = ModelRegistry()
