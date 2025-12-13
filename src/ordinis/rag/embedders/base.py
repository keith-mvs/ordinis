"""Base embedder interface."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, text: str | list[str]) -> np.ndarray:
        """Embed text into vector representation.

        Args:
            text: Single string or list of strings to embed

        Returns:
            numpy array of shape (embedding_dim,) for single text
            or (n_texts, embedding_dim) for list of texts
        """

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if embedder is available (model loaded or API accessible)."""

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory (for local models)."""
