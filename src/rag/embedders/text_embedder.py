"""Text embedder using NVIDIA NeMo Retriever 300M model."""

import os

from loguru import logger
import numpy as np

from rag.config import get_config
from rag.embedders.base import BaseEmbedder


class TextEmbedder(BaseEmbedder):
    """Text embedding using NVIDIA llama-3.2-nemoretriever-300m-embed-v2.

    Supports both local GPU inference and NVIDIA-hosted API fallback.
    """

    def __init__(
        self,
        use_local: bool | None = None,
        api_key: str | None = None,
    ):
        """Initialize text embedder.

        Args:
            use_local: Use local GPU (True) or API (False). If None, uses config default.
            api_key: NVIDIA API key. If None, uses config or env var.
        """
        config = get_config()
        self.use_local = use_local if use_local is not None else config.use_local_embeddings
        self.api_key = api_key or config.nvidia_api_key or os.getenv("NVIDIA_API_KEY")
        self.model_name = config.text_embedding_model
        self.embedding_dim = config.text_embedding_dimension

        self._model = None
        self._client = None

        if self.use_local:
            self._init_local_model()
        else:
            self._init_api_client()

    def _init_local_model(self) -> None:
        """Initialize local model using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local text embedding model: {self.model_name}")
            # Note: NVIDIA NeMo Retriever models are not directly available in sentence-transformers
            # For now, we use a compatible model. In production, use NVIDIA NIMs or API
            self._model = SentenceTransformer("all-MiniLM-L6-v2")  # Placeholder
            logger.success("Local text embedding model loaded")
        except Exception as e:
            logger.warning(f"Failed to load local model: {e}")
            if get_config().use_api_fallback:
                logger.info("Falling back to NVIDIA API")
                self.use_local = False
                self._init_api_client()
            else:
                msg = f"Failed to load local text embedding model: {e}"
                raise RuntimeError(msg) from e

    def _init_api_client(self) -> None:
        """Initialize NVIDIA API client."""
        if not self.api_key:
            msg = "NVIDIA API key required for API mode"
            raise ValueError(msg)

        try:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

            logger.info(f"Initializing NVIDIA API client for: {self.model_name}")
            self._client = NVIDIAEmbeddings(
                model=self.model_name,
                api_key=self.api_key,
            )
            logger.success("NVIDIA API client initialized")
        except Exception as e:
            msg = f"Failed to initialize NVIDIA API client: {e}"
            raise RuntimeError(msg) from e

    def embed(self, text: str | list[str]) -> np.ndarray:
        """Embed text into vector representation.

        Args:
            text: Single string or list of strings to embed

        Returns:
            numpy array of shape (embedding_dim,) for single text
            or (n_texts, embedding_dim) for list of texts
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        if not texts:
            return np.array([])

        try:
            if self.use_local and self._model is not None:
                embeddings = self._model.encode(texts, convert_to_numpy=True)
            elif self._client is not None:
                # NVIDIA API returns list of embeddings
                embeddings = np.array(self._client.embed_documents(texts))
            else:
                msg = "No embedding model or client available"
                raise RuntimeError(msg)

            # Apply Matryoshka truncation if needed
            if embeddings.shape[1] > self.embedding_dim:
                embeddings = embeddings[:, : self.embedding_dim]

            return embeddings[0] if is_single else embeddings

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Try API fallback if local failed
            if self.use_local and get_config().use_api_fallback:
                logger.info("Attempting API fallback")
                self.use_local = False
                self._init_api_client()
                return self.embed(text)
            raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def is_available(self) -> bool:
        """Check if embedder is available."""
        if self.use_local:
            return self._model is not None
        return self._client is not None

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            logger.info("Unloading local text embedding model")
            del self._model
            self._model = None

            # Clear CUDA cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared")
            except ImportError:
                pass
