"""Code embedder using NVIDIA nv-embedcode-7b-v1 model."""

import os

from loguru import logger
import numpy as np

from ordinis.rag.config import get_config
from ordinis.rag.embedders.base import BaseEmbedder


class CodeEmbedder(BaseEmbedder):
    """Code embedding using NVIDIA nv-embedcode-7b-v1.

    Optimized for code retrieval with support for text, code, and hybrid queries.
    """

    def __init__(
        self,
        use_local: bool | None = None,
        api_key: str | None = None,
    ):
        """Initialize code embedder.

        Args:
            use_local: Use local GPU (True) or API (False). If None, uses config default.
            api_key: NVIDIA API key. If None, uses config or env var.
        """
        config = get_config()
        self.use_local = use_local if use_local is not None else config.use_local_embeddings
        self.api_key = api_key or config.nvidia_api_key or os.getenv("NVIDIA_API_KEY")
        self.model_name = config.code_embedding_model
        # Use text embedding dimension since we default to the same model
        self.embedding_dim = config.text_embedding_dimension

        self._model = None
        self._client = None

        # Check VRAM before loading 7B model
        if self.use_local:
            if config.check_vram_before_load:
                if not self._check_vram():
                    logger.warning("Insufficient VRAM for 7B code model, falling back to API")
                    self.use_local = False

        if self.use_local:
            self._init_local_model()
        else:
            self._init_api_client()

    def _check_vram(self) -> bool:
        """Check if sufficient VRAM is available.

        Returns:
            True if sufficient VRAM available, False otherwise
        """
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                vram_used_mb = float(result.stdout.strip().split("\n")[0])
                vram_used_gb = vram_used_mb / 1024
                max_vram_gb = get_config().max_vram_usage_gb

                logger.debug(f"Current VRAM usage: {vram_used_gb:.2f} GB")

                if vram_used_gb < max_vram_gb:
                    logger.info(f"VRAM check passed: {vram_used_gb:.2f} GB < {max_vram_gb} GB")
                    return True

                logger.warning(
                    f"VRAM check failed: {vram_used_gb:.2f} GB >= {max_vram_gb} GB threshold"
                )
                return False

        except Exception as e:
            logger.warning(f"VRAM check failed: {e}")
            # Default to True if check fails
            return True

        return True

    def _init_local_model(self) -> None:
        """Initialize local 7B model with GPU support."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Detect device (CUDA if available)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"Loading local code embedding model: {self.model_name} on {device.upper()}"
            )

            # Load model with GPU support
            self._model = SentenceTransformer(self.model_name, device=device)

            logger.success(f"Local code embedding model loaded on {device.upper()}")
            if device == "cuda":
                logger.info(
                    f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
                )
        except Exception as e:
            logger.warning(f"Failed to load local code model: {e}")
            if get_config().use_api_fallback:
                logger.info("Falling back to NVIDIA API")
                self.use_local = False
                self._init_api_client()
            else:
                msg = f"Failed to load local code embedding model: {e}"
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
        """Embed code/text into vector representation.

        Args:
            text: Single string or list of strings to embed (code or text)

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
                embeddings = np.array(self._client.embed_documents(texts))
            else:
                msg = "No embedding model or client available"
                raise RuntimeError(msg)

            # Apply Matryoshka truncation if needed
            if embeddings.shape[1] > self.embedding_dim:
                embeddings = embeddings[:, : self.embedding_dim]

            return embeddings[0] if is_single else embeddings

        except Exception as e:
            logger.error(f"Code embedding failed: {e}")
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
        """Unload model from memory (critical for 7B model)."""
        if self._model is not None:
            logger.info("Unloading local code embedding model (7B)")
            del self._model
            self._model = None

            # Clear CUDA cache
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared after unloading 7B model")
            except ImportError:
                pass
