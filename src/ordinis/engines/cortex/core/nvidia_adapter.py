"""
NVIDIA AI client adapter for CortexEngine.

Provides decoupled client management for NVIDIA chat and embedding models
with lazy initialization, configuration updates, and error handling.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Optional NVIDIA integration
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    ChatNVIDIA = None  # type: ignore[misc, assignment]
    NVIDIAEmbeddings = None  # type: ignore[misc, assignment]


class NVIDIAAdapter:
    """
    Adapter for NVIDIA AI clients with lazy loading and configuration management.

    Separates client lifecycle from engine logic, allowing dynamic reconfiguration
    without engine restarts.
    """

    def __init__(
        self,
        api_key: str,
        chat_model: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        embedding_model: str = "nvidia/nv-embedqa-e5-v5",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize NVIDIA adapter.

        Args:
            api_key: NVIDIA API key
            chat_model: Default chat model identifier
            embedding_model: Default embedding model identifier
            temperature: Default generation temperature
            max_tokens: Default max tokens per response
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts

        Raises:
            ImportError: If langchain-nvidia-ai-endpoints not installed
        """
        if not NVIDIA_AVAILABLE:
            raise ImportError(
                "NVIDIA SDK not available. Install with: "
                "pip install langchain-nvidia-ai-endpoints"
            )

        self._api_key = api_key
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries

        # Lazy-loaded clients
        self._chat_client: ChatNVIDIA | None = None
        self._embedding_client: NVIDIAEmbeddings | None = None

        logger.info(
            f"NVIDIAAdapter initialized with chat_model={chat_model}, "
            f"embedding_model={embedding_model}"
        )

    def get_chat_client(self) -> Any:
        """
        Get or create ChatNVIDIA client.

        Returns:
            ChatNVIDIA: Configured chat client

        Raises:
            ValueError: If API key not provided
            RuntimeError: If client initialization fails
        """
        if self._chat_client is None:
            try:
                self._chat_client = ChatNVIDIA(
                    model=self._chat_model,
                    nvidia_api_key=self._api_key,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    timeout=self._timeout,
                    max_retries=self._max_retries,
                )
                logger.debug(f"ChatNVIDIA client initialized: {self._chat_model}")
            except Exception as e:
                logger.error(f"Failed to initialize ChatNVIDIA: {e}")
                raise RuntimeError(f"ChatNVIDIA initialization failed: {e}") from e

        return self._chat_client

    def get_embedding_client(self) -> Any:
        """
        Get or create NVIDIAEmbeddings client.

        Returns:
            NVIDIAEmbeddings: Configured embedding client

        Raises:
            ValueError: If API key not provided
            RuntimeError: If client initialization fails
        """
        if self._embedding_client is None:
            try:
                self._embedding_client = NVIDIAEmbeddings(
                    model=self._embedding_model,
                    nvidia_api_key=self._api_key,
                    timeout=self._timeout,
                )
                logger.debug(f"NVIDIAEmbeddings client initialized: {self._embedding_model}")
            except Exception as e:
                logger.error(f"Failed to initialize NVIDIAEmbeddings: {e}")
                raise RuntimeError(f"NVIDIAEmbeddings initialization failed: {e}") from e

        return self._embedding_client

    def update_chat_config(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """
        Update chat client configuration and invalidate cache.

        Creates new client on next get_chat_client() call.

        Args:
            model: New chat model identifier
            temperature: New generation temperature
            max_tokens: New max tokens per response
        """
        if model is not None:
            self._chat_model = model
            logger.info(f"Chat model updated to: {model}")

        if temperature is not None:
            self._temperature = temperature
            logger.info(f"Temperature updated to: {temperature}")

        if max_tokens is not None:
            self._max_tokens = max_tokens
            logger.info(f"Max tokens updated to: {max_tokens}")

        # Invalidate cached client
        self._chat_client = None
        logger.debug("Chat client cache invalidated")

    def update_embedding_config(self, model: str | None = None) -> None:
        """
        Update embedding client configuration and invalidate cache.

        Creates new client on next get_embedding_client() call.

        Args:
            model: New embedding model identifier
        """
        if model is not None:
            self._embedding_model = model
            logger.info(f"Embedding model updated to: {model}")

        # Invalidate cached client
        self._embedding_client = None
        logger.debug("Embedding client cache invalidated")

    @property
    def chat_model(self) -> str:
        """Current chat model identifier."""
        return self._chat_model

    @property
    def embedding_model(self) -> str:
        """Current embedding model identifier."""
        return self._embedding_model

    @property
    def temperature(self) -> float:
        """Current generation temperature."""
        return self._temperature

    @property
    def max_tokens(self) -> int:
        """Current max tokens setting."""
        return self._max_tokens
