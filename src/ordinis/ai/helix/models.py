"""
Helix data models.

Defines request/response structures for LLM interactions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ModelType(Enum):
    """Model capability type."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    CODE = "code"
    MULTIMODAL = "multimodal"


class ProviderType(Enum):
    """LLM provider backend."""

    NVIDIA_API = "nvidia_api"
    LOCAL_GPU = "local_gpu"
    LOCAL_CPU = "local_cpu"
    MOCK = "mock"


@dataclass(frozen=True)
class ModelInfo:
    """Model metadata and capabilities."""

    model_id: str
    display_name: str
    model_type: ModelType
    provider: ProviderType
    context_length: int
    embedding_dim: int | None = None
    supports_streaming: bool = True
    supports_function_calling: bool = False
    cost_per_1k_tokens: float = 0.0
    default_temperature: float = 0.7
    max_output_tokens: int = 4096

    def __str__(self) -> str:
        return f"{self.display_name} ({self.model_id})"


@dataclass
class ChatMessage:
    """Single message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class UsageInfo:
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float = 0.0

    @classmethod
    def empty(cls) -> "UsageInfo":
        """Create empty usage info."""
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)


@dataclass
class ChatResponse:
    """Response from chat completion."""

    content: str
    model: str
    provider: ProviderType
    usage: UsageInfo
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: dict[str, Any] | None = None
    cached: bool = False

    def __str__(self) -> str:
        return self.content


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""

    embeddings: list[list[float]]
    model: str
    provider: ProviderType
    usage: UsageInfo
    dimension: int
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    cached: bool = False

    @property
    def count(self) -> int:
        """Number of embeddings returned."""
        return len(self.embeddings)

    def as_numpy(self) -> Any:
        """Convert to numpy array (requires numpy)."""
        import numpy as np

        return np.array(self.embeddings)


class HelixError(Exception):
    """Base exception for Helix errors."""

    def __init__(
        self,
        message: str,
        provider: ProviderType | None = None,
        model: str | None = None,
        retriable: bool = False,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.retriable = retriable


class RateLimitError(HelixError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, retriable=True, **kwargs)
        self.retry_after = retry_after


class ModelNotFoundError(HelixError):
    """Requested model not available."""

    def __init__(self, model: str, available: list[str] | None = None, **kwargs: Any):
        msg = f"Model '{model}' not found"
        if available:
            msg += f". Available: {', '.join(available[:5])}"
        super().__init__(msg, model=model, **kwargs)
        self.available = available or []


class ProviderError(HelixError):
    """Provider-specific error."""

    def __init__(self, message: str, status_code: int | None = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.status_code = status_code
