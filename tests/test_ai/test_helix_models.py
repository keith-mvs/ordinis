from datetime import datetime

import numpy as np

from ordinis.ai.helix.models import (
    ModelInfo,
    ModelType,
    ProviderType,
    ChatMessage,
    UsageInfo,
    ChatResponse,
    EmbeddingResponse,
    HelixError,
    RateLimitError,
    ModelNotFoundError,
    ProviderError,
)


def test_model_info_str():
    mi = ModelInfo(
        model_id="m1",
        display_name="Display",
        model_type=ModelType.CHAT,
        provider=ProviderType.MOCK,
        context_length=1024,
    )
    assert str(mi) == "Display (m1)"


def test_chat_message_to_dict_and_metadata_is_independent():
    cm = ChatMessage(role="user", content="hello")
    assert cm.to_dict() == {"role": "user", "content": "hello"}

    cm_with_name = ChatMessage(role="assistant", content="ok", name="bot1")
    assert cm_with_name.to_dict()["name"] == "bot1"

    # metadata should be a fresh dict per instance
    cm.metadata["x"] = 1
    cm2 = ChatMessage(role="user", content="hi")
    assert "x" not in cm2.metadata


def test_usage_info_empty():
    u = UsageInfo.empty()
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0
    assert u.total_tokens == 0
    assert u.cost_usd == 0.0


def test_chat_response_str_and_timestamp_default():
    u = UsageInfo.empty()
    cr = ChatResponse(content="hi", model="m1", provider=ProviderType.MOCK, usage=u)
    assert str(cr) == "hi"
    assert isinstance(cr.timestamp, datetime)


def test_embedding_response_count_and_numpy():
    u = UsageInfo.empty()
    emb = [[1.0, 2.0], [3.0, 4.0]]
    er = EmbeddingResponse(embeddings=emb, model="m1", provider=ProviderType.MOCK, usage=u, dimension=2)
    assert er.count == 2
    arr = er.as_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)


def test_helix_errors_and_subclasses():
    e = HelixError("oops", provider=ProviderType.MOCK, model="m1", retriable=False)
    assert e.provider == ProviderType.MOCK
    assert e.model == "m1"
    assert e.retriable is False

    r = RateLimitError(retry_after=1.5)
    assert r.retriable is True
    assert r.retry_after == 1.5

    mn = ModelNotFoundError("x", available=["a", "b", "c"])
    assert "Model 'x' not found" in str(mn)
    assert mn.available == ["a", "b", "c"]

    pe = ProviderError("bad", status_code=502)
    assert pe.status_code == 502
