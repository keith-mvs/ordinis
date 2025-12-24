import pytest
import asyncio

from ordinis.ai.helix.config import HelixConfig, RetryConfig
from ordinis.ai.helix.engine import Helix
from ordinis.ai.helix.models import ChatMessage, ChatResponse, EmbeddingResponse, UsageInfo, ProviderType, ModelType, HelixError


def test_cache_and_embed_key_and_cache_behavior():
    cfg = HelixConfig(prefer_local=True)
    helix = Helix(cfg)

    msgs = [ChatMessage(role="user", content="hello")]
    key1 = helix._cache_key(msgs, "mymodel", param=1)
    key2 = helix._cache_key(msgs, "mymodel", param=1)
    assert key1 == key2
    assert isinstance(key1, str) and len(key1) == 32

    ekey = helix._embed_cache_key(["a", "b"], "emb-model")
    assert isinstance(ekey, str) and len(ekey) == 32

    # Chat caching requires enabling chat cache
    helix.config.cache.cache_chat = True
    chat_resp = ChatResponse(content="ok", model="m", provider=ProviderType.MOCK, usage=UsageInfo.empty())
    helix._set_cached(key1, chat_resp)
    got = helix._get_cached(key1)
    assert got is not None and isinstance(got, ChatResponse)
    assert got.cached is True

    # Embedding caching (enabled by default)
    emb = EmbeddingResponse(embeddings=[[0.1, 0.2]], model="e", provider=ProviderType.MOCK, usage=UsageInfo.empty(), dimension=2)
    ek = helix._embed_cache_key(["x"], "emb-model")
    helix._set_cached(ek, emb)
    got2 = helix._get_cached(ek)
    assert got2 is not None and isinstance(got2, EmbeddingResponse)
    assert got2.cached is True


@pytest.mark.asyncio
async def test_record_usage_and_retry_with_backoff():
    cfg = HelixConfig(prefer_local=True)
    # make retries fast for tests
    cfg.retry = RetryConfig(max_retries=2, initial_delay_ms=1, max_delay_ms=10, exponential_base=1.0, jitter=False)
    helix = Helix(cfg)

    helix._record_usage(5)
    assert helix._total_requests == 1
    assert helix._total_tokens == 5
    assert len(helix._rate_limit.request_times) >= 1

    # A coroutine that fails once then succeeds
    state = {"calls": 0}

    async def flaky():
        state["calls"] += 1
        if state["calls"] == 1:
            raise HelixError("temp", retriable=True)
        return "success"

    res = await helix._retry_with_backoff(flaky)
    assert res == "success"


def test_get_model_and_fallback_present():
    cfg = HelixConfig(prefer_local=True)
    helix = Helix(cfg)

    m = helix._get_model(None, ModelType.CHAT)
    assert m is not None and m.model_type == ModelType.CHAT

    fb = helix._get_fallback_model(ModelType.CHAT)
    assert fb is not None
