import asyncio
import pytest

from ordinis.ai.helix.engine import Helix
from ordinis.ai.helix.models import (
    ModelInfo,
    UsageInfo,
    ChatResponse,
    EmbeddingResponse,
    ModelType,
    ProviderType,
)


class DummyProvider:
    def __init__(self):
        self.calls = 0

    @property
    def provider_type(self):
        return ProviderType.MOCK

    @property
    def is_available(self):
        return True

    async def health_check(self):
        return self.is_available

    async def chat(self, messages, model, temperature=None, max_tokens=None, stop=None, **kwargs):
        self.calls += 1
        return ChatResponse(
            content="ok",
            model=model.model_id,
            provider=model.provider,
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def chat_stream(self, messages, model, temperature=None, max_tokens=None, **kwargs):
        yield "chunk1"
        yield "chunk2"

    async def embed(self, texts, model, **kwargs):
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2]],
            model=model.model_id,
            provider=model.provider,
            usage=UsageInfo.empty(),
            dimension=2,
        )


@pytest.mark.asyncio
async def test_helix_generate_and_metrics(monkeypatch):
    engine = Helix(config=None)

    # Make config valid (prefer_local True bypasses provider API key requirement)
    engine.config.prefer_local = True

    dummy = DummyProvider()
    engine._providers[ProviderType.MOCK] = dummy

    # Create a model that maps to our dummy provider
    model = ModelInfo(
        model_id="dummy-model",
        display_name="Dummy",
        model_type=ModelType.CHAT,
        provider=ProviderType.MOCK,
        context_length=1024,
    )

    # Monkeypatch _get_model to return our dummy model when asked
    monkeypatch.setattr(engine, "_get_model", lambda name, mt: model)

    # Enable chat caching and run two generates to exercise cache logic
    engine.config.cache.cache_chat = True

    res1 = await engine.generate([{"role": "user", "content": "hi"}], model="dummy")
    assert isinstance(res1, ChatResponse)
    assert res1.content == "ok"

    # Second call should hit cache (since same inputs) and not increase provider.calls
    res2 = await engine.generate([{"role": "user", "content": "hi"}], model="dummy")
    assert isinstance(res2, ChatResponse)
    assert res2.content == "ok"

    metrics = engine.get_metrics()
    assert metrics["total_requests"] >= 1
    assert metrics["total_tokens"] >= 2
    assert metrics["cache_hits"] >= 1 or dummy.calls == 1


@pytest.mark.asyncio
async def test_generate_stream_and_embed_and_health():
    engine = Helix(config=None)
    engine.config.prefer_local = True

    dummy = DummyProvider()
    engine._providers[ProviderType.MOCK] = dummy

    model = ModelInfo(
        model_id="dummy-model",
        display_name="Dummy",
        model_type=ModelType.CHAT,
        provider=ProviderType.MOCK,
        context_length=1024,
    )

    # Monkeypatch _get_model for both CHAT and EMBEDDING calls
    original_get_model = engine._get_model

    def _get_model(name, mt):
        if mt == ModelType.CHAT:
            return model
        # For embeddings, return a simple embedding model
        return ModelInfo(
            model_id="dummy-embed",
            display_name="DummyEmbed",
            model_type=ModelType.EMBEDDING,
            provider=ProviderType.MOCK,
            context_length=512,
            embedding_dim=2,
        )

    engine._get_model = _get_model

    # Test stream
    chunks = []
    async for c in engine.generate_stream([{"role": "user", "content": "hi"}], model="dummy"):
        chunks.append(c)
    assert chunks == ["chunk1", "chunk2"]

    # Test embedding
    emb = await engine.embed(["hello"], model="dummy-embed")
    assert isinstance(emb, EmbeddingResponse)
    assert emb.count == 1

    # Health check returns dict with provider key
    health = await engine.health_check()
    assert ProviderType.MOCK.value in health
    assert health[ProviderType.MOCK.value] is True
