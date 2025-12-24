import asyncio

from ordinis.ai.helix.providers.base import BaseProvider
from ordinis.ai.helix.models import (
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    ModelInfo,
    ProviderType,
    UsageInfo,
    ModelType,
)


class DummyProvider(BaseProvider):
    def __init__(self, available: bool = True):
        self._available = available

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.MOCK

    @property
    def is_available(self) -> bool:
        return self._available

    async def chat(self, messages, model, temperature=None, max_tokens=None, stop=None, **kwargs):
        # return a trivial ChatResponse
        content = " ".join(m.content for m in messages)
        return ChatResponse(content=content, model=model.model_id, provider=self.provider_type, usage=UsageInfo.empty())

    async def chat_stream(self, messages, model, temperature=None, max_tokens=None, **kwargs):
        # yield each message content as a chunk
        for m in messages:
            yield m.content

    async def embed(self, texts, model, **kwargs):
        # return one embedding per text with dimension 2
        embeddings = [[float(i), float(i + 1)] for i, _ in enumerate(texts)]
        return EmbeddingResponse(embeddings=embeddings, model=model.model_id, provider=self.provider_type, usage=UsageInfo.empty(), dimension=2)


async def _gather_stream(provider, messages, model):
    chunks = []
    async for c in provider.chat_stream(messages, model):
        chunks.append(c)
    return chunks


def test_supports_model_and_health_check():
    model = ModelInfo(model_id="m1", display_name="M1", model_type=ModelType.CHAT, provider=ProviderType.MOCK, context_length=512)
    p = DummyProvider(available=True)
    assert p.supports_model(model)
    assert asyncio.get_event_loop().run_until_complete(p.health_check())

    p2 = DummyProvider(available=False)
    assert p2.supports_model(model)
    assert not asyncio.get_event_loop().run_until_complete(p2.health_check())


def test_chat_and_embed_loopback():
    model = ModelInfo(model_id="m1", display_name="M1", model_type=ModelType.CHAT, provider=ProviderType.MOCK, context_length=512)
    p = DummyProvider()

    messages = [ChatMessage(role="user", content="hello"), ChatMessage(role="assistant", content="world")]

    cr = asyncio.get_event_loop().run_until_complete(p.chat(messages, model))
    assert isinstance(cr, ChatResponse)
    assert "hello world" in str(cr)

    emb = asyncio.get_event_loop().run_until_complete(p.embed(["a", "b", "c"], model))
    assert isinstance(emb, EmbeddingResponse)
    assert emb.count == 3

    chunks = asyncio.get_event_loop().run_until_complete(_gather_stream(p, messages, model))
    assert chunks == ["hello", "world"]
