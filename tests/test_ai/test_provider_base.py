import pytest
import asyncio

from ordinis.ai.helix.providers.base import BaseProvider
from ordinis.ai.helix.models import ProviderType, ModelInfo, ModelType, ChatMessage, ChatResponse, EmbeddingResponse, UsageInfo


class DummyProvider(BaseProvider):
    @property
    def provider_type(self):
        return ProviderType.MOCK

    @property
    def is_available(self):
        return True

    async def chat(self, messages, model, temperature=None, max_tokens=None, stop=None, **kwargs):
        return ChatResponse(content="ok", model=model.model_id, provider=self.provider_type, usage=UsageInfo.empty())

    async def chat_stream(self, messages, model, temperature=None, max_tokens=None, **kwargs):
        yield "chunk"

    async def embed(self, texts, model, **kwargs):
        return EmbeddingResponse(embeddings=[[0.1]], model=model.model_id, provider=self.provider_type, usage=UsageInfo.empty(), dimension=1)


@pytest.mark.asyncio
async def test_health_check_and_supports_model():
    d = DummyProvider()

    health = await d.health_check()
    assert health is True

    model = ModelInfo(model_id="m", display_name="m", model_type=ModelType.CHAT, provider=ProviderType.MOCK, context_length=10)
    assert d.supports_model(model) is True


@pytest.mark.asyncio
async def test_chat_and_embed_methods():
    d = DummyProvider()
    model = ModelInfo(model_id="m", display_name="m", model_type=ModelType.CHAT, provider=ProviderType.MOCK, context_length=10)

    resp = await d.chat([ChatMessage(role="user", content="hi")], model)
    assert isinstance(resp, ChatResponse)

    chunks = []
    async for c in d.chat_stream([ChatMessage(role="user", content="hi")], model):
        chunks.append(c)
    assert chunks and chunks[0] == "chunk"

    emb = await d.embed(["a"], model)
    assert isinstance(emb, EmbeddingResponse)
