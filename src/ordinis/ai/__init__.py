"""
Ordinis AI subsystem.

Provides unified AI infrastructure for the trading system:
- Helix: LLM provider abstraction (chat, embeddings)
- Synapse: RAG retrieval engine
"""

from ordinis.ai.helix import Helix, HelixConfig
from ordinis.ai.synapse import Synapse, SynapseConfig

__all__ = ["Helix", "HelixConfig", "Synapse", "SynapseConfig"]
