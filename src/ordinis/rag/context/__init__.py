"""Context assembly and management for RAG operations.

This package provides tools for assembling context from multiple sources
for LLM prompt construction.

Classes:
    ContextAssembler: Assembles context from SQLite, ChromaDB, and other sources
    ContextSource: Enum of available context sources
    ContextPriority: Priority levels for context chunks
    AssembledContext: Result of context assembly
"""

from ordinis.rag.context.assembler import (
    ContextAssembler,
    ContextSource,
    ContextPriority,
    ContextChunk,
    AssembledContext,
)

__all__ = [
    "ContextAssembler",
    "ContextSource",
    "ContextPriority",
    "ContextChunk",
    "AssembledContext",
]
