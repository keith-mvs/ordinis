"""RAG (Retrieval Augmented Generation) module for intelligent-investor.

This module provides:
- Text and code embeddings using NVIDIA NeMo Retriever models
- Vector storage and retrieval via ChromaDB
- Query classification and reranking
- KB and code indexing pipelines
"""

from ordinis.rag.config import RAGConfig

__all__ = ["RAGConfig"]
