"""
FastAPI server for RAG query endpoints.

Provides REST API for querying the RAG system with text and code queries.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger

from rag.api.models import (
    ConfigResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    StatsResponse,
)
from rag.retrieval.engine import RetrievalEngine

# Create FastAPI app
app = FastAPI(
    title="Intelligent Investor RAG API",
    description="Retrieval Augmented Generation API for trading knowledge base and code",
    version="1.0.0",
)

# Add CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton retrieval engine
_engine: RetrievalEngine | None = None


def get_engine() -> RetrievalEngine:
    """Get or create retrieval engine singleton."""
    global _engine
    if _engine is None:
        logger.info("Initializing RetrievalEngine...")
        _engine = RetrievalEngine()
        logger.info("RetrievalEngine initialized successfully")
    return _engine


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "Intelligent Investor RAG API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "ui": "/ui",
    }


@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint."""
    try:
        engine = get_engine()
        return HealthResponse(
            status="healthy",
            text_embedder_available=engine.text_embedder.is_available(),
            code_embedder_available=engine.code_embedder.is_available(),
            chroma_persist_directory=str(engine.config.chroma_persist_directory),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable") from e


@app.get("/stats")
async def stats() -> StatsResponse:
    """Get collection statistics."""
    try:
        engine = get_engine()

        # Get text collection stats
        text_collection = engine.chroma_client.get_text_collection()
        text_count = text_collection.count()

        # Get code collection stats
        code_collection = engine.chroma_client.get_code_collection()
        code_count = code_collection.count()

        return StatsResponse(
            text_chunks_count=text_count,
            code_chunks_count=code_count,
            total_chunks=text_count + code_count,
            text_embedder_model=engine.text_embedder.model_name,
            code_embedder_model=engine.code_embedder.model_name,
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Execute RAG query.

    Supports text, code, and hybrid queries with optional filtering.
    """
    try:
        engine = get_engine()

        # Execute query (returns schema.QueryResponse)
        schema_response = engine.query(
            query=request.query,
            query_type=request.query_type,
            top_k=request.top_k,
            filters=request.filters,
        )

        # Map schema response to API response model
        return QueryResponse(
            query=schema_response.query,
            query_type=schema_response.query_type,
            results=[
                QueryResult(
                    content=r.text,
                    score=r.score,
                    metadata=r.metadata,
                )
                for r in schema_response.results
            ],
            total_candidates=schema_response.total_candidates,
            execution_time_ms=schema_response.execution_time_ms,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/config")
async def get_config() -> ConfigResponse:
    """Get current RAG configuration."""
    try:
        engine = get_engine()
        config = engine.config

        return ConfigResponse(
            text_embedding_model=config.text_embedding_model,
            code_embedding_model=config.code_embedding_model,
            rerank_model=config.rerank_model,
            use_local_embeddings=config.use_local_embeddings,
            top_k_retrieval=config.top_k_retrieval,
            similarity_threshold=config.similarity_threshold,
        )
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/ui")
async def serve_ui() -> FileResponse:
    """Serve the web UI."""
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(ui_path)
