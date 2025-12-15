# Ordinis AI Architecture - Synthesis of Industry Patterns

**Date:** 2025-12-14
**Status:** Design - Phase 1 Complete

## Architecture Overview

Our AI system combines proven patterns from Azure, OpenAI, and Mistral AI:

```text
┌─────────────────────────────────────────────────────────────┐
│                    CodeGenService (AI Agent)                │
│              Inspired by: OpenAI Agents + Mistral Vibe      │
└──────────────────┬──────────────────────┬───────────────────┘
                   │                      │
        ┌──────────▼──────────┐  ┌───────▼────────┐
        │  Synapse (RAG)      │  │  Helix (LLM)   │
        │  Azure GPT-RAG      │  │  Multi-provider│
        │  Pattern            │  │  Routing       │
        └──────────┬──────────┘  └───────┬────────┘
                   │                      │
        ┌──────────▼──────────┐  ┌───────▼────────────────┐
        │  ChromaDB           │  │  Codestral/GPT-4.1     │
        │  Vector Store       │  │  OpenAI/Mistral/NVIDIA │
        └─────────────────────┘  └────────────────────────┘
```

## Design Patterns by Source

### 1. Azure GPT-RAG ([Source](https://github.com/Azure/GPT-RAG))

**What We Adopted:**

- Zero-Trust architecture for security
- AI Agent pattern for context-aware operations
- NL2SQL approach → adapted as NL2Code
- Enterprise observability and metrics
- Retrieval-first workflow

**Our Implementation:**

- Synapse retrieval engine with security filters
- CodeGenService as AI Agent
- Context-aware code generation
- Audit logging via GovernanceEngine

### 2. OpenAI Patterns ([Source](https://github.com/openai))

**Key Repositories Used:**

**openai-agents-python** (17.8k⭐):

- Multi-agent orchestration
- Workflow coordination
- Tool use patterns

**openai-cookbook** (69.7k⭐):

- RAG best practices
- Prompt engineering patterns
- Function calling examples

**evals** (17.4k⭐):

- Evaluation framework
- Quality benchmarks
- Testing patterns

**Our Implementation:**

- Multi-step agent workflows in CodeGenService
- Evaluation harness for generated code
- Cookbook-inspired prompt templates

### 3. Mistral AI Patterns ([Source](https://github.com/mistralai))

**Key Repositories Used:**

**mistral-vibe** (1.9k⭐):

- Minimal CLI coding agent
- Lightweight design
- Direct code generation

**mistral-inference** (10.6k⭐):

- Efficient inference
- Model optimization
- Local execution

**Our Implementation:**

- Lightweight CodeGenService API
- Optional CLI interface
- Codestral as primary code model

## Component Architecture

### Helix (LLM Provider Layer)

**Purpose:** Unified interface to multiple LLM providers
**Pattern:** OpenAI SDK abstraction + Azure enterprise features

**Providers:**

1. **Mistral AI** - Primary for code (Codestral 25.01)
2. **OpenAI** - Fallback + embeddings (GPT-4.1, GPT-4o)
3. **Azure OpenAI** - Enterprise deployments
4. **NVIDIA** - Reasoning models (Nemotron 49B)

**Features:**

- Auto-fallback between providers
- Rate limiting and caching
- Async operations
- Provider health checks

**Files:**

- `src/ordinis/ai/helix/engine.py`
- `src/ordinis/ai/helix/providers/{mistral,openai,azure,nvidia}.py`

### Synapse (RAG Engine)

**Purpose:** Context retrieval for code generation
**Pattern:** Azure GPT-RAG + OpenAI Cookbook RAG examples

**Capabilities:**

- Code pattern retrieval (NL2Code)
- Similar implementation search
- Security-filtered context
- Multi-collection search (text + code)

**Future Enhancements (Phase 2):**

- `retrieve_for_codegen()` - Agent-style context building
- `get_code_patterns()` - Pattern library access
- `find_similar_implementations()` - Codebase similarity search
- Observability metrics (retrieval quality, latency)

**Files:**

- `src/ordinis/ai/synapse/engine.py`
- `src/ordinis/rag/retrieval/engine.py`

### CodeGenService (AI Agent)

**Purpose:** AI-powered code generation and assistance
**Pattern:** OpenAI Agents + Mistral Vibe + Azure GPT-RAG

**Inspired By:**

- **OpenAI Agents:** Multi-step workflows, tool orchestration
- **Mistral Vibe:** Minimal, focused code generation
- **Azure GPT-RAG:** Context-aware, secure, observable

**Core Methods:**

```python
class CodeGenService:
    async def generate_function(request) -> CodeGenResponse:
        """Generate function with context from Synapse."""

    async def generate_nl2code(query: str) -> CodeGenResponse:
        """Natural language to code (like NL2SQL)."""

    async def refine_code(code: str, refinement: str) -> CodeGenResponse:
        """Iterative improvement with context."""

    async def generate_tests(code: str) -> CodeGenResponse:
        """Context-aware test generation."""

    async def explain_code(code: str) -> ExplanationResponse:
        """Code explanation with codebase context."""

    async def review_code(code: str, criteria) -> ReviewResponse:
        """Security + best practices review."""
```

**Safety (Responsible AI):**

- No secrets/API keys in generated code
- License compliance checking
- Pattern validation (no banned functions)
- Audit logging
- Governance engine integration

**Files (To Create - Phase 3):**

- `src/ordinis/ai/codegen/service.py`
- `src/ordinis/ai/codegen/safety.py`
- `src/ordinis/ai/codegen/prompts.py`

## Model Selection Strategy

### Code Generation Primary: Mistral Codestral 25.01

**Why:** [#1 on LMsys copilot leaderboard](https://mistral.ai/news/codestral-2501)

- 2x faster than previous version
- 80+ programming languages
- 256k context window
- Optimized for code completion

**Fallback:** OpenAI GPT-4.1 (54.6% SWE-bench)

### Chat/Reasoning: Multi-Provider

**Primary:** Codestral or GPT-4o
**Fallback:** NVIDIA Nemotron 49B
**Enterprise:** Azure GPT-4

### Embeddings: OpenAI text-embedding-3-large

**Why:** Latest model, best performance
**Fallback:** NVIDIA nv-embedqa
**Deprecation Note:** Migrate before Oct 2025

## Evaluation Strategy

**Inspired by:** [OpenAI/evals](https://github.com/openai/evals)

**Metrics (Future - Phase 5):**

- Code correctness (unit test pass rate)
- Code quality (linting, complexity)
- Security (no secrets, banned patterns)
- Performance (generation latency)
- Context relevance (retrieval quality)

**Testing:**

- Unit tests per component
- Integration tests (end-to-end)
- Safety validation tests
- Benchmark suite (code generation tasks)

## Security Architecture

**Based on:** Azure GPT-RAG Zero-Trust principles

**Layers:**

1. **Input Validation** - Sanitize all user input
2. **Context Filtering** - No secrets in RAG retrieval
3. **Output Validation** - Scan generated code for secrets
4. **Audit Logging** - All operations logged
5. **Governance** - Policy enforcement via GovernanceEngine

**Implementation:**

- `src/ordinis/ai/codegen/safety.py` - Safety filters
- `src/ordinis/engines/governance/` - Policy engine
- Synapse security filters

## Observability

**Inspired by:** Azure GPT-RAG end-to-end observability

**Metrics to Track:**

- Request latency (by provider, by model)
- Token usage and cost
- Cache hit rates
- Retrieval quality (relevance scores)
- Generation success rates
- Error rates by provider

**Tools:**

- Structured logging (loguru)
- Metrics collection (prometheus-client)
- Distributed tracing (opentelemetry)

## Implementation Status

### Phase 1: Complete ✓

- [x] Helix multi-provider system
- [x] Mistral, OpenAI, Azure providers
- [x] Model selection and fallback
- [x] Async operations

### Phase 2: Pending

- [ ] Synapse enhancements (Azure GPT-RAG pattern)
- [ ] Context-aware retrieval methods
- [ ] Security filtering
- [ ] Observability metrics

### Phase 3: Pending

- [ ] CodeGenService (AI Agent pattern)
- [ ] NL2Code capability
- [ ] Safety/validation layer
- [ ] Prompt engineering

### Phase 4: Pending

- [ ] RAG-LLM integration layer
- [ ] Code assistant helpers
- [ ] Context builders

### Phase 5: Pending

- [ ] Evaluation framework (OpenAI evals-style)
- [ ] Benchmark suite
- [ ] Performance testing
- [ ] Documentation

## Dependencies

**Current:**

- `openai>=1.0.0` (already in core)
- `langchain-nvidia-ai-endpoints` (NVIDIA provider)

**New (Phase 2-3):**

```toml
[project.optional-dependencies]
codegen = [
    "mistralai>=0.1.0",      # Mistral provider
    "tree-sitter>=0.20",     # Code parsing
    "black>=23.0",           # Code formatting
    "ruff>=0.1.0",           # Linting
]
```

## References

All implementations reference official patterns:

1. **Azure/GPT-RAG** - Enterprise RAG architecture
2. **OpenAI/openai-agents-python** - Agent orchestration
3. **OpenAI/openai-cookbook** - Best practices
4. **OpenAI/evals** - Evaluation framework
5. **Mistral/mistral-vibe** - Minimal coding agent
6. **Mistral/mistral-inference** - Inference optimization

## Next Steps

1. **Phase 2:** Enhance Synapse with Azure GPT-RAG patterns
2. **Phase 3:** Build CodeGenService as AI Agent
3. **Phase 4:** Create integration helpers
4. **Phase 5:** Implement evaluation framework

**Ready for Phase 2 implementation when context allows.**
