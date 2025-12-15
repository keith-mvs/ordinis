# Helix LLM Provider - Quick Setup Guide

Helix is Ordinis's integrated NVIDIA AI development assistant, providing access to Nemotron Super 49B for code review, architecture guidance, and strategy design.

---

## Prerequisites

- Python 3.11+
- NVIDIA API key from [build.nvidia.com](https://build.nvidia.com/)
- Ordinis environment configured

---

## Installation

### 1. Install Dependencies

```powershell
# Activate Ordinis environment
conda activate ordinis-env

# Install NVIDIA SDK
pip install langchain-nvidia-ai-endpoints
```

### 2. Configure API Key

```powershell
# Set environment variable (Windows PowerShell)
$env:NVIDIA_API_KEY = "nvapi-your-key-here"

# Or add to .env file
echo "NVIDIA_API_KEY=nvapi-your-key-here" >> .env
```

### 3. Verify Installation

```powershell
python test_helix_quick.py
```

Expected output:
```
Checking health...
Health: {'status': 'healthy', ...}

Testing code review...
Response: [Code review feedback]
Tokens used: ~150
Latency: ~500ms
```

---

## Quick Start

### Option 1: Quick Test

```powershell
# Health check and basic test
python test_helix_quick.py
```

### Option 2: Development Tests

```powershell
# Code review
python test_helix_dev.py review

# Strategy design
python test_helix_dev.py strategy

# Architecture consultation
python test_helix_dev.py architecture
```

### Option 3: Dev Assistant CLI

```powershell
# Review specific file
python scripts/helix_dev_assistant.py review src/ordinis/engines/cortex/core/engine.py

# Interactive chat
python scripts/helix_dev_assistant.py chat

# Strategy design
python scripts/helix_dev_assistant.py strategy "Iron condor with 10 delta wings, 45 DTE"

# Generate documentation
python scripts/helix_dev_assistant.py docs src/ordinis/ai/helix/engine.py

# Architecture consultation
python scripts/helix_dev_assistant.py architecture "How should engines communicate?"
```

### Option 4: Interactive Work Session

```powershell
# Start extended consultation session
python helix_work_session.py

# In session:
You: Review the CortexEngine implementation
Helix: [Provides detailed review]

You: /file src/ordinis/engines/cortex/core/engine.py
Loaded: src/ordinis/engines/cortex/core/engine.py (5000 chars)

You: What are the top 3 improvements?
Helix: [Analyzes loaded file and provides recommendations]

You: /metrics
[Shows session stats]

You: /export
Conversation exported to: session_20251214_143000_conversation.md

You: /exit
```

---

## Models Available

### Chat Models
- **nemotron-super** (default): `nvidia/llama-3.3-nemotron-super-49b-v1.5` (49B params)
  - Best for: Code review, architecture, complex reasoning
  - Latency: ~500ms per request
  - Cost: ~$0.001 per 1K tokens

- **nemotron-8b** (fallback): `nvidia/llama-3.2-nemotron-8b-instruct`
  - Best for: Quick questions, simple tasks
  - Latency: ~200ms per request
  - Cost: Lower than Super

### Embedding Models
- **nv-embedqa-e5-v5** (default): 1024-dimensional embeddings
- **nemoretriever-300m**: 300M parameter retrieval model

---

## Development Workflow

### 1. Code Review Workflow

```powershell
# Review new or modified code
python scripts/helix_dev_assistant.py review src/ordinis/new_feature.py

# Address feedback, iterate
python scripts/helix_dev_assistant.py review src/ordinis/new_feature.py

# Generate tests based on recommendations
python scripts/helix_dev_assistant.py chat
You: Generate pytest test cases for the improvements you recommended
```

### 2. Strategy Design Workflow

```powershell
# Design new strategy
python scripts/helix_dev_assistant.py strategy "Volatility breakout with dynamic position sizing"

# Refine design
python scripts/helix_dev_assistant.py chat
You: How should we handle overnight risk in this strategy?

# Implement and review
python scripts/helix_dev_assistant.py review src/ordinis/strategies/vol_breakout.py
```

### 3. Architecture Consultation Workflow

```powershell
# Ask architecture question
python scripts/helix_dev_assistant.py architecture "Should engines use direct calls or event bus?"

# Extended discussion
python helix_work_session.py
You: I need to design inter-engine communication for low-latency trading
Helix: [Provides pattern recommendations]
You: /file src/ordinis/engines/cortex/core/engine.py
You: How would you refactor this to support async messaging?
```

---

## Best Practices

### Temperature Settings
- **0.0-0.2**: Code review, documentation, testing (deterministic)
- **0.2-0.5**: Architecture, strategy design (balanced)
- **0.5-0.8**: Creative ideation, exploration (not recommended for code)

### Token Limits
- **Quick questions**: 500-1000 tokens
- **Code review**: 1500-2500 tokens
- **Strategy design**: 2000-3000 tokens
- **Architecture**: 1500-2500 tokens

### Cost Management
- Average request: 500-2000 tokens (~$0.001-$0.002)
- Code review session: ~5000 tokens (~$0.005)
- Extended consultation: ~20,000 tokens (~$0.02)

### Conversation History
- Keep last 10-15 exchanges for context
- Export important sessions with `/export`
- Clear history for new topics

---

## Troubleshooting

### Issue: Import Error

```
ImportError: No module named 'langchain_nvidia_ai_endpoints'
```

**Solution:**
```powershell
pip install langchain-nvidia-ai-endpoints
```

### Issue: API Key Not Found

```
ValueError: NVIDIA API key required
```

**Solution:**
```powershell
# Set environment variable
$env:NVIDIA_API_KEY = "nvapi-your-key-here"

# Verify
echo $env:NVIDIA_API_KEY
```

### Issue: Rate Limit Exceeded

```
RateLimitError: Too many requests
```

**Solution:**
- Wait 60 seconds
- Reduce request frequency
- Use lower max_tokens setting

### Issue: Timeout

```
TimeoutError: Request timed out
```

**Solution:**
- Reduce code size being reviewed
- Increase timeout in config
- Split large files into smaller chunks

---

## Next Steps

1. **Run Tests**: `python test_helix_quick.py`
2. **Try Dev Assistant**: `python scripts/helix_dev_assistant.py chat`
3. **Review Code**: `python scripts/helix_dev_assistant.py review <your_file>.py`
4. **Read Full Documentation**: `HELIX_DEV_SESSION_SUMMARY.md`

---

## Support

- Documentation: `docs/architecture/nvidia-integration.md`
- Issues: GitHub Issues
- API Reference: [NVIDIA Build Platform](https://build.nvidia.com/)

---

**Last Updated**: 2025-12-14
**Version**: 1.0.0
