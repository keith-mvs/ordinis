
# AI Integration & Models
**Version:** 1.0.0
**Last Updated:** December 15, 2025
**Maintainer:** Ordinis Core Team

---

## Overview
This document details the integration of AI/ML components (Cortex, Helix, Synapse) within the Ordinis system.

## Components
- **Cortex:** LLM Advisory Engine
- **Helix:** Developer Assistant
- **Synapse:** RAG & Memory System


## Model Specifications

| Engine    | Model Name                                      | Model Card Link                                                                                                                                  | Purpose/Notes                                      |
|-----------|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
| Cortex    | Meta Llama 3.1 405B Instruct                    | [meta/llama-3.1-405b-instruct](https://build.nvidia.com/meta/llama-3_1-405b-instruct)                                                            | Deep code analysis, strategy reasoning             |
| Helix     | Mistral Large 3 (675B) Instruct 2512            | [mistralai/mistral-large-3-675b-instruct-2512](https://build.nvidia.com/mistralai/mistral-large-3-675b-instruct-2512)                            | Developer Assistant, Code Generation               |
| Synapse   | NVIDIA Llama 3.2 NeMo Retriever 300M Embed V2   | [nvidia/llama-3.2-nemoretriever-300m-embed-v2](https://build.nvidia.com/nvidia/llama-3_2-nemoretriever-300m-embed-v2)                            | Semantic search embedding                          |
| Synapse   | NVIDIA Llama 3.2 NeMo Retriever 500M Rerank V2  | [nvidia/llama-3.2-nemoretriever-500m-rerank-v2](https://build.nvidia.com/nvidia/llama-3_2-nemoretriever-500m-rerank-v2)                          | RAG Reranking                                      |
| SignalCore| Meta Llama 3.3 70B Instruct                     | [meta/llama-3.3-70b-instruct](https://build.nvidia.com/meta/llama-3_3-70b-instruct)                                                              | Signal interpretation, feature suggestions         |
| RiskGuard | Meta Llama 3.3 70B Instruct                     | [meta/llama-3.3-70b-instruct](https://build.nvidia.com/meta/llama-3_3-70b-instruct)                                                              | Trade risk explanations                            |
| RiskGuard | Llama 3.1 NemoGuard 8B Content Safety           | [nvidia/llama-3.1-nemoguard-8b-content-safety](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety)                            | Content safety guardrails                          |
| AnalyticsEngine| Meta Llama 3.3 70B Instruct                     | [meta/llama-3.3-70b-instruct](https://build.nvidia.com/meta/llama-3_3-70b-instruct)                                                              | Backtest narration, optimization                   |
Portfio
---

For more details, see the linked model cards and [Security & Compliance](security-compliance.md).

---
