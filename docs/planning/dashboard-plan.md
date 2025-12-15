# Ordinis Dashboard Specification: "The Control Room"

**Version:** 1.0.0
**Date:** 2025-12-14
**Status:** Draft

## Overview
This document outlines the specification for the Ordinis Dashboard, a web-based interface designed to provide retroactive analysis, observability, and governance for the Ordinis trading system. Inspired by **Mistral AI Studio**, this dashboard serves as the "Control Room" for developers and quantitative analysts to review AI decision-making, debug strategy failures, and ensure compliance.

## Core Philosophy
*   **Retroactive Analysis:** The primary goal is not just real-time monitoring, but deep post-mortem analysis of *why* a trade was made or missed.
*   **Explainability:** Every AI output (Cortex strategy, Synapse retrieval) must be traceable to its inputs (Market Data, Prompts, Context).
*   **Governance First:** Compliance checks and "Judge" evaluations are first-class citizens, visible alongside performance metrics.

## Key Features

### 1. The "Trace" View (Deep Observability)
*   **Concept:** A timeline visualization of a single "Trading Cycle" or "Event".
*   **Data Points:**
    *   **Trigger:** The market event (e.g., "AAPL price update") that started the cycle.
    *   **Synapse Retrieval:** What documents/news were fetched? (Show the actual text chunks).
    *   **Cortex Reasoning:** The full "Chain of Thought" produced by `nemotron-ultra`.
    *   **Risk Check:** The result of the pre-trade risk guardrails.
    *   **Execution:** The final order sent to the broker.
*   **UI Inspiration:** Distributed tracing tools (Jaeger/Zipkin) but tailored for AI logic flows.

### 2. The "Judge" Panel (Automated Evaluation)
*   **Concept:** Retroactive scoring of AI decisions using the `nemotron-reward` model.
*   **Metrics:**
    *   **Safety Score:** Did the model suggest anything risky or hallucinated?
    *   **Strategy Adherence:** Did the reasoning match the provided system prompt instructions?
    *   **Outcome:** (Computed later) Did the trade result in profit or loss?
*   **Workflow:** A nightly batch job runs the "Judge" over all daily logs and populates this panel with "Red/Green" indicators.

### 3. The "Playground" (Counterfactual Analysis)
*   **Concept:** A sandbox to "replay" a historical event with modified parameters.
*   **Capabilities:**
    *   **Edit Prompt:** Change the system prompt for Cortex and re-run the specific historical tick.
    *   **Swap Model:** See how `nemotron-super` would have handled the situation vs `nemotron-ultra`.
    *   **Inject Context:** Manually add/remove RAG documents to test sensitivity to information.

### 4. System Health & Lineage
*   **Asset Registry:** A table showing which version of the Codebase, Prompts, and Models were active at any point in time.
*   **Drift Detection:** Charts showing the distribution of Model Confidence scores over time (detecting if the model is becoming "unsure").

## Technical Architecture (Proposed)
*   **Backend:** Python (FastAPI) serving data from the `artifacts/logs` and `ordinis.db`.
*   **Frontend:** Streamlit (for rapid prototyping) or Next.js (for production).
*   **Data Source:** Structured logs (JSONL) and SQLite/PostgreSQL.

## Implementation Roadmap
1.  **Phase 1 (Logging):** Ensure all Engines (Helix, Cortex, Synapse) emit structured logs with `trace_id` correlation.
2.  **Phase 2 (Backend):** Build a simple API to query logs by Time, Symbol, or Trade ID.
3.  **Phase 3 (UI):** Build the "Trace View" in Streamlit.
4.  **Phase 4 (Judges):** Implement the `nemotron-reward` evaluation pipeline.
