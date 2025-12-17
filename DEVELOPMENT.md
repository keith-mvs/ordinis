<!--
Retrieved from NVIDIA documentation
-->

# Ordinis – Local Development Guide

A lightweight, end‑to‑end setup for running the Ordinis AI‑driven trading system on a single workstation.
Production components are swapped for fast, in‑process equivalents.

---

## 1. Spin‑up a Development Stack

| Component        | Production                              | Development                              | Why? |
|------------------|----------------------------------------|------------------------------------------|------|
| **StreamingBus**| Kafka / NATS                           | **In‑memory bus**                        | Instant publish/subscribe, no external broker. |
| **Database**    | SQLite + WAL                           | SQLite (no backup)                       | Simple file‑based storage. |
| **Market Data** | Live APIs                              | **Mock CSV adapters**                    | Deterministic data, no API limits. |
| **Execution**   | Real broker / FIX                      | **Paper trader**                         | Instant fills, configurable slippage. |
| **Helix**        | Nemotron‑49B                           | **Nemotron‑8B**                          | Low latency, low memory usage. |
| **Governance**   | Strict                                 | **Relaxed**                              | Logs decisions but doesn’t abort on violations. |
| **Learning**     | CI jobs                                | **Local trainer**                        | Fast manual retraining. |

---

## 2. Configuration

The development environment is driven by `configs/dev.yaml`.

```yaml
environment: development
bus:
  type: in_memory
market_data:
  adapter: mock
  mock_path: data/mock/
execution:
  adapter: paper
helix:
  profile: dev_small
```

---

## 3. Running the System

### Live Demo Loop
Runs the full pipeline (Signal → Risk → Execution → Portfolio → Analytics) with mock data.

```shell
python scripts/demo_dev_system.py
```

### Interactive REPL
Drops you into a Python shell with the system already initialised.

```shell
python -i scripts/repl_dev.py
```

### Back‑testing
Execute the back‑test harness using the dev configuration.

```shell
python -m ordinis.engines.proofbench.run \
    --config configs/dev.yaml \
    --pack pack_01
```

---

## 4. Engine‑Specific Development Tasks

| Engine          | Command |
|-----------------|---------|
| **SignalEngine**| `python -m ordinis.engines.signalcore.train …` |
| **RiskEngine**  | `python -m ordinis.engines.riskguard.validate …` |
| **Cortex**      | `python -m ordinis.engines.cortex.analyze_code …` |
| **Synapse**     | `python -m ordinis.rag.synapse.retrieve …` |
| **CodeGen**     | `python -m ordinis.services.codegen.propose_change …` |

---

## 5. Setup

```shell
# Install all runtime and optional development dependencies
pip install -e ".[all]"
pip install -e ".[dev]"

# Run the demo to verify everything works
python scripts/demo_dev_system.py
```

---

*All commands above are intended for a **shell** (bash, zsh, PowerShell, etc.).*
If you prefer a different terminal, just replace `python` with the appropriate interpreter call.

---

*Happy hacking! 🎉*
