# Engine‑Specific CLI Commands

| Engine          | Description |
|-----------------|-------------|
| **SignalEngine** | Train the predictive models that emit trading signals. |
| **RiskEngine**   | Validate risk‑policy compliance against a portfolio or a set of signals. |
| **Cortex**       | LLM‑driven code analysis, style checks, and complexity/security reviews. |
| **Synapse**      | Retrieval‑augmented search over documentation, research papers, and code. |
| **CodeGen**      | Generate or patch code with an LLM, optionally running unit‑tests on the diff. |

Below each row is a detailed breakdown.

---

## 1️⃣ SignalEngine – `python -m ordinis.engines.signalcore.train`

**Purpose**
Train a tabular‑ML or deep‑learning model (GBM, XGBoost, LSTM, Transformer, …) that produces `Signal` objects for the downstream pipeline.

### Key Options
| Option | Meaning |
|--------|---------|
| `--config FILE` | Path to a YAML config that supplies defaults (data paths, hyper‑parameters, etc.). |
| `--data DIR` | Directory containing the training CSV/Parquet files. |
| `--model TYPE` | Model family: `gbm`, `xgboost`, `lstm`, `transformer`. |
| `--epochs N` | Number of training epochs (deep models). |
| `--batch-size N` | Mini‑batch size for GPU training. |
| `--learning-rate LR` | Learning‑rate for optimisers. |
| `--output-dir DIR` | Where the trained model, checkpoint, and metrics are saved. |
| `--log-level LEVEL` | `DEBUG` | `INFO` | `WARNING` | `ERROR`. |
| `--seed SEED` | Random seed for reproducibility. |
| `--dry-run` | Validate the config and data layout without launching training. |
| `--device ID` | GPU index (`0` by default). |

### Example

```shell
python -m ordinis.engines.signalcore.train \
    --config configs/signal.yaml \
    --data data/market/ \
    --model lstm \
    --epochs 20 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --output-dir models/signal_lstm_v1 \
    --log-level INFO \
    --seed 42
```

### Tips

- Set `MLFLOW_TRACKING_URI` (or any other experiment‑tracker) before running – the command will automatically log metrics.
- Use `--dry-run` when you only want to sanity‑check the YAML config.
- For quick prototyping, `--model gbm` runs entirely on CPU and finishes in seconds.

---

## 2️⃣ RiskEngine – `python -m ordinis.engines.riskguard.validate`

**Purpose**
Run deterministic risk checks (exposure caps, sector limits, VaR limits, etc.) against a portfolio snapshot or a batch of signals.

### Key Options
| Option | Meaning |
|--------|---------|
| `--policy FILE` | YAML file that defines the risk rule set. |
| `--portfolio FILE` | JSON/Parquet snapshot of current positions, cash, margin. |
| `--signal FILE` | Optional file containing signals to validate (e.g., a CSV export). |
| `--output FILE` | Path for a JSON report summarising passes/failures. |
| `--dry-run` | Run the checks but never exit with a non‑zero status. |
| `--log-level LEVEL` | Verbosity of the console output. |
| `--device ID` | If any auxiliary model (e.g., volatility forecaster) is GPU‑accelerated. |

### Example

```shell
python -m ordinis.engines.riskguard.validate \
    --policy policies/risk.yaml \
    --portfolio data/portfolio_snapshot.json \
    --output reports/risk_validation_20251215.json \
    --log-level DEBUG
```

### Tips

- The command returns exit code `0` on success, `1` on any violation – perfect for CI gates.
- Combine with `--dry-run` in a pre‑commit hook to surface policy breaches before code is merged.
- Use `--device 1` if you have a GPU‑based VaR model enabled.

---

## 3️⃣ Cortex – `python -m ordinis.engines.cortex.analyze_code`

**Purpose**
Leverage the LLM stack (via **Helix**) to perform static code analysis, security reviews, or complexity estimates.

### Key Options

| Option | Meaning |
|--------|---------|
| `--file PATH` | Analyse a single source file. |
| `--dir PATH` | Recursively analyse every `*.py` (or other extensions) under the directory. |
| `--type TYPE` | `review` (default), `complexity`, `security`. |
| `--output FILE` | Write a structured JSON report (`issues`, `suggestions`, `severity`). |
| `--model-id ID` | Override the default Helix model (e.g., `nemotron-8b-v3.1`). |
| `--log-level LEVEL` | Logging verbosity. |
| `--dry-run` | Show which files would be sent to the LLM without invoking it. |
| `--device ID` | GPU to use for the LLM inference (if a GPU‑enabled model is selected). |

### Example

```shell
python -m ordinis.engines.cortex.analyze_code \
    --dir src/ordinis/engines/ \
    --type security \
    --output reports/cortex_security_20251215.json \
    --log-level INFO
```

### Tips

- Set `HELIX_PROFILE=dev_small` (or export `ORDINIS_HELIX_PROFILE`) to force the lightweight Nemotron‑8B in dev environments.
- The `--dry-run` flag is handy when you want to audit the file list before incurring LLM costs.
- Results can be fed directly into the **GovernanceEngine** for automated policy enforcement.

---

## 4️⃣ Synapse – `python -m ordinis.rag.synapse.retrieve`

**Purpose**
Perform vector‑search retrieval over the knowledge base (documents, research papers, code snippets) using the **EmbedLM‑300M** embedding model and optional FAISS/Elastic index.

### Key Options
| Option | Meaning |
|--------|---------|
| `--query TEXT` | Natural‑language query string. |
| `--top-k N` | Number of results to return (default 5). |
| `--index NAME` | Name of the FAISS/Elastic index to query (e.g., `docs_index`). |
| `--output FILE` | Write the retrieved snippets as JSON (`id`, `score`, `text`, `metadata`). |
| `--max-tokens N` | Truncate each snippet to at most *N* tokens (helps keep LLM prompts short). |
| `--log-level LEVEL` | Verbosity. |
| `--device ID` | GPU for the embedding model (optional). |
| `--dry-run` | Show which index would be queried without performing the search. |

### Example
```shell
python -m ordinis.rag.synapse.retrieve \
    --query "Explain the VaR calculation used in RiskEngine" \
    --top-k 3 \
    --index docs_index \
    --output reports/synapse_vaR.json \
    --log-level INFO
```

### Tips
- If the requested index does not exist, Synapse falls back to a BM25 keyword search.
- The environment variable `SYNAPSE_EMBED_MODEL` can be set to swap the embedding model (e.g., to a larger `EmbedLM‑1B`).
- Use the `--max-tokens` flag to keep the retrieved context within the LLM token budget.

---

## 5️⃣ CodeGen – `python -m ordinis.services.codegen.propose_change`

**Purpose**
Generate a code diff (or a full file) from a natural‑language description, optionally run unit tests on the proposed change, and store a comprehensive report.

### Key Options
| Option | Meaning |
|--------|---------|
| `--prompt TEXT` | Description of the desired change (e.g., “Add logging of order latency”). |
| `--files PATH[,PATH...]` | Comma‑separated list of files that provide context for the LLM. |
| `--model-id ID` | Choose a specific Helix model (default `nemotron-super-49b-v1.5`). |
| `--output-dir DIR` | Directory where the diff, test results, and a JSON report are written. |
| `--run-tests` | After generating the diff, invoke `pytest` (or the project's test runner) and embed the outcome. |
| `--dry-run` | Show the generated diff on stdout without writing any files. |
| `--log-level LEVEL` | Logging verbosity. |
| `--device ID` | GPU for the LLM inference. |
| `--seed SEED` | Deterministic generation seed (if the model supports it). |

### Example
```shell
python -m ordinis.services.codegen.propose_change \
    --prompt "Add execution‑latency logging to ExecutionEngine" \
    --files src/ordinis/engines/execution_engine.py \
    --run-tests \
    --output-dir gen/changes/20251215 \
    --log-level INFO
```

### Tips
- The command automatically runs the **RiskGuard** content‑safety filter to strip secrets, disallowed licenses, or unsafe patterns.
- The generated JSON report (`report.json`) contains: the diff, token usage, test summary, and any safety‑filter warnings.
- For quick “one‑liner” patches, omit `--run-tests` and use `--dry-run` to preview before committing.

---

## Common Flags Across All Commands
| Flag | Description |
|------|-------------|
| `-h, --help` | Show the help message and exit. |
| `--config FILE` | Load defaults from a YAML configuration file. |
| `--log-level LEVEL` | Set logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `--dry-run` | Validate inputs and display what would happen without side‑effects. |
| `--output FILE/DIR` | Destination for generated artifacts (models, reports, diffs). |
| `--profile PROFILE` | Choose a Helix profile (`dev_small`, `prod_large`). |
| `--device ID` | GPU device index (default 0). |
| `--seed SEED` | Random seed for reproducibility (training, LLM generation). |

All commands support `--help` to list the full set of options:

```shell
python -m ordinis.engines.signalcore.train --help
python -m ordinis.engines.riskguard.validate --help
python -m ordinis.engines.cortex.analyze_code --help
python -m ordinis.rag.synapse.retrieve --help
python -m ordinis.services.codegen.propose_change --help
```

---

## Integration & Best‑Practice Tips
| Context | Recommendation |
|---------|----------------|
| **CI/CD pipelines** | Add `riskguard.validate` as a gate step; fail the build if the exit code is non‑zero. |
| **Experiment tracking** | Export `MLFLOW_TRACKING_URI` (or your preferred tracker) before running `signalcore.train`. |
| **Versioning** | Include the Git SHA in the `--output-dir` or `--output` name (e.g., `models/signal_lstm_${GIT_SHA}`). |
| **GPU utilisation** | Set `CUDA_VISIBLE_DEVICES` to the desired GPU(s) and monitor with `nvidia‑smi`. |
| **LLM cost control** | Use the `dev_small` Helix profile for local development; switch to `prod_large` only for production‑grade runs. |
| **Safety** | Keep `RiskGuard` enabled for `codegen.propose_change` to automatically strip secrets and disallowed content. |
| **Reproducibility** | Always pass `--seed` (or set `PYTHONHASHSEED`) when training or generating code. |

---

### Quick Reference Cheat‑Sheet
```text
SignalEngine   : python -m ordinis.engines.signalcore.train   [options]
RiskEngine     : python -m ordinis.engines.riskguard.validate [options]
Cortex         : python -m ordinis.engines.cortex.analyze_code [options]
Synapse        : python -m ordinis.rag.synapse.retrieve        [options]
CodeGen        : python -m ordinis.services.codegen.propose_change [options]
```

Copy the sections you need into your `ARCHITECTURE.md` (or a dedicated `USAGE.md`) and you’ll have a **complete, self‑contained guide** for all developer‑facing CLI entry points. Happy hacking! 🚀
