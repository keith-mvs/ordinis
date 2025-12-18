Repo architecture and **naming cleanup conventions** that will keep Ordinis from turning into a junk drawer as it grows.
Target repo layout (Python-first, “src/” layout, Clean Architecture)
--------------------------------------------------------------------

    ordinis/
      README.md
      LICENSE
      pyproject.toml
      ruff.toml              # or in pyproject
      mypy.ini               # optional
      .pre-commit-config.yaml
      .gitignore
      Makefile               # or justfile
      .github/
        workflows/
          ci.yaml

      src/
        ordinis/
          __init__.py

          core/              # pure domain + contracts (no I/O)
            events.py
            types.py          # Money, Qty, Price, Timeframe, etc.
            instruments.py
            orders.py
            fills.py
            portfolio.py
            risk.py
            protocols/        # Ports (Protocols)
              event_bus.py
              broker.py
              execution.py
              fill_model.py
              cost_model.py
              risk_policy.py

          application/        # use-cases / orchestration logic (still no external I/O)
            backtest/
              run_backtest.py
            paper/
              run_paper.py
            research/
              run_research.py
            services/
              signal_service.py
              risk_service.py
              execution_service.py

          engines/            # “business components” wired via ports
            signalcore/
            proofbench/
            riskguard/
            optionscore/
            cortexrag/

          adapters/           # I/O implementations (outbound + inbound)
            market_data/
              alpha_vantage.py
              finnhub.py
              polygon.py
              twelve_data.py
            broker/
              alpaca.py
            storage/
              parquet_store.py
              duckdb_store.py
              sqlite_store.py
            event_bus/
              in_memory.py
              asyncio_queue.py
            telemetry/
              prometheus.py
              logging_structured.py

          interface/          # delivery mechanisms
            cli/
              __main__.py
              commands/
                backtest.py
                paper.py
                admin.py
            api/              # future REST
              app.py

          runtime/            # glue + DI/wiring
            config.py
            container.py      # dependency injection / registry
            bootstrap.py
            logging.py

      configs/
        default.yaml
        environments/
          dev.yaml
          prod.yaml

      tests/
        unit/
        integration/
        e2e/

      docs/
      examples/
      scripts/
      tools/

      artifacts/             # GENERATED OUTPUTS ONLY (never committed)
        runs/
        reports/
        logs/
        cache/

### Dependency rules (non-negotiable if you want scale)

* `core/` imports **nothing** outside `core/`.

* `application/` imports `core/` and `core.protocols/` only.

* `engines/` imports `core/` + `core.protocols/` (and optionally `application/` services if you centralize orchestration there).

* `adapters/` can import anything it must, but **nothing in core/application depends on adapters**.

* `interface/` (CLI/API) calls **application use-cases**, not engines directly.

This keeps the repo from becoming “everything imports everything”.

* * *

File and symbol naming conventions
----------------------------------

### Python modules / packages

* **snake_case.py** for files and directories
  `risk_policy.py`, `execution_engine.py`, `order_router.py`

* Don’t abbreviate unless universal (e.g., `api`, `oms`, `pnl`).

### Classes / Protocols

* **PascalCase**: `EventBus`, `BrokerAdapter`, `RiskPolicy`

* Protocol files live in `core/protocols/`, implementation lives in `adapters/` or `engines/`.

### Events

* Event classes: `SomethingEvent`

* Event topic/type string: `domain.action` convention
  Examples: `market.bar`, `signal.generated`, `risk.rejected`, `order.submitted`, `fill.received`

### Config

* `configs/default.yaml` + `configs/environments/{dev,prod}.yaml`

* One validated config object at runtime (`runtime/config.py`), no random `os.getenv()` spread across modules.

* * *

“Admin-first” cleanup discipline (your repo stays clean automatically)
----------------------------------------------------------------------

### Rule 1: All generated output goes to `artifacts/`

No exceptions. Not `data/`, not random folders, not the repo root.

**Run folder naming:**
    artifacts/runs/YYYYMMDD/HHMMSSZ_<task>_<gitsha>/
      config.snapshot.yaml
      inputs.manifest.json
      results.parquet
      metrics.json
      logs.jsonl
      report.html

### Rule 2: Generated files never get committed

Add to `.gitignore`:
    artifacts/
    **/__pycache__/
    .pytest_cache/
    .mypy_cache/
    .ruff_cache/
    *.log
    *.tmp
    .DS_Store

### Rule 3: Built-in cleanup command

Add a CLI admin command:

* `ordinis admin clean --older-than 14d`

* `ordinis admin clean --all` (nukes artifacts/cache/logs)

Implementation policy:

* Delete only within `artifacts/`

* Support retention by age + max total size (cap runaway disk use)

### Rule 4: Task lifecycle convention

Every task should:

1. Create a run directory under `artifacts/runs/...`

2. Write a **config snapshot**

3. Write a **manifest** of inputs (symbols, date range, provider versions)

4. Write outputs only there

5. On success: keep the run folder

6. On failure: keep it but mark `status=failed` in a `run.json` metadata file

This gives you auditability _and_ controlled mess.

* * *

Scaling enhancements that pay off immediately
---------------------------------------------

* Add `pre-commit` hooks: `ruff`, `ruff-format`, `mypy`, `pytest -q` (fast checks only).

* Add `Makefile` targets: `make fmt`, `make lint`, `make test`, `make clean`.

* Adopt `Parquet + DuckDB` for research/backtests (CSV becomes a bottleneck fast).

* Use plugin entry points later (not a random `src/plugins` dump). Keep plugin contracts in `core/protocols/`.

* * *

One decision I need from you (pick one)
---------------------------------------

Is Ordinis primarily being packaged as:
**A)** a reusable **Python library** others import, or
**B)** a deployable **service/app** (CLI first, API later)?

Reply with **A** or **B** and I’ll tighten the structure (especially `interface/` + `runtime/` + packaging) to match that direction.
