# Repository Guidelines

## Project Structure & Module Organization
- `src/ordinis/` implements the clean architecture: core/application logic, adapters (storage, market_data, alerting, telemetry), engines (cortex, flowroute, proofbench, riskguard, signalcore), and interface layers (`cli`, dashboard) plus runtime, safety, plugins, analysis, visualization, and rag.
- `tests/` mirrors `src/` with fixtures in `tests/fixtures/`; `configs/` holds environment presets (e.g., `configs/dev.yaml`).
- `scripts/` contains demos and utilities; `examples/` shows usage patterns; `docs/` hosts architecture and user guides; runtime outputs stay in `data/`, `artifacts/`, and `logs/` (ignored).

## Build, Test, and Development Commands
- `make dev` installs dev dependencies and pre-commit hooks; `make fmt` formats and auto-fixes with ruff.
- `make lint` runs `ruff check` and `mypy`; `make test` or `make test-quick` runs pytest (verbose, short tracebacks).
- `make check` chains fmt+lint+tests; `make coverage` produces HTML coverage in `htmlcov/`.
- Run the mock-data demo with `python scripts/demo_dev_system.py`; back-test via `python -m ordinis.engines.proofbench.run --config configs/dev.yaml --pack pack_01`.

## Coding Style & Naming Conventions
- Target Python 3.11, 100-character lines, and full type hints; Google-style docstrings for public APIs.
- Imports are sorted (isort via ruff); prefer single-responsibility functions under ~50 lines with descriptive names.
- Branch prefixes: `feature/`, `fix/`, `docs/`, `refactor/`, `test/`; keep tests colocated with the corresponding module paths.
- Before committing, run `ruff format .`, `ruff check .`, and `mypy src/ --ignore-missing-imports` (or `make check`).

## Testing Guidelines
- Pytest is standard; structure tests to mirror `src/` modules and mock external services.
- Use deterministic fixtures; skip slow cases with `-m "not slow"` (see `make test-quick`).
- Aim for >80% coverage on business logic; generate reports with `pytest --cov=src --cov-report=html`.

## Commit & Pull Request Guidelines
- Commit subjects are imperative with optional scopes, e.g., `Fix ...`, `docs: ...`, `chore: ...`; keep the first line <72 chars.
- Include only focused, reviewable changes with updated tests/docs; never commit secrets or large data files.
- PRs should link issues, summarize behavior changes, note config impacts, attach UI screenshots if relevant, and include test results; ensure `pre-commit run --all-files` passes.

## Security & Configuration Tips
- Copy `.env.example` to `.env` and inject API keys locally; keep secrets and datasets out of Git.
- Use `configs/dev.yaml` for local runs; enable live-trading adapters only with explicit credentials and governance checks in place.
