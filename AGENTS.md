# Repository Guidelines

## Project Structure & Module Organization
- `src/ordinis/` implements the clean architecture: core/application logic, adapters (storage, market_data, alerting, telemetry), engines (cortex, flowroute, proofbench, riskguard, signalcore), and interface layers (`cli`, dashboard) plus runtime, safety, plugins, analysis, visualization, and rag.
- `tests/` mirrors `src/` with fixtures in `tests/fixtures/`; `configs/` holds environment presets (e.g., `configs/dev.yaml`).
- `scripts/` contains demos and utilities; `examples/` shows usage patterns; `docs/` hosts architecture and user guides; runtime outputs stay in `data/`, `artifacts/`, and `logs/` (ignored).

## Build, Test, and Development Commands
- `make dev` installs dev dependencies and pre-commit hooks; `make fmt` formats and auto-fixes with ruff.
- `make lint` runs `ruff check` and `mypy`; `make test` or `make test-quick` runs pytest (verbose, short tracebacks).
- `make check` chains fmt+lint+tests; `make coverage` produces HTML coverage in `htmlcov/`.

## Coding Style & Naming Conventions
- Target Python 3.11, 100-character lines, and full type hints (mypy); use ruff for linting and formatting.
- Follow PEP 8 for naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Use f-strings for formatting; prefer list/dict/set comprehensions over `map`/`filter`.
- 
- Docstrings follow Google style; use `"""Triple quotes for docstrings."""` and `#` for inline comments.
- Imports are sorted (isort via ruff); prefer single-responsibility functions under ~50 lines with descriptive names.
- Branch prefixes: `feature/`, `fix/`, `docs/`, `refactor/`, `test/`; keep tests colocated with the corresponding module paths.

## Testing Guidelines
- Pytest is standard; structure tests to mirror `src/` modules and mock external services.
- Use deterministic fixtures; skip slow cases with `-m "not slow"` (see `make test-quick`).
- Generate reports with `pytest --cov=src --cov-report=html`.

## Commit & Pull Request Guidelines
- Commit subjects are imperative with optional scopes, e.g., `Fix ...`, `docs: ...`, `chore: ...`; keep the first line <72 chars.
- Include only focused, reviewable changes with updated tests/docs; never commit secrets or large data files.
- PRs should link issues, summarize behavior changes, note config impacts, attach UI screenshots if relevant, and include test results; ensure `pre-commit run --all-files` passes.
- Before committing, run `ruff format .`, `ruff check .`, and `mypy src/ --ignore-missing-imports` (or `make check`).

## Environment Configuration
- Load all API keys and sensitive credentials from environment variables; never commit secrets to version control. The development user (kjfle) must ensure all required API keys are set in their local environment before running the system.

## Governance Configuration
- Refer `governance.yaml` for definitions and enforcement parameters throughout the development lifecycle of the artificial intelligence system, Ordinis. 
- Agents shall `governance.yaml`configuration establishes clear boundaries for safety, compliance, and decision-making, ensuring that system behavior aligns with organizational and regulatory expectations from design through
deployment.



