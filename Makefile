# Ordinis Trading System - Makefile
# Standard development targets for consistent workflow
#
# Usage:
#   make help     - Show available targets
#   make fmt      - Format code
#   make lint     - Run linters
#   make test     - Run tests
#   make clean    - Clean generated files

.PHONY: help fmt lint test clean install dev check all

# Default target
help:
	@echo "Ordinis Trading System - Available Targets"
	@echo "==========================================="
	@echo ""
	@echo "Development:"
	@echo "  make install    - Install production dependencies"
	@echo "  make dev        - Install development dependencies"
	@echo "  make fmt        - Format code with ruff"
	@echo "  make lint       - Run linters (ruff + mypy)"
	@echo "  make test       - Run pytest test suite"
	@echo "  make check      - Run fmt + lint + test"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Remove generated files"
	@echo "  make clean-all  - Remove all generated files including artifacts"
	@echo ""
	@echo "CI/CD:"
	@echo "  make all        - Full check (fmt + lint + test)"
	@echo ""

# Install production dependencies
install:
	pip install -e .

# Install development dependencies
dev:
	pip install -e ".[dev]"
	pre-commit install

# Format code
fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Run linters
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

# Run tests
test:
	pytest tests/ -v --tb=short

# Run quick tests (no slow markers)
test-quick:
	pytest tests/ -v --tb=short -m "not slow"

# Run all checks
check: fmt lint test

# Full CI check
all: check

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist .mypy_cache rmdir /s /q .mypy_cache
	@if exist .ruff_cache rmdir /s /q .ruff_cache
	@if exist .coverage del .coverage
	@if exist htmlcov rmdir /s /q htmlcov
	@if exist *.egg-info rmdir /s /q *.egg-info
	@echo "Clean complete."

# Clean all including artifacts
clean-all: clean
	@echo "Cleaning artifacts..."
	@if exist artifacts\runs rmdir /s /q artifacts\runs
	@if exist artifacts\reports rmdir /s /q artifacts\reports
	@if exist artifacts\logs rmdir /s /q artifacts\logs
	@if exist artifacts\cache rmdir /s /q artifacts\cache
	@echo "Full clean complete."

# Create required directories
setup-dirs:
	@if not exist artifacts\runs mkdir artifacts\runs
	@if not exist artifacts\reports mkdir artifacts\reports
	@if not exist artifacts\logs mkdir artifacts\logs
	@if not exist artifacts\cache mkdir artifacts\cache
	@if not exist configs mkdir configs
	@echo "Directories created."

# Run pre-commit on all files
pre-commit:
	pre-commit run --all-files

# Type check only
typecheck:
	mypy src/ --ignore-missing-imports

# Coverage report
coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"
