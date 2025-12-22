"""Tests for QPO adapter module.

Tests cover:
- QPOEnvironmentError exception
- _ensure_in_sys_path function
- _import_qpo_module function
- QPOScenarioGenerator class
- QPOPortfolioOptimizer class
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import pytest
import pandas as pd

from ordinis.quant.qpo_adapter import (
    QPOEnvironmentError,
    _ensure_in_sys_path,
    _import_qpo_module,
    QPOScenarioGenerator,
    QPOPortfolioOptimizer,
)


class TestQPOEnvironmentError:
    """Tests for QPOEnvironmentError exception."""

    @pytest.mark.unit
    def test_error_is_runtime_error(self):
        """Test QPOEnvironmentError is a RuntimeError subclass."""
        error = QPOEnvironmentError("test message")
        assert isinstance(error, RuntimeError)

    @pytest.mark.unit
    def test_error_message(self):
        """Test error message is preserved."""
        error = QPOEnvironmentError("custom error message")
        assert str(error) == "custom error message"


class TestEnsureInSysPath:
    """Tests for _ensure_in_sys_path function."""

    @pytest.mark.unit
    def test_adds_path_if_not_present(self, tmp_path):
        """Test path is added to sys.path if not present."""
        test_path = tmp_path / "test_dir"
        test_path.mkdir()

        # Ensure path is not in sys.path
        str_path = str(test_path)
        if str_path in sys.path:
            sys.path.remove(str_path)

        _ensure_in_sys_path(test_path)

        assert str_path in sys.path
        # Cleanup
        sys.path.remove(str_path)

    @pytest.mark.unit
    def test_does_not_duplicate_path(self, tmp_path):
        """Test path is not duplicated if already present."""
        test_path = tmp_path / "test_dir"
        test_path.mkdir()
        str_path = str(test_path)

        # Add path first
        if str_path not in sys.path:
            sys.path.insert(0, str_path)

        original_count = sys.path.count(str_path)
        _ensure_in_sys_path(test_path)

        assert sys.path.count(str_path) == original_count
        # Cleanup
        while str_path in sys.path:
            sys.path.remove(str_path)


class TestImportQpoModule:
    """Tests for _import_qpo_module function."""

    @pytest.mark.unit
    def test_raises_error_if_path_not_exists(self, tmp_path):
        """Test raises QPOEnvironmentError if path doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent"

        with pytest.raises(QPOEnvironmentError, match="QPO blueprint not found"):
            _import_qpo_module("some_module", nonexistent_path)

    @pytest.mark.unit
    def test_imports_module_from_path(self, tmp_path):
        """Test imports module from given path."""
        qpo_src = tmp_path / "qpo_src"
        qpo_src.mkdir()

        # Create a test module
        test_module_path = qpo_src / "test_module.py"
        test_module_path.write_text("TEST_VALUE = 42")

        try:
            module = _import_qpo_module("test_module", qpo_src)
            assert hasattr(module, "TEST_VALUE")
            assert module.TEST_VALUE == 42
        finally:
            # Cleanup sys.path and sys.modules
            str_path = str(qpo_src)
            if str_path in sys.path:
                sys.path.remove(str_path)
            if "test_module" in sys.modules:
                del sys.modules["test_module"]


class TestQPOScenarioGenerator:
    """Tests for QPOScenarioGenerator class."""

    @pytest.mark.unit
    def test_init_default_path(self):
        """Test initialization with default path."""
        gen = QPOScenarioGenerator()
        assert gen.qpo_src is not None
        assert isinstance(gen.qpo_src, Path)

    @pytest.mark.unit
    def test_init_custom_path(self, tmp_path):
        """Test initialization with custom path."""
        custom_path = tmp_path / "custom_qpo"
        gen = QPOScenarioGenerator(qpo_src=custom_path)
        assert gen.qpo_src == custom_path

    @pytest.mark.unit
    def test_loader_raises_error_if_path_missing(self):
        """Test _loader raises error if QPO path doesn't exist."""
        gen = QPOScenarioGenerator(qpo_src=Path("/nonexistent/path"))

        with pytest.raises(QPOEnvironmentError, match="QPO blueprint not found"):
            gen._loader()

    @pytest.mark.unit
    def test_generate_calls_simulator(self, tmp_path):
        """Test generate method calls ForwardPathSimulator."""
        qpo_src = tmp_path / "qpo_src"
        qpo_src.mkdir()

        # Create mock scenario_generation module
        mock_module_code = '''
class ForwardPathSimulator:
    def __init__(self, fitting_data, generation_dates, n_paths, method):
        self.simulated_paths = {"mock": "paths"}

    def generate(self, plot_paths=False, n_plots=0):
        pass
'''
        (qpo_src / "scenario_generation.py").write_text(mock_module_code)

        gen = QPOScenarioGenerator(qpo_src=qpo_src)

        fitting_data = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0]})
        generation_dates = [pd.Timestamp("2024-01-01")]

        try:
            result = gen.generate(
                fitting_data=fitting_data,
                generation_dates=generation_dates,
                n_paths=100,
            )
            assert result == {"mock": "paths"}
        finally:
            # Cleanup
            str_path = str(qpo_src)
            if str_path in sys.path:
                sys.path.remove(str_path)
            if "scenario_generation" in sys.modules:
                del sys.modules["scenario_generation"]


class TestQPOPortfolioOptimizer:
    """Tests for QPOPortfolioOptimizer class."""

    @pytest.mark.unit
    def test_init_default_path(self):
        """Test initialization with default path."""
        opt = QPOPortfolioOptimizer()
        assert opt.qpo_src is not None
        assert isinstance(opt.qpo_src, Path)

    @pytest.mark.unit
    def test_init_custom_path(self, tmp_path):
        """Test initialization with custom path."""
        custom_path = tmp_path / "custom_qpo"
        opt = QPOPortfolioOptimizer(qpo_src=custom_path)
        assert opt.qpo_src == custom_path

    @pytest.mark.unit
    def test_load_optimizer_raises_error_if_path_missing(self):
        """Test _load_optimizer raises error if QPO path doesn't exist."""
        opt = QPOPortfolioOptimizer(qpo_src=Path("/nonexistent/path"))

        with pytest.raises(QPOEnvironmentError, match="QPO blueprint not found"):
            opt._load_optimizer()

    @pytest.mark.unit
    def test_optimize_validate_only(self, tmp_path):
        """Test optimize_from_returns with execute=False validates only."""
        qpo_src = tmp_path / "qpo_src"
        qpo_src.mkdir()

        # Create mock modules
        cvar_optimizer_code = '''
class CVaR:
    pass
'''
        cvar_params_code = '''
class CvarParameters:
    pass
'''
        (qpo_src / "cvar_optimizer.py").write_text(cvar_optimizer_code)
        (qpo_src / "cvar_parameters.py").write_text(cvar_params_code)

        opt = QPOPortfolioOptimizer(qpo_src=qpo_src)

        returns = pd.DataFrame({"AAPL": [0.01, -0.02, 0.015]})

        try:
            result = opt.optimize_from_returns(
                returns=returns,
                target_return=0.001,
                execute=False,
            )
            assert result["status"] == "validated"
            assert result["api"] == "cvxpy"
            assert result["target_return"] == 0.001
        finally:
            # Cleanup
            str_path = str(qpo_src)
            if str_path in sys.path:
                sys.path.remove(str_path)
            for mod in ["cvar_optimizer", "cvar_parameters"]:
                if mod in sys.modules:
                    del sys.modules[mod]
