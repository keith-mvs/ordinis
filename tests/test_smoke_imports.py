import importlib
import pytest

# Curated list of modules that should import cleanly and exercise top-level code
MODULES = [
    "ordinis.tools.optimizer",
    "ordinis.utils.paths",
    "ordinis.utils.__init__",
    "ordinis.engines.signalcore.models.atr_optimized_rsi",
    "ordinis.engines.signalcore.core.engine",
    "ordinis.backtesting.runner",
    "ordinis.analysis.technical.indicators.moving_averages",
    "ordinis.engines.proofbench.core.engine",
    "ordinis.engines.flowroute.core.engine",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None
