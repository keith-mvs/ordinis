"""Test configuration and fixtures."""

from pathlib import Path
import sys

import pytest

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from ordinis.core.container import Container, ContainerConfig  # noqa: E402


@pytest.fixture
def container_config() -> ContainerConfig:
    """Default container config for tests."""
    return ContainerConfig(
        broker_type="paper",
        paper_slippage_bps=0.0,
        paper_commission_per_share=0.0,
        paper_fill_delay_ms=0.0,
        paper_initial_cash=100000.0,
        enable_kill_switch=False,
        enable_persistence=False,
        enable_alerting=False,
    )


@pytest.fixture
def container(container_config: ContainerConfig) -> Container:
    """Container instance for tests."""
    return Container(container_config)


@pytest.fixture
def paper_broker(container: Container):
    """Paper broker adapter for tests."""
    return container.get_broker_adapter()


@pytest.fixture
def flowroute_engine(container: Container):
    """FlowRoute engine for tests."""
    return container.get_flowroute_engine()
