"""
Tests for application bootstrap module.

Tests cover:
- ApplicationContext initialization
- Container lazy creation
- Directory creation
- Application initialization and shutdown
- Bootstrap function
- Context retrieval
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from ordinis.runtime.bootstrap import (
    ApplicationContext,
    _AppContextHolder,
    bootstrap,
    get_app_context,
    shutdown,
)


class TestApplicationContext:
    """Test ApplicationContext class."""

    @pytest.fixture
    def mock_settings(self, tmp_path):
        """Create mock settings for testing."""
        settings = MagicMock()
        settings.system.version = "0.2.0-dev"
        settings.system.environment = "test"
        settings.broker.provider = "paper"
        settings.alerting.enabled = False
        settings.database.path = ":memory:"
        settings.database.backup_dir = str(tmp_path / "backup")
        settings.artifacts.runs_dir = str(tmp_path / "runs")
        settings.artifacts.reports_dir = str(tmp_path / "reports")
        settings.artifacts.logs_dir = str(tmp_path / "logs")
        settings.artifacts.cache_dir = str(tmp_path / "cache")
        return settings

    @pytest.mark.unit
    def test_application_context_init(self, mock_settings):
        """Test ApplicationContext initialization."""
        ctx = ApplicationContext(mock_settings)

        assert ctx.settings is mock_settings
        assert ctx._container is None
        assert ctx._initialized is False

    @pytest.mark.unit
    def test_application_context_init_with_container(self, mock_settings):
        """Test ApplicationContext initialization with provided container."""
        mock_container = MagicMock()
        ctx = ApplicationContext(mock_settings, container=mock_container)

        assert ctx._container is mock_container

    @pytest.mark.unit
    def test_container_lazy_creation(self, mock_settings):
        """Test container is lazily created on first access."""
        ctx = ApplicationContext(mock_settings)

        with patch("ordinis.core.container.Container") as mock_container_cls:
            mock_instance = MagicMock()
            mock_container_cls.return_value = mock_instance

            container = ctx.container

            assert container is mock_instance
            mock_container_cls.assert_called_once()

    @pytest.mark.unit
    def test_container_cached(self, mock_settings):
        """Test container is cached after first creation."""
        mock_container = MagicMock()
        ctx = ApplicationContext(mock_settings, container=mock_container)

        container1 = ctx.container
        container2 = ctx.container

        assert container1 is container2
        assert container1 is mock_container

    @pytest.mark.unit
    def test_ensure_directories(self, mock_settings, tmp_path):
        """Test ensure_directories creates required directories."""
        mock_settings.artifacts.runs_dir = str(tmp_path / "runs")
        mock_settings.artifacts.reports_dir = str(tmp_path / "reports")
        mock_settings.artifacts.logs_dir = str(tmp_path / "logs")
        mock_settings.artifacts.cache_dir = str(tmp_path / "cache")
        mock_settings.database.backup_dir = str(tmp_path / "backup")

        ctx = ApplicationContext(mock_settings)
        ctx.ensure_directories()

        assert (tmp_path / "runs").exists()
        assert (tmp_path / "reports").exists()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "cache").exists()
        assert (tmp_path / "backup").exists()

    @pytest.mark.unit
    def test_initialize(self, mock_settings, caplog):
        """Test application initialization."""
        with patch.object(ApplicationContext, "ensure_directories"):
            ctx = ApplicationContext(mock_settings)

            with caplog.at_level(logging.INFO):
                ctx.initialize()

            assert ctx._initialized is True
            assert "Initializing Ordinis" in caplog.text

    @pytest.mark.unit
    def test_initialize_idempotent(self, mock_settings):
        """Test initialize only runs once."""
        with patch.object(ApplicationContext, "ensure_directories") as mock_dirs:
            ctx = ApplicationContext(mock_settings)

            ctx.initialize()
            ctx.initialize()  # Second call should be no-op

            mock_dirs.assert_called_once()

    @pytest.mark.unit
    def test_shutdown(self, mock_settings, caplog):
        """Test application shutdown."""
        mock_container = MagicMock()
        ctx = ApplicationContext(mock_settings, container=mock_container)

        with caplog.at_level(logging.INFO):
            ctx.shutdown()

        mock_container.reset.assert_called_once()
        assert "shutdown complete" in caplog.text

    @pytest.mark.unit
    def test_shutdown_without_container(self, mock_settings, caplog):
        """Test shutdown works when no container exists."""
        ctx = ApplicationContext(mock_settings)

        with caplog.at_level(logging.INFO):
            ctx.shutdown()

        assert "shutdown complete" in caplog.text


class TestBootstrapFunction:
    """Test bootstrap function."""

    @pytest.mark.unit
    def test_bootstrap_creates_context(self):
        """Test bootstrap creates and returns context."""
        with (
            patch("ordinis.runtime.bootstrap.get_settings") as mock_get_settings,
            patch("ordinis.runtime.bootstrap.configure_logging"),
            patch.object(ApplicationContext, "initialize"),
        ):
            mock_settings = MagicMock()
            mock_get_settings.return_value = mock_settings

            ctx = bootstrap(environment="test")

            assert ctx is not None
            assert isinstance(ctx, ApplicationContext)
            mock_get_settings.assert_called_once_with("test")

    @pytest.mark.unit
    def test_bootstrap_configures_logging(self):
        """Test bootstrap configures logging."""
        with (
            patch("ordinis.runtime.bootstrap.get_settings") as mock_get_settings,
            patch("ordinis.runtime.bootstrap.configure_logging") as mock_configure,
            patch.object(ApplicationContext, "initialize"),
        ):
            mock_settings = MagicMock()
            mock_get_settings.return_value = mock_settings

            bootstrap(log_level="DEBUG")

            mock_configure.assert_called_once_with(mock_settings.logging, level="DEBUG")

    @pytest.mark.unit
    def test_bootstrap_resets_settings(self):
        """Test bootstrap resets cached settings."""
        with (
            patch("ordinis.runtime.bootstrap.reset_settings") as mock_reset,
            patch("ordinis.runtime.bootstrap.get_settings") as mock_get_settings,
            patch("ordinis.runtime.bootstrap.configure_logging"),
            patch.object(ApplicationContext, "initialize"),
        ):
            mock_settings = MagicMock()
            mock_get_settings.return_value = mock_settings

            bootstrap()

            mock_reset.assert_called_once()


class TestGetAppContext:
    """Test get_app_context function."""

    @pytest.fixture(autouse=True)
    def _cleanup_holder(self):
        """Clean up holder before and after each test."""
        _AppContextHolder.instance = None
        yield
        _AppContextHolder.instance = None

    @pytest.mark.unit
    def test_get_app_context_returns_instance(self):
        """Test get_app_context returns stored instance."""
        mock_ctx = MagicMock(spec=ApplicationContext)
        _AppContextHolder.instance = mock_ctx

        result = get_app_context()

        assert result is mock_ctx

    @pytest.mark.unit
    def test_get_app_context_raises_when_not_bootstrapped(self):
        """Test get_app_context raises RuntimeError when not bootstrapped."""
        with pytest.raises(RuntimeError, match="not bootstrapped"):
            get_app_context()


class TestShutdownFunction:
    """Test shutdown function."""

    @pytest.fixture(autouse=True)
    def _cleanup_holder(self):
        """Clean up holder before and after each test."""
        _AppContextHolder.instance = None
        yield
        _AppContextHolder.instance = None

    @pytest.mark.unit
    def test_shutdown_calls_context_shutdown(self):
        """Test shutdown calls context's shutdown method."""
        mock_ctx = MagicMock(spec=ApplicationContext)
        _AppContextHolder.instance = mock_ctx

        shutdown()

        mock_ctx.shutdown.assert_called_once()
        assert _AppContextHolder.instance is None

    @pytest.mark.unit
    def test_shutdown_when_no_context(self):
        """Test shutdown is safe when no context exists."""
        # Should not raise
        shutdown()

        assert _AppContextHolder.instance is None
