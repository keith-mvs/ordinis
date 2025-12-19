"""
Tests for centralized logging module.

Tests cover:
- StructuredFormatter JSON output
- SimpleFormatter format string
- configure_logging with various parameters
- File handler creation
- Third-party logger level configuration
- get_logger function
"""

import json
import logging
import sys
from unittest.mock import MagicMock

import pytest

from ordinis.runtime.logging import (
    SimpleFormatter,
    StructuredFormatter,
    configure_logging,
    get_logger,
)


class TestStructuredFormatter:
    """Test StructuredFormatter class."""

    @pytest.mark.unit
    def test_format_basic_record(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    @pytest.mark.unit
    def test_format_with_args(self):
        """Test log record with format arguments."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=42,
            msg="Value is %s",
            args=("test_value",),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["message"] == "Value is test_value"

    @pytest.mark.unit
    def test_format_with_exception(self):
        """Test log record with exception info."""
        formatter = StructuredFormatter()

        try:
            1 / 0  # noqa: B018 - intentional exception for testing
        except ZeroDivisionError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="An error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert "exception" in parsed
        assert "ZeroDivisionError" in parsed["exception"]

    @pytest.mark.unit
    def test_format_with_extra_attributes(self):
        """Test log record with extra attributes."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.extra = {"custom_key": "custom_value", "count": 42}

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["custom_key"] == "custom_value"
        assert parsed["count"] == 42


class TestSimpleFormatter:
    """Test SimpleFormatter class."""

    @pytest.mark.unit
    def test_simple_formatter_init(self):
        """Test SimpleFormatter initializes with correct format."""
        formatter = SimpleFormatter()

        assert "%(levelname)s" in formatter._fmt
        assert "%(name)s" in formatter._fmt
        assert "%(message)s" in formatter._fmt
        assert "%Y-%m-%d" in formatter.datefmt

    @pytest.mark.unit
    def test_simple_formatter_format(self):
        """Test SimpleFormatter produces readable output."""
        formatter = SimpleFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "[INFO]" in result
        assert "test.logger" in result
        assert "Test message" in result


class TestConfigureLogging:
    """Test configure_logging function."""

    @pytest.fixture(autouse=True)
    def _reset_logging(self):
        """Reset root logger after each test."""
        yield
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    @pytest.mark.unit
    def test_configure_logging_default_level(self):
        """Test configure_logging with default level."""
        configure_logging()

        root = logging.getLogger()
        assert root.level == logging.INFO

    @pytest.mark.unit
    def test_configure_logging_custom_level(self):
        """Test configure_logging with custom level."""
        configure_logging(level="DEBUG")

        root = logging.getLogger()
        assert root.level == logging.DEBUG

    @pytest.mark.unit
    def test_configure_logging_with_config(self):
        """Test configure_logging with LoggingConfig."""
        mock_config = MagicMock()
        mock_config.level = "WARNING"
        mock_config.format = "simple"
        mock_config.file = None
        mock_config.max_size_mb = 50
        mock_config.backup_count = 3

        configure_logging(config=mock_config)

        root = logging.getLogger()
        assert root.level == logging.WARNING

    @pytest.mark.unit
    def test_configure_logging_structured_format(self):
        """Test configure_logging with structured format."""
        configure_logging(format_type="structured")

        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, StructuredFormatter)

    @pytest.mark.unit
    def test_configure_logging_simple_format(self):
        """Test configure_logging with simple format."""
        configure_logging(format_type="simple")

        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, SimpleFormatter)

    @pytest.mark.unit
    def test_configure_logging_clears_handlers(self):
        """Test configure_logging clears existing handlers."""
        root = logging.getLogger()
        root.addHandler(logging.NullHandler())
        root.addHandler(logging.NullHandler())

        configure_logging()

        # Should have exactly one console handler after config
        assert len(root.handlers) == 1

    @pytest.mark.unit
    def test_configure_logging_with_file(self, tmp_path):
        """Test configure_logging creates file handler."""
        log_file = tmp_path / "test.log"

        configure_logging(log_file=str(log_file))

        root = logging.getLogger()
        # Should have console + file handler
        assert len(root.handlers) == 2

    @pytest.mark.unit
    def test_configure_logging_creates_log_directory(self, tmp_path):
        """Test configure_logging creates parent directories for log file."""
        log_file = tmp_path / "subdir" / "nested" / "test.log"

        configure_logging(log_file=str(log_file))

        assert (tmp_path / "subdir" / "nested").exists()

    @pytest.mark.unit
    def test_configure_logging_sets_third_party_levels(self):
        """Test third-party loggers are set to WARNING."""
        configure_logging()

        for logger_name in ["urllib3", "httpx", "httpcore", "asyncio"]:
            logger = logging.getLogger(logger_name)
            assert logger.level == logging.WARNING


class TestGetLogger:
    """Test get_logger function."""

    @pytest.mark.unit
    def test_get_logger_returns_logger(self):
        """Test get_logger returns a Logger instance."""
        logger = get_logger("test.module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    @pytest.mark.unit
    def test_get_logger_same_name_returns_same_logger(self):
        """Test get_logger returns same logger for same name."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")

        assert logger1 is logger2

    @pytest.mark.unit
    def test_get_logger_different_names(self):
        """Test get_logger returns different loggers for different names."""
        logger1 = get_logger("module.one")
        logger2 = get_logger("module.two")

        assert logger1 is not logger2
        assert logger1.name == "module.one"
        assert logger2.name == "module.two"
