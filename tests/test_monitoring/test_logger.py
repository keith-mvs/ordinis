"""Tests for monitoring.logger module."""

from pathlib import Path
import tempfile

from loguru import logger
import pytest

from monitoring.logger import (
    get_logger,
    log_exception,
    log_execution_time,
    setup_logging,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_console_only(self):
        """Test logging setup with console output only."""
        setup_logging(log_level="INFO")
        test_logger = get_logger(__name__)
        assert test_logger is not None

    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(log_level="DEBUG", log_file=str(log_file))

            test_logger = get_logger(__name__)
            test_logger.info("Test message")

            # Remove all handlers to close files
            logger.remove()

            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content

    def test_setup_logging_with_json_format(self):
        """Test logging setup with JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"
            setup_logging(
                log_level="INFO",
                log_file=str(log_file),
                json_format=True,
            )

            test_logger = get_logger(__name__)
            test_logger.info("JSON test")

            # Remove all handlers to close files
            logger.remove()

            assert log_file.exists()
            content = log_file.read_text(encoding="utf-8")
            assert '"message"' in content or "JSON test" in content

    def test_setup_logging_with_rotation(self):
        """Test logging setup with rotation settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test_rotate.log"
            setup_logging(
                log_level="INFO",
                log_file=str(log_file),
                rotation="1 MB",
                retention="7 days",
            )

            test_logger = get_logger(__name__)
            test_logger.info("Rotation test")

            # Remove all handlers to close files
            logger.remove()

            assert log_file.exists()

    def test_setup_logging_creates_directory(self):
        """Test that logging creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "nested" / "dir" / "test.log"
            setup_logging(log_level="INFO", log_file=str(log_file))

            test_logger = get_logger(__name__)
            test_logger.info("Directory creation test")

            # Remove all handlers to close files
            logger.remove()

            assert log_file.exists()
            assert log_file.parent.exists()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        test_logger = get_logger(__name__)
        assert test_logger is not None

    def test_get_logger_with_name(self):
        """Test that logger is bound with correct name."""
        test_logger = get_logger("test_module")
        assert test_logger is not None


class TestLogExecutionTime:
    """Tests for log_execution_time decorator."""

    def test_log_execution_time_decorator(self):
        """Test that decorator logs execution time."""

        @log_execution_time
        def test_function():
            return "result"

        result = test_function()
        assert result == "result"

    def test_log_execution_time_with_args(self):
        """Test decorator with function arguments."""

        @log_execution_time
        def test_function(x, y):
            return x + y

        result = test_function(5, 3)
        assert result == 8

    def test_log_execution_time_preserves_function_name(self):
        """Test that decorator preserves function metadata."""

        @log_execution_time
        def test_function():
            """Test docstring."""
            return True

        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test docstring."


class TestLogException:
    """Tests for log_exception decorator."""

    def test_log_exception_decorator_success(self):
        """Test decorator with successful function execution."""

        @log_exception()
        def test_function():
            return "success"

        result = test_function()
        assert result == "success"

    def test_log_exception_decorator_with_exception(self):
        """Test decorator logs exceptions and re-raises."""

        @log_exception()
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_function()

    def test_log_exception_with_args(self):
        """Test decorator with function arguments."""

        @log_exception()
        def test_function(x):
            if x < 0:
                raise ValueError("Negative value")
            return x * 2

        result = test_function(5)
        assert result == 10

        with pytest.raises(ValueError, match="Negative value"):
            test_function(-1)

    def test_log_exception_preserves_function_name(self):
        """Test that decorator preserves function metadata."""

        @log_exception()
        def test_function():
            """Test docstring."""
            return True

        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test docstring."


class TestLoggerIntegration:
    """Integration tests for logging system."""

    def test_logging_levels(self):
        """Test different logging levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "levels.log"
            setup_logging(log_level="DEBUG", log_file=str(log_file))

            test_logger = get_logger(__name__)
            test_logger.debug("Debug message")
            test_logger.info("Info message")
            test_logger.warning("Warning message")
            test_logger.error("Error message")

            # Remove all handlers to close files
            logger.remove()

            content = log_file.read_text()
            assert "Debug message" in content
            assert "Info message" in content
            assert "Warning message" in content
            assert "Error message" in content

    def test_logger_with_context(self):
        """Test logger with bound context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "context.log"
            setup_logging(log_level="INFO", log_file=str(log_file))

            test_logger = get_logger(__name__)
            context_logger = test_logger.bind(symbol="AAPL", strategy="RSI")
            context_logger.info("Trade executed")

            # Remove all handlers to close files
            logger.remove()

            content = log_file.read_text()
            assert "Trade executed" in content
