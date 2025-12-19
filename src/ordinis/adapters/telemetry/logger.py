"""
Logging Configuration.

Centralized logging setup using loguru with structured logging,
rotation, and multiple output formats.
"""

from pathlib import Path
import sys

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    json_format: bool = False,
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        rotation: Log rotation size/time
        retention: How long to keep old logs
        json_format: Use JSON format for structured logging
    """
    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        colorize=True,
    )

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if json_format:
            # JSON format for structured logging
            logger.add(
                log_file,
                level=log_level,
                rotation=rotation,
                retention=retention,
                format="{message}",
                serialize=True,
            )
        else:
            # Standard format
            logger.add(
                log_file,
                level=log_level,
                rotation=rotation,
                retention=retention,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}",
            )

    logger.info(f"Logging configured: level={log_level}, file={log_file}")


def get_logger(name: str):
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Convenience functions for common logging patterns
def log_execution_time(func):
    """Decorator to log function execution time."""
    from functools import wraps
    from time import perf_counter

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start
        logger.info(f"{func.__name__} executed in {elapsed:.4f}s")
        return result

    return wrapper


def log_exception(exc_info: bool = True):
    """Decorator to log exceptions."""
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {e}")
                raise

        return wrapper

    return decorator
