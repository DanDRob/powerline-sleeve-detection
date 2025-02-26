import os
import logging
import logging.handlers
import sys
from typing import Optional, Dict, Any


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    component: str = "main"
) -> logging.Logger:
    """
    Set up logging with the specified configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Format string for log messages
        component: Component name to use in logger

    Returns:
        Configured logger instance
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create logger
    logger = logging.getLogger(f"powerline_detector.{component}")
    logger.setLevel(numeric_level)
    logger.propagate = False  # Don't propagate to root logger

    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter(log_format)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log file specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that can include contextual information."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message to include context."""
        if self.extra:
            context_str = " ".join(f"{k}={v}" for k, v in self.extra.items())
            msg = f"{msg} - {context_str}"
        return msg, kwargs


def get_logger(
    component: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> logging.LoggerAdapter:
    """
    Get a logger with the specified configuration.

    Args:
        component: Component name
        log_level: Logging level
        log_file: Optional log file path
        context: Optional context dictionary to include in logs

    Returns:
        Configured logger adapter
    """
    logger = setup_logging(log_level, log_file, component=component)
    return LoggerAdapter(logger, context or {})
