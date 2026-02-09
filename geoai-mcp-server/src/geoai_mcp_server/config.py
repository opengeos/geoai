"""Configuration management for GeoAI MCP Server.

Handles environment variables, defaults, and runtime configuration.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class GeoAIConfig:
    """Configuration settings for the GeoAI MCP Server."""

    # Workspace directories
    input_dir: Path = field(default_factory=lambda: Path(os.getenv("GEOAI_INPUT_DIR", "./input")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("GEOAI_OUTPUT_DIR", "./output")))

    # Resource limits
    timeout: int = field(default_factory=lambda: int(os.getenv("GEOAI_TIMEOUT", "300")))
    max_memory_gb: int = field(default_factory=lambda: int(os.getenv("GEOAI_MAX_MEMORY_GB", "8")))
    device: str = field(default_factory=lambda: os.getenv("GEOAI_DEVICE", "auto"))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("GEOAI_LOG_LEVEL", "INFO"))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv("GEOAI_LOG_FILE"))

    # Model configuration
    model_cache_size: int = field(default_factory=lambda: int(os.getenv("GEOAI_MODEL_CACHE_SIZE", "3")))

    def __post_init__(self) -> None:
        """Validate and initialize configuration."""
        # Convert string paths to Path objects if needed
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Resolve to absolute paths
        self.input_dir = self.input_dir.resolve()
        self.output_dir = self.output_dir.resolve()

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check input directory
        if not self.input_dir.exists():
            errors.append(f"Input directory does not exist: {self.input_dir}")
        elif not self.input_dir.is_dir():
            errors.append(f"Input path is not a directory: {self.input_dir}")

        # Check/create output directory
        if not self.output_dir.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory {self.output_dir}: {e}")
        elif not self.output_dir.is_dir():
            errors.append(f"Output path is not a directory: {self.output_dir}")

        # Validate timeout
        if self.timeout < 10:
            errors.append(f"Timeout too short (minimum 10 seconds): {self.timeout}")
        if self.timeout > 3600:
            errors.append(f"Timeout too long (maximum 3600 seconds): {self.timeout}")

        # Validate device
        valid_devices = ["auto", "cuda", "mps", "cpu"]
        if self.device not in valid_devices:
            errors.append(f"Invalid device '{self.device}'. Must be one of: {valid_devices}")

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            errors.append(f"Invalid log level '{self.log_level}'. Must be one of: {valid_levels}")

        return errors

    def get_device(self) -> str:
        """Get the actual device to use, resolving 'auto' if needed.

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'
        """
        if self.device != "auto":
            return self.device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"


def setup_logging(config: GeoAIConfig) -> logging.Logger:
    """Set up logging for the MCP server.

    IMPORTANT: For STDIO transport, we must NOT write to stdout.
    All logs go to stderr or a file.

    Args:
        config: Server configuration

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("geoai_mcp")
    logger.setLevel(getattr(logging, config.log_level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Always add stderr handler (safe for STDIO transport)
    import sys
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # Optionally add file handler
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global configuration instance
_config: Optional[GeoAIConfig] = None


def get_config() -> GeoAIConfig:
    """Get the global configuration instance.

    Returns:
        GeoAIConfig instance
    """
    global _config
    if _config is None:
        _config = GeoAIConfig()
    return _config


def load_config() -> GeoAIConfig:
    """Load and validate configuration.
    
    This is an alias for get_config() that also validates
    and sets up the configuration on first call.

    Returns:
        GeoAIConfig instance
    """
    config = get_config()
    
    # Create directories if they don't exist
    config.input_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    return config


def set_config(config: GeoAIConfig) -> None:
    """Set the global configuration instance.

    Args:
        config: Configuration to use
    """
    global _config
    _config = config
