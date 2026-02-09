"""GeoAI MCP Server - Expose GeoAI capabilities to AI agents via MCP.

This package provides a Model Context Protocol (MCP) server that exposes
GeoAI's geospatial AI capabilities to AI agents and LLM applications.

Typical usage:
    python -m geoai_mcp_server.server

For Claude Desktop integration, add to claude_desktop_config.json:
    {
        "mcpServers": {
            "geoai": {
                "command": "python",
                "args": ["-m", "geoai_mcp_server.server"],
                "env": {
                    "GEOAI_INPUT_DIR": "/path/to/input",
                    "GEOAI_OUTPUT_DIR": "/path/to/output"
                }
            }
        }
    }
"""

__version__ = "0.1.0"
__author__ = "GeoAI Team"

from .config import GeoAIConfig, load_config
from .server import mcp, main

__all__ = [
    "GeoAIConfig",
    "load_config",
    "mcp",
    "main",
    "__version__",
]

