"""Allow running the GeoAI MCP Server as a module.

Usage:
    python -m geoai_mcp_server
    python -m geoai_mcp_server.server
"""

from .server import main

if __name__ == "__main__":
    main()
