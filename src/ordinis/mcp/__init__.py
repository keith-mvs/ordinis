"""
Ordinis MCP Server - Model Context Protocol integration.

Exposes Ordinis trading system capabilities through MCP:
- Trading signals as tools
- Portfolio state as resources
- Strategy configurations as prompts
"""

from .server import ordinis_mcp

__all__ = ["ordinis_mcp"]
