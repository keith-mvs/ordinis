#!/usr/bin/env python
"""
Ordinis MCP Server Entry Point.

Run with:
    python -m ordinis.mcp
    # or
    mcp run ordinis
"""

from .server import main

if __name__ == "__main__":
    main()
