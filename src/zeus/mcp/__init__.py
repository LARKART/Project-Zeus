"""MCP: how ZEUS reaches the rest of your machine. See client.py, registry.py."""
from zeus.mcp.client import MCPError, MCPServer, MCPTool
from zeus.mcp.registry import (
    Confirmer, MCPRegistry, ServerConfig, load_server_configs, looks_destructive,
)

__all__ = [
    "Confirmer", "MCPError", "MCPRegistry", "MCPServer", "MCPTool",
    "ServerConfig", "load_server_configs", "looks_destructive",
]
