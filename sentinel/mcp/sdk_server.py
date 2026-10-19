"""Expose the forensic tools to the Claude Agent SDK as an in-process MCP server.

``create_sdk_mcp_server`` runs the tools inside this process over an in-memory
MCP transport -- no subprocess, no socket, and the handlers keep direct access
to the live ``ToolContext`` (NVR client, store, pattern library). That matters
here: the tools need an open httpx session and a SQLite handle, and marshalling
those across a process boundary would buy nothing.

For clients that are not the Agent SDK, ``mcp.stdio_server`` exposes the same
tools over real stdio MCP.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sentinel import __version__
from sentinel.tools.base import ForensicTool, ToolContext
from sentinel.tools.registry import ALL_TOOLS

logger = logging.getLogger(__name__)

# MCP tool names are namespaced by the SDK as mcp__<server>__<tool>.
SERVER_NAME = "forensics"


def qualified_name(tool_name: str) -> str:
    return f"mcp__{SERVER_NAME}__{tool_name}"


def _make_handler(tool: ForensicTool, ctx: ToolContext):
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        result = await tool.run(ctx, args or {})
        return {
            "content": [
                {"type": "text", "text": json.dumps(result, default=str, indent=2)}
            ],
            "isError": not result.get("ok", True),
        }

    handler.__name__ = f"forensic_{tool.name}"
    return handler


def build_sdk_tools(ctx: ToolContext, tools: list[ForensicTool] | None = None) -> list[Any]:
    """Wrap ForensicTools as SdkMcpTool instances."""
    from claude_agent_sdk import tool as sdk_tool

    built = []
    for t in tools or ALL_TOOLS:
        decorator = sdk_tool(t.name, t.description, t.input_schema)
        built.append(decorator(_make_handler(t, ctx)))
    return built


def build_sdk_mcp_server(ctx: ToolContext, tools: list[ForensicTool] | None = None) -> Any:
    """Return an McpSdkServerConfig for ClaudeAgentOptions.mcp_servers."""
    from claude_agent_sdk import create_sdk_mcp_server

    return create_sdk_mcp_server(
        name=SERVER_NAME, version=__version__, tools=build_sdk_tools(ctx, tools)
    )


def allowed_tool_names(tools: list[ForensicTool] | None = None) -> list[str]:
    return [qualified_name(t.name) for t in (tools or ALL_TOOLS)]
