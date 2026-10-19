"""Standalone MCP stdio server exposing the same forensic tools.

The Agent SDK path (``sdk_server.py``) is what the investigation loop uses in
production. This one exists so the tools are usable from *any* MCP client --
Claude Desktop, another agent framework, a human debugging a case by hand --
without running the whole sentinel service.

Run it with:

    python -m sentinel.mcp.stdio_server --config config.yml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

import mcp.types as mcp_types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from sentinel import __version__
from sentinel.tools.base import ToolContext
from sentinel.tools.registry import ALL_TOOLS, TOOLS_BY_NAME

logger = logging.getLogger(__name__)


def build_server(ctx: ToolContext) -> Server[Any]:
    async def on_list_tools(_ctx: Any, _params: Any) -> mcp_types.ListToolsResult:
        return mcp_types.ListToolsResult(
            tools=[
                mcp_types.Tool(
                    name=t.name,
                    description=t.description,
                    inputSchema=t.input_schema,
                )
                for t in ALL_TOOLS
            ]
        )

    async def on_call_tool(
        _ctx: Any, params: mcp_types.CallToolRequestParams
    ) -> mcp_types.CallToolResult:
        tool = TOOLS_BY_NAME.get(params.name)
        if tool is None:
            return mcp_types.CallToolResult(
                content=[
                    mcp_types.TextContent(
                        type="text", text=f"unknown tool: {params.name}"
                    )
                ],
                isError=True,
            )
        result = await tool.run(ctx, dict(params.arguments or {}))
        return mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(
                    type="text", text=json.dumps(result, default=str, indent=2)
                )
            ],
            isError=not result.get("ok", True),
        )

    return Server(
        "sentinel-forensics",
        version=__version__,
        instructions=(
            "Forensic tools over an NVR deployment: historical alert search, "
            "device and arming status, multi-camera cross-referencing, the duty "
            "roster rule base, and historical frame retrieval."
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def _amain(config_path: str | None) -> None:
    from sentinel.wiring import build_tool_context

    ctx, cleanup = await build_tool_context(config_path)
    server = build_server(ctx)
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )
    finally:
        await cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel forensic MCP server (stdio)")
    parser.add_argument("--config", default=None, help="Path to sentinel config YAML")
    args = parser.parse_args()
    # stdio is the MCP channel; logs must not go there.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(_amain(args.config))


if __name__ == "__main__":
    main()
