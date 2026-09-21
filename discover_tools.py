"""MCP discovery client - asks the server "what tools do you have?".

This is the stage 1 proof: it talks to mcp_server.py using nothing but the
standard MCP protocol, and prints back the tool catalogue the server
advertises. Claude Desktop (stage 2) and our agent (stage 3) perform exactly
this same handshake before they call anything.

Start the server first, then in a SECOND terminal run:
    .venv/Scripts/python.exe discover_tools.py
"""

import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SERVER_URL = "http://127.0.0.1:8001/mcp"


async def main() -> None:
    async with streamablehttp_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print(f"\nConnected to {SERVER_URL}")
            print(f"Server advertises {len(tools)} tools\n")

            for i, tool in enumerate(tools, 1):
                print("=" * 70)
                print(f"{i}. {tool.name}")
                print("=" * 70)

                description = (tool.description or "").strip()
                print(f"\nDESCRIPTION:\n{description}\n")

                schema = tool.inputSchema or {}
                properties = schema.get("properties", {})
                required = schema.get("required", [])

                print("INPUT SCHEMA:")
                if not properties:
                    print("  (takes no arguments)")
                for arg_name, spec in properties.items():
                    flag = "required" if arg_name in required else "optional"
                    print(f"  - {arg_name}: {spec.get('type', '?')} ({flag})")

                print(f"\nRAW SCHEMA:\n{json.dumps(schema, indent=2)}\n")

            print("=" * 70)
            print(f"TOTAL: {len(tools)} tools")
            print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
