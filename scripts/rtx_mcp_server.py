import asyncio
import subprocess
import os
import sys
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types
from mcp.server.models import InitializationOptions

server = Server("chimera-rtx-node")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="trigger_remote_training",
            description="Autonomously triggers the Chimera training loop on the RTX GPU.",
            inputSchema={
                "type": "object",
                "properties": {
                    "duration_hours": {"type": "integer", "description": "Hours to train"}
                },
                "required": ["duration_hours"]
            }
        ),
        types.Tool(
            name="check_remote_status",
            description="Checks if the RTX training loop is actively running.",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if name == "trigger_remote_training":
        duration = arguments.get("duration_hours", 1)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(repo_root)
        # Run detached
        subprocess.Popen(
            [sys.executable, os.path.join("src", "train.py"), "--duration_hours", str(duration)],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        return [types.TextContent(type="text", text=f"SUCCESS: Remote training loop detached and started on RTX GPU for {duration} hours.")]
        
    elif name == "check_remote_status":
        # Check if python process running train.py exists
        output = subprocess.check_output('wmic process where "name=\'python.exe\'" get commandline', shell=True, text=True)
        if "train.py" in output:
            return [types.TextContent(type="text", text="STATUS: ACTIVE. Training loop is currently running on the RTX GPU.")]
        return [types.TextContent(type="text", text="STATUS: IDLE. No training loop detected.")]
        
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="chimera-rtx-node",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
