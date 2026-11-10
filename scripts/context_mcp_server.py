"""Pure-Stdlib Local MCP Stdio Server for Antigravity Cross-Chat Context.

Exposes shared memory, full-text transcript search, and decision pinning
tools to every Antigravity chat session without external dependencies.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from context_hub import ContextHub  # noqa: E402

SERVER_NAME = "chimera-context-bridge"
SERVER_VERSION = "1.0.0"

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "chimera_get_shared_context",
        "description": (
            "Retrieve the latest shared context brief across all connected/parallel "
            "Antigravity chats, including pinned decisions, modified files, and recent outcomes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "workspace": {
                    "type": "string",
                    "description": "Workspace root path to filter by (omit or 'global' for all).",
                },
                "limit_chats": {
                    "type": "integer",
                    "description": "Maximum number of recent chats to summarize (default 6).",
                },
                "current_conversation_id": {
                    "type": "string",
                    "description": "Optional current conversation ID to exclude from summary.",
                },
            },
        },
    },
    {
        "name": "chimera_search_chats",
        "description": (
            "Search offline across all connected Antigravity chat transcripts using SQLite FTS5."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords or phrase to search for."},
                "workspace": {
                    "type": "string",
                    "description": "Optional workspace path filter (default 'global').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum matching snippets to return (default 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "chimera_read_chat_turns",
        "description": (
            "Read recent turns, tool calls, and edited files from another Antigravity chat "
            "by full conversation ID or ID prefix."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "Full conversation UUID or 8-char prefix.",
                },
                "last_n": {
                    "type": "integer",
                    "description": "Number of recent turns to retrieve (default 12).",
                },
            },
            "required": ["conversation_id"],
        },
    },
    {
        "name": "chimera_pin_memory",
        "description": (
            "Pin a shared architectural decision, handoff note, blocker, or fact so every "
            "other open or future Antigravity chat automatically sees it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short unique title for this note."},
                "content": {"type": "string", "description": "Details to broadcast to other chats."},
                "category": {
                    "type": "string",
                    "enum": ["decision", "handoff", "blocker", "fact", "task"],
                    "description": "Note category (default 'decision').",
                },
                "workspace": {
                    "type": "string",
                    "description": "Workspace root path (default 'global').",
                },
                "source_conversation_id": {
                    "type": "string",
                    "description": "Optional originating conversation ID.",
                },
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "chimera_sync_chats",
        "description": (
            "Scan local Antigravity brain directories and ingest any newly updated "
            "transcript.jsonl files into the shared context database."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_files": {
                    "type": "integer",
                    "description": "Max recent transcript files to inspect (default 30).",
                },
            },
        },
    },
]


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Dispatch an MCP tool call to ContextHub and return formatted text."""
    with ContextHub() as hub:
        if name == "chimera_get_shared_context":
            hub.scan_and_ingest_brains(max_files=12, recent_only_hours=72)
            return hub.get_shared_context(
                current_conversation_id=str(args.get("current_conversation_id", "")),
                workspace=str(args.get("workspace", "global")),
                limit_chats=int(args.get("limit_chats", 6)),
            )

        if name == "chimera_search_chats":
            query = str(args.get("query", "")).strip()
            if not query:
                raise ValueError("Missing required argument: query")
            results = hub.search_history(
                query=query,
                workspace=str(args.get("workspace", "global")),
                limit=int(args.get("limit", 10)),
            )
            return json.dumps(results, indent=2)

        if name == "chimera_read_chat_turns":
            conv_id = str(args.get("conversation_id", "")).strip()
            if not conv_id:
                raise ValueError("Missing required argument: conversation_id")
            data = hub.read_chat_turns(conv_id, last_n=int(args.get("last_n", 12)))
            return json.dumps(data, indent=2)

        if name == "chimera_pin_memory":
            title = str(args.get("title", "")).strip()
            content = str(args.get("content", "")).strip()
            if not title or not content:
                raise ValueError("Both 'title' and 'content' are required.")
            hub.pin_memory(
                title=title,
                content=content,
                workspace=str(args.get("workspace", "global")),
                category=str(args.get("category", "decision")),
                source_conversation_id=str(args.get("source_conversation_id", "")),
            )
            return f"Pinned shared memory [{args.get('category', 'decision')}] '{title}'."

        if name == "chimera_sync_chats":
            stats = hub.scan_and_ingest_brains(max_files=int(args.get("max_files", 30)))
            return json.dumps(stats, indent=2)

    raise ValueError(f"Unknown tool: {name}")


def handle_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single JSON-RPC 2.0 request and return response or None for notifications."""
    msg_id = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params") or {}

    # Notifications have no id and expect no reply.
    if msg_id is None:
        return None

    if method == "initialize":
        proto = params.get("protocolVersion") or "2024-11-05"
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": proto,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        tool_name = str(params.get("name", ""))
        tool_args = params.get("arguments") or {}
        try:
            text = execute_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": text}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict):
            continue

        resp = handle_message(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
