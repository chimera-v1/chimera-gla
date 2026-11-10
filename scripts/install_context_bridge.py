"""Installer for the Antigravity Chimera Context Bridge Plugin.

Deploys the self-contained plugin (`plugin.json`, `hooks.json`,
`mcp_config.json`, `rules/AGENTS.md`, and Python runtime scripts) into
`~/.gemini/config/plugins/chimera-context-bridge/` so all Antigravity chats
automatically share context across sessions and parallel tabs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from context_hub import ContextHub  # noqa: E402


RULES_AGENTS_MD = """# Cross-Chat Shared Context Bridge

All Antigravity chats on this machine are connected through the local `chimera-context-bridge` plugin (zero network, offline SQLite + FTS5).

## How to Stay Synchronized Across Chats
1. **Check Active Context**: Review the `[Cross-Chat Context Sync]` or `Shared Antigravity Context` block injected at session start and before turns when parallel chats are active.
2. **Query Deeper Details**: When another chat edited files or made decisions you need details on, call `chimera_read_chat_turns` or `chimera_search_chats` instead of asking the user to repeat themselves.
3. **Pin Major Decisions & Handoffs**: Whenever you finalize an architectural change, resolve a blocker, or finish a subtask that other chats depend on, call `chimera_pin_memory` (`category`: `decision`, `handoff`, `blocker`, `fact`, or `task`) so parallel and future chats see it immediately.
"""


def _resolve_python_executable() -> str:
    """Return the active Python path, avoiding WindowsApps store aliases."""
    exe = sys.executable
    if "WindowsApps" in exe:
        candidates = [
            os.path.join(os.path.expanduser("~"), "venv", "Scripts", "python.exe"),
            shutil.which("python3") or "",
        ]
        for cand in candidates:
            if cand and os.path.exists(cand) and "WindowsApps" not in cand:
                return cand
    return exe


def _format_hook_command(python_exe: str, event: str) -> str:
    if os.name == "nt":
        if " " in python_exe:
            return f'call "{python_exe}" context_hook.py --event {event}'
        return f"{python_exe} context_hook.py --event {event}"
    if " " in python_exe:
        return f'"{python_exe}" context_hook.py --event {event}'
    return f"{python_exe} context_hook.py --event {event}"


def install_plugin(
    gemini_config_dir: str | None = None,
    python_exe: str | None = None,
    seed_bundle: str | None = None,
    seed_workspace: str | None = None,
) -> dict:
    """Install or update the chimera-context-bridge plugin globally."""
    config_root = gemini_config_dir or os.path.join(os.path.expanduser("~"), ".gemini", "config")
    plugin_dir = os.path.join(config_root, "plugins", "chimera-context-bridge")
    rules_dir = os.path.join(plugin_dir, "rules")
    os.makedirs(rules_dir, exist_ok=True)

    py_exe = python_exe or _resolve_python_executable()

    for fname in ("context_hub.py", "context_hook.py", "context_mcp_server.py"):
        shutil.copy2(os.path.join(_SCRIPT_DIR, fname), os.path.join(plugin_dir, fname))

    plugin_manifest = {
        "name": "chimera-context-bridge",
        "displayName": "Chimera Context Bridge",
        "description": "Connects all Antigravity chats via offline shared memory, lifecycle hooks, and FTS5 transcript search.",
        "version": "1.0.0",
        "suggestedPrompts": [
            "Show shared context and recent activity from all connected chats",
            "Search previous Antigravity chats for our architecture decisions",
            "Pin a handoff summary so new chats know our current status",
        ],
    }
    with open(os.path.join(plugin_dir, "plugin.json"), "w", encoding="utf-8") as fh:
        json.dump(plugin_manifest, fh, indent=2)

    hooks_config = {
        "chimera-context-sync": {
            "SessionStart": [
                {
                    "type": "command",
                    "command": _format_hook_command(py_exe, "SessionStart"),
                    "timeout": 15,
                }
            ],
            "PreInvocation": [
                {
                    "type": "command",
                    "command": _format_hook_command(py_exe, "PreInvocation"),
                    "timeout": 10,
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "write_file|edit_file|replace_file_content|multi_replace_file_content|run_command",
                    "hooks": [
                        {
                            "type": "command",
                            "command": _format_hook_command(py_exe, "PostToolUse"),
                            "timeout": 10,
                        }
                    ],
                }
            ],
            "Stop": [
                {
                    "type": "command",
                    "command": _format_hook_command(py_exe, "Stop"),
                    "timeout": 15,
                }
            ],
        }
    }
    with open(os.path.join(plugin_dir, "hooks.json"), "w", encoding="utf-8") as fh:
        json.dump(hooks_config, fh, indent=2)

    mcp_config = {
        "mcpServers": {
            "chimera-context-bridge": {
                "command": py_exe,
                "args": [os.path.join(plugin_dir, "context_mcp_server.py")],
            }
        }
    }
    with open(os.path.join(plugin_dir, "mcp_config.json"), "w", encoding="utf-8") as fh:
        json.dump(mcp_config, fh, indent=2)

    with open(os.path.join(rules_dir, "AGENTS.md"), "w", encoding="utf-8") as fh:
        fh.write(RULES_AGENTS_MD)

    with ContextHub() as hub:
        scan_stats = hub.scan_and_ingest_brains(max_files=50)
        seed_stats = None
        if seed_bundle and os.path.exists(seed_bundle):
            seed_stats = hub.import_bundle(seed_bundle, workspace_override=seed_workspace)

    return {
        "plugin_dir": plugin_dir,
        "python": py_exe,
        "scan_stats": scan_stats,
        "seed_stats": seed_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Chimera Context Bridge Antigravity plugin")
    parser.add_argument("--gemini-config", default=None, help="Override ~/.gemini/config path")
    parser.add_argument("--python", default=None, help="Override Python executable path")
    parser.add_argument("--seed", default=None, help="Optional JSON context bundle to seed DB")
    parser.add_argument("--seed-workspace", default=None, help="Remap workspace when importing seed bundle")
    args = parser.parse_args()

    res = install_plugin(
        gemini_config_dir=args.gemini_config,
        python_exe=args.python,
        seed_bundle=args.seed,
        seed_workspace=args.seed_workspace,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
