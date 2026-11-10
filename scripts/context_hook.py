"""Antigravity Lifecycle Hook Handler for Cross-Chat Context Sync.

Implements SessionStart, PreInvocation, PostToolUse, and Stop contracts
from the Antigravity Customization System so all parallel and sequential
chats share unified workspace memory automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict

# Allow importing sibling context_hub regardless of cwd.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from context_hub import ContextHub, default_db_path  # noqa: E402


def _primary_workspace(payload: Dict[str, Any]) -> str:
    paths = payload.get("workspacePaths") or []
    if isinstance(paths, list) and paths:
        return str(paths[0])
    return "global"


def _log_hook_error(event: str, err: Exception) -> None:
    try:
        err_dir = os.path.dirname(default_db_path())
        os.makedirs(err_dir, exist_ok=True)
        err_file = os.path.join(err_dir, "hook_errors.log")
        with open(err_file, "a", encoding="utf-8") as fh:
            fh.write(f"[{event}] {err}\n{traceback.format_exc()}\n")
    except Exception:
        pass


def handle_session_start(payload: Dict[str, Any], hub: ContextHub) -> Dict[str, Any]:
    conv_id = str(payload.get("conversationId", ""))
    workspace = _primary_workspace(payload)
    transcript_path = str(payload.get("transcriptPath", ""))

    # Sync recent transcripts on disk so any prior/parallel chat is visible immediately.
    hub.scan_and_ingest_brains(max_files=20, recent_only_hours=168)
    if transcript_path:
        hub.ingest_transcript(transcript_path, conversation_id=conv_id, workspace=workspace)
    else:
        hub.touch_conversation(conv_id, workspace=workspace)

    hub.init_cursor(conv_id)
    brief = hub.get_shared_context(current_conversation_id=conv_id, workspace=workspace)
    if not brief:
        return {}

    return {
        "injectSteps": [
            {
                "systemMessage": {
                    "systemMessage": brief,
                }
            }
        ]
    }


def handle_pre_invocation(payload: Dict[str, Any], hub: ContextHub) -> Dict[str, Any]:
    conv_id = str(payload.get("conversationId", ""))
    workspace = _primary_workspace(payload)
    transcript_path = str(payload.get("transcriptPath", ""))
    last_user_input = str(payload.get("lastUserInput", ""))

    # Fast incremental sync of active parallel chats from the last 24h.
    hub.scan_and_ingest_brains(max_files=8, recent_only_hours=24)

    if conv_id:
        hub.touch_conversation(
            conv_id,
            workspace=workspace,
            transcript_path=transcript_path,
            last_user_input=last_user_input or None,
        )

    delta = hub.get_delta_update(conv_id, workspace=workspace)
    if not delta:
        return {}

    return {
        "injectSteps": [
            {
                "ephemeralMessage": delta,
            }
        ],
        "terminationBehavior": "",
    }


def handle_post_tool_use(payload: Dict[str, Any], hub: ContextHub) -> Dict[str, Any]:
    conv_id = str(payload.get("conversationId", ""))
    workspace = _primary_workspace(payload)
    transcript_path = str(payload.get("transcriptPath", ""))
    tool_call = payload.get("toolCall") or {}
    error = payload.get("error")

    if isinstance(tool_call, dict):
        hub.record_quick_event(
            conversation_id=conv_id,
            workspace=workspace,
            tool_name=tool_call.get("name"),
            tool_args=tool_call.get("args"),
            error=str(error) if error else None,
        )

    if transcript_path:
        hub.ingest_transcript(transcript_path, conversation_id=conv_id, workspace=workspace)

    return {}


def handle_stop(payload: Dict[str, Any], hub: ContextHub) -> Dict[str, Any]:
    conv_id = str(payload.get("conversationId", ""))
    workspace = _primary_workspace(payload)
    transcript_path = str(payload.get("transcriptPath", ""))
    last_user_input = str(payload.get("lastUserInput", ""))

    if conv_id:
        hub.touch_conversation(
            conv_id,
            workspace=workspace,
            transcript_path=transcript_path,
            last_user_input=last_user_input or None,
        )
    if transcript_path:
        hub.ingest_transcript(transcript_path, conversation_id=conv_id, workspace=workspace)

    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Antigravity Cross-Chat Context Hook")
    parser.add_argument(
        "--event",
        required=True,
        choices=["SessionStart", "PreInvocation", "PostToolUse", "Stop"],
    )
    args = parser.parse_args()

    result: Dict[str, Any] = {}
    try:
        raw_input = sys.stdin.read().strip()
        payload = json.loads(raw_input) if raw_input else {}
        if not isinstance(payload, dict):
            payload = {}

        with ContextHub() as hub:
            if args.event == "SessionStart":
                result = handle_session_start(payload, hub)
            elif args.event == "PreInvocation":
                result = handle_pre_invocation(payload, hub)
            elif args.event == "PostToolUse":
                result = handle_post_tool_use(payload, hub)
            elif args.event == "Stop":
                result = handle_stop(payload, hub)
    except Exception as exc:
        _log_hook_error(args.event, exc)
        result = {}

    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
