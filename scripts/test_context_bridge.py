"""End-to-end verification tests for Chimera Cross-Chat Context Bridge."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from context_hub import ContextHub  # noqa: E402
from install_context_bridge import install_plugin  # noqa: E402


class TestContextBridge(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="chimera_ctx_test_")
        self.db_path = os.path.join(self.tmp_dir, "context.db")
        self.env = os.environ.copy()
        self.env["CHIMERA_CONTEXT_DB"] = self.db_path

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_mock_transcript(self, conv_id: str, steps: list[dict]) -> str:
        path = os.path.join(self.tmp_dir, "brain", conv_id, ".system_generated", "logs", "transcript.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for step in steps:
                fh.write(json.dumps(step) + "\n")
        return path

    def _run_hook(self, event: str, payload: dict) -> dict:
        proc = subprocess.run(
            [sys.executable, os.path.join(_SCRIPT_DIR, "context_hook.py"), "--event", event],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=self.env,
            check=True,
        )
        return json.loads(proc.stdout.strip())

    def test_incremental_ingestion_and_search(self) -> None:
        conv_a = "aaaa1111-0000-0000-0000-000000000001"
        t_path = self._write_mock_transcript(
            conv_a,
            [
                {
                    "step_index": 0,
                    "type": "USER_INPUT",
                    "created_at": "2026-11-10T10:00:00Z",
                    "content": "<USER_REQUEST>\nImplement BitNet quantization\n</USER_REQUEST>\n<ADDITIONAL_METADATA>noise</ADDITIONAL_METADATA>",
                },
                {
                    "step_index": 1,
                    "type": "PLANNER_RESPONSE",
                    "created_at": "2026-11-10T10:00:05Z",
                    "content": "Added TernaryQuantize to src/model.py.",
                    "tool_calls": [
                        {"name": "edit_file", "args": {"Path": '"C:\\Users\\vinay\\ChimeraAI\\src\\model.py"'}}
                    ],
                },
            ],
        )

        with ContextHub(self.db_path) as hub:
            count = hub.ingest_transcript(t_path, workspace=r"C:\Users\vinay\ChimeraAI")
            self.assertEqual(count, 2)
            # Re-ingesting without file growth should do 0 inserts.
            self.assertEqual(hub.ingest_transcript(t_path), 0)

            hits = hub.search_history("TernaryQuantize", workspace=r"C:\Users\vinay\ChimeraAI")
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["conversation_id"], conv_a)

            turns = hub.read_chat_turns("aaaa1111")
            self.assertEqual(len(turns["turns"]), 2)
            self.assertIn("Implement BitNet quantization", turns["turns"][0]["content"])
            self.assertNotIn("noise", turns["turns"][0]["content"])

    def test_cross_chat_hooks_session_and_delta_sync(self) -> None:
        ws = r"C:\Users\vinay\ChimeraAI"
        conv_a = "aaaa1111-0000-0000-0000-000000000001"
        conv_b = "bbbb2222-0000-0000-0000-000000000002"

        t_a = self._write_mock_transcript(
            conv_a,
            [
                {
                    "step_index": 0,
                    "type": "USER_INPUT",
                    "content": "<USER_REQUEST>\nFix tokenizer bug\n</USER_REQUEST>",
                },
                {
                    "step_index": 1,
                    "type": "PLANNER_RESPONSE",
                    "content": "Patched digit isolation in src/tokenizer.py.",
                    "tool_calls": [{"name": "edit_file", "args": {"Path": "src/tokenizer.py"}}],
                },
            ],
        )
        # Chat A finishes its turn.
        stop_out = self._run_hook(
            "Stop",
            {
                "conversationId": conv_a,
                "workspacePaths": [ws],
                "transcriptPath": t_a,
                "lastUserInput": "Fix tokenizer bug",
                "terminationReason": "NO_TOOL_CALL",
            },
        )
        self.assertEqual(stop_out, {})

        # Chat B starts up and must automatically receive Chat A's context.
        start_out = self._run_hook(
            "SessionStart",
            {
                "conversationId": conv_b,
                "workspacePaths": [ws],
                "transcriptPath": "",
            },
        )
        sys_msg = start_out["injectSteps"][0]["systemMessage"]["systemMessage"]
        self.assertIn("aaaa1111", sys_msg)
        self.assertIn("tokenizer.py", sys_msg)

        # Immediately running PreInvocation on Chat B should see no new delta.
        pre_empty = self._run_hook(
            "PreInvocation",
            {"conversationId": conv_b, "workspacePaths": [ws], "lastUserInput": "hi"},
        )
        self.assertEqual(pre_empty, {})

        # Now Chat A edits another file via PostToolUse while Chat B is open!
        self._run_hook(
            "PostToolUse",
            {
                "conversationId": conv_a,
                "workspacePaths": [ws],
                "toolCall": {"name": "write_file", "args": {"Path": "src/speculative.py"}},
            },
        )

        # Chat B's next PreInvocation receives the live delta automatically!
        pre_delta = self._run_hook(
            "PreInvocation",
            {"conversationId": conv_b, "workspacePaths": [ws], "lastUserInput": "next step"},
        )
        eph = pre_delta["injectSteps"][0]["ephemeralMessage"]
        self.assertIn("Cross-Chat Context Sync", eph)
        self.assertIn("speculative.py", eph)

    def test_mcp_stdio_protocol_and_bundle_transfer(self) -> None:
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "chimera_pin_memory",
                    "arguments": {
                        "title": "RTX Target Node",
                        "content": "Primary dev and training runs on 192.168.68.115.",
                        "category": "decision",
                        "workspace": r"C:\Users\vinay\ChimeraAI",
                    },
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "chimera_get_shared_context",
                    "arguments": {"workspace": r"C:\Users\vinay\ChimeraAI"},
                },
            },
        ]
        stdin_payload = "\n".join(json.dumps(r) for r in requests) + "\n"
        proc = subprocess.run(
            [sys.executable, os.path.join(_SCRIPT_DIR, "context_mcp_server.py")],
            input=stdin_payload,
            capture_output=True,
            text=True,
            env=self.env,
            check=True,
        )
        responses = [json.loads(line) for line in proc.stdout.strip().splitlines() if line.strip()]
        self.assertEqual(len(responses), 4)
        self.assertEqual(responses[0]["result"]["serverInfo"]["name"], "chimera-context-bridge")
        self.assertEqual(len(responses[1]["result"]["tools"]), 5)
        self.assertFalse(responses[2]["result"]["isError"])
        self.assertIn("RTX Target Node", responses[3]["result"]["content"][0]["text"])

        # Verify bundle export/import.
        bundle_path = os.path.join(self.tmp_dir, "bundle.json")
        db2_path = os.path.join(self.tmp_dir, "context2.db")
        with ContextHub(self.db_path) as hub1:
            exp_stats = hub1.export_bundle(bundle_path)
            self.assertEqual(exp_stats["pins"], 1)
        with ContextHub(db2_path) as hub2:
            imp_stats = hub2.import_bundle(bundle_path)
            self.assertEqual(imp_stats["pins"], 1)
            ctx = hub2.get_shared_context(workspace=r"C:\Users\vinay\ChimeraAI")
            self.assertIn("192.168.68.115", ctx)

    def test_plugin_installer(self) -> None:
        fake_cfg = os.path.join(self.tmp_dir, "gemini_cfg")
        res = install_plugin(gemini_config_dir=fake_cfg)
        pdir = res["plugin_dir"]
        for fname in ("plugin.json", "hooks.json", "mcp_config.json", "context_hub.py", "context_hook.py", "context_mcp_server.py"):
            self.assertTrue(os.path.exists(os.path.join(pdir, fname)), f"Missing {fname}")
        self.assertTrue(os.path.exists(os.path.join(pdir, "rules", "AGENTS.md")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
