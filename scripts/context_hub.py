"""Offline Cross-Chat Shared Context Engine for Antigravity.

Maintains a lightweight local SQLite + FTS5 context hub synchronizing
conversation transcripts, file edits, command results, and pinned notes
across parallel and sequential Antigravity chats with zero network calls.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple


def default_db_path() -> str:
    """Return the default local SQLite database path for shared chat context."""
    override = os.environ.get("CHIMERA_CONTEXT_DB")
    if override:
        return override
    return os.path.join(
        os.path.expanduser("~"),
        ".gemini",
        "antigravity",
        "shared_context",
        "context_bridge.db",
    )


def normalize_workspace(path: Optional[str]) -> str:
    """Normalize workspace paths across Windows/Unix representations."""
    if not path:
        return "global"
    cleaned = os.path.normpath(path).replace("\\", "/").rstrip("/")
    if os.name == "nt" or (len(cleaned) >= 2 and cleaned[1] == ":"):
        cleaned = cleaned.lower()
    return cleaned or "global"


def extract_user_request(raw_content: str) -> str:
    """Strip Antigravity metadata envelopes and return clean user text."""
    if not raw_content:
        return ""
    match = re.search(
        r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>",
        raw_content,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    cleaned = re.sub(
        r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>",
        "",
        raw_content,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r"<USER_SETTINGS_CHANGE>.*?</USER_SETTINGS_CHANGE>",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    return cleaned.strip()


def _truncate(text: str, limit: int) -> str:
    if not text or len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _clean_assistant_summary(text: str) -> str:
    """Strip internal reasoning tags so shared briefs focus on final deliverables."""
    if not text:
        return ""
    cleaned = re.sub(r"<REASONING>.*?</REASONING>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _clean_arg_value(val: Any) -> str:
    if isinstance(val, str):
        # Transcript tool_call args sometimes serialize strings with outer quotes.
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            try:
                return str(json.loads(val))
            except Exception:
                return val.strip('"')
        return val
    return str(val)


def extract_tool_metadata(tool_calls: Any) -> Tuple[List[str], List[str]]:
    """Return compact (tool_summaries, file_paths) from Antigravity tool_calls."""
    if not isinstance(tool_calls, list):
        return [], []

    summaries: List[str] = []
    files: List[str] = []

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name", "")).strip()
        args = call.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        target = ""
        for key in ("Path", "TargetFile", "path"):
            if key in args:
                cleaned = _clean_arg_value(args[key]).replace("\\", "/")
                target = cleaned
                if name in {"write_file", "edit_file", "replace_file_content", "multi_replace_file_content"}:
                    if cleaned not in files:
                        files.append(cleaned)
                break

        if not target and "CommandLine" in args:
            target = _truncate(_clean_arg_value(args["CommandLine"]), 70)

        if name:
            summaries.append(f"{name}({target})" if target else name)

    return summaries, files


def _infer_workspace_from_text(text: str) -> Optional[str]:
    """Best-effort workspace inference from file paths inside metadata/tool calls."""
    if not text:
        return None
    # Match common Windows or Unix ChimeraAI / project paths.
    win_match = re.search(r"([A-Za-z]:[/\\][^\"'\n<>]+?ChimeraAI)", text, re.IGNORECASE)
    if win_match:
        return normalize_workspace(win_match.group(1))
    return None


def _format_age(timestamp: float, now: Optional[float] = None) -> str:
    if not timestamp:
        return "unknown"
    delta = max(0, int((now or time.time()) - timestamp))
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def _sanitize_fts_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]+", query or "")
    if not tokens:
        return ""
    return " AND ".join(f'"{t}"' for t in tokens[:8])


class ContextHub:
    """Thread/process-safe SQLite + FTS5 shared memory store for Antigravity chats."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or default_db_path()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=5.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def __enter__(self) -> "ContextHub":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                workspace TEXT NOT NULL DEFAULT 'global',
                title TEXT NOT NULL DEFAULT '',
                transcript_path TEXT NOT NULL DEFAULT '',
                transcript_mtime REAL NOT NULL DEFAULT 0,
                transcript_size INTEGER NOT NULL DEFAULT 0,
                last_step_index INTEGER NOT NULL DEFAULT -1,
                last_user_input TEXT NOT NULL DEFAULT '',
                last_assistant_summary TEXT NOT NULL DEFAULT '',
                files_touched_json TEXT NOT NULL DEFAULT '[]',
                updated_at REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                tools_json TEXT NOT NULL DEFAULT '[]',
                files_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT '',
                UNIQUE(conversation_id, step_index)
            );

            CREATE INDEX IF NOT EXISTS idx_turns_conv_step
                ON turns(conversation_id, step_index);

            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                conversation_id UNINDEXED,
                workspace UNINDEXED,
                role UNINDEXED,
                content,
                files_json
            );

            CREATE TABLE IF NOT EXISTS pinned_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace TEXT NOT NULL DEFAULT 'global',
                source_conversation_id TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT 'decision',
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(workspace, title)
            );

            CREATE TABLE IF NOT EXISTS sync_cursors (
                consumer_conversation_id TEXT PRIMARY KEY,
                last_seen_updated_at REAL NOT NULL DEFAULT 0
            );
            """
        )
        self.conn.commit()

    def touch_conversation(
        self,
        conversation_id: str,
        workspace: str = "global",
        transcript_path: str = "",
        last_user_input: Optional[str] = None,
    ) -> None:
        """Ensure conversation row exists and refresh metadata."""
        if not conversation_id:
            return
        norm_ws = normalize_workspace(workspace)
        now = time.time()
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

        if row is None:
            title = _truncate(last_user_input or "", 90)
            self.conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id, workspace, title, transcript_path,
                    last_user_input, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    norm_ws,
                    title,
                    transcript_path or "",
                    _truncate(last_user_input or "", 600),
                    now,
                ),
            )
        else:
            new_ws = norm_ws if (norm_ws != "global" or row["workspace"] == "global") else row["workspace"]
            new_tp = transcript_path or row["transcript_path"]
            new_ui = _truncate(last_user_input, 600) if last_user_input else row["last_user_input"]
            new_title = row["title"] or _truncate(new_ui, 90)
            self.conn.execute(
                """
                UPDATE conversations
                SET workspace = ?, transcript_path = ?, last_user_input = ?,
                    title = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (new_ws, new_tp, new_ui, new_title, now, conversation_id),
            )
        self.conn.commit()

    def record_quick_event(
        self,
        conversation_id: str,
        workspace: str,
        user_input: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record real-time tool edits or prompts before transcript flush."""
        if not conversation_id:
            return
        self.touch_conversation(conversation_id, workspace=workspace, last_user_input=user_input)

        if not tool_name:
            return

        _, new_files = extract_tool_metadata([{"name": tool_name, "args": tool_args or {}}])
        if not new_files and not error:
            return

        row = self.conn.execute(
            "SELECT files_touched_json FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        existing_files: List[str] = json.loads(row["files_touched_json"]) if row else []
        for path in new_files:
            if path not in existing_files:
                existing_files.append(path)
        existing_files = existing_files[-30:]

        self.conn.execute(
            """
            UPDATE conversations
            SET files_touched_json = ?, updated_at = ?
            WHERE conversation_id = ?
            """,
            (json.dumps(existing_files), time.time(), conversation_id),
        )
        self.conn.commit()

    def ingest_transcript(
        self,
        transcript_path: str,
        conversation_id: Optional[str] = None,
        workspace: Optional[str] = None,
        force: bool = False,
    ) -> int:
        """Incrementally parse new steps from an Antigravity transcript.jsonl."""
        if not transcript_path or not os.path.exists(transcript_path):
            return 0

        try:
            stat = os.stat(transcript_path)
        except OSError:
            return 0

        if not conversation_id:
            # Expected layout: .../brain/<conversation_id>/.system_generated/logs/transcript.jsonl
            parts = os.path.normpath(transcript_path).split(os.sep)
            if "brain" in parts:
                idx = parts.index("brain")
                if idx + 1 < len(parts):
                    conversation_id = parts[idx + 1]
            if not conversation_id:
                conversation_id = os.path.basename(os.path.dirname(transcript_path))

        row = self.conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

        if (
            not force
            and row is not None
            and row["transcript_mtime"] == stat.st_mtime
            and row["transcript_size"] == stat.st_size
        ):
            return 0

        last_step = row["last_step_index"] if row is not None else -1
        norm_ws = normalize_workspace(workspace) if workspace else (row["workspace"] if row else "global")
        title = row["title"] if row else ""
        last_user_input = row["last_user_input"] if row else ""
        last_assistant_summary = row["last_assistant_summary"] if row else ""
        files_touched: List[str] = json.loads(row["files_touched_json"]) if row else []
        max_step = last_step
        ingested = 0

        with open(transcript_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                step_idx = int(entry.get("step_index", -1))
                if step_idx <= last_step:
                    continue

                step_type = entry.get("type", "")
                raw_content = entry.get("content", "") or ""
                created_at = entry.get("created_at", "") or ""
                max_step = max(max_step, step_idx)

                if norm_ws == "global":
                    inferred = _infer_workspace_from_text(raw_content + json.dumps(entry.get("tool_calls", [])))
                    if inferred:
                        norm_ws = inferred

                role = ""
                stored_content = ""
                tool_summaries: List[str] = []
                step_files: List[str] = []

                if step_type == "USER_INPUT":
                    role = "user"
                    stored_content = extract_user_request(raw_content)
                    if stored_content:
                        last_user_input = _truncate(stored_content, 600)
                        if not title:
                            title = _truncate(stored_content, 90)
                elif step_type == "PLANNER_RESPONSE":
                    role = "assistant"
                    tool_summaries, step_files = extract_tool_metadata(entry.get("tool_calls"))
                    stored_content = _truncate(raw_content.strip(), 1500)
                    clean_summary = _clean_assistant_summary(raw_content)
                    if clean_summary:
                        last_assistant_summary = _truncate(clean_summary, 600)
                    for path in step_files:
                        if path not in files_touched:
                            files_touched.append(path)
                        if norm_ws == "global":
                            inferred = _infer_workspace_from_text(path)
                            if inferred:
                                norm_ws = inferred

                if role and (stored_content or tool_summaries):
                    tools_json = json.dumps(tool_summaries)
                    files_json = json.dumps(step_files)
                    cur = self.conn.execute(
                        """
                        INSERT OR IGNORE INTO turns (
                            conversation_id, step_index, role, content,
                            tools_json, files_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            conversation_id,
                            step_idx,
                            role,
                            stored_content,
                            tools_json,
                            files_json,
                            created_at,
                        ),
                    )
                    if cur.rowcount > 0:
                        fts_text = " ".join(filter(None, [stored_content, " ".join(tool_summaries)]))
                        self.conn.execute(
                            """
                            INSERT INTO turns_fts (conversation_id, workspace, role, content, files_json)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (conversation_id, norm_ws, role, fts_text, files_json),
                        )
                        ingested += 1

        files_touched = files_touched[-30:]
        now = time.time() if ingested > 0 else (row["updated_at"] if row else stat.st_mtime)

        self.conn.execute(
            """
            INSERT INTO conversations (
                conversation_id, workspace, title, transcript_path,
                transcript_mtime, transcript_size, last_step_index,
                last_user_input, last_assistant_summary, files_touched_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                workspace = CASE WHEN excluded.workspace != 'global' THEN excluded.workspace ELSE conversations.workspace END,
                title = CASE WHEN conversations.title = '' THEN excluded.title ELSE conversations.title END,
                transcript_path = excluded.transcript_path,
                transcript_mtime = excluded.transcript_mtime,
                transcript_size = excluded.transcript_size,
                last_step_index = MAX(conversations.last_step_index, excluded.last_step_index),
                last_user_input = CASE WHEN excluded.last_user_input != '' THEN excluded.last_user_input ELSE conversations.last_user_input END,
                last_assistant_summary = CASE WHEN excluded.last_assistant_summary != '' THEN excluded.last_assistant_summary ELSE conversations.last_assistant_summary END,
                files_touched_json = excluded.files_touched_json,
                updated_at = MAX(conversations.updated_at, excluded.updated_at)
            """,
            (
                conversation_id,
                norm_ws,
                title,
                transcript_path,
                stat.st_mtime,
                stat.st_size,
                max_step,
                last_user_input,
                last_assistant_summary,
                json.dumps(files_touched),
                now,
            ),
        )
        self.conn.commit()
        return ingested

    def scan_and_ingest_brains(
        self,
        gemini_root: Optional[str] = None,
        max_files: int = 30,
        recent_only_hours: Optional[float] = None,
    ) -> Dict[str, int]:
        """Discover and incrementally ingest Antigravity transcripts across surfaces."""
        root = gemini_root or os.path.join(os.path.expanduser("~"), ".gemini")
        patterns = [
            os.path.join(root, "antigravity", "brain", "*", ".system_generated", "logs", "transcript.jsonl"),
            os.path.join(root, "antigravity-ide", "brain", "*", ".system_generated", "logs", "transcript.jsonl"),
            os.path.join(root, "antigravity-cli", "brain", "*", ".system_generated", "logs", "transcript.jsonl"),
        ]

        candidates: List[Tuple[float, str]] = []
        cutoff = time.time() - (recent_only_hours * 3600) if recent_only_hours else None
        for pattern in patterns:
            for path in glob.glob(pattern):
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                if cutoff and mtime < cutoff:
                    continue
                candidates.append((mtime, path))

        candidates.sort(key=lambda item: item[0], reverse=True)
        files_checked = 0
        steps_ingested = 0

        for _, path in candidates[:max_files]:
            files_checked += 1
            steps_ingested += self.ingest_transcript(path)

        return {"files_checked": files_checked, "steps_ingested": steps_ingested}

    def pin_memory(
        self,
        title: str,
        content: str,
        workspace: str = "global",
        category: str = "decision",
        source_conversation_id: str = "",
    ) -> None:
        """Persist a shared decision, handoff note, blocker, or fact across all chats."""
        norm_ws = normalize_workspace(workspace)
        now = time.time()
        self.conn.execute(
            """
            INSERT INTO pinned_memory (
                workspace, source_conversation_id, category, title, content, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workspace, title) DO UPDATE SET
                source_conversation_id = excluded.source_conversation_id,
                category = excluded.category,
                content = excluded.content,
                updated_at = excluded.updated_at
            """,
            (norm_ws, source_conversation_id, category, title.strip(), content.strip(), now, now),
        )
        self.conn.commit()

    def init_cursor(self, conversation_id: str) -> None:
        """Set sync cursor for a conversation to current time if not set."""
        if not conversation_id:
            return
        self.conn.execute(
            """
            INSERT OR IGNORE INTO sync_cursors (consumer_conversation_id, last_seen_updated_at)
            VALUES (?, ?)
            """,
            (conversation_id, time.time()),
        )
        self.conn.commit()

    def get_delta_update(self, conversation_id: str, workspace: str = "global") -> str:
        """Return compact delta summary of updates from OTHER chats since last turn."""
        if not conversation_id:
            return ""

        norm_ws = normalize_workspace(workspace)
        cursor_row = self.conn.execute(
            "SELECT last_seen_updated_at FROM sync_cursors WHERE consumer_conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        last_seen = cursor_row["last_seen_updated_at"] if cursor_row else 0.0

        other_convs = self.conn.execute(
            """
            SELECT conversation_id, title, last_user_input, last_assistant_summary,
                   files_touched_json, updated_at
            FROM conversations
            WHERE conversation_id != ?
              AND updated_at > ?
              AND (? = 'global' OR workspace = ?)
            ORDER BY updated_at DESC
            LIMIT 4
            """,
            (conversation_id, last_seen, norm_ws, norm_ws),
        ).fetchall()

        new_pins = self.conn.execute(
            """
            SELECT category, title, content, updated_at
            FROM pinned_memory
            WHERE source_conversation_id != ?
              AND updated_at > ?
              AND (? = 'global' OR workspace = ? OR workspace = 'global')
            ORDER BY updated_at DESC
            LIMIT 4
            """,
            (conversation_id, last_seen, norm_ws, norm_ws),
        ).fetchall()

        latest_ts = last_seen
        lines: List[str] = []

        for pin in new_pins:
            latest_ts = max(latest_ts, float(pin["updated_at"]))
            lines.append(f"- [Pin:{pin['category']}] {pin['title']}: {_truncate(pin['content'], 180)}")

        for conv in other_convs:
            latest_ts = max(latest_ts, float(conv["updated_at"]))
            short_id = conv["conversation_id"][:8]
            files = json.loads(conv["files_touched_json"] or "[]")
            files_str = ", ".join(os.path.basename(f) for f in files[-4:])
            summary = conv["last_assistant_summary"] or conv["last_user_input"] or conv["title"]
            line = f"- [Chat {short_id}] {_truncate(summary, 180)}"
            if files_str:
                line += f" (Files: {files_str})"
            lines.append(line)

        if latest_ts > last_seen or cursor_row is None:
            new_cursor = max(latest_ts, time.time()) if cursor_row is None else latest_ts
            self.conn.execute(
                """
                INSERT INTO sync_cursors (consumer_conversation_id, last_seen_updated_at)
                VALUES (?, ?)
                ON CONFLICT(consumer_conversation_id) DO UPDATE SET
                    last_seen_updated_at = excluded.last_seen_updated_at
                """,
                (conversation_id, new_cursor),
            )
            self.conn.commit()

        if not lines:
            return ""
        return "[Cross-Chat Context Sync]\n" + "\n".join(lines)

    def get_shared_context(
        self,
        current_conversation_id: str = "",
        workspace: str = "global",
        limit_chats: int = 5,
        max_chars: int = 2400,
    ) -> str:
        """Build a concise shared context brief for SessionStart or MCP query."""
        norm_ws = normalize_workspace(workspace)
        now = time.time()

        pins = self.conn.execute(
            """
            SELECT category, title, content, updated_at
            FROM pinned_memory
            WHERE (? = 'global' OR workspace = ? OR workspace = 'global')
            ORDER BY updated_at DESC
            LIMIT 8
            """,
            (norm_ws, norm_ws),
        ).fetchall()

        convs = self.conn.execute(
            """
            SELECT conversation_id, workspace, title, last_user_input,
                   last_assistant_summary, files_touched_json, updated_at
            FROM conversations
            WHERE conversation_id != ?
              AND (? = 'global' OR workspace = ?)
              AND (last_user_input != '' OR last_assistant_summary != '')
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (current_conversation_id or "", norm_ws, norm_ws, limit_chats),
        ).fetchall()

        sections: List[str] = [f"### Shared Antigravity Context (`{norm_ws}`)"]

        if pins:
            sections.append("**Pinned Decisions & Handoffs:**")
            for pin in pins:
                sections.append(f"- [{pin['category']}] **{pin['title']}**: {_truncate(pin['content'], 220)}")

        if convs:
            sections.append("**Connected Recent/Parallel Chats:**")
            for conv in convs:
                short_id = conv["conversation_id"][:8]
                age = _format_age(conv["updated_at"], now)
                files = json.loads(conv["files_touched_json"] or "[]")
                files_str = ", ".join(os.path.basename(f) for f in files[-5:])
                req = _truncate(conv["last_user_input"] or conv["title"], 140)
                ans = _truncate(conv["last_assistant_summary"], 180)
                entry = f"- `{short_id}` ({age}) — Ask: \"{req}\""
                if ans:
                    entry += f" | Outcome: {ans}"
                if files_str:
                    entry += f" | Edited: `{files_str}`"
                sections.append(entry)

        if not pins and not convs:
            sections.append("- No prior chats indexed yet for this workspace.")

        sections.append(
            "Use MCP tools `chimera_search_chats`, `chimera_read_chat_turns`, and `chimera_pin_memory` "
            "to inspect full turns or share decisions with other chats."
        )
        return _truncate("\n".join(sections), max_chars)

    def search_history(
        self,
        query: str,
        workspace: str = "global",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Offline FTS5 search across all indexed Antigravity chat turns."""
        norm_ws = normalize_workspace(workspace)
        fts_query = _sanitize_fts_query(query)
        rows = []

        if fts_query:
            try:
                rows = self.conn.execute(
                    """
                    SELECT f.conversation_id, f.workspace, f.role,
                           snippet(turns_fts, 3, '[', ']', '...', 24) AS snippet,
                           f.files_json
                    FROM turns_fts f
                    WHERE turns_fts MATCH ?
                      AND (? = 'global' OR f.workspace = ? OR f.workspace = 'global')
                    LIMIT ?
                    """,
                    (fts_query, norm_ws, norm_ws, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

        if not rows and query.strip():
            like_pat = f"%{query.strip()}%"
            rows = self.conn.execute(
                """
                SELECT t.conversation_id, c.workspace, t.role,
                       substr(t.content, 1, 240) AS snippet, t.files_json
                FROM turns t
                JOIN conversations c ON c.conversation_id = t.conversation_id
                WHERE t.content LIKE ?
                  AND (? = 'global' OR c.workspace = ? OR c.workspace = 'global')
                ORDER BY t.id DESC
                LIMIT ?
                """,
                (like_pat, norm_ws, norm_ws, limit),
            ).fetchall()

        return [
            {
                "conversation_id": r["conversation_id"],
                "workspace": r["workspace"],
                "role": r["role"],
                "snippet": r["snippet"],
                "files": json.loads(r["files_json"] or "[]"),
            }
            for r in rows
        ]

    def read_chat_turns(self, conversation_id_prefix: str, last_n: int = 12) -> Dict[str, Any]:
        """Fetch the latest N turns for a conversation (full ID or prefix match)."""
        conv = self.conn.execute(
            "SELECT * FROM conversations WHERE conversation_id LIKE ? ORDER BY updated_at DESC LIMIT 1",
            (f"{conversation_id_prefix}%",),
        ).fetchone()
        if not conv:
            return {"error": f"Conversation matching '{conversation_id_prefix}' not found.", "turns": []}

        turn_rows = self.conn.execute(
            """
            SELECT step_index, role, content, tools_json, files_json, created_at
            FROM turns
            WHERE conversation_id = ?
            ORDER BY step_index DESC
            LIMIT ?
            """,
            (conv["conversation_id"], last_n),
        ).fetchall()

        turns = [
            {
                "step_index": r["step_index"],
                "role": r["role"],
                "content": r["content"],
                "tools": json.loads(r["tools_json"] or "[]"),
                "files": json.loads(r["files_json"] or "[]"),
                "created_at": r["created_at"],
            }
            for r in reversed(turn_rows)
        ]

        return {
            "conversation_id": conv["conversation_id"],
            "workspace": conv["workspace"],
            "title": conv["title"],
            "files_touched": json.loads(conv["files_touched_json"] or "[]"),
            "turns": turns,
        }

    def export_bundle(self, output_path: str, workspace: str = "global") -> Dict[str, int]:
        """Export shared context as a portable JSON bundle (zero binary bloat)."""
        norm_ws = normalize_workspace(workspace)
        convs = self.conn.execute(
            """
            SELECT * FROM conversations
            WHERE (? = 'global' OR workspace = ? OR workspace = 'global')
            """,
            (norm_ws, norm_ws),
        ).fetchall()
        conv_ids = [c["conversation_id"] for c in convs]

        turns = []
        if conv_ids:
            placeholders = ",".join("?" for _ in conv_ids)
            turns = [
                dict(r)
                for r in self.conn.execute(
                    f"SELECT conversation_id, step_index, role, content, tools_json, files_json, created_at FROM turns WHERE conversation_id IN ({placeholders})",
                    conv_ids,
                ).fetchall()
            ]

        pins = [
            dict(r)
            for r in self.conn.execute(
                "SELECT workspace, source_conversation_id, category, title, content, created_at, updated_at FROM pinned_memory WHERE (? = 'global' OR workspace = ? OR workspace = 'global')",
                (norm_ws, norm_ws),
            ).fetchall()
        ]

        bundle = {
            "version": 1,
            "exported_at": time.time(),
            "conversations": [dict(c) for c in convs],
            "turns": turns,
            "pinned_memory": pins,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, indent=2)
        return {"conversations": len(convs), "turns": len(turns), "pins": len(pins)}

    def import_bundle(self, input_path: str, workspace_override: Optional[str] = None) -> Dict[str, int]:
        """Import a portable shared context bundle into local SQLite."""
        with open(input_path, "r", encoding="utf-8") as fh:
            bundle = json.load(fh)

        convs_added = 0
        turns_added = 0
        pins_added = 0

        for c in bundle.get("conversations", []):
            ws = normalize_workspace(workspace_override) if workspace_override else c.get("workspace", "global")
            self.conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id, workspace, title, transcript_path,
                    transcript_mtime, transcript_size, last_step_index,
                    last_user_input, last_assistant_summary, files_touched_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    workspace = excluded.workspace,
                    updated_at = MAX(conversations.updated_at, excluded.updated_at)
                """,
                (
                    c["conversation_id"],
                    ws,
                    c.get("title", ""),
                    c.get("transcript_path", ""),
                    0.0,
                    0,
                    c.get("last_step_index", -1),
                    c.get("last_user_input", ""),
                    c.get("last_assistant_summary", ""),
                    c.get("files_touched_json", "[]"),
                    c.get("updated_at", time.time()),
                ),
            )
            convs_added += 1

        ws_map = {
            c["conversation_id"]: (normalize_workspace(workspace_override) if workspace_override else c.get("workspace", "global"))
            for c in bundle.get("conversations", [])
        }

        for t in bundle.get("turns", []):
            cur = self.conn.execute(
                """
                INSERT OR IGNORE INTO turns (
                    conversation_id, step_index, role, content, tools_json, files_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    t["conversation_id"],
                    t["step_index"],
                    t["role"],
                    t.get("content", ""),
                    t.get("tools_json", "[]"),
                    t.get("files_json", "[]"),
                    t.get("created_at", ""),
                ),
            )
            if cur.rowcount > 0:
                self.conn.execute(
                    """
                    INSERT INTO turns_fts (conversation_id, workspace, role, content, files_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        t["conversation_id"],
                        ws_map.get(t["conversation_id"], "global"),
                        t["role"],
                        t.get("content", ""),
                        t.get("files_json", "[]"),
                    ),
                )
                turns_added += 1

        for p in bundle.get("pinned_memory", []):
            ws = normalize_workspace(workspace_override) if workspace_override else p.get("workspace", "global")
            self.pin_memory(
                title=p["title"],
                content=p["content"],
                workspace=ws,
                category=p.get("category", "decision"),
                source_conversation_id=p.get("source_conversation_id", ""),
            )
            pins_added += 1

        self.conn.commit()
        return {"conversations": convs_added, "turns": turns_added, "pins": pins_added}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Chimera Cross-Chat Shared Context Hub CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan", help="Scan and ingest all local Antigravity transcripts")

    brief_p = sub.add_parser("brief", help="Print shared context brief")
    brief_p.add_argument("--workspace", default="global")

    search_p = sub.add_parser("search", help="Search chat history")
    search_p.add_argument("query")
    search_p.add_argument("--workspace", default="global")

    exp_p = sub.add_parser("export", help="Export shared context bundle JSON")
    exp_p.add_argument("output")
    exp_p.add_argument("--workspace", default="global")

    imp_p = sub.add_parser("import", help="Import shared context bundle JSON")
    imp_p.add_argument("input")
    imp_p.add_argument("--workspace", default=None)

    args = parser.parse_args()
    with ContextHub() as hub:
        if args.cmd == "scan":
            print(json.dumps(hub.scan_and_ingest_brains(), indent=2))
        elif args.cmd == "brief":
            print(hub.get_shared_context(workspace=args.workspace))
        elif args.cmd == "search":
            print(json.dumps(hub.search_history(args.query, workspace=args.workspace), indent=2))
        elif args.cmd == "export":
            print(json.dumps(hub.export_bundle(args.output, workspace=args.workspace), indent=2))
        elif args.cmd == "import":
            print(json.dumps(hub.import_bundle(args.input, workspace_override=args.workspace), indent=2))


if __name__ == "__main__":
    _cli()
