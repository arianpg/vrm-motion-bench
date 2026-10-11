"""結果の保存（SQLite のメタデータ。ファイル本体は data/results/<result_id>/ に置く）。"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS groups (
    id TEXT PRIMARY KEY,
    request_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    preview TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    group_id TEXT REFERENCES groups(id),
    adapter_id TEXT NOT NULL,
    adapter_version TEXT,
    request_text TEXT NOT NULL DEFAULT '',
    llm_model TEXT,
    llm_prompt TEXT,
    final_prompt TEXT,
    inputs_json TEXT NOT NULL DEFAULT '{}',
    params_json TEXT NOT NULL DEFAULT '{}',
    seed INTEGER,
    status TEXT NOT NULL,
    error TEXT,
    load_seconds REAL,
    generate_seconds REAL,
    convert_seconds REAL,
    duration REAL,
    fps REAL,
    files_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS drafts (
    id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL REFERENCES groups(id),
    adapter_id TEXT NOT NULL,
    llm_model TEXT NOT NULL,
    request_text TEXT NOT NULL,
    status TEXT NOT NULL,
    prompt TEXT,
    duration REAL,
    notes TEXT,
    violations_json TEXT NOT NULL DEFAULT '[]',
    attempts INTEGER NOT NULL DEFAULT 0,
    reasoning TEXT,
    llm_seconds REAL,
    load_seconds REAL,
    error TEXT,
    result_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS drafts_group ON drafts(group_id);
CREATE INDEX IF NOT EXISTS results_group ON results(group_id);
CREATE INDEX IF NOT EXISTS results_created ON results(created_at);
"""

_JSON_COLUMNS = {
    "inputs_json": "inputs",
    "params_json": "params",
    "files_json": "files",
    "violations_json": "violations",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Database:
    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def _row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        for col, key in _JSON_COLUMNS.items():
            if col in out:
                out[key] = json.loads(out.pop(col) or ("[]" if key == "violations" else "{}"))
        return out

    # ------------------------------------------------------------ groups

    def create_group(self, request_text: str) -> str:
        gid = new_id()
        with self._lock:
            self._conn.execute(
                "INSERT INTO groups (id, request_text, created_at) VALUES (?, ?, ?)", (gid, request_text, now_iso())
            )
            self._conn.commit()
        return gid

    def get_group(self, gid: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM groups WHERE id = ?", (gid,)).fetchone()
        return self._row(row)

    def set_group_preview(self, gid: str, preview: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE groups SET preview = ? WHERE id = ?", (preview, gid))
            self._conn.commit()

    def list_groups(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM groups ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    # ------------------------------------------------------------ results

    def create_result(
        self,
        *,
        adapter_id: str,
        adapter_version: str,
        group_id: str | None,
        request_text: str,
        inputs: dict[str, Any],
        params: dict[str, Any],
        final_prompt: str | None = None,
        llm_model: str | None = None,
        llm_prompt: str | None = None,
    ) -> str:
        rid = new_id()
        with self._lock:
            self._conn.execute(
                """INSERT INTO results (id, group_id, adapter_id, adapter_version, request_text, llm_model,
                   llm_prompt, final_prompt, inputs_json, params_json, seed, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)""",
                (
                    rid,
                    group_id,
                    adapter_id,
                    adapter_version,
                    request_text,
                    llm_model,
                    llm_prompt,
                    final_prompt,
                    json.dumps(inputs, ensure_ascii=False),
                    json.dumps(params, ensure_ascii=False),
                    params.get("seed"),
                    now_iso(),
                ),
            )
            self._conn.commit()
        return rid

    def update_result(self, rid: str, **fields: Any) -> None:
        if not fields:
            return
        cols, values = [], []
        for key, value in fields.items():
            if key in ("inputs", "params", "files"):
                key, value = f"{key}_json", json.dumps(value, ensure_ascii=False)
            cols.append(f"{key} = ?")
            values.append(value)
        with self._lock:
            self._conn.execute(f"UPDATE results SET {', '.join(cols)} WHERE id = ?", (*values, rid))
            self._conn.commit()

    def add_result_files(self, rid: str, files: dict[str, str]) -> None:
        with self._lock:
            row = self._conn.execute("SELECT files_json FROM results WHERE id = ?", (rid,)).fetchone()
            merged = {**json.loads(row["files_json"] or "{}"), **files}
            self._conn.execute(
                "UPDATE results SET files_json = ? WHERE id = ?", (json.dumps(merged, ensure_ascii=False), rid)
            )
            self._conn.commit()

    def get_result(self, rid: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM results WHERE id = ?", (rid,)).fetchone()
        return self._row(row)

    def list_results(self, *, group_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            if group_id:
                rows = self._conn.execute(
                    "SELECT * FROM results WHERE group_id = ? ORDER BY created_at", (group_id,)
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM results ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    # ------------------------------------------------------------ drafts（LLM が作ったプロンプト案）

    def create_draft(self, *, group_id: str, adapter_id: str, llm_model: str, request_text: str) -> str:
        did = new_id()
        with self._lock:
            self._conn.execute(
                """INSERT INTO drafts (id, group_id, adapter_id, llm_model, request_text, status, created_at)
                   VALUES (?, ?, ?, ?, ?, 'queued', ?)""",
                (did, group_id, adapter_id, llm_model, request_text, now_iso()),
            )
            self._conn.commit()
        return did

    def update_draft(self, did: str, **fields: Any) -> None:
        if not fields:
            return
        cols, values = [], []
        for key, value in fields.items():
            if key == "violations":
                key, value = "violations_json", json.dumps(value, ensure_ascii=False)
            cols.append(f"{key} = ?")
            values.append(value)
        with self._lock:
            self._conn.execute(f"UPDATE drafts SET {', '.join(cols)} WHERE id = ?", (*values, did))
            self._conn.commit()

    def get_draft(self, did: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM drafts WHERE id = ?", (did,)).fetchone()
        return self._row(row)

    def list_drafts(self, group_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM drafts WHERE group_id = ? ORDER BY created_at, rowid", (group_id,)
            ).fetchall()
        return [self._row(r) for r in rows]

    def mark_interrupted(self) -> int:
        """起動時: 完了していない結果とプロンプト案を failed（中断）にする。"""
        with self._lock:
            self._conn.execute(
                "UPDATE drafts SET status = 'failed', error = 'interrupted: orchestrator restarted' "
                "WHERE status NOT IN ('done', 'failed')"
            )
            cur = self._conn.execute(
                "UPDATE results SET status = 'failed', error = 'interrupted: orchestrator restarted', finished_at = ? "
                "WHERE status NOT IN ('done', 'failed')",
                (now_iso(),),
            )
            self._conn.commit()
        return cur.rowcount

    def average_seconds(self, adapter_id: str, column: str, limit: int = 10) -> float | None:
        """残り時間の見積もり用: 直近の実績の平均。"""
        if column not in ("load_seconds", "generate_seconds", "convert_seconds"):
            raise ValueError(column)
        with self._lock:
            row = self._conn.execute(
                f"SELECT AVG({column}) AS v FROM (SELECT {column} FROM results WHERE adapter_id = ? "
                f"AND {column} IS NOT NULL ORDER BY created_at DESC LIMIT ?)",
                (adapter_id, limit),
            ).fetchone()
        return row["v"]
