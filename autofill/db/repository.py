"""History repository.

``HistoryRepository`` is the interface the rest of the app depends on. ``SQLiteHistoryRepository``
is the local implementation; a PostgreSQL implementation only has to implement the same methods
(the SQL here is deliberately plain, with ``?`` placeholders and no SQLite-only functions in
queries; only the connection/migration code is SQLite specific).

Schema versions (PRAGMA user_version)
  0 -> original app: history(id, candidate_name, processed_at, source_count, raw_text, fields_json)
  1 -> adds processed_date, evidence_json, review_json, instruction + indexes. Existing rows are kept.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

from ..fields import normalize_fields

SCHEMA_VERSION = 1


@dataclass
class HistorySummary:
    id: int
    candidate_name: str
    processed_at: str
    source_count: int
    fields: dict[str, str]


@dataclass
class HistoryRecord(HistorySummary):
    raw_text: str = ""
    evidence: dict = field(default_factory=dict)
    review: dict = field(default_factory=dict)
    instruction: str = ""


class HistoryRepository(ABC):
    @abstractmethod
    def add(self, candidate_name: str, fields: dict, raw_text: str, source_count: int,
            evidence: dict | None = None, review: dict | None = None, instruction: str = "",
            processed_at: datetime | None = None) -> int: ...

    @abstractmethod
    def get(self, record_id: int) -> HistoryRecord | None: ...

    @abstractmethod
    def update_fields(self, record_id: int, candidate_name: str, fields: dict,
                      review: dict | None = None) -> bool: ...

    @abstractmethod
    def list_summaries(self, day: date, limit: int | None = None) -> list[HistorySummary]: ...

    @abstractmethod
    def count(self, day: date) -> int: ...

    @abstractmethod
    def search_name(self, day: date, query: str) -> list[HistorySummary]: ...

    @abstractmethod
    def delete_before(self, day: date) -> int: ...


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class SQLiteHistoryRepository(HistoryRepository):
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    # ------------------------------------------------------------------ connection
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self._conn() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version >= SCHEMA_VERSION:
                conn.execute("PRAGMA journal_mode=WAL")  # concurrent readers while one session writes
                return
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='history'").fetchone()
            if has_table and conn.execute("SELECT COUNT(*) FROM history").fetchone()[0] > 0:
                conn.commit()
                backup = self.path.with_suffix(self.path.suffix + ".pre-migration.bak")
                if not backup.exists():
                    shutil.copy2(self.path, backup)  # one-time safety copy of real data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_name TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    source_count INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    fields_json TEXT NOT NULL
                )""")
            existing = {r["name"] for r in conn.execute("PRAGMA table_info(history)")}
            for column, ddl in (("processed_date", "TEXT"), ("evidence_json", "TEXT NOT NULL DEFAULT '{}'"),
                                ("review_json", "TEXT NOT NULL DEFAULT '{}'"),
                                ("instruction", "TEXT NOT NULL DEFAULT ''")):
                if column not in existing:
                    conn.execute(f"ALTER TABLE history ADD COLUMN {column} {ddl}")
            conn.execute("UPDATE history SET processed_date = substr(processed_at, 1, 10) "
                         "WHERE processed_date IS NULL")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_date ON history(processed_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_name ON history(candidate_name)")
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            conn.execute("PRAGMA journal_mode=WAL")

    # ------------------------------------------------------------------ writes
    def add(self, candidate_name, fields, raw_text, source_count, evidence=None, review=None,
            instruction="", processed_at=None) -> int:
        stamp = (processed_at or datetime.now().astimezone()).isoformat(timespec="seconds")
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO history(candidate_name, processed_at, processed_date, source_count, raw_text,"
                " fields_json, evidence_json, review_json, instruction) VALUES(?,?,?,?,?,?,?,?,?)",
                (candidate_name, stamp, stamp[:10], int(source_count), raw_text or "",
                 json.dumps(normalize_fields(fields), ensure_ascii=False),
                 json.dumps(evidence or {}, ensure_ascii=False), json.dumps(review or {}, ensure_ascii=False),
                 instruction or ""))
            return int(cur.lastrowid)

    def update_fields(self, record_id, candidate_name, fields, review=None) -> bool:
        with self._conn() as conn:
            if review is None:
                cur = conn.execute("UPDATE history SET candidate_name=?, fields_json=? WHERE id=?",
                                   (candidate_name, json.dumps(normalize_fields(fields), ensure_ascii=False),
                                    record_id))
            else:
                cur = conn.execute(
                    "UPDATE history SET candidate_name=?, fields_json=?, review_json=? WHERE id=?",
                    (candidate_name, json.dumps(normalize_fields(fields), ensure_ascii=False),
                     json.dumps(review, ensure_ascii=False), record_id))
            return cur.rowcount > 0

    def delete_before(self, day: date) -> int:
        """Delete records processed before ``day``."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM history WHERE COALESCE(processed_date, substr(processed_at,1,10)) < ?",
                (day.isoformat(),))
            return cur.rowcount

    # ------------------------------------------------------------------ reads
    def get(self, record_id: int) -> HistoryRecord | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM history WHERE id=?", (record_id,)).fetchone()
        if not row:
            return None
        return HistoryRecord(
            id=row["id"], candidate_name=row["candidate_name"], processed_at=row["processed_at"],
            source_count=row["source_count"], fields=normalize_fields(json.loads(row["fields_json"] or "{}")),
            raw_text=row["raw_text"], evidence=json.loads(row["evidence_json"] or "{}"),
            review=json.loads(row["review_json"] or "{}"), instruction=row["instruction"] or "")

    def list_summaries(self, day: date, limit: int | None = None) -> list[HistorySummary]:
        sql = ("SELECT id, candidate_name, processed_at, source_count, fields_json FROM history "
               "WHERE COALESCE(processed_date, substr(processed_at,1,10)) = ? ORDER BY id DESC")
        params: list = [day.isoformat()]
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [HistorySummary(r["id"], r["candidate_name"], r["processed_at"], r["source_count"],
                               normalize_fields(json.loads(r["fields_json"] or "{}"))) for r in rows]

    def count(self, day: date) -> int:
        with self._conn() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM history WHERE COALESCE(processed_date, substr(processed_at,1,10)) = ?",
                (day.isoformat(),)).fetchone()[0])

    def search_name(self, day: date, query: str) -> list[HistorySummary]:
        """Case-insensitive name search, done in SQL so large days stay fast."""
        like = f"%{_like_escape(query.strip().lower())}%"
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, candidate_name, processed_at, source_count, fields_json FROM history "
                "WHERE COALESCE(processed_date, substr(processed_at,1,10)) = ? "
                "AND lower(candidate_name) LIKE ? ESCAPE '\\' ORDER BY id DESC", (day.isoformat(), like)).fetchall()
        return [HistorySummary(r["id"], r["candidate_name"], r["processed_at"], r["source_count"],
                               normalize_fields(json.loads(r["fields_json"] or "{}"))) for r in rows]
