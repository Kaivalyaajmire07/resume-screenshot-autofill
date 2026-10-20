import json
import sqlite3
from datetime import date, datetime, timedelta

from autofill.history import candidate_name, save_candidate, search_today
from autofill.retention import cutoff_day, purge_expired


def test_legacy_database_is_migrated_without_data_loss(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE history (id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_name TEXT NOT NULL,
        processed_at TEXT NOT NULL, source_count INTEGER NOT NULL, raw_text TEXT NOT NULL, fields_json TEXT NOT NULL)""")
    today = datetime.now().astimezone().isoformat(timespec="seconds")
    conn.execute("INSERT INTO history(candidate_name, processed_at, source_count, raw_text, fields_json) VALUES(?,?,?,?,?)",
                 ("Old Row", today, 3, "raw", json.dumps({"email": "x@y.com"})))
    conn.commit(); conn.close()

    from autofill.db import SQLiteHistoryRepository
    repo = SQLiteHistoryRepository(path)
    rec = repo.get(1)
    assert rec.candidate_name == "Old Row" and rec.fields["email"] == "x@y.com" and rec.source_count == 3
    assert rec.fields["first_name"] == ""            # missing keys are filled in
    assert repo.count(date.today()) == 1             # processed_date backfilled
    assert (tmp_path / "legacy.db.pre-migration.bak").exists()
    SQLiteHistoryRepository(path)                    # re-opening is idempotent
    assert repo.count(date.today()) == 1


def test_save_search_update(repo):
    rid = save_candidate(repo, {"first_name": "Anita", "last_name": "Rao", "email": "a@b.com"}, "raw", 2,
                         evidence={"email": {"source_index": 1}}, instruction="note")
    assert candidate_name({"first_name": "Anita", "last_name": "Rao"}) == "Anita Rao"
    assert [r.id for r in search_today(repo, "anita")] == [rid]
    assert search_today(repo, "zzz") == []
    assert [r.id for r in search_today(repo, "")] == [rid]
    # saving the same draft again updates instead of duplicating
    assert save_candidate(repo, {"first_name": "Anita", "last_name": "R"}, "raw", 2, existing_id=rid) == rid
    assert repo.count(date.today()) == 1 and repo.get(rid).candidate_name == "Anita R"
    assert repo.get(rid).instruction == "note" and repo.get(rid).evidence == {"email": {"source_index": 1}}


def test_repeat_processing_creates_separate_entries(repo):
    a = save_candidate(repo, {"first_name": "Same"}, "r", 1)
    b = save_candidate(repo, {"first_name": "Same"}, "r", 1)
    assert a != b and repo.count(date.today()) == 2


def test_search_escapes_like_wildcards(repo):
    save_candidate(repo, {"first_name": "100%"}, "r", 1)
    save_candidate(repo, {"first_name": "Bob"}, "r", 1)
    assert [r.candidate_name for r in search_today(repo, "%")] == ["100%"]
    assert search_today(repo, "_ob") == []


def test_unknown_candidate_name(repo):
    rid = save_candidate(repo, {}, "r", 1)
    assert repo.get(rid).candidate_name == "Unknown Candidate"


def test_retention_deletes_only_previous_days(repo):
    now = datetime.now().astimezone()
    old = repo.add("Old", {}, "r", 1, processed_at=now - timedelta(days=1))
    new = repo.add("New", {}, "r", 1, processed_at=now)
    assert purge_expired(repo) == 1
    assert repo.get(old) is None and repo.get(new) is not None


def test_retention_window_and_throttle(repo):
    now = datetime.now().astimezone()
    repo.add("Y", {}, "r", 1, processed_at=now - timedelta(days=1))
    assert cutoff_day(date(2026, 10, 19), 2) == date(2026, 10, 17)
    assert purge_expired(repo, keep_days=1) == 0
    assert purge_expired(repo, keep_days=0, min_interval=60) == 1
    assert purge_expired(repo, keep_days=0, min_interval=60) == 0   # throttled


def test_many_records_stay_fast(repo):
    import time
    for i in range(1000):
        repo.add(f"Cand {i}", {"first_name": f"Cand{i}"}, "raw", 1)
    t = time.perf_counter()
    assert len(search_today(repo, "cand 99")) == 11
    assert repo.count(date.today()) == 1000
    assert time.perf_counter() - t < 1.0
