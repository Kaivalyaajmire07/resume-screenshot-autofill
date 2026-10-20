"""Headless run of the real Streamlit script (OCR itself is covered by the fake-engine tests)."""
from datetime import date

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture
def app(tmp_path, monkeypatch):
    import streamlit as st
    from autofill import config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    st.cache_resource.clear()
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    monkeypatch.chdir(root)
    return st_testing.AppTest.from_file(str(root / "app.py"), default_timeout=30)


def test_app_starts_clean(app):
    at = app.run()
    assert not at.exception
    assert any("Resume Screenshot Autofill" in t.value for t in at.title)
    assert [b.label for b in at.button if b.key == "extract"] and at.button(key="extract").disabled


def test_history_tab_lists_and_edits_selected_record(app):
    from autofill.db import SQLiteHistoryRepository
    from autofill import config
    from autofill.history import save_candidate
    repo = SQLiteHistoryRepository(config.DB_PATH)
    save_candidate(repo, {"first_name": "Anita", "last_name": "Rao"}, "raw", 2)
    at = app.run()
    assert not at.exception
    assert any("1 of 1 record" in c.value for c in at.caption)


def test_review_form_renders_after_extraction_and_saves(app, fake_engine, png_bytes):
    from autofill.pipeline import process
    from autofill.attachments import AttachmentQueue
    q = AttachmentQueue(); q.add("a.png", png_bytes())
    result = process(list(q.items), fake_engine)
    app.session_state["queue"] = q
    app.session_state["result"] = result
    app.session_state["run_id"] = 1
    app.session_state["saved_id"] = None
    app.session_state["screenshots"] = 1
    at = app.run()
    assert not at.exception
    assert at.text_input(key="draft1_email").value == "anita.rao@example.com"
    assert at.text_input(key="draft1_first_name").value == "Anita"
    at.text_input(key="draft1_last_name").set_value("Rao-Edited")
    at.button[[b.label for b in at.button].index("Save to today's history")].click()
    at = at.run()
    assert not at.exception
    from autofill import config
    from autofill.db import SQLiteHistoryRepository
    repo = SQLiteHistoryRepository(config.DB_PATH)
    rows = repo.list_summaries(date.today())
    assert len(rows) == 1 and rows[0].candidate_name == "Anita Rao-Edited"
    # saving again updates the same entry rather than duplicating it
    at.button[[b.label for b in at.button].index("Update saved entry")].click()
    at.run()
    assert repo.count(date.today()) == 1
