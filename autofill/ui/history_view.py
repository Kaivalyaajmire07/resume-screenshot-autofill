"""Searchable history: a table of today's candidates plus ONE editor for the selected row.

Rendering one editor (instead of one form per record, as the original did) keeps the page fast
with hundreds of records per day.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .. import nlq
from ..db import HistoryRepository
from ..geo import fill_location
from ..history import candidate_name, search_today
from ..validation import validate
from .forms import render_fields

PAGE_SIZE = 200


def render(repo: HistoryRepository) -> None:
    today = date.today()
    st.subheader("Search today's processed candidates")
    total = repo.count(today)
    query = st.text_input("Search candidate name", placeholder="e.g. Anita Rao", key="hist_search")
    rows = search_today(repo, query, today)
    st.caption(f"{len(rows)} of {total} record(s) shown for today. Older dates are cleared automatically.")

    with st.expander("💬 Ask a question about today's candidates (read-only)"):
        q = st.text_input("Question", key="nlq_q",
                          placeholder="how many candidates today · email of Anita · who knows Python · from Pune")
        if q:
            ans = nlq.answer(repo, q, today)
            st.write(ans.message)
            if ans.rows:
                st.dataframe(pd.DataFrame(ans.rows), hide_index=True)

    if not rows:
        st.info("No matching history yet. Process a resume and save it to see it here.")
        return
    shown = rows[:PAGE_SIZE]
    if len(rows) > PAGE_SIZE:
        st.caption(f"Showing the newest {PAGE_SIZE}. Narrow the search to see others.")
    df = pd.DataFrame([{"ID": r.id, "Candidate": r.candidate_name, "Mobile": r.fields.get("mobile", ""),
                        "Email": r.fields.get("email", ""), "Screenshots": r.source_count,
                        "Processed": r.processed_at.replace("T", " ")[:19]} for r in shown])
    event = st.dataframe(df, hide_index=True, on_select="rerun", selection_mode="single-row", key="hist_table")
    selected = event.selection.rows if event and event.selection else []
    if not selected:
        st.caption("Select a row to view or edit that candidate.")
        return

    record = repo.get(int(df.iloc[selected[0]]["ID"]))
    if record is None:
        st.error("That record no longer exists (it may have expired).")
        return
    st.markdown(f"### {record.candidate_name}  ·  #{record.id}")
    if record.instruction:
        st.caption("Note: " + record.instruction)
    with st.form(f"history_form_{record.id}"):
        report = validate(record.fields)
        edited = render_fields(record.fields, f"hist{record.id}", report)
        saved = st.form_submit_button("Save edits", type="primary")
    if saved:
        edited = fill_location(edited)
        new_report = validate(edited)
        repo.update_fields(record.id, candidate_name(edited), edited, new_report.to_dict())
        st.success("Changes saved.")
        st.rerun()
    with st.expander("Original OCR text"):
        st.text(record.raw_text)
