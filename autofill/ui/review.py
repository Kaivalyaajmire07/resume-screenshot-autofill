"""Extraction results + editable review form + save."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..db import HistoryRepository
from ..fields import FIELD_KEYS, LABELS
from ..history import save_candidate
from ..pipeline import PipelineResult
from ..structure import education_table, jobs_table, lines_table, plain_text_sections, sections_table
from ..validation import validate
from ..jobs import split_jobs
from .forms import render_fields


def _summary_pills(result: PipelineResult, screenshots: int) -> None:
    filled = sum(1 for k in FIELD_KEYS if k != "qualification_records" and result.extraction.values.get(k))
    flagged = len(result.report.flagged)
    st.markdown(
        f"<span class='pill ok'>{result.images_ok}/{screenshots} screenshots read</span>"
        f"<span class='pill'>{filled}/{len(FIELD_KEYS) - 1} fields filled</span>"
        f"<span class='pill {'warn' if flagged else 'ok'}'>{flagged} need review</span>"
        f"<span class='pill'>{result.seconds:.1f}s</span>", unsafe_allow_html=True)


def _evidence_table(result: PipelineResult) -> None:
    rows = [{"Field": LABELS[k], "Value": (result.extraction.values[k] or "")[:80],
             "Screenshot": ev.source_index, "Confidence": ev.confidence, "OCR score": round(ev.ocr_score, 2),
             "Source text": ev.text[:100]}
            for k, ev in result.extraction.evidence.items() if result.extraction.values.get(k)]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.caption("No field could be matched to the OCR text.")


def render(repo: HistoryRepository) -> None:
    result: PipelineResult | None = st.session_state.get("result")
    if result is None:
        return
    run_id = st.session_state["run_id"]
    saved_id = st.session_state.get("saved_id")

    st.subheader("Review and edit candidate details")
    _summary_pills(result, st.session_state.get("screenshots", 0))
    st.caption("Nothing is guessed: fields not clearly present in the screenshots stay blank. "
               "⚠️ marks a field to verify — hover the field's ? for the reason and the source text.")
    if st.session_state.get("instruction"):
        st.info("Your note: " + st.session_state["instruction"])
    for w in result.warnings:
        st.warning(w)

    with st.form(f"review_form_{run_id}"):
        edited = render_fields(result.extraction.values, f"draft{run_id}", result.report, result.extraction)
        label = "Update saved entry" if saved_id else "Save to today's history"
        submitted = st.form_submit_button(label, type="primary")
    if submitted:
        report = validate(edited, result.extraction)
        new_id = save_candidate(repo, edited, result.raw_text, st.session_state.get("screenshots", 0),
                                evidence=result.extraction.evidence_dict(), report=report,
                                instruction=st.session_state.get("instruction", ""), existing_id=saved_id)
        st.session_state["saved_id"] = new_id
        st.session_state["save_notice"] = (new_id, sorted(report.flagged))
        st.rerun()
    notice = st.session_state.pop("save_notice", None)
    if notice:
        st.success(f"Saved as history entry #{notice[0]}.")
        if notice[1]:
            st.caption("Saved with fields still marked for review: " + ", ".join(LABELS[k] for k in notice[1]))

    with st.expander("Where each value came from"):
        _evidence_table(result)
    with st.expander("Timing per screenshot"):
        for name, secs in result.image_seconds:
            st.text(f"{name}: {secs:.2f}s" + ("  (cached)" if secs < 0.05 else ""))
    with st.expander("OCR text as tables (troubleshooting)"):
        t_kv, t_education, t_experience, t_sections, t_lines, t_raw = st.tabs(
            ["Key / value", "Education", "Professional experience", "By section", "Content lines", "Plain text"])
        with t_kv:
            rows = [{"Key": LABELS[k], "Value": result.extraction.values.get(k, "")}
                    for k in FIELD_KEYS if k != "qualification_records"]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"Key": st.column_config.TextColumn(width="medium"),
                                        "Value": st.column_config.TextColumn(width="large")})
        with t_education:
            rows = education_table(result.extraction.values.get("qualification_records", ""))
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            else:
                st.caption("No education records were identified in the resume.")
        with t_experience:
            jobs = split_jobs(result.extraction.values.get("employment_details", ""))
            if jobs:
                rows = jobs_table(result.extraction.values.get("employment_details", ""))
                rows = [{"Experience": row.get("Job", i), **{k: v for k, v in row.items() if k != "Job"}}
                        for i, row in enumerate(rows, start=1)]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            else:
                st.caption("No professional experience text was identified. This is not an error for freshers.")
        with t_sections:
            rows = sections_table(result.lines)
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                    "Content": st.column_config.TextColumn(width="large")})
            else:
                st.caption("No text found.")
        with t_lines:
            rows = lines_table(result.lines)
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, hide_index=True, width="stretch", column_config={
                    "Content": st.column_config.TextColumn(width="large")})
                st.download_button("Download content lines as CSV", df.to_csv(index=False).encode("utf-8"),
                                   "ocr_content_lines.csv", "text/csv")
        with t_raw:
            # Real line breaks (the old code joined with a literal backslash-n), grouped by section.
            st.code(plain_text_sections(result.lines, result.extraction.values) or "No text found.", language=None, wrap_lines=True)
