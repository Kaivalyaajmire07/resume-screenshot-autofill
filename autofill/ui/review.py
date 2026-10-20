"""Extraction results + editable review form + save."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..db import HistoryRepository
from ..fields import FIELD_KEYS, LABELS
from ..history import save_candidate
from ..pipeline import PipelineResult
from ..structure import (EDUCATION_COLUMNS, EXPERIENCE_COLUMNS, education_table, experience_table, lines_table,
                         sections_table)
from ..validation import validate
from .forms import render_fields


def _summary_pills(result: PipelineResult, screenshots: int) -> None:
    filled = sum(1 for k in FIELD_KEYS if result.extraction.values.get(k))
    flagged = len(result.report.flagged)
    st.markdown(
        f"<span class='pill ok'>{result.images_ok}/{screenshots} screenshots read</span>"
        f"<span class='pill'>{filled}/{len(FIELD_KEYS)} fields filled</span>"
        f"<span class='pill {'warn' if flagged else 'ok'}'>{flagged} need review</span>"
        f"<span class='pill'>{result.seconds:.1f}s</span>", unsafe_allow_html=True)


def _evidence_table(result: PipelineResult) -> None:
    rows = [{"Field": LABELS[k], "Value": (result.extraction.values[k] or "")[:80],
             "Source #": ev.source_index, "Confidence": ev.confidence, "OCR score": round(ev.ocr_score, 2),
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

    with st.expander("Structured result (diagnostics)"):
        t_kv, t_edu, t_exp, t_warn = st.tabs(["Fields", "Education records", "Experience records", "Warnings & conflicts"])
        with t_kv:
            kv = [{"Field": LABELS[k], "Value": v} for k, v in result.extraction.values.items() if v]
            st.dataframe(pd.DataFrame(kv or [{"Field": "", "Value": ""}]), hide_index=True)
        with t_edu:  # every column stays visible even when a cell is blank
            st.dataframe(pd.DataFrame(education_table(result.extraction.education), columns=EDUCATION_COLUMNS),
                         hide_index=True)
        with t_exp:
            if result.extraction.experience:
                st.dataframe(pd.DataFrame(experience_table(result.extraction.experience), columns=EXPERIENCE_COLUMNS),
                             hide_index=True)
            else:
                st.caption("No professional experience found.")
        with t_warn:
            for n in result.report.notes:
                st.info(n)
            for key, msgs in {**{LABELS.get(k, k): m for k, m in result.report.issues.items()},
                              **result.report.record_issues}.items():
                st.warning(f"{key}: " + "; ".join(msgs))
            if not (result.report.notes or result.report.issues or result.report.record_issues):
                st.caption("Nothing to review.")
    with st.expander("Where each value came from"):
        _evidence_table(result)
    with st.expander("Timing per screenshot"):
        for name, secs in result.image_seconds:
            st.text(f"{name}: {secs:.2f}s" + ("  (cached)" if secs < 0.05 else ""))
    try:
        from ..export_docx import build_docx
        st.download_button("⬇ Download as Word (.docx)",
                           build_docx(result.extraction.values, result.extraction.education,
                                      result.extraction.experience),
                           "resume.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           key=f"docx_{st.session_state.get('run_id', 0)}")
    except ImportError:
        st.caption("Install python-docx to enable Word export.")
    with st.expander("OCR text as tables (troubleshooting)"):
        t_sections, t_lines, t_raw = st.tabs(["Structured resume", "Line by line", "Plain text"])
        with t_sections:
            _structured_resume(result)
        with t_lines:
            rows = lines_table(result.lines)
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, hide_index=True, column_config={"Text": st.column_config.TextColumn(width="large")})
                st.download_button("Download as CSV", df.to_csv(index=False).encode("utf-8"), "ocr_lines.csv", "text/csv")
        with t_raw:
            st.text(result.raw_text)


def _structured_resume(result) -> None:
    """The OCR text arranged as a resume: contact, summary, experience, education, skills."""
    v = result.extraction.values
    contact = [(lbl, v.get(k, "")) for lbl, k in [("Name", None), ("Address", "address"), ("City", "city"),
               ("State", "state"), ("Pincode", "pincode"), ("Email", "email"), ("Mobile", "mobile")]
               if (lbl == "Name" or v.get(k))]
    name = " ".join(x for x in (v.get("first_name"), v.get("middle_name"), v.get("last_name")) if x)
    contact = [(a, name if a == "Name" else b) for a, b in contact]
    st.markdown("##### Contact")
    st.dataframe(pd.DataFrame(contact, columns=["Item", "Value"]), hide_index=True)
    for title, key in (("Objective", "objectives"), ("Summary", "summary")):
        if v.get(key):
            st.markdown(f"##### {title}")
            st.write(v[key])
    st.markdown("##### Professional experience")
    if result.extraction.experience:
        rows = [{"#": i, "Company": e.company, "Role": e.title, "Duration": e.duration or "",
                 "Start": e.start_date, "End": "Present" if e.current else e.end_date}
                for i, e in enumerate(result.extraction.experience, 1)]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.caption("No professional experience found.")
    st.markdown("##### Education")
    if result.extraction.education:
        rows = [{"Qualification": r.degree or r.level.upper().replace("_", " "), "Institution": r.institution,
                 "Board / university": r.board_or_body, "Status": r.status,
                 "Year": r.passing_year or (f"Expected {r.expected_year}" if r.expected_year else ""),
                 "Marks / %": r.percentage or r.cgpa or r.marks_obtained}
                for r in result.extraction.education]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.caption("No education found.")
    if v.get("key_skills"):
        st.markdown("##### Key skills")
        st.write("\n".join(f"- {x.lstrip('•-* ').strip()}" for x in v["key_skills"].splitlines() if x.strip()))
