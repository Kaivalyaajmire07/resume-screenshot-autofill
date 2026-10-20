"""Candidate form rendering shared by the review step and the history editor."""
from __future__ import annotations

import streamlit as st
import json
import re

from ..extraction import ExtractionResult
from ..fields import GROUPS, LABELS, LONG_TEXT_FIELDS, TALL_FIELDS, normalize_fields
import pandas as pd

from ..jobs import join_jobs, split_jobs
from ..structure import jobs_table

MAX_JOB_BOXES = 10
from ..validation import ValidationReport



EDUCATION_COLUMNS = [
    ("qualification", "Qualification"),
    ("degree", "Degree / Stream"),
    ("institution", "Institution / College"),
    ("board_awarding_body", "Board / Awarding Body"),
    ("start_year", "Start Year"),
    ("passing_year", "Passing Year"),
    ("expected_completion_year", "Expected Year"),
    ("status", "Status"),
    ("marks_obtained", "Marks Obtained"),
    ("maximum_marks", "Maximum Marks"),
    ("percentage", "Percentage"),
    ("cgpa", "CGPA"),
    ("grade", "Grade"),
    ("result", "Result"),
]
EDUCATION_TEMPLATES = [
    "SSC / Class 10", "HSC / Class 12", "Diploma", "Graduation",
    "Post-graduation", "Higher education",
]


def _qualification_rows(values: dict[str, str]) -> list[dict[str, str]]:
    """Load structured rows, or build a backward-compatible view from older flat fields."""
    raw = values.get("qualification_records", "")
    try:
        parsed = json.loads(raw) if raw else []
    except (TypeError, json.JSONDecodeError):
        parsed = []
    by_name = {str(row.get("qualification", "")): row for row in parsed if isinstance(row, dict)}
    rows = []
    legacy = {
        "SSC / Class 10": ("ssc_result", "ssc_board", "ssc_pass_year", "", ""),
        "HSC / Class 12": ("hsc_result", "hsc_board", "hsc_pass_year", "", ""),
        "Graduation": ("graduation_result", "graduation_university", "graduation_year", "graduation_degree", "graduation_university"),
        "Post-graduation": ("post_graduation_result", "post_graduation_university", "post_graduation_year",
                            "post_graduation_degree", "post_graduation_university"),
    }
    for name in EDUCATION_TEMPLATES:
        row = {k: "" for k, _ in EDUCATION_COLUMNS}
        row["qualification"] = name
        row.update({k: str(v or "") for k, v in by_name.get(name, {}).items() if k in row})
        if name in legacy:
            result_key, board_key, year_key, degree_key, institution_key = legacy[name]
            row["result"] = row["result"] or values.get(result_key, "")
            row["board_awarding_body"] = row["board_awarding_body"] or values.get(board_key, "")
            row["passing_year"] = row["passing_year"] or values.get(year_key, "")
            if degree_key:
                row["degree"] = row["degree"] or values.get(degree_key, "")
            if institution_key:
                row["institution"] = row["institution"] or values.get(institution_key, "")
            result = row["result"]
            if result and "%" in result and not row["percentage"]:
                row["percentage"] = result
            elif result and re.match(r"(?i)^(?:CGPA|GPA|CPI|SGPA)", result) and not row["cgpa"]:
                row["cgpa"] = result
        rows.append(row)
    # Preserve any additional qualifications that are present in stored data.
    for row in parsed:
        if isinstance(row, dict) and row.get("qualification") and row["qualification"] not in EDUCATION_TEMPLATES:
            normalized = {k: str(row.get(k, "") or "") for k, _ in EDUCATION_COLUMNS}
            rows.append(normalized)
    return rows


def _render_qualification_table(values: dict[str, str], key_prefix: str) -> tuple[str, dict[str, str]]:
    st.caption("One row per qualification. Empty cells mean the resume did not provide that value.")
    rows = _qualification_rows(values)
    df = pd.DataFrame(rows, columns=[k for k, _ in EDUCATION_COLUMNS])
    df = df.rename(columns=dict(EDUCATION_COLUMNS))
    edited_df = st.data_editor(
        df,
        key=f"{key_prefix}_qualification_records_editor",
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        disabled=["Qualification"],
        column_config={
            "Qualification": st.column_config.TextColumn("Qualification", width="medium"),
            "Degree / Stream": st.column_config.TextColumn("Degree / Stream", width="medium"),
            "Institution / College": st.column_config.TextColumn("Institution / College", width="large"),
            "Board / Awarding Body": st.column_config.TextColumn("Board / Awarding Body", width="medium"),
            "Start Year": st.column_config.TextColumn("Start Year", width="small"),
            "Passing Year": st.column_config.TextColumn("Passing Year", width="small"),
            "Expected Year": st.column_config.TextColumn("Expected Year", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Marks Obtained": st.column_config.TextColumn("Marks Obtained", width="small"),
            "Maximum Marks": st.column_config.TextColumn("Maximum Marks", width="small"),
            "Percentage": st.column_config.TextColumn("Percentage", width="small"),
            "CGPA": st.column_config.TextColumn("CGPA", width="small"),
            "Grade": st.column_config.TextColumn("Grade", width="small"),
            "Result": st.column_config.TextColumn("Result", width="small"),
        },
    )
    # Convert visible labels back to stable internal keys.
    inverse = {label: key for key, label in EDUCATION_COLUMNS}
    normalized_rows = []
    for record in edited_df.to_dict(orient="records"):
        normalized_rows.append({inverse[k]: ("" if pd.isna(v) else str(v).strip())
                                for k, v in record.items() if k in inverse})
    legacy_values = {}
    by_name = {r.get("qualification", ""): r for r in normalized_rows}
    for name, prefix in (("SSC / Class 10", "ssc"), ("HSC / Class 12", "hsc")):
        row = by_name.get(name, {})
        legacy_values[f"{prefix}_result"] = row.get("percentage") or row.get("cgpa") or row.get("result") or (
            f"{row.get('marks_obtained')}/{row.get('maximum_marks')}" if row.get("marks_obtained") and row.get("maximum_marks") else "")
        legacy_values[f"{prefix}_board"] = row.get("board_awarding_body", "")
        legacy_values[f"{prefix}_pass_year"] = row.get("passing_year") or row.get("expected_completion_year", "")
    for name, prefix in (("Graduation", "graduation"), ("Post-graduation", "post_graduation")):
        row = by_name.get(name, {})
        legacy_values[f"{prefix}_degree"] = row.get("degree", "")
        legacy_values[f"{prefix}_result"] = row.get("percentage") or row.get("cgpa") or row.get("result") or (
            f"{row.get('marks_obtained')}/{row.get('maximum_marks')}" if row.get("marks_obtained") and row.get("maximum_marks") else "")
        legacy_values[f"{prefix}_university"] = row.get("institution", "")
        legacy_values[f"{prefix}_year"] = row.get("passing_year") or row.get("expected_completion_year", "")
    higher = by_name.get("Higher education", {})
    diploma = by_name.get("Diploma", {})
    legacy_values["higher_level_education"] = higher.get("degree") or higher.get("institution") or diploma.get("degree") or diploma.get("institution") or ""
    return json.dumps(normalized_rows, ensure_ascii=False), legacy_values


def _help(key: str, report: ValidationReport | None, extraction: ExtractionResult | None) -> str | None:
    lines = []
    if report and key in report.issues:
        lines += [f"• {m}" for m in report.issues[key]]
    ev = extraction.evidence.get(key) if extraction else None
    if ev and ev.source_index:
        lines.append(f"Source: screenshot {ev.source_index}, “{ev.text[:120]}” ({ev.confidence} confidence)")
    return "\n".join(lines) or None


def _render_jobs(value: str, key_prefix: str, label: str, help_text: str | None) -> str:
    """One text box per job, plus one spare box to add another. Returns the jobs joined for storage."""
    jobs = split_jobs(value)
    boxes = min(max(len(jobs) + 1, 1), max(MAX_JOB_BOXES, len(jobs)))
    st.markdown(f"**{label}**  ·  {len(jobs)} job(s) found")
    if len(jobs) > 1:  # quick overview; the boxes below are what gets saved
        st.dataframe(pd.DataFrame(jobs_table(value)), hide_index=True, column_config={
            "Details": st.column_config.TextColumn(width="large")})
    texts = []
    for i in range(boxes):
        found = i < len(jobs)
        texts.append(st.text_area(
            f"Job {i + 1}" if found else f"Job {i + 1} (optional, add another)",
            value=jobs[i] if found else "", height=130, key=f"{key_prefix}_employment_details_{i}",
            help=help_text if i == 0 else None,
            placeholder="" if found else "Company, role, dates, what you did"))
    return join_jobs(texts)


def render_fields(values: dict[str, str], key_prefix: str, report: ValidationReport | None = None,
                  extraction: ExtractionResult | None = None) -> dict[str, str]:
    """Render all groups; call inside an ``st.form``. Returns the edited values."""
    values = normalize_fields(values)
    # Start with the complete value set so fields hidden in the structured UI remain
    # backward-compatible in saved history.
    edited: dict[str, str] = normalize_fields(values)
    flagged = report.flagged if report else set()
    for group, keys in GROUPS.items():
        st.markdown(f"#### {group}")
        if group == "Qualification Details":
            serialized, legacy_values = _render_qualification_table(values, key_prefix)
            edited["qualification_records"] = serialized
            edited.update(legacy_values)
            continue
        cols = st.columns(2)
        for i, key in enumerate(keys):
            label = ("⚠️ " if key in flagged else "") + LABELS[key]
            kwargs = dict(value=values[key], key=f"{key_prefix}_{key}", help=_help(key, report, extraction),
                          placeholder="" if values[key] else "Not found — left blank")
            if key == "employment_details":
                edited[key] = _render_jobs(values[key], key_prefix, label, kwargs["help"])
            elif key in LONG_TEXT_FIELDS:
                edited[key] = st.text_area(label, height=110 if key in TALL_FIELDS else 72, **kwargs)
            else:
                edited[key] = cols[i % 2].text_input(label, **kwargs)
    return edited
