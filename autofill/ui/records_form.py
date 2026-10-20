"""Editable cards for education and experience records. Every column is always shown, even when blank."""
from __future__ import annotations

import streamlit as st

from ..records import (EDUCATION_KEY, EMPLOYMENT_TYPES, EXPERIENCE_KEY, LEVEL_LABELS, SCHOOL_LEVELS, STATUSES,
                       EducationRecord, ExperienceRecord, dump_records)
from ..validation import ValidationReport

MAX_EXPERIENCE_CARDS = 12
MAX_EXTRA_QUALIFICATIONS = 8
DEGREE_LEVELS = ["diploma", "graduation", "post_graduation", "higher"]

SCHOOL_COLUMNS = [("institution", "School name"), ("board_or_body", "Board"), ("specialization", "Stream (if stated)"),
                  ("passing_year", "Passing year"), ("expected_year", "Expected year"), ("status", "Status"),
                  ("marks_obtained", "Marks obtained"), ("max_marks", "Maximum marks"), ("percentage", "Percentage"),
                  ("cgpa", "CGPA"), ("grade", "Grade"), ("result", "Result")]
DEGREE_COLUMNS = [("degree", "Degree / diploma name"), ("institution", "College / institution"),
                  ("board_or_body", "University / awarding body"), ("specialization", "Specialization"),
                  ("start_year", "Start year"), ("passing_year", "Passing year"), ("expected_year", "Expected year"),
                  ("status", "Status"), ("current_year", "Current year / semester"),
                  ("marks_obtained", "Marks obtained"), ("max_marks", "Maximum marks"), ("percentage", "Percentage"),
                  ("cgpa", "CGPA"), ("grade", "Grade"), ("result", "Result")]


def _issues(report: ValidationReport | None, key: str) -> list[str]:
    return list(report.record_issues.get(key, [])) if report else []


def _show_issues(issues: list[str], rec) -> None:
    for msg in issues:
        st.warning(msg, icon="⚠️")
    if rec is not None and rec.source_text:
        with st.popover("Source text", help="The text this record was read from (diagnostics)"):
            st.caption(rec.source_text)


def _education_card(rec: EducationRecord, slot: str, prefix: str, columns, issues: list[str],
                    level_choice: bool = False) -> EducationRecord:
    out = EducationRecord(**{k: getattr(rec, k) for k in rec.__dict__})
    if level_choice:
        levels = DEGREE_LEVELS
        out.level = st.selectbox("Qualification type", levels, index=levels.index(rec.level) if rec.level in levels else 1,
                                 format_func=LEVEL_LABELS.get, key=f"{prefix}_edu_{slot}_level")
    cols = st.columns(3)
    for i, (attr, label) in enumerate(columns):
        key, current = f"{prefix}_edu_{slot}_{attr}", getattr(rec, attr)
        col = cols[i % 3]
        if attr == "status":
            value = col.selectbox(label, STATUSES, index=STATUSES.index(current) if current in STATUSES else 0, key=key)
        else:
            value = col.text_input(label, value=current, key=key)
        setattr(out, attr, value)
    _show_issues(issues, rec)
    return out


def render_education(records: list[EducationRecord], prefix: str, report: ValidationReport | None) -> list[EducationRecord]:
    """SSC and HSC cards are always shown; each other qualification gets its own card, plus spare cards."""
    st.markdown("#### Qualification Details")
    st.caption("Each qualification is a separate record. Blank boxes mean the value was not found in the document.")
    result: list[EducationRecord] = []
    index_of = {id(r): i for i, r in enumerate(records)}
    for lvl in SCHOOL_LEVELS:
        rec = next((r for r in records if r.level == lvl), EducationRecord(level=lvl))
        with st.container(border=True):
            st.markdown(f"**{LEVEL_LABELS[lvl]}**")
            cols = SCHOOL_COLUMNS if lvl == "hsc" else [c for c in SCHOOL_COLUMNS if c[0] != "specialization"]
            result.append(_education_card(rec, lvl, prefix, cols, _issues(report, f"education:{index_of.get(id(rec), -1)}")))
    others = [r for r in records if r.level not in SCHOOL_LEVELS]
    n_cards = min(len(others) + 1, MAX_EXTRA_QUALIFICATIONS)
    for i in range(n_cards):
        rec = others[i] if i < len(others) else EducationRecord(level="graduation")
        title = LEVEL_LABELS.get(rec.level, "Qualification") if i < len(others) else "Add another qualification (optional)"
        with st.container(border=True):
            st.markdown(f"**{title}**")
            out = _education_card(rec, f"x{i}", prefix, DEGREE_COLUMNS,
                                  _issues(report, f"education:{index_of.get(id(rec), -1)}"), level_choice=True)
            result.append(out)
    return [r for r in result if not r.is_blank()]


def _experience_card(rec: ExperienceRecord, i: int, prefix: str, issues: list[str]) -> ExperienceRecord:
    out = ExperienceRecord(**{k: getattr(rec, k) for k in rec.__dict__})
    k = lambda name: f"{prefix}_exp{i}_{name}"  # noqa: E731
    c1, c2, c3 = st.columns(3)
    out.company = c1.text_input("Employer / company", value=rec.company, key=k("company"))
    out.title = c2.text_input("Job title / designation", value=rec.title, key=k("title"))
    types = EMPLOYMENT_TYPES if rec.employment_type in EMPLOYMENT_TYPES else EMPLOYMENT_TYPES + [rec.employment_type]
    out.employment_type = c3.selectbox("Employment type", types, index=types.index(rec.employment_type), key=k("type"))
    c4, c5, c6, c7 = st.columns(4)
    out.location = c4.text_input("Work location", value=rec.location, key=k("location"))
    out.start_date = c5.text_input("Start date", value=rec.start_date, key=k("start"))
    out.end_date = c6.text_input("End date", value=rec.end_date, key=k("end"))
    out.duration = c7.text_input("Duration", value=rec.duration, key=k("duration"))
    out.current = st.checkbox("Currently working here", value=rec.current, key=k("current"))
    out.responsibilities = st.text_area("Responsibilities", value=rec.responsibilities, height=90, key=k("resp"))
    c8, c9 = st.columns(2)
    out.achievements = c8.text_area("Achievements", value=rec.achievements, height=70, key=k("ach"))
    out.technologies = c9.text_area("Technologies / skills used", value=rec.technologies, height=70, key=k("tech"))
    _show_issues(issues, rec)
    return out


def render_experience(records: list[ExperienceRecord], prefix: str, report: ValidationReport | None,
                      notes: list[str] | None = None) -> list[ExperienceRecord]:
    """One card per job (Experience 1, 2, ... is only a label) plus one spare card to add another."""
    st.markdown("#### Professional Experience")
    if not records:
        st.info("No professional experience found. Leave blank for a fresher, or fill the card below.")
    else:
        st.caption(f"{len(records)} professional experience record(s) found. Each job is a separate record.")
    result: list[ExperienceRecord] = []
    for i in range(min(len(records) + 1, MAX_EXPERIENCE_CARDS)):
        rec = records[i] if i < len(records) else ExperienceRecord()
        label = f"Experience {i + 1}" + ("" if i < len(records) else " (optional — add another)")
        with st.expander(label, expanded=i < len(records) or not records):
            result.append(_experience_card(rec, i, prefix, _issues(report, f"experience:{i}")))
    return [r for r in result if not r.is_blank()]


def to_fields(education: list[EducationRecord], experience: list[ExperienceRecord]) -> dict[str, str]:
    return {EDUCATION_KEY: dump_records(education), EXPERIENCE_KEY: dump_records(experience)}
