"""Candidate form rendering shared by the review step and the history editor."""
from __future__ import annotations

import streamlit as st

from ..extraction import ExtractionResult
from ..fields import GROUPS, LABELS, LONG_TEXT_FIELDS, TALL_FIELDS, normalize_fields
from ..records import (education_from_fields, education_from_flat, experience_from_fields,
                       experience_from_text, sync_flat)
from ..validation import ValidationReport
from .records_form import render_education, render_experience, to_fields


def _help(key: str, report: ValidationReport | None, extraction: ExtractionResult | None) -> str | None:
    lines = []
    if report and key in report.issues:
        lines += [f"• {m}" for m in report.issues[key]]
    ev = extraction.evidence.get(key) if extraction else None
    if ev and ev.text:
        lines.append(f"Source text: “{ev.text[:120]}” ({ev.confidence} confidence)")
    return "\n".join(lines) or None


def render_fields(values: dict[str, str], key_prefix: str, report: ValidationReport | None = None,
                  extraction: ExtractionResult | None = None) -> dict[str, str]:
    """Render all groups; call inside an ``st.form``. Returns the edited values (flat fields + record JSON)."""
    values = normalize_fields(values)
    edu = education_from_fields(values) or (list(extraction.education) if extraction else []) \
        or education_from_flat(values)
    exp = experience_from_fields(values) or (list(extraction.experience) if extraction else []) \
        or experience_from_text(values.get("employment_details", ""))
    edited: dict[str, str] = {}
    flagged = report.flagged if report else set()
    for group, keys in GROUPS.items():
        if group == "Qualification Details":
            edu_out = render_education(edu, key_prefix, report)
            continue
        st.markdown(f"#### {group}")
        cols = st.columns(2)
        for i, key in enumerate(keys):
            if key == "employment_details":
                continue  # replaced by the Professional Experience records below
            label = ("⚠️ " if key in flagged else "") + LABELS[key]
            kwargs = dict(value=values[key], key=f"{key_prefix}_{key}", help=_help(key, report, extraction),
                          placeholder="" if values[key] else "Not found — left blank")
            if key in LONG_TEXT_FIELDS:
                edited[key] = st.text_area(label, height=110 if key in TALL_FIELDS else 72, **kwargs)
            else:
                edited[key] = cols[i % 2].text_input(label, **kwargs)
        if group == "Employment & Profile":
            exp_out = render_experience(exp, key_prefix, report)
    edited.update(to_fields(edu_out, exp_out))
    edited = sync_flat(edited)
    # legacy flat qualification fields stay in the data model; shown read-only because the records drive them
    with st.expander("Summary fields derived from the records (read-only)"):
        for key in GROUPS["Qualification Details"] + ["employment_details"]:
            st.text_input(LABELS[key], value=edited.get(key, ""), disabled=True,
                          key=f"{key_prefix}_derived_{key}_{abs(hash(edited.get(key, ''))) % 99991}")
    return edited
