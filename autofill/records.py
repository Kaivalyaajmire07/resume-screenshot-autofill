"""Structured candidate records: one EducationRecord per qualification, one ExperienceRecord per job.

Records are stored as JSON strings inside the existing ``fields_json`` column (keys
``education_records`` / ``experience_records``), so no schema migration is needed and old
history rows keep working. The flat legacy fields (ssc_result, graduation_degree,
employment_details, ...) are *derived* from the records by :func:`sync_flat`, which keeps the
history table, search and older records consistent.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields as dc_fields
from typing import Any

from .jobs import SEP, join_jobs

EDUCATION_KEY = "education_records"
EXPERIENCE_KEY = "experience_records"

LEVELS = ["ssc", "hsc", "diploma", "graduation", "post_graduation", "higher"]
LEVEL_LABELS = {"ssc": "SSC / Class 10", "hsc": "HSC / Class 12", "diploma": "Diploma",
                "graduation": "Graduation", "post_graduation": "Post-graduation",
                "higher": "Higher / other qualification"}
SCHOOL_LEVELS = ("ssc", "hsc")
STATUSES = ["Not Available", "Completed", "Pursuing", "Appearing"]
EMPLOYMENT_TYPES = ["", "Full-time", "Part-time", "Internship", "Contract", "Freelance", "Apprenticeship"]


@dataclass
class EducationRecord:
    level: str = "graduation"
    degree: str = ""              # degree / diploma name (blank for Class 10 / 12)
    institution: str = ""         # school / college name
    board_or_body: str = ""       # school board, or university / awarding body
    specialization: str = ""      # stream (Class 12) or specialization (degrees)
    start_year: str = ""
    passing_year: str = ""        # only when completed
    expected_year: str = ""       # only when pursuing / appearing
    status: str = "Not Available"
    current_year: str = ""        # "Final year", "3rd year", "Semester 6" - only if written
    marks_obtained: str = ""
    max_marks: str = ""
    percentage: str = ""
    cgpa: str = ""
    grade: str = ""
    result: str = ""
    # diagnostics (never shown as candidate content)
    source_index: int = 0
    source_page: int = 0
    source_text: str = ""
    confidence: str = "medium"
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return LEVEL_LABELS.get(self.level, self.level)

    def is_blank(self) -> bool:
        skip = {"level", "status", "source_index", "source_page", "source_text", "confidence", "notes"}
        return not any(getattr(self, f.name) for f in dc_fields(self) if f.name not in skip)


@dataclass
class ExperienceRecord:
    company: str = ""
    title: str = ""
    employment_type: str = ""
    location: str = ""
    start_date: str = ""
    end_date: str = ""
    current: bool = False
    duration: str = ""
    responsibilities: str = ""
    achievements: str = ""
    technologies: str = ""
    source_index: int = 0
    source_page: int = 0
    source_text: str = ""
    confidence: str = "medium"
    notes: list[str] = field(default_factory=list)

    def is_blank(self) -> bool:
        skip = {"source_index", "source_page", "source_text", "confidence", "notes", "current"}
        return not any(getattr(self, f.name) for f in dc_fields(self) if f.name not in skip)

    def as_text(self) -> str:
        """Plain-text form used for the legacy ``employment_details`` field."""
        dates = " – ".join(p for p in (self.start_date, "Present" if self.current and not self.end_date
                                       else self.end_date) if p)
        head = [x for x in (self.title, self.company, self.employment_type, self.location) if x]
        parts = [", ".join(head)] if head else []
        if dates or self.duration:
            parts.append(" | ".join(x for x in (dates, self.duration) if x))
        for text in (self.responsibilities, self.achievements):
            if text:
                parts.append(text)
        if self.technologies:
            parts.append("Technologies: " + self.technologies)
        return "\n".join(parts)


# ------------------------------------------------------------------ (de)serialisation
def _load(raw: Any, cls):
    if not raw:
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    names = {f.name for f in dc_fields(cls)}
    out = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict):
            rec = cls(**{k: v for k, v in item.items() if k in names})
            if isinstance(rec.notes, str):
                rec.notes = [rec.notes]
            out.append(rec)
    return out


def dump_records(records: list) -> str:
    return json.dumps([asdict(r) for r in records], ensure_ascii=False)


def education_from_fields(values: dict) -> list[EducationRecord]:
    return _load(values.get(EDUCATION_KEY), EducationRecord)


def experience_from_fields(values: dict) -> list[ExperienceRecord]:
    return _load(values.get(EXPERIENCE_KEY), ExperienceRecord)


# ------------------------------------------------------------------ legacy flat fields
def _result_text(r: EducationRecord) -> str:
    if r.percentage:
        return r.percentage if "%" in r.percentage else f"{r.percentage}%"
    if r.cgpa:
        return r.cgpa if "cgpa" in r.cgpa.lower() else f"{r.cgpa} CGPA"
    if r.marks_obtained:
        return f"{r.marks_obtained}/{r.max_marks}" if r.max_marks else r.marks_obtained
    return r.grade


def _year_text(r: EducationRecord) -> str:
    if r.passing_year:
        return r.passing_year
    return f"Expected {r.expected_year}" if r.expected_year else ""


def _degree_text(r: EducationRecord) -> str:
    return r.degree + (f" in {r.specialization}" if r.degree and r.specialization else "")


def education_to_flat(records: list[EducationRecord]) -> dict[str, str]:
    """Derive the legacy flat qualification fields from the records (first record per level)."""
    out: dict[str, str] = {}
    first = {lvl: next((r for r in records if r.level == lvl), None) for lvl in LEVELS}
    for lvl in SCHOOL_LEVELS:
        r = first[lvl]
        out[f"{lvl}_result"] = _result_text(r) if r else ""
        out[f"{lvl}_board"] = r.board_or_body if r else ""
        out[f"{lvl}_pass_year"] = r.passing_year if r else ""
    for lvl in ("graduation", "post_graduation"):
        r = first[lvl]
        out[f"{lvl}_degree"] = _degree_text(r) if r else ""
        out[f"{lvl}_result"] = _result_text(r) if r else ""
        out[f"{lvl}_university"] = (r.institution if r else "")
        out[f"{lvl}_year"] = _year_text(r) if r else ""
    other = [r for r in records if r.level in ("diploma", "higher")]
    out["higher_level_education"] = "\n".join(
        " – ".join(x for x in (f"{r.label}: {_degree_text(r)}".strip(": "), r.institution, _year_text(r)) if x)
        for r in other)
    return out


def sync_flat(values: dict) -> dict:
    """Return ``values`` with legacy flat fields re-derived from the structured records.

    Records are the source of truth: if records exist for a level, the flat fields for that level
    follow them. If there are no records at all (old history rows), flat fields are left untouched.
    """
    out = dict(values)
    edu = education_from_fields(out)
    edu = [r for r in edu if not r.is_blank()]
    if edu:
        out.update(education_to_flat(edu))
    exp = [r for r in experience_from_fields(out) if not r.is_blank()]
    if exp:
        out["employment_details"] = join_jobs([r.as_text() for r in exp])
    return out


def education_from_flat(values: dict) -> list[EducationRecord]:
    """Build records from the old flat fields so history rows saved before this version stay editable."""
    import re

    def split_result(text: str) -> dict:
        t = (text or "").strip()
        if not t:
            return {}
        if "%" in t:
            return {"percentage": t}
        num = re.search(r"\d+(?:\.\d+)?", t)
        if "cgpa" in t.lower() or (num and float(num.group(0)) <= 10):
            return {"cgpa": num.group(0) if num else t}
        return {"percentage": (num.group(0) + "%") if num else "", "grade": "" if num else t}

    def year_kwargs(text: str) -> dict:
        t = (text or "").strip()
        if t.lower().startswith("expected"):
            return {"expected_year": t.split(None, 1)[-1], "status": "Pursuing"}
        return {"passing_year": t, "status": "Completed" if t else "Not Available"}

    out: list[EducationRecord] = []
    for lvl in SCHOOL_LEVELS:
        res, board, yr = values.get(f"{lvl}_result", ""), values.get(f"{lvl}_board", ""), values.get(f"{lvl}_pass_year", "")
        if any((res, board, yr)):
            out.append(EducationRecord(level=lvl, board_or_body=board, **split_result(res), **year_kwargs(yr)))
    for lvl in ("graduation", "post_graduation"):
        deg, res, uni, yr = (values.get(f"{lvl}_{k}", "") for k in ("degree", "result", "university", "year"))
        if any((deg, res, uni, yr)):
            out.append(EducationRecord(level=lvl, degree=deg, institution=uni, **split_result(res), **year_kwargs(yr)))
    if (values.get("higher_level_education") or "").strip():
        out.append(EducationRecord(level="higher", degree=values["higher_level_education"].strip()))
    return out


def experience_from_text(text: str) -> list[ExperienceRecord]:
    """Parse the legacy employment_details text (one job per SEP-separated block)."""
    from .experience import parse_group
    from .jobs import split_jobs

    return [r for r in (parse_group([ln for ln in job.splitlines() if ln.strip()]) for job in split_jobs(text))
            if not r.is_blank()]


__all__ = ["education_from_flat", "experience_from_text", "EducationRecord", "ExperienceRecord", "EDUCATION_KEY", "EXPERIENCE_KEY", "LEVELS", "LEVEL_LABELS",
           "SCHOOL_LEVELS", "STATUSES", "EMPLOYMENT_TYPES", "dump_records", "education_from_fields",
           "experience_from_fields", "education_to_flat", "sync_flat", "SEP"]
