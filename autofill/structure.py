"""Turn flat OCR lines into tables: by section (Contact, Experience, Education, ...) and line by line."""
from __future__ import annotations

from .extraction import _clean, _is_heading
from .jobs import JOB_COLUMNS, parse_job, split_jobs
from .ocr_engine import OcrLine

TOP = "Header / Contact"


def sections_table(lines: list[OcrLine]) -> list[dict]:
    """One row per resume section: Section | Screenshot | Content. Text before the first heading is the header."""
    rows: list[dict] = []
    title, shots, body = TOP, [], []

    def flush() -> None:
        if body:
            rows.append({"Section": title, "Screenshot": ", ".join(str(s) for s in sorted(set(shots)) if s) or "",
                         "Content": "\n".join(body)})

    for ln in lines:
        text = _clean(ln.text)
        if not text:
            continue
        if _is_heading(text):
            flush()
            title, shots, body = text.title(), [], []
        else:
            body.append(text)
            shots.append(ln.source_index)
    flush()
    return rows


def lines_table(lines: list[OcrLine]) -> list[dict]:
    """One row per OCR line: # | Screenshot | (Page) | Text | Confidence (%). Diagnostics only."""
    has_pages = any(getattr(ln, "page", 0) for ln in lines)
    rows = []
    for i, ln in enumerate(lines, start=1):
        row = {"#": i, "Screenshot": ln.source_index}
        if has_pages:
            row["Page"] = getattr(ln, "page", 0) or ""
        row.update({"Text": ln.text, "Confidence %": round(ln.score * 100)})
        rows.append(row)
    return rows


EDUCATION_COLUMNS = ["Qualification", "Degree", "Institution", "Board / university", "Stream / specialization",
                     "Start year", "Passing year", "Expected year", "Status", "Current year", "Marks", "Max marks",
                     "Percentage", "CGPA", "Grade", "Result"]
EXPERIENCE_COLUMNS = ["Experience", "Company", "Title", "Type", "Location", "Start", "End", "Current", "Duration",
                      "Responsibilities", "Achievements", "Technologies"]


def education_table(records) -> list[dict]:
    """One row per qualification with ALL columns present (blank when not found)."""
    return [{"Qualification": r.label, "Degree": r.degree, "Institution": r.institution,
             "Board / university": r.board_or_body, "Stream / specialization": r.specialization,
             "Start year": r.start_year, "Passing year": r.passing_year, "Expected year": r.expected_year,
             "Status": r.status, "Current year": r.current_year, "Marks": r.marks_obtained,
             "Max marks": r.max_marks, "Percentage": r.percentage, "CGPA": r.cgpa, "Grade": r.grade,
             "Result": r.result} for r in records]


def experience_table(records) -> list[dict]:
    return [{"Experience": i, "Company": r.company, "Title": r.title, "Type": r.employment_type,
             "Location": r.location, "Start": r.start_date, "End": r.end_date, "Current": "Yes" if r.current else "",
             "Duration": r.duration, "Responsibilities": r.responsibilities, "Achievements": r.achievements,
             "Technologies": r.technologies} for i, r in enumerate(records, start=1)]


def jobs_table(employment_text: str) -> list[dict]:
    """One row per job: Job | Company | Role | Duration | Details."""
    return [{"Job": i, **{c: parse_job(j)[c] for c in JOB_COLUMNS}} for i, j in enumerate(split_jobs(employment_text), start=1)]
