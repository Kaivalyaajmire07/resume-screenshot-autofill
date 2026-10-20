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
        if not body:
            return
        for row in rows:  # the same heading seen again (overlapping screenshots): merge, keep order
            if row["Section"] == title:
                row["Content"] += "\n" + "\n".join(body)
                return
        rows.append({"Section": title, "Content": "\n".join(body)})

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
    """One row per recognized content line; omit internal screenshot indexes from the UI."""
    return [{"Content": ln.text, "Confidence %": round(ln.score * 100)} for ln in lines]


def jobs_table(employment_text: str) -> list[dict]:
    """One row per job: Job | Company | Role | Duration | Details."""
    return [{"Job": i, **{c: parse_job(j)[c] for c in JOB_COLUMNS}} for i, j in enumerate(split_jobs(employment_text), start=1)]


EDUCATION_DISPLAY_COLUMNS = [
    ("qualification", "Qualification"), ("degree", "Degree / Stream"),
    ("institution", "Institution / College"), ("board_awarding_body", "Board / Awarding Body"),
    ("start_year", "Start Year"), ("passing_year", "Passing Year"),
    ("expected_completion_year", "Expected Year"), ("status", "Status"),
    ("marks_obtained", "Marks Obtained"), ("maximum_marks", "Maximum Marks"),
    ("percentage", "Percentage"), ("cgpa", "CGPA"), ("grade", "Grade"), ("result", "Result"),
]


def education_table(qualification_json: str) -> list[dict]:
    """Structured qualification rows with stable columns, no screenshot/file metadata."""
    import json
    try:
        rows = json.loads(qualification_json or "[]")
    except (TypeError, json.JSONDecodeError):
        rows = []
    if not isinstance(rows, list):
        return []
    output = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        output.append({label: str(row.get(key, "") or "") for key, label in EDUCATION_DISPLAY_COLUMNS})
    return output


def _format_jobs(content: str) -> str:
    out = []
    for n, job in enumerate(split_jobs(content), start=1):
        p = parse_job(job)
        out.append(f"{n}. {p['Company'] or '(company not read)'}")
        for label in ("Role", "Duration", "Details"):
            if p[label]:
                out.append(f"   {label}: {p[label]}")
    return "\n".join(out)


def _format_education(qualification_json: str) -> str:
    rows = education_table(qualification_json)
    out = []
    for row in reversed(rows):  # highest qualification first
        filled = {k: v for k, v in row.items() if v and k != "Qualification"}
        if not filled:
            continue
        out.append(f"{row['Qualification']}")
        for key in ("Degree / Stream", "Institution / College", "Board / Awarding Body", "Status",
                    "Start Year", "Passing Year", "Expected Year", "Result"):
            if row.get(key):
                out.append(f"   {key}: {row[key]}")
    return "\n".join(out)


def plain_text_sections(lines: list[OcrLine], values: dict | None = None) -> str:
    """Readable document: one titled block per section, jobs and qualifications laid out as records."""
    values = values or {}
    blocks = []
    for row in sections_table(lines):
        title, content = row["Section"], row["Content"]
        low = title.lower()
        if "experience" in low or "employment" in low or "internship" in low:
            content = _format_jobs(content) or content
        elif ("education" in low or "qualification" in low or "academic" in low) and values.get("qualification_records"):
            content = _format_education(values["qualification_records"]) or content
        elif "skill" in low:
            content = "\n".join("• " + ln.lstrip("•·-* ") for ln in content.splitlines())
        blocks.append(f"{title.upper()}\n{'-' * len(title)}\n{content}")
    return "\n\n".join(blocks)
