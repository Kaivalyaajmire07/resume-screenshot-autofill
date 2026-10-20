"""Split employment text into one entry per job (and join the entries back for storage).

The database keeps a single ``employment_details`` string, so existing records keep working.
Jobs are separated inside it by ``SEP``. Old records without that separator are split on the
fly by looking for date ranges such as ``Jan 2020 - Mar 2022`` or ``2019 – Present``.
"""
from __future__ import annotations

import re

SEP = "\n\n---\n\n"
_SEP_RE = re.compile(r"\n\s*-{3,}\s*\n")

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_POINT = rf"(?:(?:{_MONTH}[\s,'’]*|\d{{1,2}}[/.\-])?(?:19|20)\d{{2}})"
DURATION_RE = re.compile(
    rf"{_POINT}\s*(?:-|–|—|to|till|until)\s*(?:{_POINT}|present|current|till\s*date|to\s*date|now|ongoing)",
    re.I)


# "Worked as a Telecaller", "Role: Sales Executive", "Designation - Team Lead" ...
ROLE_RE = re.compile(r"^\W*(?:(?:worked|working|work)\s+(?:as|at)\b|(?:role|designation|position|job\s*title|profile)\s*[:\-–])", re.I)
# A line that is only a length of service: "1.5 Year", "2Year", "6 months", "3+ yrs"
SHORT_DURATION_RE = re.compile(r"^\W*(?:\(?\s*)?\d{1,2}(?:\.\d+)?\s*\+?\s*(?:years?|yrs?|months?|mos?)\b\W*$", re.I)


def _is_duration(line: str) -> bool:
    return bool(SHORT_DURATION_RE.match(line)) or (bool(DURATION_RE.search(line)) and len(line) < 40)


_LABELLED = re.compile(r"^\W*(?:technolog\w*|tech\w*\s*stack|tools|environment|skills?|responsibilit\w+|achievements?|"
                       r"description|location)\s*:", re.I)
_SMALL = {"of", "and", "at", "in", "for", "the", "&", "to", "a", "an", "as", "on", "pvt", "ltd", "by", "with", "|", "-", "–"}


def _title_like(line: str) -> bool:
    """Short, capitalised, non-sentence line: plausibly a job title or company name."""
    if not (0 < len(line) <= 60 and len(line.split()) <= 7 and line[0].isupper()
            and not line.endswith((".", ";", ",")) and not DURATION_RE.search(line)) or _LABELLED.match(line):
        return False
    if ROLE_RE.match(line):
        return True
    # a bullet such as "Handled customer calls" is not a heading: its words are not capitalised
    return not any(w[0].islower() and w.lower() not in _SMALL for w in line.split() if w[0].isalpha())


def _bounds_to_groups(lines: list[str], starts: list[int]) -> list[list[str]]:
    starts = sorted(set(starts))
    starts[0] = 0  # anything before the first entry belongs to the first job
    bounds = starts + [len(lines)]
    return [lines[bounds[i]:bounds[i + 1]] for i in range(len(starts))]


def group_lines(lines: list[str]) -> list[list[str]]:
    """Group OCR/text lines into jobs.

    Tried in order, first one that finds 2+ jobs wins:
    1. date ranges      ``Jan 2020 - Mar 2022``
    2. role lines       ``Worked as a Telecaller`` (the company line just above starts the job)
    3. lone durations   ``1.5 Year`` (each one ends a job)
    One group when none of them applies.
    """
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        return []

    anchors = [i for i, ln in enumerate(lines) if DURATION_RE.search(ln)]
    if len(anchors) >= 2:
        starts, prev = [], -1
        for k in anchors:
            s = k
            while s - 1 > prev and k - s < 2 and _title_like(lines[s - 1]):
                s -= 1  # pull the role / company line(s) printed just above the dates
            starts.append(s)
            prev = k
        return _bounds_to_groups(lines, starts)

    roles = [i for i, ln in enumerate(lines) if ROLE_RE.match(ln) and len(ln) <= 80]
    if len(roles) >= 2:
        starts = []
        for i in roles:
            above = lines[i - 1] if i > 0 else ""
            starts.append(i - 1 if (above and (i - 1) not in roles and not _is_duration(above)
                                    and not above.endswith((".", ";", ",")) and len(above) <= 80) else i)
        return _bounds_to_groups(lines, starts)

    ends = [i for i, ln in enumerate(lines) if SHORT_DURATION_RE.match(ln)]
    if len(ends) >= 2:
        starts = [0] + [i + 1 for i in ends if i + 1 < len(lines)]
        return _bounds_to_groups(lines, starts)
    return [lines]


def split_jobs(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SEP_RE.split(text) if p.strip()]
    if len(parts) > 1:
        return parts
    return ["\n".join(g) for g in group_lines(text.splitlines())]


def join_jobs(jobs: list[str]) -> str:
    return SEP.join(j.strip() for j in jobs if j and j.strip())


JOB_COLUMNS = ("Company", "Role", "Duration", "Details")


def parse_job(text: str) -> dict[str, str]:
    """Break one job's text into Company / Role / Duration / Details (for the table view).

    Only copies text that is already there: a part that cannot be recognised stays blank.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    company = role = duration = ""
    rest: list[str] = []
    for ln in lines:
        if not role and ROLE_RE.match(ln):
            role = re.sub(r"^\W*(?:worked|working|work)\s+(?:as|at)\s+(?:an?\s+|the\s+)?", "", ln, flags=re.I).strip()
            role = re.sub(r"^\W*(?:role|designation|position|job\s*title|profile)\s*[:\-–]\s*", "", role, flags=re.I) or ln
        elif not duration and _is_duration(ln):
            duration = ln.strip("() ")
        elif not duration and DURATION_RE.search(ln):
            m = DURATION_RE.search(ln)
            duration = m.group(0)
            leftover = (ln[:m.start()] + " " + ln[m.end():]).strip(" ,;|-–()")
            if leftover:
                if company:
                    rest.append(leftover)
                else:
                    company = leftover
        elif not company:
            company = ln
        else:
            rest.append(ln)
    return {"Company": company, "Role": role, "Duration": duration, "Details": " ".join(rest)}
