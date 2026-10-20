"""Experience extraction: one ExperienceRecord per job / internship.

Job boundaries come from jobs.group_lines (date ranges, "Worked as ..." lines, lone durations).
Each job is then split into company, title, type, location, dates, duration, responsibilities,
achievements and technologies. Only text that is present is copied; a duration is calculated only
when both dates carry a month and a year. Projects are never classified as employment.
"""
from __future__ import annotations

import re
from datetime import date

from .jobs import DURATION_RE, ROLE_RE, SHORT_DURATION_RE, group_lines
from .ocr_engine import OcrLine
from .records import ExperienceRecord
from .sectioning import clean, split_sections

_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
COMPANY_WORDS = re.compile(r"\b(?:Pvt\.?|Private|Ltd\.?|Limited|Inc\.?|LLP|LLC|Technologies|Technology|Solutions|Systems|Services|"
                           r"Corp(?:oration)?\.?|Company|Co\.|Enterprises?|Labs?|Bank|Institute|University|Hospital|Store|"
                           r"Mart|Industries|Group|Consultanc(?:y|ies)|Software|Infotech|Broadband|Education|Telecom|"
                           r"Analytics|Softwares|Infosystems|Agency|Studio|Foundation|Traders|Motors)\b", re.I)
TITLE_WORDS = re.compile(r"\b(?:engineer|developer|analyst|intern|trainee|manager|executive|associate|consultant|lead|officer|"
                         r"assistant|designer|specialist|administrator|head|director|representative|supervisor|telecaller|"
                         r"telecounsellor|counsel+or|coordinator|technician|accountant|clerk|teacher|tutor|recruiter|"
                         r"programmer|architect|scientist|researcher|operator|agent|advisor|adviser|support|sales|"
                         r"marketing|hr|apprentice|freelancer|volunteer)\b", re.I)
TYPE_RE = re.compile(r"\b(full[\s-]?time|part[\s-]?time|internship|contract|freelance|apprentice(?:ship)?)\b", re.I)
TECH_RE = re.compile(r"^\W*(?:technolog(?:y|ies)(?:\s+used)?|tech(?:nical)?\s*stack|tools(?:\s+used)?|environment|skills?\s+used)"
                     r"\s*[:\-–]\s*(.+)$", re.I)
ACH_RE = re.compile(r"\b(?:achiev\w*|award\w*|increas\w*|improv\w*|reduc\w*|saved|exceed\w*|recogni[sz]\w*|top\s+performer|"
                    r"promoted|rank\w*|won)\b|\d+\s*%", re.I)
POINT_RE = re.compile(r"(?:([A-Za-z]{3})[a-z]*\.?[\s,'’]*)?(\d{4})")
PRESENT_RE = re.compile(r"^(?:present|current|till\s*date|to\s*date|now|ongoing)$", re.I)
LOCATION_LABEL = re.compile(r"^\W*(?:location|place)\s*[:\-–]\s*(.+)$", re.I)


def _point(text: str):
    m = re.search(r"(?:(\d{1,2})[/.\-])?((?:19|20)\d{2})", text)
    mon = re.match(r"\s*([A-Za-z]{3})", text)
    if mon and mon.group(1).lower() in _MONTHS and m:
        return _MONTHS[mon.group(1).lower()], int(m.group(2))
    mm = re.match(r"\s*(\d{1,2})[/.\-]((?:19|20)\d{2})", text)
    if mm and 1 <= int(mm.group(1)) <= 12:
        return int(mm.group(1)), int(mm.group(2))
    return None


def _months_text(start: str, end: str, current: bool, today: date) -> str:
    a = _point(start)
    b = _point(end) if end else ((today.month, today.year) if current else None)
    if not a or not b:
        return ""
    months = (b[1] - a[1]) * 12 + (b[0] - a[0])
    if months < 0:
        return ""
    y, m = divmod(months, 12)
    parts = ([f"{y} year{'s' if y != 1 else ''}"] if y else []) + ([f"{m} month{'s' if m != 1 else ''}"] if m else [])
    return " ".join(parts) or "Less than a month"


def _split_range(text: str) -> tuple[str, str, bool]:
    a, b = re.split(r"\s*(?:-|–|—|\bto\b|\btill\b|\buntil\b)\s*", text, maxsplit=1, flags=re.I)[:2]
    if PRESENT_RE.match(b.strip()):
        return a.strip(), "", True
    return a.strip(), b.strip(), False


def _is_body_line(line: str) -> bool:
    return len(line) > 70 or len(line.split()) > 9 or line.endswith(".")


def parse_group(lines: list[str], kind: str = "Employment", today: date | None = None) -> ExperienceRecord:
    today = today or date.today()
    rec = ExperienceRecord()
    header: list[str] = []
    body: list[str] = []
    seen_body = False
    leftovers: list[str] = []
    for ln in lines:
        mt = TECH_RE.match(ln)
        if mt:
            rec.technologies = (rec.technologies + ", " if rec.technologies else "") + mt.group(1).strip()
            continue
        ml = LOCATION_LABEL.match(ln)
        if ml and not rec.location:
            rec.location = ml.group(1).strip()
            continue
        if SHORT_DURATION_RE.match(ln):
            rec.duration = ln.strip("() ")
            continue
        md = DURATION_RE.search(ln)
        if md and not rec.start_date and len(ln) < 90:
            rec.start_date, rec.end_date, rec.current = _split_range(md.group(0))
            rest = (ln[:md.start()] + " " + ln[md.end():]).strip(" ,;|-–—()")
            if rest and not seen_body:
                header.append(rest)
            continue
        if ROLE_RE.match(ln) and not rec.title:
            t = re.sub(r"^\W*(?:worked|working|work)\s+(?:as|at)\s+(?:an?\s+|the\s+)?", "", ln, flags=re.I)
            rec.title = re.sub(r"^\W*(?:role|designation|position|job\s*title|profile)\s*[:\-–]\s*", "", t, flags=re.I).strip()
            continue
        if not seen_body and len(header) < 3 and not _is_body_line(ln):
            header.append(ln)
        else:
            seen_body = True
            body.append(ln)

    segs: list[str] = []
    for line in header:
        m = re.search(r"\(([^()]{2,40})\)\s*$", line)
        if m and not DURATION_RE.search(m.group(1)) and not rec.location:
            rec.location = m.group(1).strip()
            line = line[:m.start()].strip()
        for seg in [x.strip(" :-–—.") for x in re.split(r"\s*[|,]\s*|\s+[–—-]\s+|\s+@\s+|\s+\bat\b\s+", line, flags=re.I)]:
            if not seg:
                continue
            if re.fullmatch(r"[\d\s./-]+", seg):
                leftovers.append(seg)
            else:
                segs.append(seg)

    def _is_title(seg: str) -> bool:
        return bool(TITLE_WORDS.search(seg)) and (
            not COMPANY_WORDS.search(seg) or bool(TITLE_WORDS.fullmatch(seg.split()[-1].strip(".,"))))

    titles = [x for x in segs if _is_title(x)]
    companies = [x for x in segs if not _is_title(x) and COMPANY_WORDS.search(x)]
    unknown = [x for x in segs if x not in titles and x not in companies]
    if not rec.title and titles:
        rec.title = titles.pop(0)
    if companies:
        rec.company = companies.pop(0)
    rest = titles + companies + unknown
    if not rec.company and rest:
        rec.company = rest.pop(0)
    if not rec.title and rest:
        rec.title = rest.pop(0)
    if not rec.location and rest:
        rec.location = rest.pop(0)
    if rec.title and rec.company and TITLE_WORDS.search(rec.company) and not COMPANY_WORDS.search(rec.company) \
            and not TITLE_WORDS.search(rec.title):
        rec.company, rec.title = rec.title, rec.company  # "Developer" printed above "Acme Ltd"

    mtype = TYPE_RE.search(" ".join(header + [rec.title]))
    if kind == "Internship" or (mtype and "intern" in mtype.group(1).lower()) or re.search(r"\bintern\b", rec.title, re.I):
        rec.employment_type = "Internship"
    elif mtype:
        t = mtype.group(1).lower().replace(" ", "-")
        rec.employment_type = {"full-time": "Full-time", "fulltime": "Full-time", "part-time": "Part-time",
                               "parttime": "Part-time", "contract": "Contract", "freelance": "Freelance"}.get(t, "Apprenticeship")

    ach = [b for b in body if ACH_RE.search(b)]
    rec.achievements = "\n".join(ach)
    rec.responsibilities = "\n".join(b for b in body if b not in ach)
    if not rec.duration:
        calc = _months_text(rec.start_date, rec.end_date, rec.current, today)
        if calc:
            rec.duration = calc
            rec.notes.append("Duration was calculated from the start and end dates.")
    for leftover in leftovers:
        if re.fullmatch(r"(?:19|20)\d{2}", leftover):
            rec.notes.append(f"A year ({leftover}) appears without a full date range; start/end dates left blank.")
    if not rec.company:
        rec.notes.append("Company name not found.")
    if not rec.title:
        rec.notes.append("Job title not found.")
    rec.source_text = " | ".join(lines)[:240]
    rec.confidence = "medium" if rec.company and rec.title else "low"
    return rec


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def extract_experience(lines: list[OcrLine], today: date | None = None) -> tuple[list[ExperienceRecord], list[str]]:
    """Return (records, notes). ``notes`` carries 'No professional experience found' for freshers."""
    today = today or date.today()
    texts, meta = [], []
    for ln in lines:
        t = clean(ln.text)
        if t:
            texts.append(t)
            meta.append(ln)
    records: list[ExperienceRecord] = []
    found_section = False
    for sec in split_sections(texts):
        if re.search(r"project", sec.title) or not re.search(r"experience|employment|work\s+history|internship", sec.title):
            continue
        found_section = True
        kind = "Internship" if "internship" in sec.title else "Employment"
        body = [texts[i] for i in sec.body]
        for group in group_lines(body):
            rec = parse_group(group, kind, today)
            first = sec.body[0] if sec.body else 0
            rec.source_index = meta[first].source_index
            rec.source_page = getattr(meta[first], "page", 0)
            if rec.is_blank():
                continue
            dup = next((r for r in records if _norm(r.company) == _norm(rec.company) and _norm(r.title) == _norm(rec.title)
                        and _norm(r.start_date) == _norm(rec.start_date)), None)
            if dup:  # repeated screenshot / page: keep one record, fill any blanks
                for f in ("employment_type", "location", "end_date", "duration", "responsibilities", "achievements", "technologies"):
                    if not getattr(dup, f) and getattr(rec, f):
                        setattr(dup, f, getattr(rec, f))
                continue
            records.append(rec)
    notes = []
    if not records:
        notes.append("No professional experience found." if not found_section else
                     "An experience section was found, but no job could be read from it.")
    return records, notes
