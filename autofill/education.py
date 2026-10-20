"""Education extraction: one EducationRecord per qualification.

Method
1. Merge OCR fragments that sit on the same visual row (tables), per screenshot.
2. Find the Education section(s) (or the whole text if there is no heading).
3. Anchor each qualification on a line that names its level (10th/SSC, 12th/HSC, Diploma,
   Graduation/degree names, Post-graduation, Higher) - the *leftmost* level word wins, so
   "Class 12: ABC Senior Secondary School" is Class 12, not Class 10.
4. A record spans from its anchor to the next anchor. Inside the span, comma/pipe/bracket
   separated segments and labelled values are classified independently (institution, board,
   degree, specialisation, year, status, marks, percentage, CGPA, grade, result).
5. Records of the same level are merged; conflicting values are noted, never silently chosen.
Nothing is calculated: a percentage or CGPA is copied only if the document states it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from . import config
from .ocr_engine import OcrLine
from .records import EducationRecord, SCHOOL_LEVELS
from .sectioning import clean, is_heading, split_sections

YEAR = r"(?:19[5-9]\d|20[0-4]\d)"
_I = re.I

# ----------------------------------------------------------------------------- level anchors
# (level, regex, concrete?, flags). Concrete = names an actual degree/class, generic = a label word.
_LEVEL_PATTERNS: list[tuple[str, re.Pattern, bool]] = [
    ("post_graduation", re.compile(r"\b(?:M\.?\s?Tech|M\.?\s?Sc|MBA|MCA|M\.?\s?Com|M\.?\s?Pharm|M\.?\s?Arch|M\.?\s?Ed|"
                                   r"LL\.?M|PGDM|PG\s?Diploma|Post[\s-]?Graduate\s+Diploma|"
                                   r"Masters?(?:'s)?\s+(?:of|in)\s+[A-Z][A-Za-z&]*(?:\s+(?:and|&)\s+[A-Z][a-z]+)?|"
                                   r"Masters?(?:'s)?(?:\s+Degree)?)\b", _I), True),
    ("post_graduation", re.compile(r"(?<![A-Za-z])(?:M\.?A\.?|M\.?E\.?)(?![A-Za-z])"), True),
    ("post_graduation", re.compile(r"\bPost[\s-]?Graduat\w*", _I), False),
    ("higher", re.compile(r"\b(?:Ph\.?\s?D|Doctorate|M\.?\s?Phil|Post[\s-]?Doctoral|Chartered\s+Accountant|"
                          r"CFA|ICWA|CMA\s+(?:Final|Inter)|CS\s+(?:Executive|Professional))\b", _I), True),
    ("graduation", re.compile(r"\b(?:B\.?\s?Tech|B\.?\s?Sc|B\.?\s?Com|BCA|BBA|BMS|B\.?\s?Pharm|B\.?\s?Arch|B\.?\s?Ed|"
                              r"LL\.?B|MBBS|BDS|"
                              r"Bachelors?(?:'s)?\s+(?:of|in)\s+[A-Z][A-Za-z&]*(?:\s+(?:and|&)\s+[A-Z][a-z]+)?|"
                              r"Bachelors?(?:'s)?(?:\s+Degree)?)\b", _I), True),
    ("graduation", re.compile(r"(?<![A-Za-z])(?:B\.?E\.?|B\.?A\.?)(?![A-Za-z])"), True),
    ("graduation", re.compile(r"\b(?:Under[\s-]?Graduat\w*|Graduat(?:ion|e)\b)", _I), False),
    ("diploma", re.compile(r"\b(?:Diploma|Polytechnic|D\.?\s?Pharm)\b", _I), True),
    ("hsc", re.compile(r"(?<![\w])(?:H\.?\s?S\.?\s?C\.?|12th|XII(?:th)?|^\W*XII(?=\s*(?:[-–—:|)]|$))|Class\s*(?:XII|12)|(?:Std|Standard)\.?\s*(?:XII|12)|"
                       r"Senior\s+Secondary|Higher\s+Secondary|Intermediate|\+2|Plus\s*Two|Pre[\s-]?University|PUC)"
                       r"(?![\w])", _I), True),
    ("ssc", re.compile(r"(?<![\w])(?:S\.?\s?S\.?\s?C\.?|SSLC|10th|Xth|^\W*X(?=\s*(?:[-–—:|)]|$))|Class\s*(?:X|10)(?![\w])|"
                       r"(?:Std|Standard)\.?\s*(?:X|10)(?![\w])|Matriculation|Secondary(?:\s+School)?(?:\s+Certificate)?|"
                       r"High\s+School|\(X\))(?![\w])", _I), True),
]
_PRIORITY = {"post_graduation": 0, "higher": 1, "graduation": 2, "diploma": 3, "hsc": 4, "ssc": 5}
_LEVEL_ONLY = re.compile(r"^\W*(?:class\s*)?(?:X{1,2}I{0,2}|10|12|10th|12th|XIIth?|Xth)\W*$", _I)

_DEGREE_TOKEN = {  # level -> regexes that capture the degree *name*
    "graduation": [p for lvl, p, c in _LEVEL_PATTERNS if lvl == "graduation" and c],
    "post_graduation": [p for lvl, p, c in _LEVEL_PATTERNS if lvl == "post_graduation" and c],
    "diploma": [p for lvl, p, c in _LEVEL_PATTERNS if lvl == "diploma"],
    "higher": [p for lvl, p, c in _LEVEL_PATTERNS if lvl == "higher"],
}

# ----------------------------------------------------------------------------- value patterns
RANGE_RE = re.compile(rf"({YEAR})\s*(?:-|–|—|to)\s*({YEAR}|\d{{2}}(?!\d)|present|current|ongoing|continuing|pursuing)", _I)
YEAR_RE = re.compile(rf"(?<!\d)({YEAR})(?!\d)")
EXPECTED_BEFORE = re.compile(r"(?:expected|expecting|exp\.?|likely|anticipated|by|completing|completion)(?:\s+\w+){0,2}\s*[:\-]?\s*$", _I)
EXPECTED_AFTER = re.compile(r"^\W*(?:\(?\s*)?(?:expected|exp\b)", _I)
PERCENT_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,2})?)\s*%")
PERCENT_LABEL_RE = re.compile(r"\bpercentage\b\s*[:\-]?\s*(\d{1,3}(?:\.\d{1,2})?)\s*%?", _I)
CGPA_LABEL_RE = re.compile(r"\b(?:CGPA|GPA|CPI|SGPA|DGPA)\b\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,2})?)(?:\s*/\s*(\d{1,2}(?:\.\d+)?))?", _I)
CGPA_AFTER_RE = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d{1,2})?)(?:\s*/\s*(\d{1,2}(?:\.\d+)?))?\s*(?:CGPA|GPA|CPI|SGPA)\b", _I)
OUT_OF_RE = re.compile(r"(?<![\d/])(\d{1,4}(?:\.\d+)?)\s*(?:/|out\s+of)\s*(\d{1,4})(?![\d/])", _I)
MARKS_LABEL_RE = re.compile(r"\bmarks?(?:\s+obtained)?\b\s*[:\-]?\s*(\d{2,4}(?:\.\d+)?)(?!\s*%)", _I)
MAXMARKS_LABEL_RE = re.compile(r"\b(?:max(?:imum)?\.?\s+marks?|total\s+marks?)\b\s*[:\-]?\s*(\d{2,4})", _I)
GRADE_RE = re.compile(r"\bgrade\b\s*[:\-]?\s*([A-F][+\-]?\d?)(?![A-Za-z])|\b(First\s+(?:Class|Division)(?:\s+with\s+Distinction)?|"
                      r"Second\s+(?:Class|Division)|Distinction)\b", _I)
RESULT_RE = re.compile(r"\b(?:result\s*[:\-]?\s*)?(passed|pass|fail(?:ed)?|promoted)\b", _I)
PURSUING_RE = re.compile(r"\b(?:pursuing|persuing|pursueing|pursing|currently\s+(?:studying|pursuing|enrolled)|ongoing|"
                         r"continuing|in\s+progress|studying)\b", _I)
APPEARING_RE = re.compile(r"\b(?:appearing|result\s+awaited|awaited)\b", _I)
COMPLETED_RE = re.compile(r"\b(?:completed|graduated|passed(?:\s+out)?|awarded|degree\s+(?:received|completed))\b", _I)
CURRENT_YEAR_RE = re.compile(r"\b(?:final\s+year|(?:1st|2nd|3rd|4th|5th|first|second|third|fourth|fifth)\s+year|"
                             r"(?:semester|sem)\.?\s*[-:]?\s*(?:[1-9]|I{1,3}|IV|VI{0,3}|IX)\b|"
                             r"(?:1st|2nd|3rd|4th|5th|6th|7th|8th)\s+sem(?:ester)?)\b", _I)
BOARD_RE = re.compile(r"\b(?:CBSE|ICSE|CISCE|ISC|NIOS|IGCSE|IB)\b|\bBoard\b", _I)
SCHOOL_WORDS = re.compile(r"\b(?:School|Vidyalaya|Vidyalay|Vidya\s?Mandir|Vidyamandir|Academy|Convent|Public|Model|Kendriya|"
                          r"Navodaya|Junior\s+College|High|Sr\.?\s?Sec|Gurukul|College|Institute|Mahavidyalaya)\b", _I)
DEGREE_INST_WORDS = re.compile(r"\b(?:College|University|Institute|Institution|Polytechnic|Vidyapeeth|Mahavidyalaya|Academy|"
                               r"Campus|IIT|NIT|IIIT|School\s+of|Autonomous|Faculty|Deemed)\b", _I)
UNIVERSITY_RE = re.compile(r"\b(?:University|Univ\.?|Vidyapeeth|Deemed|Autonomous|Affiliated)(?![A-Za-z])", _I)
STREAM_RE = re.compile(r"^(?:Science|Commerce|Arts|Humanities|PCM|PCB|PCMB|PCME|Vocational|General|"
                       r"(?:Science|Commerce|Arts)\s*\(?\w*\)?)(?:\s+Stream)?$", _I)
LABEL_ONLY = re.compile(r"^(?:course|qualification|institute|institution|school|college|university|board|year|"
                        r"percentage|marks?|cgpa|gpa|grade|result|status|stream|degree|class|std|passing\s+year|"
                        r"year\s+of\s+passing|school/college|board/university|name|duration|specialization)$", _I)


@dataclass
class _Row:
    text: str
    source: int
    page: int
    score: float


def merge_rows(lines: list[OcrLine]) -> list[_Row]:
    """Join OCR fragments that share a visual row (table cells) into one ' | ' separated line."""
    rows: list[_Row] = []
    prev: OcrLine | None = None
    for ln in lines:
        text = clean(ln.text)
        if not text:
            continue
        merged = False
        if prev is not None and rows and prev.bbox and ln.bbox and prev.source_index == ln.source_index:
            py0, py1 = prev.bbox[1], prev.bbox[3]
            cy0, cy1 = ln.bbox[1], ln.bbox[3]
            height = max(1, min(py1 - py0, cy1 - cy0))
            same_row = abs((py0 + py1) / 2 - (cy0 + cy1) / 2) < 0.5 * height
            to_the_right = ln.bbox[0] >= prev.bbox[2] - 4
            if same_row and to_the_right:
                rows[-1].text += " | " + text
                rows[-1].score = min(rows[-1].score, ln.score)
                merged = True
        if not merged:
            rows.append(_Row(text, ln.source_index, getattr(ln, "page", 0), ln.score))
        prev = ln
    return rows


def classify_line(text: str) -> tuple[str, bool, re.Match] | None:
    """Return (level, concrete, match) for the leftmost level word in ``text``."""
    best = None
    for level, rx, concrete in _LEVEL_PATTERNS:
        m = rx.search(text)
        if m:
            key = (m.start(), _PRIORITY[level], not concrete)
            if best is None or key < best[0]:
                best = (key, level, concrete, m)
    if not best:
        return None
    _, level, concrete, m = best
    if not concrete:  # a generic label ("Graduation:") on the same line as a real degree name is concrete
        for rx in _DEGREE_TOKEN.get(level, []):
            if rx.search(text):
                concrete = True
                break
    return level, concrete, m


def _segments(text: str) -> list[str]:
    s = re.sub(r"[()\[\]]", " | ", text)
    parts = re.split(r"\s*[|;•·]\s*|\s*,\s*|\s+[-–—]\s+|\s*:\s+|\t+", s)
    return [p.strip(" :-–—.") for p in parts if p.strip(" :-–—.")]


def _find_degree(text: str, level: str) -> tuple[str, str, str]:
    """Return (degree name, specialisation, text after) for degree-type levels."""
    for rx in _DEGREE_TOKEN.get(level, []):
        m = rx.search(text)
        if not m:
            continue
        degree = re.sub(r"\s+", " ", m.group(0)).strip()
        rest = text[m.end():]
        spec = ""
        mi = re.match(r"\s*(?:in|\(\s*in)\s+([A-Za-z][A-Za-z&/ ]{1,60}?)(?=\s*(?:[,|;()\d]|\s[-–—]\s|$|\b(?:pursuing|persuing|completed|final)\b))",
                      rest, _I)
        if mi:
            spec = mi.group(1).strip()
        else:
            mp = re.match(r"\s*\(\s*([A-Za-z][A-Za-z&/ ]{1,40})\s*\)", rest)
            if mp and not (PURSUING_RE.search(mp.group(1)) or APPEARING_RE.search(mp.group(1)) or
                           re.fullmatch(r"hons?\.?|honou?rs|regular|distance", mp.group(1).strip(), _I) or
                           COMPLETED_RE.search(mp.group(1))):
                spec = mp.group(1).strip()
        if not spec:
            tail = re.sub(r"^\s*\(\s*(?:hons?\.?|honou?rs|regular|distance)\s*\)", "", rest, flags=_I)
            ms = re.match(r"\s*\.?\s+([A-Z][A-Za-z&/]*(?:\s+(?:&|and|of)?\s*[A-Z][A-Za-z&/]*){0,4})(?=\s*(?:[,|;(\d]|\s[-–—]\s|$))", tail)
            if ms and not (SCHOOL_WORDS.search(ms.group(1)) or DEGREE_INST_WORDS.search(ms.group(1)) or
                           BOARD_RE.search(ms.group(1)) or PURSUING_RE.search(ms.group(1)) or
                           CURRENT_YEAR_RE.search(ms.group(1)) or COMPLETED_RE.search(ms.group(1)) or
                           APPEARING_RE.search(ms.group(1))):
                spec = ms.group(1).strip()
        return degree, spec, rest
    return "", "", text


def _status_and_years(span: str, level: str, today: date) -> tuple[dict, list[str]]:
    notes: list[str] = []
    out = {"status": "Not Available", "start_year": "", "passing_year": "", "expected_year": "", "current_year": ""}
    school = level in SCHOOL_LEVELS
    pursuing = PURSUING_RE.search(span) is not None
    appearing = APPEARING_RE.search(span) is not None
    completed = COMPLETED_RE.search(span) is not None
    cy = CURRENT_YEAR_RE.search(span)
    if cy:
        out["current_year"] = cy.group(0).strip()
        pursuing = pursuing or True
    year_end, expected_flag, open_range = None, False, False
    rng = RANGE_RE.search(span)
    if rng:
        out["start_year"] = rng.group(1)
        end = rng.group(2)
        if end.isdigit():
            year_end = int(end) if len(end) == 4 else int(rng.group(1)[:2] + end)
        else:
            open_range = True
    else:
        singles = list(YEAR_RE.finditer(span))
        for m in singles:
            before, after = span[max(0, m.start() - 28):m.start()], span[m.end():m.end() + 14]
            if EXPECTED_BEFORE.search(before) or EXPECTED_AFTER.search(after):
                expected_flag = True
                year_end = int(m.group(1))
                break
        if year_end is None and singles:
            year_end = int(singles[-1].group(1))
    active = "Appearing" if (appearing or (school and (pursuing or expected_flag))) else "Pursuing"
    if completed and not pursuing and not appearing and not open_range:
        out["status"] = "Completed"
        if year_end:
            out["passing_year"] = str(year_end)
    elif pursuing or appearing or open_range:
        out["status"] = active
        if year_end:
            out["expected_year"] = str(year_end)
            if year_end < today.year:
                notes.append(f"The document says {active.lower()} but the year {year_end} is in the past — please verify.")
    elif expected_flag and year_end:
        out["status"], out["expected_year"] = active, str(year_end)
        notes.append("Marked as expected in the document, so it is not treated as a passing year.")
    elif year_end:
        if year_end < today.year:
            out["status"], out["passing_year"] = "Completed", str(year_end)
        elif year_end == today.year:
            out["expected_year"] = str(year_end)
            notes.append(f"{year_end} is the current year: it could be a passing or an expected year. Please verify "
                         "the status.")
        else:
            out["status"], out["expected_year"] = active, str(year_end)
            notes.append(f"{year_end} is in the future, so it is treated as an expected year.")
    return out, notes


def _numbers(span: str) -> dict[str, str]:
    out = {"percentage": "", "cgpa": "", "marks_obtained": "", "max_marks": "", "grade": "", "result": ""}
    m = PERCENT_RE.search(span)
    if m:
        out["percentage"] = m.group(1) + "%"
    else:
        m = PERCENT_LABEL_RE.search(span)
        if m:
            out["percentage"] = m.group(1) + "%"
    for rx in (CGPA_LABEL_RE, CGPA_AFTER_RE):
        m = rx.search(span)
        if m:
            out["cgpa"] = m.group(1) + (f"/{m.group(2)}" if m.group(2) else "")
            break
    mm = MARKS_LABEL_RE.search(span)
    mx = MAXMARKS_LABEL_RE.search(span)
    if mm:
        out["marks_obtained"] = mm.group(1)
    if mx:
        out["max_marks"] = mx.group(1)
    for m in OUT_OF_RE.finditer(span):
        num, den = m.group(1), m.group(2)
        if int(den) in (4, 5, 7, 10) and not out["cgpa"] and float(num) <= float(den):
            out["cgpa"] = f"{num}/{den}"
        elif int(den) > 10 and not out["marks_obtained"]:
            out["marks_obtained"], out["max_marks"] = num, den
        elif int(den) > 10 and out["marks_obtained"] == num and not out["max_marks"]:
            out["max_marks"] = den
    g = GRADE_RE.search(span)
    if g:
        out["grade"] = (g.group(1) or g.group(2)).strip()
    r = RESULT_RE.search(span)
    if r:
        out["result"] = r.group(1).capitalize()
    return out


def _is_noise_segment(seg: str) -> bool:
    return bool(LABEL_ONLY.match(seg) or _LEVEL_ONLY.match(seg) or re.fullmatch(r"[\d\s./%:-]+", seg)
                or PURSUING_RE.fullmatch(seg) or APPEARING_RE.fullmatch(seg) or COMPLETED_RE.fullmatch(seg)
                or CURRENT_YEAR_RE.fullmatch(seg) or re.fullmatch(r"(?:expected|exp\.?)(?:\s+\d{4})?", seg, _I)
                or (classify_line(seg) is not None and len(seg.split()) <= 3 and not SCHOOL_WORDS.search(seg)
                    and not DEGREE_INST_WORDS.search(seg)))


def _strip_values(seg: str) -> str:
    """Remove years, percentages, CGPA, marks and status words so only the name text remains."""
    for rx in (RANGE_RE, PERCENT_LABEL_RE, PERCENT_RE, CGPA_LABEL_RE, CGPA_AFTER_RE, OUT_OF_RE, YEAR_RE,
               PURSUING_RE, APPEARING_RE, COMPLETED_RE, CURRENT_YEAR_RE):
        seg = rx.sub(" ", seg)
    seg = re.sub(r"\b(?:expected|exp\.?|marks?|grade|result)\b", " ", seg, flags=_I)
    seg = re.sub(r"^\s*(?:from|in|at|under)\s+", "", seg, flags=_I)
    seg = re.sub(r"^\s*((?:CBSE|ICSE|CISCE|ISC|NIOS|IGCSE|IB))\s+board\s*$", r"\1", seg, flags=_I)
    return re.sub(r"\s+", " ", seg).strip(" :-–—.,|/")


def _classify_segments(segs: list[str], level: str, degree: str, spec: str) -> dict[str, str]:
    inst = body = stream = ""
    guess = ""
    school = level in SCHOOL_LEVELS
    inst_words = SCHOOL_WORDS if school else DEGREE_INST_WORDS
    for raw in segs:
        seg = _strip_values(raw)
        low = seg.lower()
        if not seg or (degree and low == degree.lower()):
            continue
        if _is_noise_segment(seg):
            continue
        if school and STREAM_RE.match(seg) and not stream:
            stream = seg
        elif BOARD_RE.search(seg) and not inst_words.search(seg):
            body = body or seg
        elif not school and UNIVERSITY_RE.search(seg) and not re.search(r"\b(?:College|Institute|Campus)\b", seg, _I):
            body = body or seg
        elif inst_words.search(seg):
            inst = inst or seg
        elif not guess and re.search(r"[A-Za-z]{3}", seg) and len(seg) <= 70 and not (degree and seg.lower() in degree.lower()):
            guess = seg
    notes: list[str] = []
    if not inst and guess:
        inst = guess
        notes.append("Institution name is a best guess (no school/college keyword) — please verify.")
    return {"institution": inst, "board_or_body": body, "stream": stream, "_notes": notes}


NUMBER_UNIT = re.compile(r"\d\s*(?:%|/|cgpa|gpa)|\b(?:cgpa|gpa)\b", _I)


def extract_education(lines: list[OcrLine], today: date | None = None) -> list[EducationRecord]:
    today = today or date.today()
    rows = merge_rows(lines)
    texts = [r.text for r in rows]
    if not texts:
        return []

    edu_idx: list[int] = []
    for sec in split_sections(texts):
        if re.search(r"\b(?:education|academic|qualification)", sec.title):
            edu_idx.extend(sec.body)
    scoped = bool(edu_idx)
    if not scoped:
        edu_idx = list(range(len(texts)))
    edu_set = set(edu_idx)

    # --- anchors
    anchors: list[tuple[int, str, bool, re.Match]] = []
    for i in edu_idx:
        t = texts[i]
        if len(t) > 160 or is_heading(t):
            continue
        hit = classify_line(t)
        if not hit:
            continue
        level, concrete, m = hit
        if not scoped:  # without an Education heading require supporting evidence nearby
            window = " ".join(texts[i:i + 3])
            if not (YEAR_RE.search(window) or PERCENT_RE.search(window) or CGPA_AFTER_RE.search(window) or
                    CGPA_LABEL_RE.search(window) or SCHOOL_WORDS.search(window) or DEGREE_INST_WORDS.search(window)):
                continue
        anchors.append((i, level, concrete, m))
    if not anchors:
        return []

    # --- spans: anchor .. next anchor. Generic same-level anchors extend the previous record.
    def _name_above(i: int, floor: int) -> int | None:
        """Index of a lone institution-name line directly above anchor line ``i`` (name-first layouts)."""
        j = i - 1
        if j < floor or j not in edu_set:
            return None
        t = texts[j]
        if (len(t) > 80 or re.search(r"\d", t) or is_heading(t) or classify_line(t) or
                _is_noise_segment(t) or not re.search(r"[A-Za-z]{3}", t)):
            return None
        own = classify_line(texts[i])
        if own and (SCHOOL_WORDS.search(texts[i]) or DEGREE_INST_WORDS.search(texts[i])):
            return None  # the anchor line names its own institution
        return j

    pre: dict[int, int] = {}
    for k, (i, *_r) in enumerate(anchors):
        j = _name_above(i, (anchors[k - 1][0] + 1) if k else min(edu_idx))
        if j is not None:
            pre[k] = j
    spans: list[tuple[str, list[int]]] = []
    span_pre: dict[int, int] = {}
    for k, (i, level, concrete, _) in enumerate(anchors):
        end = anchors[k + 1][0] if k + 1 < len(anchors) else max(edu_idx) + 1
        if k + 1 in pre:
            end = pre[k + 1]
        idxs = [j for j in range(i, end) if j in edu_set]
        if spans and spans[-1][0] == level and not concrete and level not in SCHOOL_LEVELS and \
                i == spans[-1][1][-1] + 1:
            spans[-1][1].extend(idxs)
        else:
            spans.append((level, idxs))
            if k in pre:
                span_pre[idxs[0]] = pre[k]

    records: list[EducationRecord] = []
    for level, idxs in spans:
        anchor_text = texts[idxs[0]]
        degree = spec = ""
        rest_text = anchor_text
        if level in _DEGREE_TOKEN:
            for j in idxs[:2]:  # the degree may be on the line after a generic label like "Graduation"
                degree, spec, rest = _find_degree(texts[j], level)
                if degree:
                    break
        span_text = " | ".join(texts[j] for j in idxs)
        # strip a leading level label ("Class 10:", "10th -") from the anchor line only
        label = re.match(r"^\W*(?:class\s*)?(?:S\.?S\.?C\.?|H\.?S\.?C\.?|SSLC|10th|12th|XII(?:th)?|Xth|XII|X|Class\s*(?:X|XII|10|12)|"
                         r"Graduation|Post[\s-]?Graduation|Under[\s-]?Graduation|Diploma)\b\W*", anchor_text, _I) \
            if level in SCHOOL_LEVELS or not degree else None
        first_line = anchor_text[label.end():] if label and label.end() < len(anchor_text) else (
            "" if label else anchor_text)
        above = [texts[span_pre[idxs[0]]]] if idxs[0] in span_pre else []
        seg_lines = above + ([first_line] if first_line else []) + [texts[j] for j in idxs[1:]]
        segs = [s for ln in seg_lines for s in _segments(ln)]
        info = _classify_segments(segs, level, degree, spec)
        status = _status_and_years(span_text, level, today)
        nums = _numbers(span_text)
        score = min(rows[j].score for j in idxs)
        rec = EducationRecord(
            level=level, degree=degree, institution=info["institution"], board_or_body=info["board_or_body"],
            specialization=spec or info["stream"], start_year=status[0]["start_year"],
            passing_year=status[0]["passing_year"], expected_year=status[0]["expected_year"],
            status=status[0]["status"], current_year=status[0]["current_year"], **nums,
            source_index=rows[idxs[0]].source, source_page=rows[idxs[0]].page,
            source_text=span_text[:240], notes=list(status[1]) + list(info["_notes"]))
        if score < config.OCR_REVIEW_SCORE:
            rec.notes.append(f"Low OCR confidence ({score:.2f}) — please verify.")
            rec.confidence = "low"
        elif info["_notes"]:
            rec.confidence = "low"
        if level in SCHOOL_LEVELS:
            rec.degree = ""
        if not (rec.is_blank()):
            records.append(rec)
    return merge_records(records)


# ----------------------------------------------------------------------------- merge / dedupe
_MERGE_FIELDS = ["degree", "institution", "board_or_body", "specialization", "start_year", "passing_year",
                 "expected_year", "current_year", "marks_obtained", "max_marks", "percentage", "cgpa", "grade",
                 "result"]
_CONFLICT_FIELDS = {"institution", "board_or_body", "passing_year", "expected_year", "percentage", "cgpa", "degree"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _same_record(a: EducationRecord, b: EducationRecord) -> bool:
    if a.level != b.level:
        return False
    if a.level in SCHOOL_LEVELS:
        return True
    if a.degree and b.degree and _norm(a.degree) != _norm(b.degree):
        return False
    if a.institution and b.institution and _norm(a.institution) != _norm(b.institution):
        return bool(a.degree and b.degree)
    return True


def merge_records(records: list[EducationRecord]) -> list[EducationRecord]:
    merged: list[EducationRecord] = []
    for rec in records:
        target = next((m for m in merged if _same_record(m, rec)), None)
        if target is None:
            merged.append(rec)
            continue
        for f in _MERGE_FIELDS:
            mine, theirs = getattr(target, f), getattr(rec, f)
            if not mine and theirs:
                setattr(target, f, theirs)
            elif mine and theirs and _norm(mine) != _norm(theirs) and f in _CONFLICT_FIELDS:
                note = f"Conflicting {f.replace('_', ' ')}: “{mine}” vs “{theirs}” — kept the first."
                if note not in target.notes:
                    target.notes.append(note)
        if target.status == "Not Available" and rec.status != "Not Available":
            target.status = rec.status
        for n in rec.notes:
            if n not in target.notes:
                target.notes.append(n)
        if rec.confidence == "low":
            target.confidence = "low"
    order = {lvl: i for i, lvl in enumerate(["ssc", "hsc", "diploma", "graduation", "post_graduation", "higher"])}
    return sorted(merged, key=lambda r: order.get(r.level, 9))
