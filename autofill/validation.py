"""Field validation, conflict detection and review flags.

Validation never blocks saving and never edits a value: it produces human-readable issues
so the reviewer can decide. Blank fields are reported separately as 'not found'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from . import config
from .extraction import EMAIL_RE, ExtractionResult, INDIAN_STATES
from .fields import CORE_FIELDS, FIELD_KEYS, LABELS

_DATE_FORMATS = [r"(\d{1,2})[\s/.\-](\d{1,2})[\s/.\-](\d{4})", r"(\d{4})[\s/.\-](\d{1,2})[\s/.\-](\d{1,2})"]
_MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


@dataclass
class ValidationReport:
    issues: dict[str, list[str]] = field(default_factory=dict)  # field -> reasons (needs review)
    record_issues: dict[str, list[str]] = field(default_factory=dict)  # "education:0" / "experience:1" -> reasons
    notes: list[str] = field(default_factory=list)  # document-level notes, e.g. "No professional experience found."
    blank: list[str] = field(default_factory=list)  # fields not found

    def add(self, key: str, message: str) -> None:
        self.issues.setdefault(key, [])
        if message not in self.issues[key]:
            self.issues[key].append(message)

    def add_record(self, key: str, message: str) -> None:
        self.record_issues.setdefault(key, [])
        if message not in self.record_issues[key]:
            self.record_issues[key].append(message)

    @property
    def flagged(self) -> set[str]:
        return set(self.issues)

    def to_dict(self) -> dict:
        return {"issues": self.issues, "blank": self.blank, "record_issues": self.record_issues, "notes": self.notes}


def parse_dob(value: str) -> date | None:
    v = value.strip()
    m = re.fullmatch(_DATE_FORMATS[0], v)
    if m:
        d, mo, y = (int(x) for x in m.groups())
    else:
        m = re.fullmatch(_DATE_FORMATS[1], v)
        if m:
            y, mo, d = (int(x) for x in m.groups())
        else:
            m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?[\s\-]+([A-Za-z]{3})[a-z]*[\s,.\-]+(\d{4})", v)
            if not m or m.group(2).lower() not in _MONTHS:
                return None
            d, mo, y = int(m.group(1)), _MONTHS[m.group(2).lower()], int(m.group(3))
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def _years(value: str) -> list[int]:
    return [int(y) for y in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", value)]


def validate(values: dict[str, str], extraction: ExtractionResult | None = None,
             today: date | None = None) -> ValidationReport:
    today = today or date.today()
    rep = ValidationReport()
    v = {k: (values.get(k) or "").strip() for k in FIELD_KEYS}

    # --- format checks
    if v["email"] and not (EMAIL_RE.fullmatch(v["email"])):
        rep.add("email", "Not a valid email address")
    if v["mobile"]:
        digits = re.sub(r"\D", "", v["mobile"])
        national = digits[-10:] if len(digits) > 10 and digits[:-10] in ("0", "91", "091") else digits
        if len(national) != 10:
            rep.add("mobile", f"Expected 10 digits, found {len(national)}")
        elif national[0] not in "6789":
            rep.add("mobile", "Indian mobile numbers start with 6–9")
    if v["pincode"] and not re.fullmatch(r"[1-9]\d{5}", v["pincode"]):
        rep.add("pincode", "Pincode must be 6 digits")
    if v["date_of_birth"]:
        dob = parse_dob(v["date_of_birth"])
        if dob is None:
            rep.add("date_of_birth", "Could not read this as a date")
        else:
            age = (today - dob).days / 365.25
            if not 15 <= age <= 75:
                rep.add("date_of_birth", f"Implausible age ({age:.0f})")
    for key in ("ssc_pass_year", "hsc_pass_year", "graduation_year", "post_graduation_year"):
        if v[key]:
            ys = _years(v[key])
            if not ys:
                rep.add(key, "Expected a 4-digit year")
            elif any(y > today.year + 6 or y < 1950 for y in ys):
                rep.add(key, "Year is outside a plausible range")
    for key in ("ssc_result", "hsc_result", "graduation_result", "post_graduation_result"):
        if v[key]:
            num = re.search(r"\d+(?:\.\d+)?", v[key])
            if not num:
                rep.add(key, "No numeric result found")
            else:
                n = float(num.group(0))
                limit = 100 if "%" in v[key] or n > 10 else 10
                if n > limit or n <= 0:
                    rep.add(key, "Result value is outside a plausible range")
    if v["state"] and v["state"].title() not in [s.title() for s in INDIAN_STATES]:
        rep.add("state", "Not a recognised Indian state/UT")

    # --- cross-field consistency
    ssc, hsc, grad = (_years(v["ssc_pass_year"]), _years(v["hsc_pass_year"]), _years(v["graduation_year"]))
    if ssc and hsc and max(ssc) >= max(hsc):
        rep.add("hsc_pass_year", "HSC year should be after SSC year")
    if hsc and grad and max(hsc) >= max(grad):
        rep.add("graduation_year", "Graduation year should be after HSC year")

    # --- extraction-derived flags
    if extraction:
        for key, alts in extraction.conflicts.items():
            rep.add(key, "Multiple different values found: " + ", ".join(alts))
        for key, ev in extraction.evidence.items():
            if not v.get(key) or v[key] != (extraction.values.get(key) or "").strip():
                continue  # blank, or edited by the reviewer since extraction
            if ev.confidence == "low":
                rep.add(key, "Guessed from layout, not a labelled value — please verify")
            if ev.ocr_score < config.OCR_REVIEW_SCORE:
                rep.add(key, f"Low OCR confidence ({ev.ocr_score:.2f})")

    _validate_records(values, extraction, rep, today)

    # --- completeness
    rep.blank = [k for k in FIELD_KEYS if not v[k]]
    for key in CORE_FIELDS:
        if not v[key]:
            rep.add(key, "Not found — left blank")
    return rep


def describe(key: str, report: ValidationReport) -> str:
    return f"{LABELS[key]}: " + "; ".join(report.issues.get(key, []))


def _num(text: str) -> float | None:
    m = re.search(r"\d+(?:\.\d+)?", text or "")
    return float(m.group(0)) if m else None


def _validate_records(values: dict, extraction: ExtractionResult | None, rep: ValidationReport, today: date) -> None:
    from .records import education_from_fields, experience_from_fields, LEVEL_LABELS

    edu = education_from_fields(values) or (list(extraction.education) if extraction else [])
    exp = experience_from_fields(values) or (list(extraction.experience) if extraction else [])
    for i, r in enumerate(edu):
        key = f"education:{i}"
        for n in r.notes:
            rep.add_record(key, n)
        pct, cg = _num(r.percentage), _num(r.cgpa)
        if r.percentage and (pct is None or not 0 < pct <= 100):
            rep.add_record(key, "Percentage is outside 0–100.")
        if r.cgpa and cg is not None:
            scale = _num(r.cgpa.split("/", 1)[1]) if "/" in r.cgpa else 10
            if cg <= 0 or cg > (scale or 10):
                rep.add_record(key, f"CGPA {r.cgpa} is outside the 0–{scale:g} scale.")
        mo, mx = _num(r.marks_obtained), _num(r.max_marks)
        if mo is not None and mx is not None and mo > mx:
            rep.add_record(key, "Marks obtained exceed the maximum marks.")
        for f in ("start_year", "passing_year", "expected_year"):
            y = _num(getattr(r, f))
            if y is not None and not 1950 <= y <= today.year + 8:
                rep.add_record(key, f"{f.replace('_', ' ').capitalize()} is outside a plausible range.")
        if r.status == "Completed" and r.expected_year:
            rep.add_record(key, "Status is Completed but an expected year is set.")
        if r.status in ("Pursuing", "Appearing") and r.passing_year:
            rep.add_record(key, f"Status is {r.status} but a passing year is set — an expected year is not a passing year.")
        if r.start_year and r.passing_year and _num(r.start_year) and _num(r.passing_year) and \
                _num(r.start_year) > _num(r.passing_year):
            rep.add_record(key, "Start year is after the passing year.")
    ordered = [(lvl, next((_num(r.passing_year) for r in edu if r.level == lvl and r.passing_year), None))
               for lvl in ("ssc", "hsc", "graduation", "post_graduation")]
    known = [(lvl, y) for lvl, y in ordered if y]
    for (la, ya), (lb, yb) in zip(known, known[1:]):
        if ya >= yb:
            idx = next(i for i, r in enumerate(edu) if r.level == lb)
            rep.add_record(f"education:{idx}", f"{LEVEL_LABELS[lb]} year should be after {LEVEL_LABELS[la]} year.")
    for lvl in ("ssc", "hsc"):
        if edu and not any(r.level == lvl for r in edu):
            rep.notes.append(f"No {LEVEL_LABELS[lvl]} details found — left blank.")
    if not edu:
        rep.notes.append("No education details found.")
    for i, r in enumerate(exp):
        key = f"experience:{i}"
        for n in r.notes:
            if not n.startswith("Duration was calculated"):
                rep.add_record(key, n)
        sy, ey = _num(r.start_date), _num(r.end_date)
        if sy and ey and sy > ey:
            rep.add_record(key, "End date is before the start date.")
        if r.current and r.end_date:
            rep.add_record(key, "Marked as current but an end date is set.")
    if extraction:
        rep.notes.extend(n for n in extraction.notes if n not in rep.notes)
    elif not exp:
        rep.notes.append("No professional experience found.")
