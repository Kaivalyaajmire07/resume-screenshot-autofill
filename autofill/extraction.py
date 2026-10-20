"""Rule-based candidate field extraction with source evidence.

Design rules
* Never invent: a value is only ever copied (or lightly normalised) from OCR text.
* Unknown stays blank. Heuristic guesses are tagged low confidence so the UI flags them.
* Every extracted field records which screenshot and which OCR line it came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config, geo
from .sectioning import HEADING_RE as _HEADING_RE, clean as _clean_line, is_heading as _is_heading
from .fields import empty_fields
from .jobs import group_lines, join_jobs
from .ocr_engine import OcrLine

HIGH, MEDIUM, LOW = "high", "medium", "low"


@dataclass
class Evidence:
    source_index: int
    source_name: str
    text: str
    confidence: str  # high | medium | low
    ocr_score: float = 1.0

    def to_dict(self) -> dict:
        return {"source_index": self.source_index, "source_name": self.source_name,
                "text": self.text, "confidence": self.confidence, "ocr_score": round(self.ocr_score, 3)}


@dataclass
class ExtractionResult:
    values: dict[str, str] = field(default_factory=empty_fields)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    conflicts: dict[str, list[str]] = field(default_factory=dict)
    education: list = field(default_factory=list)   # list[EducationRecord]
    experience: list = field(default_factory=list)  # list[ExperienceRecord]
    notes: list[str] = field(default_factory=list)  # e.g. 'No professional experience found.'

    def evidence_dict(self) -> dict[str, dict]:
        return {k: e.to_dict() for k, e in self.evidence.items()}


# ----------------------------------------------------------------------------- patterns
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<![\d.])(?:\+?\d[\d\s().-]{8,}\d)(?![\d])")
PIN_RE = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")
YEAR_RE = re.compile(r"(?<!\d)(?:19[5-9]\d|20[0-4]\d)(?!\d)")
PERCENT_RE = re.compile(r"(?<!\d)\d{1,3}(?:\.\d{1,2})?\s*%")
CGPA_RE = re.compile(r"(?:\b(?:CGPA|GPA|CPI|SGPA)\b\s*[:\-–]?\s*\d{1,2}(?:\.\d{1,2})?(?:\s*/\s*\d{1,2})?)"
                     r"|(?:(?<!\d)\d{1,2}(?:\.\d{1,2})?\s*/\s*(?:10|4)(?:\.0+)?(?!\d))"
                     r"|(?:(?<![\d.])\d{1,2}(?:\.\d{1,2})?\s*(?:CGPA|GPA|CPI|SGPA)\b)", re.I)
DATE_RE = re.compile(r"(?<!\d)(\d{1,2}[\s/.\-]\d{1,2}[\s/.\-]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?[\s\-]+"
                     r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*[\s,.\-]+\d{2,4}"
                     r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*[\s.]+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})",
                     re.I)

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir", "Ladakh",
    "Chandigarh", "Puducherry", "Andaman and Nicobar Islands", "Lakshadweep",
    "The Dadra and Nagar Haveli and Daman and Diu",
]

HEADING_RE = _HEADING_RE

NAME_BLOCK = re.compile(
    r"\b(resume|curriculum|vitae|cv|profile|contact|address|email|phone|mobile|objective|skills?|education|"
    r"experience|summary|declaration|details|personal|career|project|engineer|developer|analyst|student|"
    r"intern|manager|fresher|linkedin|github|portfolio|page|name|date|birth|gender|nationality)\b", re.I)

DEGREE_PATTERNS = {
    "ssc": re.compile(r"\b(?:S\.?\s?S\.?\s?C\.?|SSLC|10th|X(?:th)?\s+(?:std|standard)|class\s*(?:X|10)|"
                      r"matriculation|secondary\s+school(?:\s+certificate)?|high\s+school)\b", re.I),
    "hsc": re.compile(r"\b(?:H\.?\s?S\.?\s?C\.?|12th|XII|class\s*(?:XII|12)|higher\s+secondary|senior\s+secondary|"
                      r"intermediate|\+2|plus\s*two)\b", re.I),
    "post_graduation": re.compile(r"\b(?:M\.?\s?Tech|M\.?\s?E\.|M\.?\s?Sc|MBA|MCA|M\.?\s?Com|M\.?\s?A\.|M\.?\s?Pharm|"
                                  r"PGDM|Master(?:s)?\s+of|Post[\s-]?Graduat\w*)\b", re.I),
    "graduation": re.compile(r"\b(?:B\.?\s?Tech|B\.?\s?E\.|B\.?\s?Sc|B\.?\s?Com|BCA|BBA|B\.?\s?A\.|B\.?\s?Pharm|"
                             r"B\.?\s?Arch|Bachelor(?:s)?\s+of|Graduat\w*)\b", re.I),
    "higher": re.compile(r"\b(?:Ph\.?\s?D|Doctorate|Diploma|Post\s+Graduate\s+Diploma)\b", re.I),
}
BOARD_RE = re.compile(r"\b((?:[A-Z][A-Za-z.&]+\s+){0,4}(?:Board|CBSE|ICSE|ISC|NIOS)\b(?:\s+of\s+[A-Z][A-Za-z ]+)?)")
UNIVERSITY_RE = re.compile(r"[^,|]*\b(?:University|Institute|College|Autonomous|Vidyapeeth|IIT|NIT)\b[^,|]*", re.I)

LABELS = {
    "name": r"(?:full\s+|candidate\s+)?name",
    "date_of_birth": r"date\s+of\s+birth|d\.?\s?o\.?\s?b\.?|birth\s*date",
    "gender": r"gender|sex",
    "nationality": r"nationality|citizenship",
    "marital_status": r"marital\s+status|marital",
    "passport": r"passport(?:\s+(?:no\.?|number|status))?",
    "hobbies": r"hobbies(?:\s*(?:&|and)\s*interests)?|interests",
    "language_known": r"languages?(?:\s+known)?|language\s+proficiency",
    "address": r"(?:permanent\s+|current\s+|residential\s+|correspondence\s+)?address",
    "landmark": r"landmark",
    "city": r"city|town",
    "state": r"state",
    "pincode": r"pin\s*code|pincode|pin|zip(?:\s*code)?|postal\s+code",
    "mobile": r"mobile(?:\s+(?:no\.?|number))?|phone(?:\s+(?:no\.?|number))?|contact(?:\s+(?:no\.?|number))?|cell",
    "email": r"e-?mail(?:\s+(?:id|address))?",
}
MARITAL_VALUES = {"single", "married", "unmarried", "divorced", "widowed", "separated"}
GENDER_VALUES = {"male": "Male", "female": "Female", "m": "Male", "f": "Female", "other": "Other",
                 "transgender": "Transgender", "non-binary": "Non-binary"}


_clean = _clean_line


class _Extractor:
    def __init__(self, lines: list[OcrLine]):
        self.recs: list[OcrLine] = []
        for ln in lines:
            text = _clean(ln.text)
            if text:
                self.recs.append(OcrLine(text=text, score=ln.score, bbox=ln.bbox,
                                         source_index=ln.source_index, source_name=ln.source_name))
        self.texts = [r.text for r in self.recs]
        self.out = ExtractionResult()

    # ------------------------------------------------------------------ helpers
    def _set(self, key: str, value: str, idx: int | None, confidence: str, snippet: str | None = None,
             overwrite: bool = False) -> None:
        value = value.strip()
        if not value or (self.out.values.get(key) and not overwrite):
            return
        score = 1.0
        src_i, src_n = 0, ""
        if idx is not None and 0 <= idx < len(self.recs):
            rec = self.recs[idx]
            score, src_i, src_n = rec.score, rec.source_index, rec.source_name
            snippet = snippet or rec.text
            if score < config.OCR_REVIEW_SCORE and confidence == HIGH:
                confidence = MEDIUM
        self.out.values[key] = value
        self.out.evidence[key] = Evidence(src_i, src_n, snippet or "", confidence, score)

    def _label_value(self, pattern: str) -> tuple[int, str] | None:
        """Find 'Label: value' (or 'Label' followed by the value on the next OCR line)."""
        rx = re.compile(rf"^(?:{pattern})\b\s*[:\-–=]*\s*(.*)$", re.I)
        for i, text in enumerate(self.texts):
            m = rx.match(text)
            if not m:
                continue
            rest = m.group(1).strip(" :-–=")
            if rest:
                return i, rest
            if i + 1 < len(self.texts) and not self._looks_like_label(self.texts[i + 1]):
                return i + 1, self.texts[i + 1].strip(" :-–=")
        return None

    def _looks_like_label(self, text: str) -> bool:
        if _is_heading(text):
            return True
        return any(re.match(rf"^(?:{p})\b\s*[:\-–=]", text, re.I) for p in LABELS.values())

    def _section(self, start: str, max_lines: int = 60) -> tuple[int, list[str]]:
        rx = re.compile(rf"^\W*(?:{start})\W*$", re.I)
        for i, text in enumerate(self.texts):
            if rx.match(text) and len(text) < 45:
                body: list[str] = []
                for t in self.texts[i + 1:i + 1 + max_lines]:
                    if _is_heading(t):
                        break
                    body.append(t)
                return i, body
        return -1, []

    # ------------------------------------------------------------------ extractors
    def contacts(self) -> None:
        emails: list[tuple[int, str]] = []
        for i, text in enumerate(self.texts):
            for m in EMAIL_RE.finditer(text):
                emails.append((i, m.group(0).rstrip(".")))
        if emails:
            first_i, first = emails[0]
            self._set("email", first, first_i, HIGH)
            distinct = []
            for _, e in emails:
                if e.lower() not in [d.lower() for d in distinct]:
                    distinct.append(e)
            if len(distinct) > 1:
                self.out.conflicts["email"] = distinct

        mobiles: list[tuple[int, str, str]] = []
        for i, text in enumerate(self.texts):
            for m in PHONE_RE.finditer(text):
                raw = m.group(0).strip()
                digits = re.sub(r"\D", "", raw)
                if len(digits) < 10 or len(digits) > 13:
                    continue
                if YEAR_RE.fullmatch(digits[:4]) and re.search(r"\d{4}\s*[-–]\s*\d{4}", raw):
                    continue  # a year range, not a phone number
                national = digits[-10:] if (len(digits) in (11, 12, 13) and digits[:-10] in ("0", "91", "091")) else digits
                mobiles.append((i, raw, national))
        good = [(i, raw, n) for i, raw, n in mobiles if len(n) == 10 and n[0] in "6789"]
        pick = good[0] if good else (mobiles[0] if mobiles else None)
        if pick:
            i, raw, national = pick
            self._set("mobile", national if len(national) == 10 else raw, i, HIGH if good else LOW, snippet=raw)
            distinct = []
            for _, _, n in (good or mobiles):
                if n not in distinct:
                    distinct.append(n)
            if len(distinct) > 1:
                self.out.conflicts["mobile"] = distinct

    def pincode_and_address(self) -> None:
        lv = self._label_value(LABELS["address"])
        addr_idx, addr_end = None, 0
        if lv:
            addr_idx, value = lv
            # address may continue over the next OCR lines until a label/heading/contact line
            parts = [value]
            j = addr_idx + 1
            while j < len(self.texts) and len(parts) < 4:
                t = self.texts[j]
                if self._looks_like_label(t) or EMAIL_RE.search(t) or PHONE_RE.search(t):
                    break
                parts.append(t)
                j += 1
            addr_end = j
            self._set("address", ", ".join(p.strip(", ") for p in parts), addr_idx, HIGH if len(parts) == 1 else MEDIUM)
        else:
            rx = re.compile(r"\b(?:near|road|rd\.?|street|st\.|sector|extension|nagar|village|district|tal\.?|"
                            r"at\s+post|plot|flat|colony|lane)\b", re.I)
            for i, t in enumerate(self.texts[:20]):
                if rx.search(t) and PIN_RE.search(t):
                    self._set("address", t, i, MEDIUM)
                    addr_idx, addr_end = i, i + 1
                    break

        pins: list[tuple[int, str]] = []
        lv_pin = self._label_value(LABELS["pincode"])
        if lv_pin and PIN_RE.fullmatch(lv_pin[1].strip()):
            pins.append((lv_pin[0], lv_pin[1].strip()))
        scan = range(addr_idx, max(addr_end, addr_idx + 1)) if addr_idx is not None else []
        for i in scan:  # the pincode is often on the last line of a multi-line address
            pins += [(i, m.group(0)) for m in PIN_RE.finditer(self.texts[i])]
        if pins:
            self._set("pincode", pins[0][1], pins[0][0], HIGH if lv_pin else MEDIUM)
            distinct = list(dict.fromkeys(p for _, p in pins))
            if len(distinct) > 1:
                self.out.conflicts["pincode"] = distinct

        for key in ("landmark", "city", "state"):
            lv = self._label_value(LABELS[key])
            if lv and not self._looks_like_label(lv[1]) and len(lv[1]) < 80:
                self._set(key, lv[1], lv[0], HIGH)
        self._fill_from_pincode()
        if not self.out.values["state"] and self.out.values["address"]:
            found = [s for s in INDIAN_STATES if re.search(rf"\b{s}\b", self.out.values["address"], re.I)]
            if len(found) == 1:
                self._set("state", found[0], addr_idx, LOW, snippet=self.out.values["address"])

    def _fill_from_pincode(self) -> None:
        """City/state are rarely labelled on a resume, so derive blanks from the pincode."""
        pin = self.out.values["pincode"]
        info = geo.lookup(pin)
        pin_ev = self.out.evidence.get("pincode")
        if not info or not pin_ev:
            return
        note = f"Pincode {pin} → {info.city}, {info.state} (India Post data)"
        for key, value in (("city", info.city), ("state", info.state)):
            current = self.out.values[key]
            if not current:
                self.out.values[key] = value
                self.out.evidence[key] = Evidence(pin_ev.source_index, pin_ev.source_name, note,
                                                  MEDIUM, pin_ev.ocr_score)
            elif key == "state" and current.strip().lower() != value.lower():
                self.out.conflicts["state"] = [current, value]  # surfaced as "multiple values" warning

    def personal(self) -> None:
        # Name: label first, then a conservative heading heuristic.
        name_idx, name = None, ""
        lv = self._label_value(LABELS["name"])
        conf = HIGH
        if lv and self._valid_name(lv[1]):
            name_idx, name = lv
        else:
            conf = LOW
            for i, t in enumerate(self.texts[:15]):
                if self._valid_name(t):
                    name_idx, name = i, t
                    break
        if name:
            words = [w.title() if w.isupper() else w for w in name.split()]
            self._set("first_name", words[0], name_idx, conf, snippet=name)
            if len(words) > 2:
                self._set("middle_name", " ".join(words[1:-1]), name_idx, conf, snippet=name)
            self._set("last_name", words[-1], name_idx, conf, snippet=name)

        lv = self._label_value(LABELS["date_of_birth"])
        if lv:
            m = DATE_RE.search(lv[1])
            if m:
                self._set("date_of_birth", m.group(0).strip(), lv[0], HIGH, snippet=lv[1])
        lv = self._label_value(LABELS["gender"])
        if lv:
            token = re.split(r"[\s,/|]", lv[1].strip().lower())[0]
            if token in GENDER_VALUES:
                self._set("gender", GENDER_VALUES[token], lv[0], HIGH, snippet=lv[1])
        lv = self._label_value(LABELS["marital_status"])
        if lv:
            token = re.split(r"[\s,/|]", lv[1].strip().lower())[0]
            if token in MARITAL_VALUES:
                self._set("marital_status", token.title(), lv[0], HIGH, snippet=lv[1])
        for key in ("nationality", "passport"):
            lv = self._label_value(LABELS[key])
            if lv and len(lv[1]) < 60 and not self._looks_like_label(lv[1]):
                self._set(key, lv[1], lv[0], HIGH)
        for key in ("hobbies", "language_known"):
            lv = self._label_value(LABELS[key])
            if lv and not self._looks_like_label(lv[1]):
                self._set(key, lv[1], lv[0], HIGH)
            else:  # section form: heading followed by lines
                head = {"hobbies": r"hobbies|interests", "language_known": r"languages?(?:\s+known)?"}[key]
                i, body = self._section(head, max_lines=6)
                if body:
                    self._set(key, ", ".join(b.strip(", ") for b in body), i + 1, MEDIUM, snippet=" ".join(body))

    def _valid_name(self, text: str) -> bool:
        if re.search(r"\d|@|[/\\:]", text) or NAME_BLOCK.search(text):
            return False
        words = text.split()
        return 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Za-z][A-Za-z.'’-]*", w) for w in words) \
            and all(len(w.strip(".")) >= 1 for w in words)

    def structured(self, lines: list[OcrLine], today) -> None:
        """Education + experience as separate records; legacy flat fields are derived from them."""
        from .education import extract_education
        from .experience import extract_experience
        from .records import education_to_flat

        edu = extract_education(lines, today)
        exp, notes = extract_experience(lines, today)
        self.out.education, self.out.experience, self.out.notes = edu, exp, notes
        for key, value in education_to_flat(edu).items():
            if value:
                rec = next((r for r in edu if key.startswith(r.level) or
                            (key == "higher_level_education" and r.level in ("diploma", "higher"))), edu[0])
                self.out.values[key] = value
                self.out.evidence[key] = Evidence(rec.source_index, "", rec.source_text[:140], rec.confidence,
                                                  0.5 if rec.confidence == "low" and any("OCR" in n for n in rec.notes)
                                                  else 1.0)
        if exp:
            from .jobs import join_jobs
            self.out.values["employment_details"] = join_jobs([r.as_text() for r in exp])
            self.out.evidence["employment_details"] = Evidence(exp[0].source_index, "", exp[0].source_text[:140],
                                                                exp[0].confidence, 1.0)

    def sections(self) -> None:
        for key, start in (("objectives", r"(?:career\s+)?objectives?"),
                           ("summary", r"(?:professional\s+)?summary|profile(?:\s+summary)?|about\s+me"),
                           ("employment_details", r"(?:work|professional)\s+experience|employment(?:\s+(?:history|details))?|"
                                                  r"experience|internships?")):
            i, body = self._section(start)
            if body and key == "employment_details":
                jobs = ["\n".join(g) for g in group_lines(body)]
                self._set(key, join_jobs(jobs), i + 1, LOW if len(jobs) > 1 else MEDIUM, snippet=body[0])
            elif body:
                self._set(key, "\n".join(body), i + 1, MEDIUM, snippet=body[0])
        for key, start in (("certifications", r"certifications?"), ("projects", r"(?:academic\s+)?projects?"),
                           ("academic_achievements", r"(?:academic\s+)?achievements?|awards?")):
            i, body = self._section(start)
            if body:
                self._set(key, "\n".join(body), i + 1, MEDIUM, snippet=body[0])
        i, body = self._section(r"(?:key|technical|core|professional)?\s*skills?(?:\s+summary)?")
        if body:
            self._set("key_skills", "\n".join(f"• {b.lstrip('•-*· ')}" for b in body), i + 1, MEDIUM, snippet=body[0])


def extract(lines: list[OcrLine], today=None) -> ExtractionResult:
    """Extract candidate fields from OCR lines (reading order, across all screenshots)."""
    ex = _Extractor(lines)
    ex.contacts()
    ex.personal()
    ex.pincode_and_address()
    ex.sections()
    ex.structured(lines, today)
    return ex.out


def lines_from_text(text: str) -> list[OcrLine]:
    """Rebuild OcrLines from stored raw_text (``--- Screenshot N: name ---`` separators)."""
    lines: list[OcrLine] = []
    idx, name = 1, ""
    for raw in text.splitlines():
        m = re.match(r"^--- Screenshot (\d+): (.*) ---$", raw.strip())
        if m:
            idx, name = int(m.group(1)), m.group(2)
        elif raw.strip():
            lines.append(OcrLine(text=raw, score=1.0, source_index=idx, source_name=name))
    return lines


def parse_fields(text: str) -> dict[str, str]:
    """Backward-compatible entry point used by the original app."""
    return extract(lines_from_text(text)).values
