"""Rule-based candidate field extraction with source evidence.

Design rules
* Never invent: a value is only ever copied (or lightly normalised) from OCR text.
* Unknown stays blank. Heuristic guesses are tagged low confidence so the UI flags them.
* Every extracted field records which screenshot and which OCR line it came from.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import config, geo
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

HEADING_WORDS = (
    r"(?:career\s+)?objectives?|professional\s+summary|summary|profile(?:\s+summary)?|about\s+me|"
    r"(?:key|technical|core|professional)?\s*skills?(?:\s+summary)?|work\s+experience|"
    r"professional\s+experience|employment(?:\s+(?:history|details))?|experience|internships?|"
    r"education(?:al)?(?:\s+(?:qualifications?|details))?|academics?|qualifications?|projects?|"
    r"certifications?|achievements?|awards?|personal\s+(?:details|information|profile)|languages?(?:\s+known)?|"
    r"hobbies|interests?|declaration|references?|contact(?:\s+(?:details|information))?|"
    r"strengths?|extra[\s-]?curricular(?:\s+activities)?|training"
)
HEADING_RE = re.compile(rf"^\W*(?:{HEADING_WORDS})\W*$", re.I)

NAME_BLOCK = re.compile(
    r"\b(resume|curriculum|vitae|cv|profile|contact|address|email|phone|mobile|objective|skills?|education|"
    r"experience|summary|declaration|details|personal|career|project|engineer|developer|analyst|student|"
    r"intern|manager|fresher|linkedin|github|portfolio|page|name|date|birth|gender|nationality)\b", re.I)

DEGREE_PATTERNS = {
    "ssc": re.compile(r"\b(?:S\.?\s?S\.?\s?C\.?|SSLC|10th|X(?:th)?\s+(?:std|standard)|class\s*(?:X|10)|"
                      r"matriculation|secondary\s+school(?:\s+certificate)?|high\s+school)\b", re.I),
    "hsc": re.compile(r"\b(?:H\.?\s?S\.?\s?C\.?|12th|XII|class\s*(?:XII|12)|higher\s+secondary|senior\s+secondary|"
                      r"intermediate|\+2|plus\s*two)\b", re.I),
    "post_graduation": re.compile(r"\b(?:M\.?\s?Tech|M\.?\s?E\.|M\.?\s?Sc|MBA|MCA|M\.?\s?Com|M\.\s?A\.?|(?-i:MA)|M\.?\s?Pharm|"
                                  r"PGDM|Master(?:s)?\s+of|Post[\s-]?Graduat\w*)(?=\s|$|[,/(\-])", re.I),
    "graduation": re.compile(r"\b(?:B\.?\s?Tech|B\.?\s?E\.|B\.?\s?Sc|B\.?\s?Com|BCA|BBA|B\.\s?A\.?|(?-i:BA)|B\.?\s?Pharm|"
                             r"B\.?\s?Arch|Bachelor(?:s)?\s+of|Graduat\w*)(?=\s|$|[,/(\-])", re.I),
    "higher": re.compile(r"\b(?:Ph\.?\s?D|Doctorate|Diploma|Post\s+Graduate\s+Diploma)\b", re.I),
}
BOARD_RE = re.compile(r"\b((?:[A-Z][A-Za-z.&]+\s+){0,4}(?:Board|CBSE|ICSE|ISC|NIOS)\b(?:\s+of\s+[A-Z][A-Za-z ]+)?)")
UNIVERSITY_RE = re.compile(r"[^,|]*\b(?:University|Institute|College|School|Academy|Autonomous|Vidyapeeth|IIT|NIT)\b[^,|]*", re.I)
MARKS_RE = re.compile(r"(?<!\d)(\d{1,4}(?:,\d{3})?(?:\.\d{1,2})?)\s*/\s*(\d{1,4}(?:,\d{3})?(?:\.\d{1,2})?)(?!\d)")

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


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" \t•·▪●◦*-–—|")


NOT_INSTITUTION_RE = re.compile(
    r"^\W*\d+(?:\.\d+)?\s*(?:years?|yrs?|months?)\W*$|^\W*worked\s+as\b|^\W*role\s*:", re.I)


def _is_heading(line: str) -> bool:
    return len(line) < 45 and bool(HEADING_RE.match(line))


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

    def education(self) -> None:
        """Extract qualification-specific records plus legacy flat fields.

        Context is gathered around each qualification anchor, rather than assuming
        all relevant values occur after the anchor. This handles common resume
        layouts where the institution is printed immediately before the class/degree.
        """
        row_templates = [
            {"qualification": "SSC / Class 10"},
            {"qualification": "HSC / Class 12"},
            {"qualification": "Diploma"},
            {"qualification": "Graduation"},
            {"qualification": "Post-graduation"},
            {"qualification": "Higher education"},
        ]
        records = [dict(
            qualification=x["qualification"], degree="", institution="", board_awarding_body="",
            stream="", start_year="", passing_year="", expected_completion_year="", status="",
            marks_obtained="", maximum_marks="", percentage="", cgpa="", grade="", result=""
        ) for x in row_templates]

        # Limit education anchors to the education section when headings are present.
        edu_start = next((i for i, t in enumerate(self.texts)
                          if re.fullmatch(r"\W*(?:education|educational qualifications?|academic qualifications?|academics?)\W*", t, re.I)), -1)
        edu_end = len(self.texts)
        if edu_start >= 0:
            for j in range(edu_start + 1, len(self.texts)):
                if _is_heading(self.texts[j]) and not re.search(r"education|academic|qualification", self.texts[j], re.I):
                    edu_end = j
                    break
            search_range = range(edu_start + 1, edu_end)
        else:
            search_range = range(len(self.texts))

        def classify_education_line(text: str) -> str | None:
            # Prefer explicit qualification labels. Generic phrases such as
            # "Senior Secondary School" are often part of an institution name,
            # not a qualification record, and must not steal the HSC anchor.
            if re.search(r"\bpost[ -]?graduate diploma\b", text, re.I):
                return "higher"
            if re.search(r"\b(?:H\.?S\.?C\.?|12th|XII|class\s*12|intermediate|\+2|plus\s*two|higher secondary)\b", text, re.I):
                return "hsc"
            if re.search(r"\b(?:S\.?S\.?C\.?|SSLC|10th|class\s*10|class\s*X|matriculation|secondary school certificate)\b", text, re.I):
                return "ssc"
            if DEGREE_PATTERNS["post_graduation"].search(text):
                return "post_graduation"
            if DEGREE_PATTERNS["graduation"].search(text):
                return "graduation"
            if re.search(r"\bDiploma\b", text, re.I):
                return "diploma"
            if re.search(r"\b(?:Ph\.?D|Doctorate)\b", text, re.I):
                return "higher"
            # Generic level phrases count only when the line itself has qualification
            # evidence, not when it is simply an institution name.
            if re.search(r"\b(?:senior secondary|high school|secondary school)\b", text, re.I):
                if not re.search(r"\b(?:school|college|academy)\b", text, re.I) or (
                    YEAR_RE.search(text) or PERCENT_RE.search(text) or BOARD_RE.search(text)):
                    return "hsc" if re.search(r"senior secondary", text, re.I) else "ssc"
            return None

        anchors: list[tuple[int, str]] = []
        for i in search_range:
            text = self.texts[i]
            if _is_heading(text):
                continue
            kind = classify_education_line(text)
            if kind:
                anchors.append((i, kind))

        # Layout: many resumes print the institution on the line BEFORE the qualification
        # ("IMS Noida / Pursuing BA 3rd Year / Diwakar Model School / 12th passed ...").
        def plausible_institution(j: int) -> bool:
            if j < 0 or j >= len(self.texts):
                return False
            t = self.texts[j]
            return not (_is_heading(t) or classify_education_line(t) is not None or NOT_INSTITUTION_RE.search(t)
                        or YEAR_RE.search(t) or PERCENT_RE.search(t) or CGPA_RE.search(t) or len(t) < 3
                        or re.fullmatch(r"\W*(?:CBSE|ICSE|ISC|NIOS)\W*", t, re.I))

        inst_first = False
        if anchors:
            first_i, last_i = anchors[0][0], anchors[-1][0]
            lead = first_i - 1 >= (edu_start + 1 if edu_start >= 0 else 0) and plausible_institution(first_i - 1)

            trail = last_i + 1 < edu_end and plausible_institution(last_i + 1)
            inst_first = lead and not trail
        claimed: set[int] = set()
        anchor_idx = [a[0] for a in anchors]

        def institution_above(idx: int) -> int | None:
            """Nearest institution line above an anchor, skipping stray durations / page counters."""
            for j in range(idx - 1, max(idx - 4, -1), -1):
                if j in anchor_idx or _is_heading(self.texts[j]):
                    return None
                if NOT_INSTITUTION_RE.search(self.texts[j]) or re.fullmatch(r"\W*\d{1,2}\W*", self.texts[j]):
                    continue
                return j if plausible_institution(j) else None
            return None

        # Include only one anchor per legacy level; table has one canonical row per level.
        used: set[str] = set()
        kind_to_record = {
            "ssc": 0, "hsc": 1, "diploma": 2, "graduation": 3,
            "post_graduation": 4, "higher": 5,
        }
        for pos, (idx, kind) in enumerate(anchors):
            if kind in used:
                continue
            # Nearby lines before and after the anchor can carry institution, board,
            # grade, and status. Stop at another qualification or section heading.
            prev_idx = idx - 1
            start_idx = idx
            if prev_idx >= 0 and not _is_heading(self.texts[prev_idx]):
                previous_line = self.texts[prev_idx]
                previous_is_institution = bool(re.search(r"\b(?:college|university|institute|vidyapeeth|IIT|NIT)\b", previous_line, re.I))
                # A school immediately before SSC/HSC usually belongs to that row;
                # a college immediately before HSC often belongs to the prior degree.
                if classify_education_line(previous_line) is None and (
                    not previous_is_institution or kind in ("graduation", "post_graduation", "diploma", "higher")
                ):
                    start_idx = prev_idx
            end_idx = min(len(self.texts), idx + 4)
            if inst_first:  # the line before the next qualification belongs to that qualification
                nxt = next((a for a in anchor_idx if a > idx), None)
                if nxt is not None:
                    nxt_inst = institution_above(nxt)
                    end_idx = min(end_idx, max(idx + 1, nxt_inst if nxt_inst is not None else nxt))
                above = institution_above(idx)
                start_idx = above if above is not None and above not in claimed else idx
            for j in range(idx + 1, end_idx):
                if _is_heading(self.texts[j]) or classify_education_line(self.texts[j]) is not None:
                    end_idx = j
                    break
            window = self.texts[start_idx:end_idx]
            blob = " | ".join(window)
            board_match = BOARD_RE.search(blob)
            percent_match = PERCENT_RE.search(blob)
            cgpa_match = CGPA_RE.search(blob)
            marks_match = MARKS_RE.search(blob)
            years = YEAR_RE.findall(blob)
            status_match = re.search(r"\b(pursuing|currently studying|in progress|appearing|final year|completed|graduated|passed|ongoing)\b", blob, re.I)
            status = ""
            if status_match:
                token = status_match.group(1).lower()
                status = "Pursuing" if token in {"pursuing", "currently studying", "in progress", "final year", "ongoing"} else (
                    "Appearing" if token == "appearing" else "Completed")

            # Prefer a named institution line; do not mistake board/grade/date lines for it.
            institution = ""
            institution_line_idx = None
            if inst_first and start_idx < idx:
                institution, institution_line_idx = self.texts[start_idx].strip(" ,|:-"), start_idx
                m = UNIVERSITY_RE.search(institution)
                if m and m.group(0).strip(" ,|:-"):
                    institution = m.group(0).strip(" ,|:-")
            for j in ([] if institution else range(start_idx, end_idx)):
                line = self.texts[j]
                if j in claimed or NOT_INSTITUTION_RE.search(line):
                    continue
                if board_match and board_match.group(1).lower() in line.lower() and not re.search(
                        r"school|college|university|institute", line, re.I):
                    continue
                match = UNIVERSITY_RE.search(line)
                if match and not re.search(r"\b(?:board|CBSE|ICSE|ISC|NIOS)\b", line, re.I):
                    candidate = match.group(0).strip(" ,|:-")
                    if candidate and not re.fullmatch(r"(?:school|college|university|institute)", candidate, re.I):
                        institution = candidate
                        institution_line_idx = j
                        break
            if not institution:
                # In a compact education row, the line next to the qualification may
                # be the institution even if its name has no "school/college" suffix.
                for j in range(start_idx, end_idx):
                    line = self.texts[j]
                    if j == idx or j in claimed or NOT_INSTITUTION_RE.search(line) or _is_heading(line) or re.search(
                        r"\b(?:CBSE|ICSE|ISC|NIOS|Board|CGPA|GPA|CPI|SGPA|percentage|marks?|result|pursuing|completed|appearing)\b",
                        line, re.I) or YEAR_RE.search(line) or PERCENT_RE.search(line) or CGPA_RE.search(line):
                        continue
                    if len(line) >= 3:
                        institution, institution_line_idx = line.strip(" ,|:-"), j
                        break

            if institution_line_idx is not None:
                claimed.add(institution_line_idx)
            rec = records[kind_to_record[kind]]
            degree_text = self.texts[idx].strip(" ,|:-")
            degree_text = re.sub(r"\s*[\(\[]?\b(?:19|20)\d{2}\b.*$", "", degree_text).strip(" ,|:-")
            degree_text = re.sub(r"\s+(?:final year|currently studying|pursuing|appearing)$", "", degree_text, flags=re.I).strip()
            degree_text = re.sub(r"^(?:pursuing|currently studying|appearing|completed)\s+", "", degree_text, flags=re.I)
            degree_text = re.sub(r"\s*(\d(?:st|nd|rd|th)\s+year)$", r" (\1)", degree_text, flags=re.I).strip()
            rec["degree"] = degree_text if kind in ("diploma", "graduation", "post_graduation", "higher") else ""
            rec["institution"] = institution
            board_value = board_match.group(1).strip() if board_match else ""
            board_value = re.sub(r"^(?:SSC|HSC|10th|12th|Class\s*(?:X|10|XII|12))\s+", "", board_value, flags=re.I)
            rec["board_awarding_body"] = board_value
            rec["stream"] = ""
            rec["status"] = status
            pursuing = bool(re.search(r"\b(pursuing|currently studying|in progress|final year|ongoing)\b", blob, re.I))
            if pursuing and years:
                rec["start_year"] = years[0] if len(years) >= 2 else ""
                rec["expected_completion_year"] = years[-1]
            elif years:
                rec["passing_year"] = years[-1]
                if len(years) >= 2:
                    rec["start_year"] = years[0]
            rec["marks_obtained"] = marks_match.group(1) if marks_match and not re.search(r"\b(?:CGPA|GPA|CPI|SGPA)\b", blob, re.I) else ""
            rec["maximum_marks"] = marks_match.group(2) if rec["marks_obtained"] else ""
            rec["percentage"] = percent_match.group(0).replace(" ", "") if percent_match else ""
            rec["cgpa"] = cgpa_match.group(0).strip() if cgpa_match else ""
            grade_match = re.search(r"\b(?:grade|class)\s*[:\-]?\s*([A-F][+-]?|Distinction|First Class|Second Class)\b", blob, re.I)
            rec["grade"] = grade_match.group(1) if grade_match else ""
            rec["result"] = rec["percentage"] or rec["cgpa"] or (
                f"{rec['marks_obtained']}/{rec['maximum_marks']}" if rec["marks_obtained"] else rec["grade"])
            used.add(kind)

            # Backward-compatible scalar fields used by existing forms and history.
            if kind in ("ssc", "hsc"):
                prefix = kind
                result_value = rec["result"]
                self._set(f"{prefix}_result", result_value, idx, MEDIUM, snippet=blob)
                self._set(f"{prefix}_board", rec["board_awarding_body"], idx, MEDIUM, snippet=blob)
                self._set(f"{prefix}_pass_year", rec["passing_year"] or rec["expected_completion_year"], idx, MEDIUM, snippet=blob)
            elif kind in ("graduation", "post_graduation"):
                self._set(f"{kind}_degree", rec["degree"], idx, MEDIUM, snippet=blob)
                self._set(f"{kind}_result", rec["result"], idx, MEDIUM, snippet=blob)
                self._set(f"{kind}_university", rec["institution"], institution_line_idx if institution_line_idx is not None else idx,
                          MEDIUM, snippet=blob)
                self._set(f"{kind}_year", rec["passing_year"] or rec["expected_completion_year"], idx, MEDIUM, snippet=blob)
            elif kind in ("higher", "diploma"):
                self._set("higher_level_education", rec["degree"] or blob, idx, MEDIUM, snippet=blob)

        # Keep structured rows for the UI/history without changing the existing DB schema:
        # fields are already stored as JSON and unknown/new keys remain backward compatible.
        if anchors:
            self._set("qualification_records", json.dumps(records, ensure_ascii=False), anchors[0][0],
                      MEDIUM, snippet="Structured qualification records extracted from OCR")
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
        i, body = self._section(r"(?:key|technical|core|professional)?\s*skills?(?:\s+summary)?")
        if body:
            self._set("key_skills", "\n".join(f"• {b.lstrip('•-*· ')}" for b in body), i + 1, MEDIUM, snippet=body[0])


def extract(lines: list[OcrLine]) -> ExtractionResult:
    """Extract candidate fields from OCR lines (reading order, across all screenshots)."""
    ex = _Extractor(lines)
    ex.contacts()
    ex.personal()
    ex.pincode_and_address()
    ex.education()
    ex.sections()
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
