"""Optional read-only natural-language questions about today's saved candidates.

Deliberately rule-based: no LLM, no network, and no SQL is ever generated from user text.
Questions are mapped onto the repository's read methods, so the feature cannot modify data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .db import HistoryRepository, HistorySummary
from .fields import LABELS

_FIELD_WORDS = {"email": "email", "mail": "email", "mobile": "mobile", "phone": "mobile", "number": "mobile",
                "address": "address", "city": "city", "state": "state", "pincode": "pincode",
                "skills": "key_skills", "dob": "date_of_birth", "birthday": "date_of_birth",
                "gender": "gender", "languages": "language_known"}


@dataclass
class QueryAnswer:
    message: str
    rows: list[dict] = field(default_factory=list)


def _row(s: HistorySummary) -> dict:
    return {"ID": s.id, "Candidate": s.candidate_name, "Mobile": s.fields.get("mobile", ""),
            "Email": s.fields.get("email", ""), "Processed": s.processed_at.replace("T", " ")}


def answer(repo: HistoryRepository, question: str, today: date | None = None) -> QueryAnswer:
    today = today or date.today()
    q = question.strip()
    if not q:
        return QueryAnswer("Ask something like “how many candidates today?”, “email of Anita Rao” or "
                           "“who knows Python?”.")
    low = q.lower()
    items = repo.list_summaries(today)

    if re.search(r"\b(how many|count|total|number of)\b", low) and not re.search(r"\bwith|know|skill", low):
        return QueryAnswer(f"{len(items)} candidate record(s) saved today.")

    m = re.search(r"\b(email|mail|mobile|phone|number|address|city|state|pincode|dob|birthday|gender)\b"
                  r"(?:\s+id|\s+number)?\s+(?:of|for)\s+(.+)$", low)
    if m:
        key = _FIELD_WORDS[m.group(1)]
        target = m.group(2).strip(" ?.")
        hits = [s for s in items if target in s.candidate_name.lower()]
        if not hits:
            return QueryAnswer(f"No candidate matching “{target}” was saved today.")
        return QueryAnswer(f"{LABELS[key]} for {len(hits)} match(es):",
                           [{"Candidate": s.candidate_name, LABELS[key]: s.fields.get(key, "") or "(blank)"}
                            for s in hits])

    m = re.search(r"\b(?:who\s+(?:knows?|has|have)|with|skilled\s+in|skills?\s*(?:in|:)?|knows?)\s+(.+)$", low)
    if m:
        term = m.group(1).strip(" ?.")
        hits = [s for s in items if term in " ".join([s.fields.get("key_skills", ""),
                                                      s.fields.get("employment_details", "")]).lower()]
        return QueryAnswer(f"{len(hits)} candidate(s) mention “{term}”.", [_row(s) for s in hits])

    m = re.search(r"\b(?:from|in|living in|located in)\s+([a-z ]+)$", low)
    if m:
        term = m.group(1).strip()
        hits = [s for s in items if term in " ".join([s.fields.get("city", ""), s.fields.get("state", ""),
                                                      s.fields.get("address", "")]).lower()]
        return QueryAnswer(f"{len(hits)} candidate(s) with “{term}” in their location.", [_row(s) for s in hits])

    hits = [s for s in items if low.strip(" ?.") in s.candidate_name.lower()]
    if hits:
        return QueryAnswer(f"{len(hits)} candidate(s) matching the name.", [_row(s) for s in hits])
    return QueryAnswer("I couldn't map that to a supported question. Try a name, “how many candidates”, "
                       "“email of <name>”, “who knows <skill>” or “candidates from <city>”.")
