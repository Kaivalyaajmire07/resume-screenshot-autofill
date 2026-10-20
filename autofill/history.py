"""History service: save/update/search on top of a repository."""
from __future__ import annotations

from datetime import date

from .db import HistoryRepository, HistorySummary
from .fields import normalize_fields
from .geo import fill_location
from .validation import ValidationReport, validate


def candidate_name(fields: dict) -> str:
    parts = [fields.get(k, "") for k in ("first_name", "middle_name", "last_name")]
    return " ".join(p.strip() for p in parts if p and p.strip()) or "Unknown Candidate"


def save_candidate(repo: HistoryRepository, fields: dict, raw_text: str, source_count: int,
                   evidence: dict | None = None, report: ValidationReport | None = None,
                   instruction: str = "", existing_id: int | None = None) -> int:
    """Insert a new history entry, or update ``existing_id`` (prevents duplicate saves of one draft)."""
    clean = fill_location(normalize_fields(fields))  # blank city/state <- pincode
    name = candidate_name(clean)
    review = (report or validate(clean)).to_dict()
    if existing_id is not None and repo.update_fields(existing_id, name, clean, review):
        return existing_id
    return repo.add(name, clean, raw_text, source_count, evidence=evidence, review=review,
                    instruction=instruction)


def search_today(repo: HistoryRepository, query: str, today: date | None = None) -> list[HistorySummary]:
    today = today or date.today()
    if not query.strip():
        return repo.list_summaries(today)
    return repo.search_name(today, query)
