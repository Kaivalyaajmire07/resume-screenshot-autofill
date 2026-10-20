"""Clean merged OCR lines: drop UI noise and lines repeated by overlapping screenshots."""
from __future__ import annotations

import re
from dataclasses import replace

from .ocr_engine import OcrLine

# Website/UI chrome that is not resume content ("Resume in PDF", "Skip", "4 of 8", ...).
NOISE_RE = re.compile(
    r"^\W*(?:skip|resume\s+(?:in|ih|1n)\s+pdf|download(?:\s+resume)?|next|previous|back|"
    r"\d{1,3}\s*(?:of|/)\s*\d{1,3}|n?\d{1,2}\s*of\s*\d{1,2})\W*$", re.I)


def _norm(text: str) -> str:
    """Letters and digits only, so '1.5 Year' and '1.5Year' count as the same line."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def is_noise(text: str) -> bool:
    t = text.strip()
    return bool(NOISE_RE.match(t)) or bool(re.fullmatch(r"\W*\d{1,2}\W*", t))  # page counters like "4"


def clean_lines(lines: list[OcrLine]) -> tuple[list[OcrLine], int]:
    """Remove UI noise, and lines already seen in an EARLIER screenshot (scroll overlap).

    A repeat is dropped when it is long enough to be distinctive (12+ chars) or sits next to
    another repeat, so a short legitimate line such as "1 Year" in a new screenshot survives.
    Repeats inside the same screenshot are never touched.
    """
    kept: list[OcrLine] = []
    seen: set[str] = set()
    dropped = 0
    by_shot: dict[int, list[OcrLine]] = {}
    order: list[int] = []
    for ln in lines:
        if ln.source_index not in by_shot:
            by_shot[ln.source_index] = []
            order.append(ln.source_index)
        by_shot[ln.source_index].append(ln)
    for idx in order:
        shot = [ln for ln in by_shot[idx] if not is_noise(ln.text)]
        dropped += len(by_shot[idx]) - len(shot)
        norms = [_norm(ln.text) for ln in shot]
        repeat = [bool(n) and n in seen for n in norms]
        for i, ln in enumerate(shot):
            neighbour = (i > 0 and repeat[i - 1]) or (i + 1 < len(shot) and repeat[i + 1])
            if repeat[i] and (len(norms[i]) >= 10 or neighbour):
                dropped += 1
                continue
            kept.append(ln)
        seen.update(n for n in norms if n)
    return kept, dropped


def split_embedded_newlines(lines: list[OcrLine]) -> list[OcrLine]:
    """Some pages OCR as one line holding literal '\\n' text; turn those into real lines."""
    out: list[OcrLine] = []
    for ln in lines:
        if "\\n" in ln.text:
            parts = [p.strip() for p in ln.text.split("\\n") if p.strip()]
            out.extend(replace(ln, text=p) for p in parts)
        else:
            out.append(ln)
    return out
