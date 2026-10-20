"""Attachment queue -> validation/preprocessing -> OCR -> extraction -> validation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

from . import config, pdfs
from .cleanup import clean_lines, split_embedded_newlines
from .extraction import ExtractionResult, extract
from .imaging import ImageError, image_hash, load_image, preprocess
from .overlap import COVERED_RATIO, OverlapTracker, fingerprint
from .ocr_engine import OcrEngine, OcrError, OcrLine, filter_lines, get_engine
from .validation import ValidationReport, validate


class Attachment(Protocol):
    name: str
    data: bytes


@dataclass
class PipelineResult:
    raw_text: str = ""
    lines: list[OcrLine] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extraction: ExtractionResult = field(default_factory=ExtractionResult)
    report: ValidationReport = field(default_factory=ValidationReport)
    seconds: float = 0.0
    images_ok: int = 0
    image_seconds: list[tuple[str, float]] = field(default_factory=list)


def _read_pdf(att: Attachment, engine: OcrEngine) -> tuple[list[OcrLine], list[str]]:
    """Text-based PDF pages are read directly (no OCR); scanned pages are rendered and OCR'd."""
    pages, total = pdfs.read_pages(att.data)
    notes: list[str] = []
    if total > len(pages):
        notes.append(f"{att.name}: read the first {len(pages)} of {total} pages.")
    lines: list[OcrLine] = []
    for page in pages:
        if page.image is None:
            lines.extend(OcrLine(text=t.strip()) for t in page.text.splitlines() if t.strip())
        else:
            found = engine.recognize(preprocess(page.image), cache_key=f"{image_hash(att.data)}:p{page.number}")
            lines.extend(filter_lines(found)[0])
    return lines, notes


def _ocr_one(att: Attachment, idx: int, engine: OcrEngine, prepared: tuple | None = None
             ) -> tuple[list[OcrLine], list[str], float, bool]:
    """OCR a single attachment. Never raises: problems come back as warnings.

    ``prepared`` is (image, seen_share, error) from the overlap pass: rows already read in an
    earlier screenshot are blanked so they cost no OCR time.
    """
    t0 = time.perf_counter()
    warnings: list[str] = []
    lines: list[OcrLine] = []
    covered = False
    try:
        if prepared is not None and prepared[2] is not None:
            raise prepared[2]
        if prepared is not None:
            image, seen_share, _ = prepared
            if seen_share >= COVERED_RATIO:
                covered = True
                warnings.append(f"{att.name}: already covered by earlier screenshot(s) — skipped.")
                dropped = 0
            else:
                if seen_share > 0:
                    warnings.append(f"{att.name}: {round(seen_share * 100)}% of it was already read in an "
                                    "earlier screenshot, so only the new part was scanned.")
                key = fingerprint(image) if seen_share > 0 else image_hash(att.data)
                lines = engine.recognize(preprocess(image), cache_key=key)
                lines, dropped = filter_lines(lines)
        elif pdfs.is_pdf(att.data):
            lines, pdf_notes = _read_pdf(att, engine)
            warnings.extend(pdf_notes)
            dropped = 0
        else:
            image = load_image(att.data)
            lines = engine.recognize(preprocess(image), cache_key=image_hash(att.data))
            lines, dropped = filter_lines(lines)
        if dropped:
            warnings.append(f"{att.name}: ignored {dropped} unreadable low-confidence text fragment(s).")
        if not lines and not covered:
            warnings.append(f"{att.name}: OCR returned no readable text.")
    except (ImageError, pdfs.PdfError) as exc:
        warnings.append(f"{att.name}: skipped ({exc}).")
    except OcrError as exc:
        warnings.append(f"{att.name}: OCR engine error — {exc}")
    except Exception as exc:  # keep the batch going; surface the reason
        warnings.append(f"{att.name}: processing failed ({type(exc).__name__}: {exc}).")
    return ([replace(ln, source_index=idx, source_name=att.name) for ln in lines], warnings,
            time.perf_counter() - t0, bool(lines) or covered)


def run_ocr(attachments: list[Attachment], engine: OcrEngine | None = None,
            progress: Callable[[int, int], None] | None = None) -> tuple[list[OcrLine], list[str], int, list[tuple[str, float]]]:
    """OCR every attachment independently (in parallel when the engine allows it).

    One bad image never aborts the batch; results are always returned in upload order.
    """
    engine = engine or get_engine()
    total = len(attachments)
    results: dict[int, tuple[list[OcrLine], list[str], float, bool]] = {}
    workers = min(config.OCR_WORKERS, total)
    # Overlap pass (fast, in upload order): blank rows already seen in an earlier screenshot.
    tracker = OverlapTracker()
    prepared: dict[int, tuple] = {}
    for idx, att in enumerate(attachments, start=1):
        if pdfs.is_pdf(att.data):
            continue
        try:
            image, share = tracker.process(load_image(att.data))
            prepared[idx] = (image, share, None)
        except Exception as exc:  # reported per attachment by _ocr_one
            prepared[idx] = (None, 0.0, exc)
    if workers > 1 and engine.parallel_safe:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_ocr_one, att, i, engine, prepared.get(i)): i for i, att in enumerate(attachments, start=1)}
            for done, fut in enumerate(as_completed(futures), start=1):
                results[futures[fut]] = fut.result()
                if progress:
                    progress(done, total)
    else:
        for idx, att in enumerate(attachments, start=1):
            results[idx] = _ocr_one(att, idx, engine, prepared.get(idx))
            if progress:
                progress(idx, total)
    all_lines: list[OcrLine] = []
    warnings: list[str] = []
    timings: list[tuple[str, float]] = []
    ok = 0
    for idx, att in enumerate(attachments, start=1):
        lines, warns, secs, read = results[idx]
        all_lines.extend(lines)
        warnings.extend(warns)
        timings.append((att.name, secs))
        ok += bool(read)
    return all_lines, warnings, ok, timings


def lines_to_text(lines: list[OcrLine], attachments: list[Attachment]) -> str:
    """Raw text in the same ``--- Screenshot N: name ---`` layout the original app stored."""
    chunks = []
    for idx, att in enumerate(attachments, start=1):
        body = "\n".join(ln.text for ln in lines if ln.source_index == idx)
        chunks.append(f"--- Screenshot {idx}: {att.name} ---\n{body}")
    return "\n\n".join(chunks)


def process(attachments: list[Attachment], engine: OcrEngine | None = None,
            progress: Callable[[int, int], None] | None = None) -> PipelineResult:
    started = time.perf_counter()
    if len(attachments) > config.MAX_UPLOADS:
        raise ValueError(f"At most {config.MAX_UPLOADS} screenshots per run.")
    lines, warnings, ok, timings = run_ocr(attachments, engine, progress)
    lines, removed = clean_lines(split_embedded_newlines(lines))
    if removed:
        warnings.append(f"Removed {removed} repeated or non-resume line(s) (overlapping screenshots, buttons).")
    extraction = extract(lines)
    report = validate(extraction.values, extraction)
    return PipelineResult(raw_text=lines_to_text(lines, attachments), lines=lines, warnings=warnings,
                          extraction=extraction, report=report, seconds=time.perf_counter() - started,
                          images_ok=ok, image_seconds=timings)
