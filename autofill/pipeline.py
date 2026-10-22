"""Attachment queue -> validation/preprocessing -> OCR -> extraction -> validation."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

from . import config, pdfs
from .extraction import ExtractionResult, extract
from .imaging import ImageError, image_hash, load_image, preprocess
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
            lines.extend(OcrLine(text=t.strip(), page=page.number) for t in page.text.splitlines() if t.strip())
        else:
            found = engine.recognize(preprocess(page.image), cache_key=f"{image_hash(att.data)}:p{page.number}")
            lines.extend(replace(ln, page=page.number) for ln in filter_lines(found)[0])
    return lines, notes


def _ocr_one(att: Attachment, idx: int, engine: OcrEngine):
    """OCR a single attachment. Never raises: returns (lines, warnings, ok, seconds)."""
    t0 = time.perf_counter()
    warnings: list[str] = []
    lines: list[OcrLine] = []
    try:
        if pdfs.is_pdf(att.data):
            lines, pdf_notes = _read_pdf(att, engine)
            warnings.extend(pdf_notes)
            dropped = 0
        else:
            image = load_image(att.data)
            lines = engine.recognize(preprocess(image), cache_key=image_hash(att.data))
            lines, dropped = filter_lines(lines)
        if dropped:
            warnings.append(f"{att.name}: ignored {dropped} unreadable low-confidence text fragment(s).")
        if not lines:
            warnings.append(f"{att.name}: OCR returned no readable text.")
        lines = [replace(ln, source_index=idx, source_name=att.name) for ln in lines]
    except (ImageError, pdfs.PdfError) as exc:
        warnings.append(f"{att.name}: skipped ({exc}).")
        lines = []
    except OcrError as exc:
        warnings.append(f"{att.name}: OCR engine error — {exc}")
        lines = []
    except Exception as exc:  # keep the batch going; surface the reason
        warnings.append(f"{att.name}: processing failed ({type(exc).__name__}: {exc}).")
        lines = []
    return lines, warnings, bool(lines), time.perf_counter() - t0


def run_ocr(attachments: list[Attachment], engine: OcrEngine | None = None,
            progress: Callable[[int, int], None] | None = None) -> tuple[list[OcrLine], list[str], int, list[tuple[str, float]]]:
    """OCR each attachment independently (in parallel when the backend allows): one bad image never aborts the batch."""
    engine = engine or get_engine()
    total = len(attachments)
    workers = min(config.OCR_WORKERS, total)
    results: list = [None] * total
    done = 0
    if workers > 1:
        try:
            parallel = engine.parallel_ok
        except Exception:
            parallel = False
    else:
        parallel = False
    if parallel:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_ocr_one, att, i, engine): i - 1 for i, att in enumerate(attachments, start=1)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                done += 1
                if progress:
                    progress(done, total)
    else:
        for i, att in enumerate(attachments, start=1):
            results[i - 1] = _ocr_one(att, i, engine)
            if progress:
                progress(i, total)
    all_lines: list[OcrLine] = []
    warnings: list[str] = []
    ok = 0
    timings: list[tuple[str, float]] = []
    for att, (lines, warns, good, secs) in zip(attachments, results):  # original order preserved
        all_lines.extend(lines)
        warnings.extend(warns)
        ok += int(good)
        timings.append((att.name, secs))
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
    extraction = extract(lines)
    report = validate(extraction.values, extraction)
    return PipelineResult(raw_text=lines_to_text(lines, attachments), lines=lines, warnings=warnings,
                          extraction=extraction, report=report, seconds=time.perf_counter() - started,
                          images_ok=ok, image_seconds=timings)
