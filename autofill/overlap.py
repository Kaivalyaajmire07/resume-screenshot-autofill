"""Skip OCR work that has already been done: blank the rows of a screenshot that were already
visible in an earlier screenshot (scrolled / overlapping captures of the same page).

OCR cost grows with the amount of text, so reading six overlapping screenshots costs about six
times one page. Rows are compared by a small content signature; a long run of already-seen rows
is painted with the background colour, so the OCR detector finds no text there and does not
spend time on it. Nothing is guessed: if rows do not match exactly, nothing is blanked and the
screenshot is read in full.
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

SIG_COLUMNS = 128      # each row is shrunk to this many columns before comparing
MIN_INK_ROWS = 40      # a repeated run must contain at least this many rows with text/graphics
MIN_CONTRAST = 40      # a row is "ink" when its brightest and darkest pixels differ by this much
COVERED_RATIO = 0.95   # screenshot counts as fully covered when this share of ink rows was seen


def _row_signatures(img: Image.Image) -> tuple[list[bytes | None], np.ndarray]:
    gray = img.convert("L")
    full = np.asarray(gray)
    ink = (full.max(axis=1).astype(int) - full.min(axis=1).astype(int)) >= MIN_CONTRAST
    small = np.asarray(gray.resize((SIG_COLUMNS, gray.height), Image.Resampling.BOX)) >> 4
    sigs = [small[i].tobytes() if ink[i] else None for i in range(gray.height)]
    return sigs, ink


def _background(img: Image.Image) -> tuple[int, int, int]:
    arr = np.asarray(img.convert("RGB"))
    return tuple(int(v) for v in np.median(arr.reshape(-1, 3), axis=0))  # type: ignore[return-value]


class OverlapTracker:
    """Feed screenshots in upload order; each call returns (image_to_ocr, share_already_seen)."""

    def __init__(self) -> None:
        self._seen: set[bytes] = set()

    def process(self, img: Image.Image) -> tuple[Image.Image, float]:
        sigs, ink = _row_signatures(img)
        total_ink = int(ink.sum())
        flagged = [s is not None and s in self._seen for s in sigs]
        self._seen.update(s for s in sigs if s is not None)

        bands: list[tuple[int, int]] = []
        start = last = None
        count = 0
        for i, s in enumerate(sigs):
            if flagged[i]:
                if start is None:
                    start = i
                last, count = i, count + 1
            elif s is not None:  # ink row not seen before: the run ends here
                if start is not None and count >= MIN_INK_ROWS:
                    bands.append((start, last))
                start, last, count = None, None, 0
        if start is not None and count >= MIN_INK_ROWS:
            bands.append((start, last))

        covered_rows = sum(int(ink[a:b + 1].sum()) for a, b in bands)
        if not bands or total_ink == 0:
            return img, 0.0
        out = img.convert("RGB").copy()
        fill = _background(out)
        for a, b in bands:
            out.paste(fill, (0, a, out.width, b + 1))
        return out, covered_rows / total_ink


def fingerprint(img: Image.Image) -> str:
    """Cache key for an image after blanking (the OCR result depends on what was blanked)."""
    return hashlib.sha256(np.asarray(img.convert("RGB")).tobytes()).hexdigest()
