from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from autofill.overlap import OverlapTracker
from autofill.ocr_engine import OcrEngine
from autofill.pipeline import run_ocr
from conftest import FakeOcr


@dataclass
class Att:
    name: str
    data: bytes


def page(lines=60, width=700, step=24):
    img = Image.new("RGB", (width, lines * step + 20), "white")
    d = ImageDraw.Draw(img)
    for i in range(lines):
        d.text((20, 10 + i * step), f"Section line {i} unique text number {i * 7919 % 1013}", fill="black")
        d.rectangle((20, 10 + i * step + 12, 20 + (i * 37) % 400, 10 + i * step + 14), fill="black")  # varies per line
    return img


def png(img):
    b = BytesIO(); img.save(b, format="PNG"); return b.getvalue()


def ink_rows(img):
    a = np.asarray(img.convert("L")).astype(int)
    return int(((a.max(1) - a.min(1)) >= 40).sum())


def test_overlapping_scroll_only_new_rows_are_kept():
    full = page()
    a, b = full.crop((0, 0, 700, 800)), full.crop((0, 480, 700, 1280))   # 320 px overlap
    t = OverlapTracker()
    out_a, share_a = t.process(a)
    out_b, share_b = t.process(b)
    assert share_a == 0 and ink_rows(out_a) == ink_rows(a)
    assert 0.3 < share_b < 0.6 and ink_rows(out_b) < ink_rows(b)
    # new part of b (below the overlap) is untouched
    assert np.array_equal(np.asarray(out_b)[330:], np.asarray(b)[330:])


def test_identical_screenshot_is_fully_covered_and_not_ocrd():
    img = page(20)
    eng = OcrEngine(factory=lambda: FakeOcr())
    fake = eng._get()
    lines, warns, ok, _ = run_ocr([Att("a.png", png(img)), Att("same.png", png(img))], eng)
    assert fake.calls == 1 and ok == 2 and any("already covered" in w for w in warns)


def test_unrelated_screenshots_are_untouched():
    t = OverlapTracker()
    t.process(page(20))
    other = Image.new("RGB", (700, 500), "white")
    ImageDraw.Draw(other).text((10, 10), "completely different", fill="black")
    out, share = t.process(other)
    assert share == 0 and np.array_equal(np.asarray(out), np.asarray(other))
