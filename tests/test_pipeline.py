from dataclasses import dataclass

import pytest

from autofill import config
from autofill.ocr_engine import OcrEngine
from autofill.pipeline import process
from conftest import FakeOcr


@dataclass
class Att:
    name: str
    data: bytes


def test_end_to_end_with_fake_ocr(fake_engine, png_bytes):
    res = process([Att("a.png", png_bytes()), Att("b.png", png_bytes("gray"))], fake_engine)
    assert res.images_ok == 2 and res.extraction.values["email"] == "anita.rao@example.com"
    assert res.raw_text.startswith("--- Screenshot 1: a.png ---") and "--- Screenshot 2: b.png ---" in res.raw_text
    assert res.extraction.evidence["email"].source_name in ("a.png", "b.png")
    assert "mobile" not in res.report.issues


def test_bad_image_does_not_abort_batch(fake_engine, png_bytes):
    res = process([Att("junk.png", b"not an image"), Att("ok.png", png_bytes())], fake_engine)
    assert res.images_ok == 1 and any("junk.png: skipped" in w for w in res.warnings)
    assert res.extraction.values["first_name"] == "Anita"


def test_ocr_exception_is_reported_per_image(png_bytes):
    eng = OcrEngine(factory=lambda: FakeOcr(fail_times=1, fail_msg="boom"))
    res = process([Att("a.png", png_bytes()), Att("b.png", png_bytes("gray"))], eng)
    assert res.images_ok == 1 and any("a.png: processing failed" in w for w in res.warnings)


def test_empty_ocr_result_warns_and_stays_blank(png_bytes):
    eng = OcrEngine(factory=lambda: FakeOcr(texts=[" "]))
    res = process([Att("a.png", png_bytes())], eng)
    assert any("no readable text" in w for w in res.warnings)
    assert all(v == "" for v in res.extraction.values.values())


def test_low_confidence_lines_dropped(png_bytes):
    eng = OcrEngine(factory=lambda: FakeOcr(texts=["Email: a@b.com"], score=0.1))
    res = process([Att("a.png", png_bytes())], eng)
    assert res.extraction.values["email"] == "" and any("low-confidence" in w for w in res.warnings)


def test_batch_limit(fake_engine, png_bytes):
    with pytest.raises(ValueError):
        process([Att(f"{i}.png", png_bytes()) for i in range(config.MAX_UPLOADS + 1)], fake_engine)


def test_progress_callback(fake_engine, png_bytes):
    seen = []
    process([Att("a.png", png_bytes())], fake_engine, progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 1)]
