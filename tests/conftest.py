import os
import sys
from pathlib import Path

import pytest
from PIL import Image

os.environ["RESUME_NO_WARMUP"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autofill.db import SQLiteHistoryRepository  # noqa: E402
from autofill.ocr_engine import OcrEngine, OcrLine  # noqa: E402

RESUME_LINES = [
    "ANITA RAO", "Mobile: +91 98765 43210", "Email: anita.rao@example.com",
    "Address: Plot 12, Shivaji Nagar, Near City Mall, Pune, Maharashtra 411005",
    "Date of Birth: 12/05/1998", "Gender: Female", "Marital Status: Single", "Nationality: Indian",
    "Languages Known: English, Hindi, Marathi",
    "Career Objective", "To obtain a challenging data analyst role.",
    "Education",
    "B.Tech in Artificial Intelligence & Data Science 2022 - 2026",
    "P.R. Pote Patil College of Engineering, Amravati 8.3 CGPA",
    "HSC Maharashtra State Board 2021 78.5%",
    "SSC Maharashtra State Board 2019 88.20%",
    "Key Skills", "Python", "SQL", "Power BI",
    "Work Experience", "Data Intern, Acme Analytics, 2025",
]


class FakeOcr:
    """Stands in for PaddleOCR: returns canned PaddleOCR-3.x style results."""
    def __init__(self, texts=None, score=0.97, fail_times=0, fail_msg=""):
        self.texts, self.score, self.calls = texts or RESUME_LINES, score, 0
        self.fail_times, self.fail_msg = fail_times, fail_msg

    def predict(self, array):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise NotImplementedError(self.fail_msg)
        polys = [[[0, i * 20], [100, i * 20], [100, i * 20 + 15], [0, i * 20 + 15]] for i in range(len(self.texts))]
        return [{"res": {"rec_texts": self.texts, "rec_scores": [self.score] * len(self.texts), "rec_polys": polys}}]


@pytest.fixture
def repo(tmp_path):
    return SQLiteHistoryRepository(tmp_path / "t.db")


@pytest.fixture
def png_bytes():
    from io import BytesIO
    def make(color="white", size=(400, 300)):
        buf = BytesIO(); Image.new("RGB", size, color).save(buf, format="PNG"); return buf.getvalue()
    return make


@pytest.fixture
def fake_engine():
    return OcrEngine(factory=lambda: FakeOcr())


@pytest.fixture
def resume_lines():
    return [OcrLine(text=t, score=0.97, source_index=1 if i < 10 else 2, source_name="s.png")
            for i, t in enumerate(RESUME_LINES)]
