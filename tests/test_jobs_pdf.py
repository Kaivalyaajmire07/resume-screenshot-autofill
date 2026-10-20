from io import BytesIO

import pytest

from autofill import pdfs
from autofill.attachments import AttachmentQueue
from autofill.jobs import parse_job, split_jobs
from autofill.pipeline import process
from autofill.structure import jobs_table, lines_table, sections_table
from autofill.ocr_engine import OcrLine

BLOB = ("Airtel Broadband (Kirti Nagar)\nWorked as a Telecaller\n1.5 Year\n"
        "Gyaanjyoti Education (Vaishali)\nWorked as a Telecounsellor\n2Year")


def test_jobs_without_date_ranges_split_on_role_lines():
    jobs = split_jobs(BLOB)
    assert len(jobs) == 2 and jobs[0].startswith("Airtel") and jobs[1].startswith("Gyaanjyoti")


def test_jobs_split_on_lone_durations_when_no_role_lines():
    assert len(split_jobs("Acme\nSales exec\n1 Year\nBeta\nCaller\n6 months")) == 2


def test_parse_job_columns():
    assert parse_job(split_jobs(BLOB)[1]) == {"Company": "Gyaanjyoti Education (Vaishali)", "Role": "Telecounsellor",
                                              "Duration": "2Year", "Details": ""}
    assert [r["Job"] for r in jobs_table(BLOB)] == [1, 2]


def test_sections_and_lines_tables():
    lines = [OcrLine(t, 0.9, source_index=1) for t in ["Anita", "Professional Experience", "Acme", "Skills", "SQL"]]
    assert [r["Section"] for r in sections_table(lines)] == ["Header / Contact", "Professional Experience", "Skills"]
    assert lines_table(lines)[0]["Confidence %"] == 90


def _pdf(lines):
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    buf = BytesIO()
    c = reportlab.Canvas(buf)
    y = 800
    for t in lines:
        c.drawString(60, y, t)
        y -= 22
    c.save()
    return buf.getvalue()


def test_text_pdf_is_read_without_ocr():
    pytest.importorskip("pypdfium2")
    data = _pdf(["Simran Saxena", "simran@example.com | +918077571727", "Professional Experience", BLOB.replace("\n", " | ")[:0] or "Acme Ltd"])
    q = AttachmentQueue()
    assert q.add("cv.pdf", data) == (True, "") and q.items[0].is_pdf

    class NoOcr:  # would raise if OCR were used
        def recognize(self, *a, **k):
            raise AssertionError("OCR must not run for a text PDF")

    result = process(list(q.items), NoOcr())
    assert result.extraction.values["email"] == "simran@example.com" and result.images_ok == 1


def test_broken_pdf_is_rejected():
    pytest.importorskip("pypdfium2")
    ok, msg = AttachmentQueue().add("bad.pdf", b"%PDF-1.4 junk")
    assert not ok and "PDF" in msg
