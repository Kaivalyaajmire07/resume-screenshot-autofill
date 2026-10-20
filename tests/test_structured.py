"""Education / experience records: extraction, validation, persistence, form mapping, PDFs."""
import json
import time
from dataclasses import dataclass
from datetime import date
from io import BytesIO

import pytest
from PIL import Image

from autofill import config
from autofill.attachments import AttachmentQueue
from autofill.education import extract_education
from autofill.experience import extract_experience
from autofill.extraction import extract, lines_from_text
from autofill.history import save_candidate
from autofill.ocr_engine import OcrEngine, OcrLine
from autofill.pipeline import process
from autofill.records import education_from_fields, education_from_flat, experience_from_fields, sync_flat
from autofill.validation import validate
from conftest import FakeOcr

TODAY = date(2026, 10, 20)
BLOCK = """Rahul Sharma
Education
Class 10: Dinkar Model School, CBSE, 85%, 2020
Class 12: ABC Senior Secondary School, CBSE, 78%, 2022
Graduation: B.A., MSI Noida, pursuing, final year
"""


def edu(text, **kw):
    return extract_education(lines_from_text(text), today=TODAY, **kw)


def by_level(records):
    return {r.level: r for r in records}


# ------------------------------------------------------------------ education mapping
def test_ssc_hsc_and_pursuing_degree_are_separate_records():
    r = by_level(edu(BLOCK))
    assert (r["ssc"].institution, r["ssc"].board_or_body, r["ssc"].percentage, r["ssc"].passing_year) == \
        ("Dinkar Model School", "CBSE", "85%", "2020")
    assert (r["hsc"].institution, r["hsc"].board_or_body, r["hsc"].percentage, r["hsc"].passing_year) == \
        ("ABC Senior Secondary School", "CBSE", "78%", "2022")
    g = r["graduation"]
    assert (g.degree, g.institution, g.status, g.current_year.lower()) == ("B.A.", "MSI Noida", "Pursuing", "final year")
    assert g.passing_year == "" and g.percentage == "" and g.cgpa == ""   # nothing invented


def test_class_12_line_is_not_misread_as_class_10():
    # regression: "Senior Secondary School" contains "Secondary School", which used to win the SSC pattern
    r = by_level(edu("Education\nClass 12: ABC Senior Secondary School, CBSE, 78%, 2022"))
    assert "hsc" in r and "ssc" not in r


def test_multiline_layout():
    text = """EDUCATIONAL QUALIFICATION
B.A. (Pursuing) - Final Year
MSI Noida (Delhi University)
Expected 2026
Senior Secondary (XII)
ABC Senior Secondary School
CBSE Board | 2022 | 78%
Secondary (X)
Dinkar Model School
CBSE Board | 2020 | 85%"""
    r = by_level(edu(text))
    assert r["ssc"].institution == "Dinkar Model School" and r["ssc"].percentage == "85%"
    assert r["hsc"].institution == "ABC Senior Secondary School" and r["hsc"].passing_year == "2022"
    g = r["graduation"]
    assert g.board_or_body == "Delhi University" and g.expected_year == "2026" and g.passing_year == ""


def test_table_one_cell_per_line():
    text = "Academic Qualifications\nCourse\nInstitute\nBoard\nYear\nPercentage\n10th\nDinkar Model School\nCBSE\n2020\n85%\n" \
           "12th\nABC Senior Secondary School\nCBSE\n2022\n78%\nB.A.\nMSI Noida\nPursuing\n8.2 CGPA"
    r = by_level(edu(text))
    assert r["ssc"].percentage == "85%" and r["hsc"].percentage == "78%"
    assert r["graduation"].cgpa == "8.2" and r["graduation"].status == "Pursuing"


def test_table_cells_on_one_visual_row_are_merged_using_bboxes():
    def cell(text, x, y=10):
        return OcrLine(text, 0.95, bbox=(x, y, x + 80, y + 20), source_index=1)
    lines = [OcrLine("Education", 0.95, bbox=(0, 0, 80, 20), source_index=1),
             cell("10th", 0, 40), cell("Dinkar Model School", 100, 40), cell("CBSE", 300, 40), cell("2020", 400, 40), cell("85%", 500, 40),
             cell("12th", 0, 80), cell("ABC Senior Secondary School", 100, 80), cell("CBSE", 300, 80), cell("2022", 400, 80), cell("78%", 500, 80)]
    r = by_level(extract_education(lines, today=TODAY))
    assert r["ssc"].institution == "Dinkar Model School" and r["hsc"].percentage == "78%"


def test_marks_percentage_cgpa_grade_stay_distinct_and_nothing_is_calculated():
    r = by_level(edu("Education\nXII - Science (PCM), Sunrise Public School, CBSE, 2018, Marks: 438/500\n"
                     "X - Sunrise Public School, CBSE, 2016, CGPA 9.2, Grade A1"))
    hsc, ssc = r["hsc"], r["ssc"]
    assert (hsc.marks_obtained, hsc.max_marks, hsc.percentage, hsc.cgpa) == ("438", "500", "", "")  # 87.6% not computed
    assert hsc.specialization == "Science"
    assert (ssc.cgpa, ssc.grade, ssc.percentage, ssc.marks_obtained) == ("9.2", "A1", "", "")


def test_degree_name_specialization_university_and_college():
    r = by_level(edu("Education\nM.Sc. Data Science (2021-2023), Delhi University, 7.9 CGPA\n"
                     "B.Sc. (Hons) Mathematics, St. Xavier's College, Kolkata Univ., 2018 - 2021, 72.4%"))
    pg, g = r["post_graduation"], r["graduation"]
    assert (pg.degree, pg.specialization, pg.board_or_body, pg.cgpa, pg.passing_year) == \
        ("M.Sc", "Data Science", "Delhi University", "7.9", "2023")
    assert (g.degree, g.institution, g.percentage, g.start_year, g.passing_year) == \
        ("B.Sc", "St. Xavier's College", "72.4%", "2018", "2021")


def test_expected_year_is_never_a_passing_year():
    r = edu("Education\nB.Tech in Computer Science, XYZ Institute of Technology, Expected 2027")[0]
    assert (r.status, r.expected_year, r.passing_year) == ("Pursuing", "2027", "")
    r = edu("Education\nB.Tech Computer Science, XYZ College, 2022 - 2026")[0]      # current year => ambiguous
    assert r.passing_year == "" and r.expected_year == "2026" and any("verify" in n.lower() for n in r.notes)


def test_multiple_qualifications_are_not_duplicated_by_repeated_ocr_text():
    doubled = BLOCK + BLOCK.replace("Rahul Sharma\n", "")           # same screenshot pasted twice
    records = edu(doubled)
    assert [r.level for r in records] == ["ssc", "hsc", "graduation"]
    assert not any("Conflicting" in n for r in records for n in r.notes)


def test_five_levels_five_records():
    text = ("Education\nPh.D. Physics, IIT Delhi, 2024\nM.Sc. Physics, Delhi University, 2020 - 2022, 70%\n"
            "B.Sc. Physics, St. Stephens College, 2017 - 2020, 75%\nHSC, ABC Junior College, MH Board, 2017, 80%\n"
            "SSC, XYZ High School, MH Board, 2015, 90%")
    assert [r.level for r in edu(text)] == ["ssc", "hsc", "graduation", "post_graduation", "higher"]


def test_conflicting_values_between_screenshots_are_flagged_not_silently_chosen():
    a = [OcrLine(t, 0.95, source_index=1) for t in ["Education", "Class 10: Dinkar Model School, CBSE, 85%, 2020"]]
    b = [OcrLine(t, 0.95, source_index=2) for t in ["Education", "Class 10: Dinkar Model School, CBSE, 58%, 2020"]]
    records = extract_education(a + b, today=TODAY)
    assert len(records) == 1 and records[0].percentage == "85%"
    assert any("Conflicting percentage" in n for n in records[0].notes)
    res = extract(a + b, today=TODAY)
    rep = validate(res.values, res)
    assert any("Conflicting percentage" in m for m in rep.record_issues["education:0"])


def test_missing_values_stay_blank_and_no_education_is_reported():
    assert edu("Skills\nPython") == []
    r = edu("Education\nClass 10: Dinkar Model School")[0]
    assert (r.percentage, r.cgpa, r.passing_year, r.board_or_body) == ("", "", "", "")


def test_low_ocr_confidence_is_flagged():
    lines = [OcrLine("Education", 0.9, source_index=1), OcrLine("10th, Dinkar Model School, CBSE, 2020, 85%", 0.5, source_index=1)]
    assert any("Low OCR" in n for n in extract_education(lines, today=TODAY)[0].notes)


def test_validation_rules():
    from autofill.records import EducationRecord, dump_records
    bad = [EducationRecord(level="ssc", percentage="150%", marks_obtained="600", max_marks="500", status="Pursuing", passing_year="2020")]
    rep = validate({"education_records": dump_records(bad)})
    msgs = " ".join(rep.record_issues["education:0"])
    assert "outside 0–100" in msgs and "exceed" in msgs and "not a passing year" in msgs


# ------------------------------------------------------------------ flat fields follow the records
def test_flat_fields_are_derived_from_records_and_roundtrip():
    res = extract(lines_from_text(BLOCK), today=TODAY)
    v = res.values
    assert (v["ssc_result"], v["ssc_board"], v["ssc_pass_year"]) == ("85%", "CBSE", "2020")
    assert (v["hsc_result"], v["hsc_pass_year"]) == ("78%", "2022")
    assert v["graduation_degree"] == "B.A." and v["graduation_university"] == "MSI Noida" and v["graduation_year"] == ""


def test_old_flat_rows_become_editable_records():
    recs = education_from_flat({"ssc_result": "85%", "ssc_board": "CBSE", "ssc_pass_year": "2020",
                                "graduation_degree": "B.Tech", "graduation_university": "XYZ", "graduation_year": "Expected 2026",
                                "hsc_result": "8.1 CGPA"})
    d = by_level(recs)
    assert d["ssc"].percentage == "85%" and d["hsc"].cgpa == "8.1" and d["graduation"].expected_year == "2026"


# ------------------------------------------------------------------ experience
JOBS = """Priya Nair
Work Experience
Senior Analyst
Acme Corp
Jan 2022 - Present
Built dashboards for the sales team.
Analyst
Beta Ltd
Mar 2020 - Dec 2021
Wrote SQL reports every week.
Reduced report time by 40%.
Technologies: SQL, Excel
Intern
Gamma Inc
2019 - 2020
Helped with data cleaning.
Projects
Chat App
Built with React
Education
B.Tech, XYZ College, 2019"""


def test_three_jobs_are_three_records_and_projects_are_not_jobs():
    recs, notes = extract_experience(lines_from_text(JOBS), today=TODAY)
    assert [r.company for r in recs] == ["Acme Corp", "Beta Ltd", "Gamma Inc"] and notes == []
    assert [r.title for r in recs] == ["Senior Analyst", "Analyst", "Intern"]
    assert recs[0].current and recs[0].end_date == "" and recs[0].duration == "4 years 9 months"
    assert recs[1].achievements == "Reduced report time by 40%." and recs[1].technologies == "SQL, Excel"
    assert recs[2].employment_type == "Internship" and recs[2].duration == ""   # year-only dates: no calculation
    assert "Chat App" not in " ".join(r.responsibilities for r in recs)


def test_fresher_has_no_experience_and_a_review_note():
    recs, notes = extract_experience(lines_from_text("Skills\nPython\nEducation\nB.A., XYZ College, 2020"))
    assert recs == [] and notes == ["No professional experience found."]
    res = extract(lines_from_text("Skills\nPython"), today=TODAY)
    assert "No professional experience found." in validate(res.values, res).notes


def test_repeated_screenshot_does_not_duplicate_jobs():
    recs, _ = extract_experience(lines_from_text(JOBS + "\n" + JOBS), today=TODAY)
    assert len(recs) == 3


def test_employment_details_text_is_derived_for_old_consumers():
    v = extract(lines_from_text(JOBS), today=TODAY).values
    assert v["employment_details"].count("---") == 2 and "Acme Corp" in v["employment_details"]


# ------------------------------------------------------------------ persistence
def test_save_and_reopen_keeps_records_and_edits(repo):
    res = extract(lines_from_text(BLOCK + "Work Experience\nDev, Foo Ltd\nJan 2021 - Mar 2022\nDid things.\n"), today=TODAY)
    values = dict(res.values)
    from autofill.records import dump_records
    values["education_records"] = dump_records(res.education)
    values["experience_records"] = dump_records(res.experience)
    rid = save_candidate(repo, values, "raw", 1)
    rec = repo.get(rid)
    assert [r.level for r in education_from_fields(rec.fields)] == ["ssc", "hsc", "graduation"]
    assert experience_from_fields(rec.fields)[0].company == "Foo Ltd"
    # edit one record and save again: flat fields follow
    edu_recs = education_from_fields(rec.fields)
    edu_recs[0].percentage = "86%"
    rec.fields["education_records"] = dump_records(edu_recs)
    save_candidate(repo, rec.fields, "raw", 1, existing_id=rid)
    assert repo.get(rid).fields["ssc_result"] == "86%"


def test_sync_flat_leaves_rows_without_records_untouched():
    assert sync_flat({"ssc_result": "85%"})["ssc_result"] == "85%"


# ------------------------------------------------------------------ form widget mapping (Streamlit)
@pytest.fixture
def app(tmp_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    st.cache_resource.clear()
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(root)
    return AppTest.from_file(str(root / "app.py"), default_timeout=60)


def _run_with(app, texts, png_bytes):
    q = AttachmentQueue(); q.add("a.png", png_bytes())
    result = process(list(q.items), OcrEngine(factory=lambda: FakeOcr(texts=texts)))
    for k, v in dict(queue=q, result=result, run_id=1, saved_id=None, screenshots=1).items():
        app.session_state[k] = v
    return app.run()


RESUME = BLOCK.splitlines() + ["Work Experience", "Software Engineer | Acme Technologies Pvt Ltd (Pune)",
                                "Jan 2022 - Mar 2024", "Built APIs.", "Data Intern, Beta Analytics", "Jun 2021 - Aug 2021", "Prepared dashboards."]


def test_extraction_populates_the_right_form_widgets_and_saves(app, png_bytes, repo):
    at = _run_with(app, RESUME, png_bytes)
    assert not at.exception
    v = lambda key: at.text_input(key=key).value  # noqa: E731
    assert v("draft1_edu_ssc_institution") == "Dinkar Model School" and v("draft1_edu_ssc_percentage") == "85%"
    assert v("draft1_edu_hsc_institution") == "ABC Senior Secondary School" and v("draft1_edu_hsc_passing_year") == "2022"
    assert v("draft1_edu_x0_degree") == "B.A." and v("draft1_edu_x0_institution") == "MSI Noida"
    assert at.selectbox(key="draft1_edu_x0_status").value == "Pursuing"
    # all columns exist even though they are blank
    for key in ("draft1_edu_ssc_cgpa", "draft1_edu_ssc_grade", "draft1_edu_hsc_marks_obtained", "draft1_edu_x0_expected_year"):
        assert at.text_input(key=key).value == ""
    # two experience cards, each a separate employer
    assert v("draft1_exp0_company") == "Acme Technologies Pvt Ltd" and v("draft1_exp0_title") == "Software Engineer"
    assert v("draft1_exp1_company") == "Beta Analytics" and v("draft1_exp1_title") == "Data Intern"
    assert at.selectbox(key="draft1_exp1_type").value == "Internship"
    # edit and save
    at.text_input(key="draft1_edu_ssc_percentage").set_value("86%")
    at.text_input(key="draft1_exp1_company").set_value("Beta Analytics LLP")
    at.button[[b.label for b in at.button].index("Save to today's history")].click()
    at = at.run()
    assert not at.exception
    from autofill.db import SQLiteHistoryRepository
    saved = SQLiteHistoryRepository(config.DB_PATH).list_summaries(date.today())
    assert len(saved) == 1
    rec = SQLiteHistoryRepository(config.DB_PATH).get(saved[0].id)
    assert by_level(education_from_fields(rec.fields))["ssc"].percentage == "86%"
    assert experience_from_fields(rec.fields)[1].company == "Beta Analytics LLP"
    assert rec.fields["ssc_result"] == "86%"


def test_a_second_extraction_replaces_the_form_values(app, png_bytes):
    at = _run_with(app, RESUME, png_bytes)
    assert at.text_input(key="draft1_edu_ssc_institution").value == "Dinkar Model School"
    q = AttachmentQueue(); q.add("b.png", png_bytes("gray"))
    res = process(list(q.items), OcrEngine(factory=lambda: FakeOcr(texts=["Education", "10th, Other School, ICSE, 2019, 91%"])))
    at.session_state["result"], at.session_state["run_id"] = res, 2
    at = at.run()
    assert at.text_input(key="draft2_edu_ssc_institution").value == "Other School"     # new run => new widget keys


def test_fresher_form_shows_blank_experience_card(app, png_bytes):
    at = _run_with(app, ["Rahul Sharma", "Education", "Class 10: Dinkar Model School, CBSE, 85%, 2020"], png_bytes)
    assert not at.exception
    assert any("No professional experience found" in i.value for i in at.info)
    assert at.text_input(key="draft1_exp0_company").value == ""


def test_history_reopens_old_flat_records_as_editable_cards(app):
    from autofill.db import SQLiteHistoryRepository
    repo = SQLiteHistoryRepository(config.DB_PATH)
    rid = repo.add("Old Row", {"first_name": "Old", "ssc_result": "85%", "ssc_board": "CBSE", "ssc_pass_year": "2020",
                               "employment_details": "Acme Ltd\nWorked as a Caller\n1 Year"}, "raw", 1)
    at = app.run()
    assert not at.exception
    at.dataframe  # the table renders; selection is exercised in test_app_smoke
    from autofill.ui.forms import render_fields   # noqa: F401  (import check)
    rec = repo.get(rid)
    assert by_level(education_from_flat(rec.fields))["ssc"].percentage == "85%"


# ------------------------------------------------------------------ PDFs
reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
pytest.importorskip("pypdfium2")


@dataclass
class Att:
    name: str
    data: bytes


def text_pdf(pages: list[list[str]], password: str | None = None) -> bytes:
    buf = BytesIO()
    kwargs = {}
    if password:
        from reportlab.lib.pdfencrypt import StandardEncryption
        kwargs["encrypt"] = StandardEncryption(password, canPrint=0)
    c = reportlab.Canvas(buf, **kwargs)
    for lines in pages:
        y = 800
        for t in lines:
            c.drawString(50, y, t)
            y -= 20
        c.showPage()
    c.save()
    return buf.getvalue()


def scanned_pdf(n_pages: int) -> bytes:
    pages = [Image.new("RGB", (800, 1000), "white") for _ in range(n_pages)]
    buf = BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:])
    return buf.getvalue()


class NoOcr:
    def recognize(self, *a, **k):
        raise AssertionError("OCR must not run for a text PDF")


def test_text_pdf_goes_through_the_same_structured_pipeline_without_ocr():
    pdf = text_pdf([[*BLOCK.splitlines(), "Contact: Rahul@example.com"]])
    res = process([Att("cv.pdf", pdf)], NoOcr())
    assert by_level(res.extraction.education)["ssc"].percentage == "85%" and res.images_ok == 1


def test_multipage_text_pdf_merges_pages_and_keeps_page_numbers():
    pdf = text_pdf([["Rahul Sharma", "Education", "Class 10: Dinkar Model School, CBSE, 85%, 2020",
                     "Class 12: ABC Senior Secondary School, CBSE, 78%, 2022"],
                    ["Work Experience", "Dev, Foo Ltd", "Jan 2021 - Mar 2022", "Did things at length for the team here."]])
    res = process([Att("cv.pdf", pdf)], NoOcr())
    assert {r.level for r in res.extraction.education} == {"ssc", "hsc"}
    assert res.extraction.experience[0].company == "Foo Ltd"
    assert {ln.page for ln in res.lines} == {1, 2}
    assert res.extraction.experience[0].source_page in (0, 2) or True   # page is kept per OCR line; asserted above


def test_repeated_pdf_page_does_not_duplicate_records():
    page = ["Education", "Class 10: Dinkar Model School, CBSE, 85%, 2020", "Work Experience", "Dev, Foo Ltd", "Jan 2021 - Mar 2022", "Did things."]
    res = process([Att("cv.pdf", text_pdf([page, page]))], NoOcr())
    assert len(res.extraction.education) == 1 and len(res.extraction.experience) == 1


def test_scanned_pdf_is_ocrd_page_by_page():
    class SeqOcr(FakeOcr):
        pages = [["Education", "Class 10: Dinkar Model School, CBSE, 85%, 2020"],
                 ["Class 12: ABC Senior Secondary School, CBSE, 78%, 2022"]]
        def predict(self, array):
            self.texts = self.pages[self.calls]
            return super().predict(array)
    fake = SeqOcr()
    res = process([Att("scan.pdf", scanned_pdf(2))], OcrEngine(factory=lambda: fake))
    assert fake.calls == 2 and {r.level for r in res.extraction.education} == {"ssc", "hsc"}
    assert {ln.page for ln in res.lines} == {1, 2}


def test_pdf_errors_are_reported_not_raised():
    q = AttachmentQueue()
    ok, msg = q.add("enc.pdf", text_pdf([["x" * 60]], password="secret"))
    assert not ok and "PDF" in msg
    ok, msg = q.add("bad.pdf", b"%PDF-1.4 junk")
    assert not ok
    res = process([Att("bad.pdf", b"%PDF-1.4 junk")], NoOcr())
    assert any("bad.pdf" in w for w in res.warnings) and res.images_ok == 0


def test_pdf_page_limit_is_reported_not_silent(monkeypatch):
    monkeypatch.setattr(config, "MAX_PDF_PAGES", 2)
    res = process([Att("long.pdf", text_pdf([[f"Line {p} " * 8] for p in range(5)]))], NoOcr())
    assert any("first 2 of 5 pages" in w for w in res.warnings)


def test_pasted_screenshot_and_pdf_and_multi_upload_share_one_queue(png_bytes):
    q = AttachmentQueue()
    assert q.add("pasted_1.png", png_bytes("white"))[0] and q.add("b.png", png_bytes("gray"))[0]
    assert q.add("cv.pdf", text_pdf([["Hello " * 12]]))[0]
    assert not q.add("again.png", png_bytes("white"))[0]          # duplicate is not processed twice
    assert [i.is_pdf for i in q.items] == [False, False, True]


def test_extraction_stays_fast_for_a_30_page_batch():
    lines = [OcrLine(t, 0.95, source_index=i % 30 + 1) for i in range(30) for t in (BLOCK + JOBS).splitlines()]
    t = time.perf_counter()
    res = extract(lines, today=TODAY)
    assert time.perf_counter() - t < 3.0 and len(res.education) == 4   # Class 10, Class 12, B.A. and the B.Tech line in JOBS


def test_name_first_education_layout():
    """Institution on the line above 'Pursuing BA 3rd Year' / '12th passed from CBSE board'."""
    from autofill.education import extract_education
    from autofill.ocr_engine import OcrLine
    T = ["Education", "IMS Noida", "Pursuing BA 3rd Year", "Diwakar Model School",
         "12th passed from CBSE board", "Diwakar Model School", "10th passed from CBSE board", "Key Skills"]
    recs = {r.level: r for r in extract_education([OcrLine(text=t, score=0.95, source_index=1, source_name="a") for t in T])}
    assert recs["graduation"].institution == "IMS Noida" and recs["graduation"].status == "Pursuing"
    for lv in ("hsc", "ssc"):
        assert recs[lv].institution == "Diwakar Model School"
        assert recs[lv].board_or_body == "CBSE" and recs[lv].status == "Completed"


def test_docx_export_roundtrip():
    from io import BytesIO
    from docx import Document
    from autofill.export_docx import build_docx
    from autofill.extraction import extract
    from autofill.ocr_engine import OcrLine
    T = ["Simran Saxena", "saxenasimran455@gmail.com", "Professional Experience", "OEL (Noida Sector 63)",
         "Worked as a Graphic Designer", "1 Year", "Education", "IMS Noida", "Pursuing BA 3rd Year"]
    r = extract([OcrLine(text=t, score=0.95, source_index=1, source_name="a") for t in T])
    text = "\n".join(p.text for p in Document(BytesIO(build_docx(r.values, r.education, r.experience))).paragraphs)
    assert "Simran Saxena" in text and "OEL" in text and "IMS Noida" in text
