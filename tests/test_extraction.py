from autofill.extraction import extract, lines_from_text, parse_fields
from autofill.ocr_engine import OcrLine


def test_contacts_and_identity(resume_lines):
    r = extract(resume_lines).values
    assert (r["first_name"], r["last_name"]) == ("Anita", "Rao")
    assert r["email"] == "anita.rao@example.com"
    assert r["mobile"] == "9876543210"          # +91 prefix and spaces normalised away
    assert r["date_of_birth"] == "12/05/1998"
    assert r["gender"] == "Female" and r["marital_status"] == "Single" and r["nationality"] == "Indian"
    assert r["language_known"] == "English, Hindi, Marathi"


def test_address_pincode_state(resume_lines):
    res = extract(resume_lines)
    assert res.values["pincode"] == "411005"
    assert "Shivaji Nagar" in res.values["address"]
    assert res.values["state"] == "Maharashtra" and res.values["city"] == "Pune"
    assert "411005" in res.evidence["city"].text           # derived from the pincode, shows its source


def test_education(resume_lines):
    r = extract(resume_lines).values
    assert r["graduation_degree"].startswith("B.Tech")
    assert r["graduation_result"].lower().startswith(("8.3", "cgpa")) and "8.3" in r["graduation_result"]
    assert r["graduation_year"] == "2026"
    assert r["hsc_pass_year"] == "2021" and r["hsc_result"] == "78.5%" and "Maharashtra State Board" in r["hsc_board"]
    assert r["ssc_pass_year"] == "2019" and r["ssc_result"] == "88.20%"


def test_sections(resume_lines):
    r = extract(resume_lines).values
    assert r["objectives"] == "To obtain a challenging data analyst role."
    assert r["key_skills"] == "• Python\n• SQL\n• Power BI"
    assert "Acme Analytics" in r["employment_details"]


def test_unknown_stays_blank():
    res = extract([OcrLine("Lorem ipsum dolor sit amet", 0.95, source_index=1)])
    assert all(v == "" for v in res.values.values())
    assert res.evidence == {}


def test_multiple_emails_and_phones_are_conflicts():
    lines = [OcrLine(t, 0.95, source_index=1) for t in
             ["Name: Ravi Kumar", "ravi@a.com", "ravi.k@b.org", "9876543210", "9123456780"]]
    res = extract(lines)
    assert res.values["email"] == "ravi@a.com" and res.conflicts["email"] == ["ravi@a.com", "ravi.k@b.org"]
    assert res.conflicts["mobile"] == ["9876543210", "9123456780"]


def test_year_range_is_not_a_phone_number():
    res = extract([OcrLine("2019 - 2023 2023 - 2025", 0.95, source_index=1)])
    assert res.values["mobile"] == ""


def test_name_heuristic_is_low_confidence_and_skips_headings():
    res = extract([OcrLine(t, 0.95, source_index=1) for t in ["Curriculum Vitae", "Priya Sharma", "Skills"]])
    assert res.values["first_name"] == "Priya"
    assert res.evidence["first_name"].confidence == "low"


def test_label_value_on_next_line():
    res = extract([OcrLine(t, 0.95, source_index=1) for t in ["Date of Birth", "03 Feb 1999", "Gender", "Male"]])
    assert res.values["date_of_birth"] == "03 Feb 1999" and res.values["gender"] == "Male"


def test_low_ocr_score_downgrades_confidence():
    res = extract([OcrLine("Email: a@b.com", 0.55, source_index=1)])
    assert res.evidence["email"].confidence == "medium" and res.evidence["email"].ocr_score == 0.55


def test_evidence_points_to_source_screenshot(resume_lines):
    res = extract(resume_lines)
    assert res.evidence["email"].source_index == 1
    assert res.evidence["hsc_pass_year"].source_index == 2


def test_backward_compatible_parse_fields():
    text = "--- Screenshot 1: a.png ---\nANITA RAO\nEmail: anita@x.com\n"
    assert parse_fields(text)["email"] == "anita@x.com"
    assert lines_from_text(text)[0].source_name == "a.png"



def test_structured_qualification_records_keep_levels_and_values_separate():
    import json

    lines = [OcrLine(t, 0.97, source_index=1, source_name="resume.png") for t in [
        "Education",
        "Dinkar Model School",
        "Class 10 / SSC CBSE 85% 2020",
        "ABC Senior Secondary School",
        "Class 12 HSC CBSE 78% 2022",
        "B.A. Final Year",
        "MSI Noida",
        "Pursuing 2026",
    ]]
    result = extract(lines)
    rows = {row["qualification"]: row for row in json.loads(result.values["qualification_records"])}
    assert rows["SSC / Class 10"]["institution"] == "Dinkar Model School"
    assert rows["SSC / Class 10"]["board_awarding_body"] == "CBSE"
    assert rows["SSC / Class 10"]["percentage"] == "85%"
    assert rows["HSC / Class 12"]["institution"] == "ABC Senior Secondary School"
    assert rows["HSC / Class 12"]["board_awarding_body"] == "CBSE"
    assert rows["HSC / Class 12"]["passing_year"] == "2022"
    assert rows["Graduation"]["degree"] == "B.A."
    assert rows["Graduation"]["institution"] == "MSI Noida"
    assert rows["Graduation"]["status"] == "Pursuing"
    assert rows["Graduation"]["expected_completion_year"] == "2026"


def test_structured_qualification_does_not_copy_graduation_cgpa_into_hsc():
    import json

    lines = [OcrLine(t, 0.97, source_index=1, source_name="resume.png") for t in [
        "Education",
        "B.Tech in Artificial Intelligence & Data Science 2022 - 2026",
        "P.R. Pote Patil College of Engineering, Amravati 8.3 CGPA",
        "HSC Maharashtra State Board 2021 78.5%",
        "SSC Maharashtra State Board 2019 88.20%",
    ]]
    result = extract(lines)
    rows = {row["qualification"]: row for row in json.loads(result.values["qualification_records"])}
    assert rows["Graduation"]["degree"] == "B.Tech in Artificial Intelligence & Data Science"
    assert rows["Graduation"]["cgpa"] == "8.3 CGPA"
    assert rows["Graduation"]["institution"].startswith("P.R. Pote Patil College")
    assert rows["HSC / Class 12"]["percentage"] == "78.5%"
    assert rows["HSC / Class 12"]["cgpa"] == ""
    assert rows["HSC / Class 12"]["institution"] == ""
    assert rows["SSC / Class 10"]["percentage"] == "88.20%"
    assert rows["SSC / Class 10"]["board_awarding_body"] == "Maharashtra State Board"


def _education(texts):
    import json
    from autofill.cleanup import clean_lines
    from autofill.extraction import extract
    from autofill.ocr_engine import OcrLine
    lines, _ = clean_lines([OcrLine(text=t, source_index=i) for i, group in enumerate(texts, 1) for t in group])
    return {r["qualification"]: r for r in json.loads(extract(lines).values["qualification_records"])}


PAGE = ["Professional Experience", "Airtel Broadband (Kirti Nagar)", "Worked as a Telecaller", "1.5 Year",
        "Gyaanjyoti Education (Vaishali)", "Worked as a Telecounsellor", "2Year",
        "Education", "IMS Noida", "Pursuing BA 3rd Year", "Diwakar Model School",
        "12th passed from CBSE board", "Diwakar Model School", "10th passed from CBSE board", "Key Skills"]


def test_institution_before_qualification_layout():
    rec = _education([PAGE])
    assert rec["SSC / Class 10"]["institution"] == "Diwakar Model School" and rec["SSC / Class 10"]["board_awarding_body"] == "CBSE"
    assert rec["HSC / Class 12"]["institution"] == "Diwakar Model School"
    grad = rec["Graduation"]
    assert grad["institution"] == "IMS Noida" and grad["status"] == "Pursuing" and grad["degree"].startswith("BA")


def test_overlapping_screenshots_and_job_durations_do_not_leak_into_education():
    rec = _education([PAGE, ["Resume in PDF", "Skip"] + PAGE[:7] + PAGE[7:]])
    assert rec["SSC / Class 10"]["institution"] == "Diwakar Model School"
    assert all("Year" not in r["institution"] for r in rec.values())


def test_stray_duration_between_institution_and_qualification():
    rec = _education([["Education", "IMS Noida", "Pursuing BA 3rd Year", "Diwakar Model School", "12th passed from CBSE board",
                       "Diwakar Model School", "1.5Year", "10th passed from CBSE board", "Key Skills"]])
    assert rec["SSC / Class 10"]["institution"] == "Diwakar Model School"
