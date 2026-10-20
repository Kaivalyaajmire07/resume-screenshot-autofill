from autofill.cleanup import clean_lines, is_noise, split_embedded_newlines
from autofill.ocr_engine import OcrLine
from autofill.structure import plain_text_sections


def L(text, idx=1):
    return OcrLine(text=text, source_index=idx)


def test_noise_lines():
    assert all(is_noise(t) for t in ["Skip", "Resume in PDF", "Resume Ih PDF", "4 of 8", "n4 of 8"])
    assert not is_noise("Worked as a Telecaller")


def test_overlapping_screenshots_are_deduped():
    a = [L("Professional Experience"), L("Airtel Broadband (Kirti Nagar)"), L("Worked as a Telecaller"), L("1.5 Year")]
    b = [L("Airtel Broadband (Kirti Nagar)", 2), L("Worked as a Telecaller", 2), L("1.5 Year", 2),
         L("Gyaanjyoti Education (Vaishali)", 2), L("Resume in PDF", 2), L("Skip", 2)]
    out, dropped = clean_lines(a + b)
    texts = [x.text for x in out]
    assert texts.count("Airtel Broadband (Kirti Nagar)") == 1
    assert texts.count("1.5 Year") == 1
    assert "Gyaanjyoti Education (Vaishali)" in texts and "Skip" not in texts and dropped == 5


def test_short_unique_repeat_in_new_screenshot_is_kept():
    out, _ = clean_lines([L("1 Year"), L("Some distinct line here"), L("OEL (Noida Sector 63)", 2), L("1 Year", 2)])
    assert [x.text for x in out].count("1 Year") == 2


def test_literal_backslash_n_is_split():
    out = split_embedded_newlines([L("Simran Saxena\\nB-75 Shyam Park\\nGhaziabad")])
    assert [x.text for x in out] == ["Simran Saxena", "B-75 Shyam Park", "Ghaziabad"]


def test_plain_text_has_real_newlines_and_sections():
    text = plain_text_sections([L("Simran"), L("Professional Experience"), L("Airtel"), L("Education"), L("IMS Noida")])
    assert "\\n" not in text and "PROFESSIONAL EXPERIENCE\n" + "-" * 23 + "\n1. Airtel" in text
    assert "\n\nEDUCATION\n---------\nIMS Noida" in text


def test_full_plain_text_for_overlapping_screenshots():
    from autofill.cleanup import clean_lines
    from autofill.extraction import extract
    page1 = ["Simran Saxena", "saxenasimran455@gmail.com|+918077571727", "Professional Experience",
             "Airtel Broadband (Kirti Nagar)", "Worked as a Telecaller", "1.5 Year",
             "Gyaanjyoti Education (Vaishali)", "Worked as a Telecounsellor", "2Year",
             "Education", "IMS Noida", "Pursuing BA 3rd Year", "Diwakar Model School",
             "12th passed from CBSE board", "Diwakar Model School", "10th passed from CBSE board", "Key Skills", "Teamwork"]
    page2 = ["Resume in PDF", "Skip", "Education", "IMS Noida", "Pursuing BA 3rd Year", "Diwakar Model School",
             "12th passed from CBSE board", "Diwakar Model School", "4", "1.5Year", "10th passed from CBSE board",
             "Key Skills", "Teamwork"]
    lines, _ = clean_lines([L(t, 1) for t in page1] + [L(t, 2) for t in page2])
    text = plain_text_sections(lines, extract(lines).values)
    assert text.count("EDUCATION") == 1 and text.count("KEY SKILLS") == 1 and "1.5Year" not in text
    assert "Role: Telecaller" in text and "Duration: 1.5 Year" in text
    assert "Institution / College: IMS Noida" in text and "Status: Pursuing" in text
