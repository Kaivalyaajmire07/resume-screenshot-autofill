from datetime import date

from autofill.extraction import extract
from autofill.fields import empty_fields
from autofill.validation import parse_dob, validate


def f(**kw):
    d = empty_fields(); d.update(kw); return d


def test_format_checks():
    r = validate(f(email="bad@", mobile="12345", pincode="12", date_of_birth="31/02/1999",
                   graduation_year="2090", ssc_result="150%"), today=date(2026, 10, 19))
    for key in ("email", "mobile", "pincode", "date_of_birth", "graduation_year", "ssc_result"):
        assert key in r.issues, key


def test_clean_record_only_flags_missing_core():
    r = validate(f(first_name="A", last_name="B", mobile="9876543210", email="a@b.com", pincode="411005"))
    assert r.issues == {}


def test_blank_core_fields_marked_for_review():
    r = validate(f())
    assert {"first_name", "last_name", "mobile", "email"} <= r.flagged
    from autofill.fields import FIELD_KEYS
    assert len(r.blank) == len(FIELD_KEYS)


def test_year_ordering_conflict():
    r = validate(f(ssc_pass_year="2021", hsc_pass_year="2019", graduation_year="2018"))
    assert "hsc_pass_year" in r.issues and "graduation_year" in r.issues


def test_dob_formats():
    assert parse_dob("12/05/1998") == date(1998, 5, 12)
    assert parse_dob("1998-05-12") == date(1998, 5, 12)
    assert parse_dob("12 May 1998") == date(1998, 5, 12)
    assert parse_dob("garbage") is None


def test_conflicts_and_low_confidence_flow_from_extraction(resume_lines):
    ex = extract(resume_lines + [type(resume_lines[0])("second@mail.com", 0.4, source_index=2)])
    r = validate(ex.values, ex)
    assert any("Multiple different values" in m for m in r.issues["email"])
    assert any("Guessed" in m for m in r.issues["first_name"])


def test_edited_value_drops_evidence_flags(resume_lines):
    ex = extract(resume_lines)
    values = dict(ex.values, first_name="Anita ")  # unchanged after strip
    assert "first_name" in validate(values, ex).issues
    values["first_name"] = "Anitha"                # reviewer changed it
    assert "first_name" not in validate(values, ex).issues
