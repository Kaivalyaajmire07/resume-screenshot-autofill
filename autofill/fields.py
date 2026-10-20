"""Candidate form schema. Keys are stable: they are stored in the database."""
from __future__ import annotations

FIELDS = [
    ("first_name", "First Name"), ("middle_name", "Middle Name"), ("last_name", "Last Name"),
    ("date_of_birth", "Date of Birth"), ("gender", "Gender"), ("nationality", "Nationality"),
    ("marital_status", "Marital Status"), ("passport", "Passport"), ("hobbies", "Hobbies"),
    ("language_known", "Language Known"), ("address", "Address"), ("landmark", "Landmark"),
    ("city", "City"), ("state", "State"), ("pincode", "Pincode"), ("mobile", "Mobile"),
    ("email", "Email"), ("ssc_result", "SSC Result"), ("ssc_board", "SSC Board"),
    ("ssc_pass_year", "SSC Pass Year"), ("hsc_result", "HSC Result"), ("hsc_board", "HSC Board"),
    ("hsc_pass_year", "HSC Pass Year"), ("graduation_degree", "Graduation Degree"),
    ("graduation_result", "Graduation Result"), ("graduation_university", "Graduation University"),
    ("graduation_year", "Graduation Year"), ("post_graduation_degree", "Post Graduation Degree"),
    ("post_graduation_result", "Post Graduation Result"),
    ("post_graduation_university", "Post Graduation University"),
    ("post_graduation_year", "Post Graduation Year"), ("higher_level_education", "Higher Level Education"),
    ("objectives", "Objectives"), ("summary", "Summary"), ("key_skills", "Key Skills"),
    ("employment_details", "Employment Details"),
    ("certifications", "Certifications"), ("projects", "Projects"),
    ("academic_achievements", "Academic Achievements"),
]
LABELS = dict(FIELDS)
FIELD_KEYS = [k for k, _ in FIELDS]

GROUPS = {
    "Personal Details": ["first_name", "middle_name", "last_name", "date_of_birth", "gender", "nationality",
                         "marital_status", "passport", "hobbies", "language_known"],
    "Communication Details": ["address", "landmark", "city", "state", "pincode", "mobile", "email"],
    "Qualification Details": ["ssc_result", "ssc_board", "ssc_pass_year", "hsc_result", "hsc_board",
                              "hsc_pass_year", "graduation_degree", "graduation_result",
                              "graduation_university", "graduation_year", "post_graduation_degree",
                              "post_graduation_result", "post_graduation_university", "post_graduation_year",
                              "higher_level_education"],
    "Employment & Profile": ["objectives", "summary", "employment_details", "key_skills", "certifications",
                             "projects", "academic_achievements"],
}

LONG_TEXT_FIELDS = {"objectives", "summary", "employment_details", "key_skills", "address",
                    "hobbies", "language_known", "certifications", "projects", "academic_achievements"}
TALL_FIELDS = {"employment_details", "summary", "objectives"}

# Fields that must be present for a usable candidate record.
CORE_FIELDS = ["first_name", "last_name", "mobile", "email"]


def empty_fields() -> dict[str, str]:
    return {k: "" for k in FIELD_KEYS}


# Structured records travel inside the same dict as JSON strings (see records.py).
STRUCTURED_KEYS = ("education_records", "experience_records")


def normalize_fields(values: dict | None) -> dict[str, str]:
    """Return a complete, string-only field dict (tolerates older/partial stored records).

    The structured-record keys are kept only when present, so old rows stay unchanged.
    """
    out = empty_fields()
    for k, v in (values or {}).items():
        if k in out or k in STRUCTURED_KEYS:
            out[k] = "" if v is None else str(v)
    return out
