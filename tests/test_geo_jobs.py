from autofill import geo
from autofill.extraction import extract
from autofill.jobs import SEP, join_jobs, split_jobs
from autofill.ocr_engine import OcrLine


def L(*texts):
    return [OcrLine(t, 0.97, source_index=1) for t in texts]


def test_pincode_lookup():
    info = geo.lookup("422001")
    assert (info.city, info.state) == ("Nashik", "Maharashtra")
    assert geo.lookup("999999") is None and geo.lookup("abc") is None and geo.lookup(None) is None


def test_fill_location_never_overwrites():
    out = geo.fill_location({"pincode": "411005", "city": "Pimpri", "state": ""})
    assert out["city"] == "Pimpri" and out["state"] == "Maharashtra"


def test_city_state_from_pincode_when_address_has_neither():
    res = extract(L("Name: Ravi Kumar", "Address: Flat 4, Gandhi Road", "Nashik 422001"))
    # pincode sits on the 2nd address line (multi-line address)
    assert res.values["pincode"] == "422001"
    assert (res.values["city"], res.values["state"]) == ("Nashik", "Maharashtra")


def test_state_conflict_is_flagged():
    res = extract(L("Address: Main Road, Pune 411005", "State: Gujarat"))
    assert res.conflicts["state"] == ["Gujarat", "Maharashtra"]


def test_three_jobs_split_into_three():
    lines = L("Work Experience",
              "Senior Analyst", "Acme Corp", "Jan 2022 - Present", "Built dashboards for the sales team.",
              "Analyst", "Beta Ltd", "Mar 2020 - Dec 2021", "Wrote SQL reports every week.",
              "Intern", "Gamma Inc", "2019 - 2020", "Helped with data cleaning.",
              "Education")
    value = extract(lines).values["employment_details"]
    jobs = split_jobs(value)
    assert len(jobs) == 3 and SEP in value
    assert jobs[0].startswith("Senior Analyst") and "Beta Ltd" in jobs[1] and jobs[2].startswith("Intern")


def test_old_blob_without_separator_is_split_on_the_fly():
    blob = "Dev, Foo Ltd\nJan 2021 - Mar 2022\nDid things.\nTester, Bar Ltd\n2019 - 2020\nTested things."
    assert len(split_jobs(blob)) == 2


def test_single_or_undated_experience_stays_one_box():
    assert split_jobs("Data Intern, Acme Analytics, 2025") == ["Data Intern, Acme Analytics, 2025"]
    assert split_jobs("") == [] and join_jobs(["a", " ", "b"]) == "a" + SEP + "b"
