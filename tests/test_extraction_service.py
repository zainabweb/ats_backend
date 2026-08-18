from app.services.extraction_service import (
    check_role_alignment,
    education_meets_requirement,
    has_any_dates,
    highest_education,
    normalize_and_sum,
    normalize_education_level,
    try_direct_experience,
)


def test_direct_experience_found():
    text = "Experienced administrator with 5+ years of experience in office management."
    assert try_direct_experience(text) == 5.0


def test_direct_experience_reversed_phrasing():
    text = "Summary: experience of 15 years in software engineering."
    assert try_direct_experience(text) == 15.0


def test_direct_experience_professional_phrasing():
    text = "Professional Experience: 12 Years"
    assert try_direct_experience(text) == 12.0


def test_direct_experience_not_found():
    text = "Managed scheduling and correspondence for a busy office."
    assert try_direct_experience(text) is None


def test_has_any_dates():
    assert has_any_dates("Jan 2020 - Present") is True
    assert has_any_dates("Responsible for filing") is False


def test_normalize_and_sum_single_role():
    companies = [{"company": "Acme", "start_date_raw": "Jan 2020", "end_date_raw": "Jan 2022"}]
    assert normalize_and_sum(companies) == 2.0


def test_normalize_and_sum_merges_overlapping_roles():
    companies = [
        {"company": "Acme", "start_date_raw": "Jan 2020", "end_date_raw": "Jan 2022"},
        {"company": "Acme Side Gig", "start_date_raw": "Jun 2021", "end_date_raw": "Dec 2021"},
    ]
    assert normalize_and_sum(companies) == 2.0


def test_normalize_and_sum_handles_till_date():
    # From the spec's own example: Company A 2006-2010, Company B 2010-2023,
    # Company C 2024-Till Date -> continuous experience through today, no
    # double-counting at the touching boundaries.
    companies = [
        {"company": "Company A", "start_date_raw": "2006", "end_date_raw": "2010"},
        {"company": "Company B", "start_date_raw": "2010", "end_date_raw": "2023"},
        {"company": "Company C", "start_date_raw": "2024", "end_date_raw": "Till Date"},
    ]
    total = normalize_and_sum(companies)
    assert total > 18  # roughly 2006 -> today, allowing for the current year


def test_normalize_and_sum_empty():
    assert normalize_and_sum([]) == 0.0


def test_education_normalize_handles_variants():
    assert normalize_education_level("Bachelor of Computer Science") == "Bachelor's"
    assert normalize_education_level("Matriculation") == "Matric / Secondary School"
    assert normalize_education_level("FSc Pre-Engineering (Intermediate)") == "Intermediate / High School"
    assert normalize_education_level("6-month bootcamp certificate") == "Certificate/Diploma (non-academic, e.g. course/bootcamp)"
    assert normalize_education_level("") == "No formal education stated"


def test_highest_education_picks_the_top_qualification_regardless_of_order():
    records = [
        {"level": "Matric", "detail": "Matriculation, ABC School"},
        {"level": "Bachelor of Computer Science", "detail": "BSCS, XYZ University"},
        {"level": "PhD in Artificial Intelligence", "detail": "PhD AI, State University"},
        {"level": "Intermediate", "detail": "FSc Pre-Engineering"},
    ]
    level, detail = highest_education(records)
    assert level == "PhD / Doctorate"
    assert "PhD AI" in detail


def test_highest_education_with_only_a_certificate():
    records = [{"level": "Online UX bootcamp certificate", "detail": "Google UX Certificate"}]
    level, _ = highest_education(records)
    assert level == "Certificate/Diploma (non-academic, e.g. course/bootcamp)"


def test_education_meets_requirement_strict():
    assert education_meets_requirement("Certificate/Diploma (non-academic, e.g. course/bootcamp)", "Bachelor's") is False
    assert education_meets_requirement("Master's", "Bachelor's") is True
    assert education_meets_requirement("PhD / Doctorate", "Master's") is True
    assert education_meets_requirement("Matric / Secondary School", "Intermediate / High School") is False


def test_role_alignment_mismatch():
    status, note = check_role_alignment("AI Engineer", "Administrative Assistant")
    assert status == "different_field"
    assert "AI Engineer" in note


def test_role_alignment_match():
    status, _ = check_role_alignment("Senior Administrative Assistant", "Administrative Assistant")
    assert status == "aligned"


def test_role_alignment_unknown():
    status, _ = check_role_alignment("", "Administrative Assistant")
    assert status == "unknown"
