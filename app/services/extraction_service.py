"""
Turns a raw resume file into structured fields. Two things are deliberately
NEVER left to the LLM's judgment - they're computed by Python instead:
  - total_experience (date arithmetic)
  - which education record is the HIGHEST qualification (hierarchy lookup)
The LLM's job is comprehensive extraction/structuring of the whole document;
Python's job is anything that's actually arithmetic or a strict comparison.
"""
import re
from datetime import date, datetime

from dateutil import parser as dateparser

from app.core.logging import get_logger
from app.services import llm_service

logger = get_logger(__name__)

# Catches "5+ years of experience", "experience: 15 years", "over 20 years
# of experience", "12 Years Professional Experience" - only searched within
# the summary/profile area (see try_direct_experience), never the whole
# document, so a duration next to one specific role isn't mistaken for the total.
DIRECT_EXP_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\+?\s*[- ]?\s*years?\s*(?:of)?\s*(?:relevant\s+|professional\s+)?experience)"
    r"|(?:(?:professional\s+)?experience\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\+?\s*years?)",
    re.I,
)

DATE_TOKEN_RE = re.compile(
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}"
    r"|\b\d{1,2}[/-]\d{4}\b|\b\d{4}\s*[-–—/]\s*(?:\d{4}|present|current)\b|\b\d{4}\b"
    r"|present|current|till date|to date",
    re.I,
)

# A role stated as a plain duration with no calendar dates at all - either
# digit form ("3 years experience", "(2.5 yrs)") or WORD form ("One year
# experience", "Six-month experience", "a couple of years"). Used both to
# widen the pre-LLM sanity check and to keep Rule 1 scoped to the summary
# only. Deliberately generic (a word-number list, not any specific phrasing)
# so it covers any resume written this way, not one hardcoded example.
_NUM_WORDS = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"a\s+couple(?:\s+of)?|a\s+few|half\s+(?:a|an)?"
)
DURATION_TOKEN_RE = re.compile(
    rf"\d+(?:\.\d+)?\+?\s*(?:years?|months?)"
    rf"|\b(?:{_NUM_WORDS})[\s-]*(?:years?|months?)\b",
    re.I,
)

# Common section headings that mark the start of the actual role-by-role
# history - Rule 1 (stated total) is only ever read from BEFORE this point,
# so a duration mentioned next to one specific role is never mistaken for
# the candidate's overall total experience. Optional trailing colon: "Work Experience:".
EXPERIENCE_SECTION_RE = re.compile(
    r"\n\s*(work experience|employment history|professional experience|"
    r"career history|experience)\s*:?\s*\n",
    re.I,
)

# Ordered low -> high, matching common international + South-Asian schooling
# systems (Matric/Intermediate are widely used qualification names in
# Pakistan/India). "Certificate/Diploma (non-academic)" is deliberately BELOW
# Matric - a bootcamp/online-course certificate is not a substitute for a
# recognized academic qualification, but it's also not "no education", so it
# gets its own honest rung instead of being force-fit into a degree bucket.
EDUCATION_LEVELS = [
    "No formal education stated",
    "Certificate/Diploma (non-academic, e.g. course/bootcamp)",
    "Matric / Secondary School",
    "Intermediate / High School",
    "Diploma (technical/associate)",
    "Bachelor's",
    "Master's",
    "MPhil",
    "PhD / Doctorate",
]


def normalize_education_level(raw_level: str) -> str:
    """Snap whatever the LLM returned to the nearest known bucket, defensively."""
    if not raw_level:
        return "No formal education stated"
    raw = raw_level.strip().lower()
    for level in EDUCATION_LEVELS:
        if level.lower() == raw:
            return level
    if "phd" in raw or "doctor" in raw:
        return "PhD / Doctorate"
    if "mphil" in raw or "m.phil" in raw:
        return "MPhil"
    if "master" in raw or "msc" in raw or "mba" in raw or "m.tech" in raw or "ms " in raw:
        return "Master's"
    if "bachelor" in raw or "bsc" in raw or "b.a" in raw or "b.tech" in raw or "bs " in raw or "undergraduate degree" in raw:
        return "Bachelor's"
    if "associate" in raw or ("diploma" in raw and ("technical" in raw or "engineering" in raw or "polytechnic" in raw)):
        return "Diploma (technical/associate)"
    if "intermediate" in raw or "hssc" in raw or "a-level" in raw or "a level" in raw or "12th" in raw or "high school" in raw:
        return "Intermediate / High School"
    if "matric" in raw or "ssc" in raw or "o-level" in raw or "o level" in raw or "10th" in raw or "secondary school" in raw:
        return "Matric / Secondary School"
    if "diploma" in raw or "certificate" in raw or "course" in raw or "bootcamp" in raw:
        return "Certificate/Diploma (non-academic, e.g. course/bootcamp)"
    return "No formal education stated"


def highest_education(education_records: list[dict]) -> tuple[str, str]:
    """
    A candidate may list several qualifications (Matric, Intermediate,
    Bachelor's...). We compare ONLY the highest one against a job's
    requirement - never the first one found, never all of them.
    Returns (highest_level, highest_detail).
    """
    if not education_records:
        return "No formal education stated", ""
    best_level = "No formal education stated"
    best_detail = ""
    for rec in education_records:
        level = normalize_education_level(rec.get("level", ""))
        if EDUCATION_LEVELS.index(level) >= EDUCATION_LEVELS.index(best_level):
            best_level = level
            best_detail = rec.get("detail", "")
    return best_level, best_detail


def education_meets_requirement(candidate_level: str, required_level: str) -> bool:
    """Strict hierarchical check - candidate's HIGHEST level must be at or above the required rung."""
    candidate_level = normalize_education_level(candidate_level)
    required_level = normalize_education_level(required_level)
    try:
        return EDUCATION_LEVELS.index(candidate_level) >= EDUCATION_LEVELS.index(required_level)
    except ValueError:
        return False


def try_direct_experience(resume_text: str) -> float | None:
    """Rule 1 - if the SUMMARY/PROFILE area (before the first Experience-section
    heading) states a total outright, use it directly, never recalculated from
    dates. Deliberately scoped to that area only - a duration mentioned next to
    one specific role further down (Rule 2's job) must never be mistaken for
    the candidate's overall total."""
    section_match = EXPERIENCE_SECTION_RE.search(resume_text)
    summary_area = resume_text[: section_match.start()] if section_match else resume_text[:800]
    match = DIRECT_EXP_RE.search(summary_area)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return float(value) if value else None


def has_any_dates(resume_text: str) -> bool:
    """Cheap sanity check before spending an LLM call - regex only confirms
    date-like OR stated-duration tokens exist somewhere, never computes a
    year count itself. Widened to also catch resumes where roles have no
    calendar dates at all but state a plain duration ("3 years") instead -
    those must still reach the LLM/Python pipeline, not be skipped to 0."""
    return bool(DATE_TOKEN_RE.search(resume_text) or DURATION_TOKEN_RE.search(resume_text))


_YEAR_ONLY_RE = re.compile(r"^(19|20)\d{2}$")


def _parse_date(raw: str, is_end: bool = False) -> date:
    raw = raw.strip()
    lowered = raw.lower()
    if lowered in ("present", "current", "till date", "to date", ""):
        return date.today()
    # A bare year with no month at all ("2025") is ambiguous by itself - as
    # a START it should mean the beginning of that year (dateutil's default
    # of month=1/day=1 already gets this right), but as an END it should
    # mean the END of that year (December), not silently collapse to
    # January and undercount the role by up to 11 months.
    if is_end and _YEAR_ONLY_RE.match(raw):
        raw = f"December {raw}"
    try:
        return dateparser.parse(raw, default=datetime(2000, 1, 1)).date()
    except (ValueError, OverflowError):
        return date.today()


_YEAR_RE = re.compile(r"(19|20)\d{2}")
_PRESENT_WORDS = {"present", "current", "till date", "to date", ""}


def _fill_missing_years(start_raw: str, end_raw: str) -> tuple[str, str]:
    """The LLM sometimes copies a role's start (or end) date without its
    year when the resume itself only states it once for the pair - e.g.
    "February – November 2017" is written expecting the reader to infer
    "February 2017". Left as-is, dateutil silently defaults any missing
    year to 2000 - turning that 9-month role into an 18-year one once
    summed, and (via the overlap-merge below) swallowing every other role
    that falls inside that phantom 2000-2017 span too. Borrow the year from
    whichever side of the pair does state one (or from today, if the other
    side is "Present") instead of ever letting a bare 2000 default through."""
    end_is_present = end_raw.strip().lower() in _PRESENT_WORDS

    if not _YEAR_RE.search(start_raw):
        if end_is_present:
            start_raw = f"{start_raw} {date.today().year}"
        else:
            match = _YEAR_RE.search(end_raw)
            if match:
                start_raw = f"{start_raw} {match.group(0)}"

    if not end_is_present and not _YEAR_RE.search(end_raw):
        match = _YEAR_RE.search(start_raw)
        if match:
            end_raw = f"{end_raw} {match.group(0)}"

    return start_raw, end_raw


# word -> number, used only to convert a duration phrase written in words
# ("One year", "half a year") into a decimal - Python's job, not the LLM's.
_WORD_TO_NUM = {
    "half": 0.5, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "couple": 2, "few": 3,
}
_DURATION_PARSE_RE = re.compile(
    r"(\d+(?:\.\d+)?|[a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s*(?:a|an)?\s*[-]?\s*(years?|months?)", re.I
)


def parse_duration_years(raw: str) -> float | None:
    """Deterministically parses a duration phrase COPIED VERBATIM from a
    resume (the LLM's only job for this field) into decimal years - handles
    both digit form ("3 years", "2.5 yrs") and word form ("One year",
    "Six-month", "a couple of years", "half a year"). Returns None if the
    phrase can't be parsed (never a silent 0)."""
    if not raw or not raw.strip():
        return None
    raw = raw.strip().lower()

    match = _DURATION_PARSE_RE.search(raw)
    if not match:
        return None
    qty_text, unit = match.group(1).strip(), match.group(2).lower()

    try:
        qty = float(qty_text)
    except ValueError:
        # word form - "a"/"an" is filler unless it's the ONLY word present
        # ("a year" = 1 year); a real number word beside it always wins
        # ("half a year" = 0.5, not 1; "a couple of years" = 2, not 1).
        all_words = [w for w in re.findall(r"[a-z]+", qty_text) if w in _WORD_TO_NUM]
        words = [w for w in all_words if w not in ("a", "an")] or all_words
        if not words:
            return None
        qty = _WORD_TO_NUM[words[0]]

    years = qty / 12 if unit.startswith("month") else qty
    return round(years, 2)


def normalize_and_sum(companies: list[dict]) -> float:
    """Rule 2 - Python does the actual date math: parses whatever date
    formats came back, treats Present/Current/Till Date as today, merges
    OVERLAPPING roles so concurrent jobs aren't double-counted, and sums
    real month/year durations. The LLM never computes this number.

    A role with NO usable start/end dates no longer contributes 0 - if the
    resume states a duration directly for THAT role (any form - "3 years",
    "One year", "Six-month" - the LLM only copies the raw phrase, Python
    parses it via parse_duration_years), that duration is added on top
    instead. Only a role with neither dates nor a stated duration contributes
    nothing."""
    if not companies:
        return 0.0

    periods: list[tuple[date, date]] = []
    extra_months = 0.0
    for c in companies:
        start_raw = (c.get("start_date_raw") or "").strip()
        end_raw = (c.get("end_date_raw") or "").strip()
        if start_raw and end_raw:
            start_raw, end_raw = _fill_missing_years(start_raw, end_raw)
            today = date.today()
            start_year_match = _YEAR_ONLY_RE.match(start_raw)
            end_year_match = _YEAR_ONLY_RE.match(end_raw)
            if start_year_match and end_year_match and start_raw != end_raw:
                # A genuine year-only RANGE across two different years - e.g.
                # "2024-2025", "2025/2026", "2025 to 2026" - however it's
                # punctuated. Expanding each side to Jan/Dec would turn an
                # adjacent-year range into nearly 2 years; resumes stating
                # bare years like this mean simple year subtraction (1 year
                # for adjacent years), so anchor both sides to January and
                # let the plain year/month difference do that math.
                start = date(int(start_raw), 1, 1)
                end = date(int(end_raw), 1, 1)
            else:
                # Either full dates, or the SAME bare year on both sides
                # (a single-year entry, not a range) - keep the normal
                # Jan-start/Dec-end handling.
                start = _parse_date(start_raw)
                end = _parse_date(end_raw, is_end=True)
            if end > today:
                # A year-only end ("2026") can default to December of that
                # year, but if we're still mid-year (e.g. it's only August
                # 2026), December hasn't happened yet - a role can't run
                # into the future, so cap it at today instead of overcounting.
                end = today
            if end < start:
                # A borrowed year can land start a year late (e.g. "November"
                # .. "February 2018" means Nov 2017, not Nov 2018) - step
                # start back a year before falling back to a plain swap.
                stepped_back = date(start.year - 1, start.month, start.day)
                start = stepped_back if stepped_back < end else end
            if end < start:
                start, end = end, start
            periods.append((start, end))
        else:
            stated_years = parse_duration_years(c.get("duration_raw", ""))
            if stated_years is not None:
                extra_months += stated_years * 12

    periods.sort()
    merged: list[tuple[date, date]] = []
    for s, e in periods:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Each merged period counts for at least 1 month - a role whose start
    # and end land in the same calendar month (e.g. a one-off "July 2011 -
    # July 2011" internship) was still worked for that month; the raw
    # (year*12+month) difference would otherwise compute 0 and erase it
    # from the total entirely.
    months = sum(max((e.year - s.year) * 12 + (e.month - s.month), 1) for s, e in merged)
    months += extra_months
    return round(max(months, 0) / 12, 1)


def get_total_experience(resume_text: str) -> tuple[float, str, list[dict]]:
    """
    Rule 1 then Rule 2, exactly as specified:
      1. Total stated in the summary/profile area (before the first Experience
         heading) -> use it, skip date math entirely.
      2. Otherwise -> LLM reads the WHOLE resume (Work Experience, Employment
         History, Internships, Freelance Work, Research - wherever roles with
         dates OR a stated duration appear, not just one heading-sliced
         section) and structures every role; Python then normalizes and sums
         the dates, falling back to a role's own stated duration ("3 years")
         when that role has no dates at all, instead of contributing 0.
    Returns (total_experience, source, work_history) where source is "stated" or "computed".
    """
    direct = try_direct_experience(resume_text)
    if direct is not None:
        return direct, "stated", []

    if not has_any_dates(resume_text):
        logger.info("no_dates_found_in_resume")
        return 0.0, "computed", []

    companies = llm_service.structure_companies(resume_text)  # LLM reads the FULL text
    total = normalize_and_sum(companies)                      # Python does the math
    return total, "computed", companies


def check_role_alignment(detected_title: str, posted_title: str) -> tuple[str, str]:
    """
    Rough, deterministic pre-check comparing what the resume actually looks
    like (detected_job_title) against the posted role - catches only the
    OBVIOUS case (zero word overlap, e.g. "AI Engineer" vs "Administrator").
    This is intentionally cheap; nuanced judgment (e.g. "Office Coordinator"
    is basically the same field as "Administrative Assistant") is left to the
    scoring LLM call, which sees the full resume and can reason about it
    properly instead of matching title words.
    """
    if not detected_title:
        return "unknown", "Could not detect a clear job title from this resume."

    stop_words = {"senior", "junior", "lead", "the", "a", "of", "and", "i", "ii", "iii"}
    detected_words = {w for w in re.findall(r"[a-z]+", detected_title.lower()) if w not in stop_words}
    posted_words = {w for w in re.findall(r"[a-z]+", posted_title.lower()) if w not in stop_words}

    if detected_words & posted_words:
        return "aligned", ""

    return (
        "different_field",
        f"Resume profile reads as '{detected_title}', which doesn't overlap with the posted role '{posted_title}'.",
    )


def analyze_resume(resume_text: str, required_skills: list[str]) -> dict:
    """
    Single-shot ATS analysis matching the exact spec:
      Step 1/2 - total_experience_years (stated total wins outright; otherwise
                 Python sums normalized, non-overlapping date ranges - reuses
                 get_total_experience exactly, no separate code path).
      Step 3   - highest_education only (never the full list) - reuses
                 highest_education() over the LLM-extracted records.
      Step 4   - skill matching by meaning, LLM-judged per skill, no hardcoded
                 skill/synonym list (see llm_service.analyze_skills).
      Step 5   - match_score computed in Python: (explicit + inferred) /
                 total_required * 100. Never left to the LLM.
    Internal-only data (dates, companies, full education list) is deliberately
    NOT included in the returned dict.
    """
    total_experience, _source, _work_history = get_total_experience(resume_text)

    fields = llm_service.extract_resume_fields(resume_text)
    education_records = fields.get("education_records", [])
    highest_level, highest_detail = highest_education(education_records)
    highest_education_str = f"{highest_level} in {highest_detail}" if highest_detail else highest_level

    skills_analysis = llm_service.analyze_skills(resume_text, required_skills)

    total_required = len(required_skills)
    matched = sum(1 for s in skills_analysis if s.get("status") in ("explicit", "inferred"))
    match_score = round((matched / total_required) * 100, 1) if total_required else 0.0

    return {
        "total_experience_years": total_experience,
        "highest_education": highest_education_str,
        "skills_analysis": skills_analysis,
        "match_score": match_score,
    }


def extract_all_fields(resume_text: str) -> dict:
    """
    Full extraction pipeline:
      - LLM does a DEEP read of the entire document (every section - Work
        Experience, Employment History, Projects, Education, Certifications,
        Skills, Internships, Freelance Work, Research) and returns name/
        contact/education_records/skills_explicit/skills_inferred/
        certifications/projects/detected_job_title.
      - Python picks the HIGHEST education record (never the first found).
      - Experience is handled by the hybrid Rule 1 / Rule 2 pipeline above.
    """
    fields = llm_service.extract_resume_fields(resume_text)
    total_experience, source, work_history = get_total_experience(resume_text)

    education_records = fields.get("education_records", [])
    highest_level, highest_detail = highest_education(education_records)

    return {
        "full_name": fields.get("full_name", ""),
        "email": fields.get("email", ""),
        "phone": fields.get("phone", ""),
        "skills_explicit": fields.get("skills_explicit", []),
        "skills_inferred": fields.get("skills_inferred", []),  # [{skill, confidence, evidence}]
        "education_records": education_records,                # every qualification found, as-is
        "highest_education_level": highest_level,               # normalized bucket, used for strict comparison
        "highest_education_detail": highest_detail,
        "certifications": fields.get("certifications", []),
        "projects": fields.get("projects", []),
        "total_experience": total_experience,
        "experience_source": source,
        "work_history": work_history,
        "detected_job_title": fields.get("detected_job_title", ""),
    }
