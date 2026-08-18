"""
Pydantic models for request/response validation.
ChromaDB metadata is flat (str | int | float | bool only), so list-like
fields (skills, work history) are stored as comma-separated / JSON strings
and converted to real lists here at the API boundary.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, EmailStr, Field


# ---------- Jobs ----------

class JobCreate(BaseModel):
    job_title: str
    required_skills: list[str] = Field(default_factory=list)
    minimum_experience: float = 0
    education_requirement: str = "Intermediate / High School"
    job_description: str


class RawJobText(BaseModel):
    raw_text: str  # HR pastes an unstructured job posting here


class ParsedJobSuggestion(BaseModel):
    """What POST /jobs/parse returns - HR reviews/edits this before saving,
    it does NOT get auto-saved to ChromaDB."""
    job_title: str
    required_skills: list[str]
    minimum_experience: float
    education_requirement: str
    job_description: str  # the original raw_text, passed through for the JD field


class JobUpdate(BaseModel):
    job_title: str | None = None
    required_skills: list[str] | None = None
    minimum_experience: float | None = None
    education_requirement: str | None = None
    job_description: str | None = None
    status: Literal["active", "closed"] | None = None


class JobResponse(BaseModel):
    job_id: str
    job_title: str
    required_skills: list[str]
    minimum_experience: float
    education_requirement: str
    job_description: str
    status: str
    created_at: str
    updated_at: str


# ---------- Resumes ----------

class PendingFile(BaseModel):
    file_id: str
    filename: str
    status: Literal["saved"] = "saved"


class WorkHistoryItem(BaseModel):
    company: str
    title: str
    start_date_raw: str
    end_date_raw: str
    duration_raw: str = ""  # e.g. "One year", "3 years" - used only when this role has no dates
    type: str = "job"  # job | internship | freelance | research


class EducationRecord(BaseModel):
    level: str   # one of the normalized EDUCATION_LEVELS buckets
    detail: str  # raw text, e.g. "BSc Computer Science, XYZ University"


class InferredSkill(BaseModel):
    skill: str
    confidence: float           # 0.0-1.0, only surfaced when strongly supported
    evidence: str                # which project/experience line supports this


class MatchedSkill(BaseModel):
    skill: str
    match_type: Literal["exact", "related"]
    evidence: str


class CandidateResponse(BaseModel):
    resume_id: str
    full_name: str
    email: str
    phone: str
    skills_explicit: list[str]           # literally stated in the resume
    skills_inferred: list[InferredSkill]  # semantically inferred, confidence-scored, never hallucinated
    total_experience: float
    experience_source: Literal["stated", "computed"]
    work_history: list[WorkHistoryItem]
    education_records: list[EducationRecord]  # every qualification found (Matric, Bachelor's, etc.)
    highest_education_level: str              # the HIGHEST one - only this is compared against a job
    highest_education_detail: str
    detected_job_title: str    # what this resume actually looks like, independent of any posting
    certifications: list[str]
    projects: list[str]
    resume_file_path: str
    applied_job_id: str | None
    uploaded_at: str
    # Last persisted /search score for this candidate - survives a page
    # refresh instead of disappearing until /search runs again. None until
    # the candidate has been scored against at least one job.
    last_score: int | None = None
    last_score_job_id: str | None = None
    last_matched_skills: list[MatchedSkill] = Field(default_factory=list)
    last_missing_skills: list[str] = Field(default_factory=list)
    last_relevant_experience: float | None = None
    last_education_match: bool | None = None
    last_role_alignment: Literal["aligned", "different_field", "unknown"] | None = None
    last_explanation: str | None = None
    last_scored_at: str | None = None


class SkillAnalysisItem(BaseModel):
    skill: str
    status: Literal["explicit", "inferred", "missing"]
    evidence: str


class AnalyzeRequest(BaseModel):
    resume_text: str
    required_skills: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    """Exact output shape for the single-shot ATS analysis. Internal-only data
    (dates, company names, full education list) is intentionally never included."""
    total_experience_years: float
    highest_education: str
    skills_analysis: list[SkillAnalysisItem]
    match_score: float


class ScreenRequest(BaseModel):
    file_ids: list[str] | None = None  # None = screen ALL pending files
    applied_job_id: str | None = None


class ScreenFailure(BaseModel):
    file: str
    reason: str


class ScreenResponse(BaseModel):
    results: list[CandidateResponse]        # successfully screened - already on the dashboard
    total_files: int                        # how many files this call looked at
    screened_count: int                     # succeeded, moved out of pending, now in results
    failed_count: int                       # attempted but failed for a per-file reason (stays in pending)
    pending_remaining: int                  # not yet attempted at all (stays in pending)
    stopped_early: bool                     # True if an API quota/rate limit stopped the batch early
    failures: list[ScreenFailure] = []


# ---------- Search ----------

class SearchRequest(BaseModel):
    job_id: str
    query: str
    top_k: int = 5
    # When true, score and return EVERY screened candidate (100, 200,
    # whatever the count is) instead of only the top_k nearest by embedding.
    # top_k is ignored when this is set.
    show_all: bool = False


class SearchResult(BaseModel):
    resume_id: str
    full_name: str
    score: int | None
    matched_skills: list[MatchedSkill]  # semantic matches, each tagged exact/related with evidence
    missing_skills: list[str]
    total_experience: float
    relevant_experience: float
    education_match: bool
    highest_education_level: str
    detected_job_title: str
    role_alignment: Literal["aligned", "different_field", "unknown"]
    explanation: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    # Natural-language answer to the recruiter's query, grounded in the same
    # scores as `results` above (replaces what used to be a separate /ask
    # endpoint - one call now does both).
    answer: str
    in_scope: bool  # false when the query wasn't CV/candidate/hiring related


class ApiError(BaseModel):
    code: str
    message: str


class ApiEnvelope(BaseModel):
    success: bool
    data: object | None = None
    error: ApiError | None = None
