import uuid

from fastapi import APIRouter, Depends

from app.core.security import require_api_key
from app.models.schemas import JobCreate, JobResponse, JobUpdate, ParsedJobSuggestion, RawJobText
from app.services import chroma_service, embedding_service, llm_service

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(require_api_key)])


def _job_text(job: dict | JobCreate) -> str:
    """Text used to build the JD's single embedding."""
    if isinstance(job, JobCreate):
        skills = ", ".join(job.required_skills)
        return f"{job.job_title}\nRequired skills: {skills}\n{job.job_description}"
    skills = ", ".join(job.get("required_skills", []))
    return f"{job.get('job_title')}\nRequired skills: {skills}\n{job.get('job_description')}"


@router.post("/parse", response_model=ParsedJobSuggestion)
def parse_job_description(payload: RawJobText):
    """
    HR pastes a rough, unstructured JD (skills/experience/education mixed in
    any order). LLM extracts suggested structured fields - this does NOT save
    anything; the frontend fills the form fields for HR to review/edit, then
    calls POST /jobs as normal.
    """
    parsed = llm_service.parse_job_description(payload.raw_text)
    return ParsedJobSuggestion(
        job_title=parsed.get("job_title", ""),
        required_skills=parsed.get("required_skills", []),
        minimum_experience=float(parsed.get("minimum_experience", 0) or 0),
        education_requirement=parsed.get("education_requirement", "No formal education stated"),
        job_description=payload.raw_text,
    )


@router.post("", response_model=JobResponse)
def create_job(payload: JobCreate):
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    embedding = embedding_service.embed_text(_job_text(payload))
    metadata = payload.model_dump()
    metadata.pop("job_description")
    chroma_service.add_job(job_id, embedding, payload.job_description, metadata)
    return chroma_service.get_job(job_id)


@router.get("", response_model=list[JobResponse])
def list_jobs():
    return chroma_service.list_jobs()


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    return chroma_service.get_job(job_id)


@router.put("/{job_id}", response_model=JobResponse)
def update_job(job_id: str, payload: JobUpdate):
    existing = chroma_service.get_job(job_id)
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    merged = {**existing, **updates}
    embedding = embedding_service.embed_text(_job_text(merged))
    doc = merged.pop("job_description")
    merged.pop("job_id", None)
    chroma_service.update_job(job_id, embedding, doc, merged)
    return chroma_service.get_job(job_id)


@router.delete("/{job_id}")
def delete_job(job_id: str):
    chroma_service.delete_job(job_id)
    return {"success": True, "deleted": job_id}
