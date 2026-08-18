"""
Resumes are handled in two explicit steps, matching the frontend's
"save files" -> "Screen CVs" flow:

  POST /resumes/upload   -> saves files to disk only. No parsing, no embedding.
  GET  /resumes/pending   -> lists what's saved but not yet screened.
  POST /resumes/screen    -> runs extraction + the hybrid experience pipeline
                              + embedding + ChromaDB write for pending files.
  GET  /resumes           -> lists screened candidates (from ChromaDB).
  DELETE /resumes/{id}    -> remove one candidate.
  DELETE /resumes         -> remove ALL candidates ("Delete all").
"""
import os
import uuid

from fastapi import APIRouter, Depends, UploadFile

from app.config import get_settings
from app.core.logging import get_logger
from app.core.exceptions import LLMQuotaExhaustedError
from app.core.security import require_api_key
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CandidateResponse,
    PendingFile,
    ScreenFailure,
    ScreenRequest,
    ScreenResponse,
)
from app.services import chroma_service, embedding_service, extraction_service
from app.utils.file_parser import extract_text
from app.utils.validators import validate_upload

router = APIRouter(prefix="/resumes", tags=["resumes"])
logger = get_logger(__name__)
settings = get_settings()

PENDING_DIR = os.path.join(settings.STORAGE_DIR, settings.PENDING_SUBDIR)
PROCESSED_DIR = os.path.join(settings.STORAGE_DIR, settings.PROCESSED_SUBDIR)
os.makedirs(PENDING_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------- Step 1: save ----------------

@router.post("/upload", response_model=list[PendingFile])
async def upload_resumes(files: list[UploadFile]):
    saved = []
    for file in files:
        content = await file.read()
        validate_upload(file, len(content))

        ext = os.path.splitext(file.filename)[1].lower()
        file_id = uuid.uuid4().hex[:12]
        disk_name = f"{file_id}{ext}"
        with open(os.path.join(PENDING_DIR, disk_name), "wb") as f:
            f.write(content)

        saved.append(PendingFile(file_id=file_id, filename=file.filename))
    return saved


@router.get("/pending", response_model=list[PendingFile])
def list_pending():
    items = []
    for name in os.listdir(PENDING_DIR):
        if name.startswith("."):
            continue
        file_id = os.path.splitext(name)[0]
        items.append(PendingFile(file_id=file_id, filename=name))
    return items


@router.delete("/pending/{file_id}")
def delete_pending(file_id: str):
    for name in os.listdir(PENDING_DIR):
        if name.startswith(file_id):
            os.remove(os.path.join(PENDING_DIR, name))
            return {"success": True, "deleted": file_id}
    return {"success": False, "error": {"code": "NOT_FOUND", "message": "Pending file not found."}}


# ---------------- Step 2: screen ----------------

@router.post("/screen", response_model=ScreenResponse, dependencies=[Depends(require_api_key)])
def screen_resumes(payload: ScreenRequest):
    pending_names = [n for n in os.listdir(PENDING_DIR) if not n.startswith(".")]
    if payload.file_ids:
        pending_names = [n for n in pending_names if os.path.splitext(n)[0] in payload.file_ids]

    results = []
    failures: list[ScreenFailure] = []
    stopped_early = False

    for name in pending_names:
        pending_path = os.path.join(PENDING_DIR, name)

        # One bad file (corrupted PDF, an LLM/parsing hiccup, an unreadable
        # scan, etc.) must not take the whole batch down with it - without
        # this try/except, a single failure anywhere in the loop raised past
        # this function, FastAPI returned a 500 for the ENTIRE request, and
        # every file after the failing one - even ones that would have
        # screened fine - never got processed at all.
        try:
            resume_text = extract_text(pending_path)
            fields = extraction_service.extract_all_fields(resume_text)
            embedding = embedding_service.embed_text(resume_text)

            resume_id = f"resume_{uuid.uuid4().hex[:10]}"
            processed_path = os.path.join(PROCESSED_DIR, f"{resume_id}{os.path.splitext(name)[1]}")
            os.replace(pending_path, processed_path)

            metadata = {
                **fields,
                "resume_file_path": processed_path,
                "applied_job_id": payload.applied_job_id,
            }
            chroma_service.add_resume(resume_id, embedding, resume_text, metadata)
            results.append(chroma_service.get_resume(resume_id))
        except LLMQuotaExhaustedError as exc:
            # Every configured key, on every provider, is rate-limited or
            # quota-exhausted right now - every remaining file in this batch
            # would fail the exact same way, so continuing would just burn
            # time without producing a single extra result. Stop here
            # instead: this file (and everything after it) is left
            # completely untouched in "pending", ready to be screened again
            # with a single click once the quota resets - no re-upload needed.
            logger.error("resume_screen_quota_exhausted", extra={"file": name, "error": str(exc)})
            stopped_early = True
            break
        except Exception as exc:
            # A genuine per-file problem (corrupt PDF, no extractable text,
            # etc). This file also stays in "pending" untouched (it's never
            # moved to processed/ until every step above succeeds), so
            # pressing "Screen CVs" again will simply retry it.
            logger.error("resume_screen_failed", extra={"file": name, "error": str(exc)})
            failures.append(ScreenFailure(file=name, reason=str(exc)[:300]))
            continue

    pending_remaining = len(pending_names) - len(results) - len(failures)
    return ScreenResponse(
        results=results,
        total_files=len(pending_names),
        screened_count=len(results),
        failed_count=len(failures),
        pending_remaining=pending_remaining,
        stopped_early=stopped_early,
        failures=failures,
    )


# ---------------- Single-shot ATS analysis (exact 5-step spec) ----------------

@router.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
def analyze_resume(payload: AnalyzeRequest):
    """
    Steps 1-2: total_experience_years (stated total takes priority; otherwise
               computed from the full experience section, no overlap double-count).
    Step 3:    highest_education only (single highest qualification, never the full list).
    Step 4:    skills_analysis - explicit / inferred / missing per required skill,
               judged by meaning, no hardcoded skill/synonym list.
    Step 5:    match_score = (explicit + inferred) / total_required * 100.
    Never returns raw job entries, dates, or company names.
    """
    return extraction_service.analyze_resume(payload.resume_text, payload.required_skills)


# ---------------- Read / delete screened candidates ----------------

@router.get("", response_model=list[CandidateResponse])
def list_resumes(applied_job_id: str | None = None):
    return chroma_service.list_resumes(applied_job_id)


@router.get("/{resume_id}", response_model=CandidateResponse)
def get_resume(resume_id: str):
    return chroma_service.get_resume(resume_id)


@router.delete("/{resume_id}", dependencies=[Depends(require_api_key)])
def delete_resume(resume_id: str):
    chroma_service.delete_resume(resume_id)
    return {"success": True, "deleted": resume_id}


@router.delete("", dependencies=[Depends(require_api_key)])
def delete_all_resumes():
    count = chroma_service.delete_all_resumes()
    return {"success": True, "deleted_count": count}
