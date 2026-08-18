"""
Every read/write to the only database lives here. ChromaDB metadata is flat
(str | int | float | bool), so list/dict fields are serialized to comma-separated
strings or JSON strings on write, and parsed back on read.
"""
import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache

import chromadb

from app.config import get_settings
from app.core.exceptions import ChromaDBError, JobNotFoundError, ResumeNotFoundError

settings = get_settings()


@lru_cache
def _client() -> chromadb.ClientAPI:
    if settings.CHROMA_MODE == "server":
        return chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
    return chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)


def _jobs():
    return _client().get_or_create_collection(settings.JOBS_COLLECTION)


def _resumes():
    return _client().get_or_create_collection(settings.RESUMES_COLLECTION)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _list_to_csv(items: list[str]) -> str:
    return ", ".join(i.strip() for i in items if i and i.strip())


def _csv_to_list(csv: str) -> list[str]:
    return [i.strip() for i in csv.split(",") if i.strip()] if csv else []


# ---------------- Jobs ----------------

def add_job(job_id: str, embedding: list[float], document: str, metadata: dict) -> None:
    metadata = {
        **metadata,
        "required_skills": _list_to_csv(metadata.get("required_skills", [])),
        "status": metadata.get("status", "active"),
        "created_at": _now(),
        "updated_at": _now(),
    }
    try:
        _jobs().add(ids=[job_id], embeddings=[embedding], documents=[document], metadatas=[metadata])
    except Exception as exc:  # noqa: BLE001
        raise ChromaDBError(str(exc)) from exc


def update_job(job_id: str, embedding: list[float], document: str, metadata: dict) -> None:
    existing = get_job(job_id)  # raises JobNotFoundError if missing
    metadata = {**existing, **metadata, "updated_at": _now()}
    metadata["required_skills"] = _list_to_csv(metadata.get("required_skills", []))
    try:
        _jobs().update(ids=[job_id], embeddings=[embedding], documents=[document], metadatas=[metadata])
    except Exception as exc:  # noqa: BLE001
        raise ChromaDBError(str(exc)) from exc


def get_job(job_id: str) -> dict:
    result = _jobs().get(ids=[job_id], include=["metadatas", "documents"])
    if not result["ids"]:
        raise JobNotFoundError(f"Job '{job_id}' not found.")
    meta = dict(result["metadatas"][0])
    meta["job_id"] = job_id
    meta["job_description"] = result["documents"][0]
    meta["required_skills"] = _csv_to_list(meta.get("required_skills", ""))
    return meta


def list_jobs() -> list[dict]:
    result = _jobs().get(include=["metadatas", "documents"])
    jobs = []
    for job_id, meta, doc in zip(result["ids"], result["metadatas"], result["documents"]):
        m = dict(meta)
        m["job_id"] = job_id
        m["job_description"] = doc
        m["required_skills"] = _csv_to_list(m.get("required_skills", ""))
        jobs.append(m)
    return jobs


def delete_job(job_id: str) -> None:
    get_job(job_id)  # 404 if missing
    _jobs().delete(ids=[job_id])


# ---------------- Resumes ----------------

def add_resume(resume_id: str, embedding: list[float], document: str, metadata: dict) -> None:
    # Hash of the extracted resume text - identical text (e.g. the exact same
    # CV file uploaded more than once) gets the exact same hash, which is how
    # a repeat upload is recognized and given the SAME score instead of a
    # fresh LLM call that could come back slightly different each time.
    content_hash = hashlib.sha256((document or "").strip().encode("utf-8")).hexdigest()
    flat = {
        **metadata,
        "skills_explicit": _list_to_csv(metadata.get("skills_explicit", [])),
        "skills_inferred_json": json.dumps(metadata.get("skills_inferred", [])),
        "certifications": _list_to_csv(metadata.get("certifications", [])),
        "projects": _list_to_csv(metadata.get("projects", [])),
        "education_records_json": json.dumps(metadata.get("education_records", [])),
        "work_history_json": json.dumps(metadata.get("work_history", [])),
        "content_hash": content_hash,
        "uploaded_at": _now(),
    }
    for key in ("skills_inferred", "education_records", "work_history"):
        flat.pop(key, None)
    try:
        _resumes().add(ids=[resume_id], embeddings=[embedding], documents=[document], metadatas=[flat])
    except Exception as exc:  # noqa: BLE001
        raise ChromaDBError(str(exc)) from exc


def get_resume(resume_id: str) -> dict:
    result = _resumes().get(ids=[resume_id], include=["metadatas", "documents"])
    if not result["ids"]:
        raise ResumeNotFoundError(f"Resume '{resume_id}' not found.")
    return _inflate_resume(resume_id, result["metadatas"][0], result["documents"][0])


def list_resumes(applied_job_id: str | None = None) -> list[dict]:
    where = {"applied_job_id": applied_job_id} if applied_job_id else None
    result = _resumes().get(include=["metadatas", "documents"], where=where)
    return [
        _inflate_resume(rid, meta, doc)
        for rid, meta, doc in zip(result["ids"], result["metadatas"], result["documents"])
    ]


def find_cached_score(content_hash: str, job_id: str) -> dict | None:
    """Look up a score already persisted for ANY resume with the exact same
    extracted text, scored against this same job - so re-uploading the
    identical CV (accidentally or on purpose) returns the same score every
    time instead of a fresh LLM call that could drift slightly. Returns None
    if no matching cached score exists yet."""
    if not content_hash:
        return None
    try:
        result = _resumes().get(
            where={"$and": [{"content_hash": content_hash}, {"last_score_job_id": job_id}]},
            include=["metadatas"],
        )
    except Exception as exc:  # noqa: BLE001
        raise ChromaDBError(str(exc)) from exc
    for meta in result.get("metadatas", []):
        if meta.get("last_score") is not None:
            return {
                "score": meta.get("last_score"),
                "matched_skills": json.loads(meta.get("last_matched_skills_json", "[]") or "[]"),
                "missing_skills": json.loads(meta.get("last_missing_skills_json", "[]") or "[]"),
                "relevant_experience": meta.get("last_relevant_experience", 0.0),
                "education_match": meta.get("last_education_match", False),
                "role_alignment": meta.get("last_role_alignment", "unknown"),
                "explanation": meta.get("last_explanation", ""),
            }
    return None


def update_resume_score(resume_id: str, job_id: str, score_data: dict) -> None:
    """Persist the last /search result for this candidate against a specific
    job onto the resume's own ChromaDB record, so it survives a page refresh
    instead of vanishing until someone runs /search again. Only called after
    a successful scoring call - a failed/None score is never persisted, so a
    later rate-limit error can't wipe out a previously good score. Does not
    touch the resume's embedding, document, or any extracted field."""
    if score_data.get("score") is None:
        return
    result = _resumes().get(ids=[resume_id], include=["metadatas"])
    if not result["ids"]:
        return
    meta = dict(result["metadatas"][0])
    meta["last_score_job_id"] = job_id
    meta["last_score"] = int(score_data.get("score", 0))
    meta["last_matched_skills_json"] = json.dumps(score_data.get("matched_skills", []))
    meta["last_missing_skills_json"] = json.dumps(score_data.get("missing_skills", []))
    meta["last_relevant_experience"] = float(score_data.get("relevant_experience", 0.0))
    meta["last_education_match"] = bool(score_data.get("education_match", False))
    meta["last_role_alignment"] = score_data.get("role_alignment", "unknown")
    meta["last_explanation"] = score_data.get("explanation", "")
    meta["last_scored_at"] = _now()
    try:
        _resumes().update(ids=[resume_id], metadatas=[meta])
    except Exception as exc:  # noqa: BLE001
        raise ChromaDBError(str(exc)) from exc


def delete_resume(resume_id: str) -> None:
    get_resume(resume_id)  # 404 if missing
    _resumes().delete(ids=[resume_id])


def delete_all_resumes() -> int:
    result = _resumes().get(include=[])
    ids = result["ids"]
    if ids:
        _resumes().delete(ids=ids)
    return len(ids)


def query_resumes(query_embedding: list[float], top_k: int, applied_job_id: str | None = None) -> list[dict]:
    where = {"applied_job_id": applied_job_id} if applied_job_id else None
    try:
        result = _resumes().query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as exc:  # noqa: BLE001
        raise ChromaDBError(str(exc)) from exc

    if not result["ids"] or not result["ids"][0]:
        return []

    candidates = []
    for rid, meta, doc, dist in zip(
        result["ids"][0], result["metadatas"][0], result["documents"][0], result["distances"][0]
    ):
        candidate = _inflate_resume(rid, meta, doc)
        candidate["_distance"] = dist
        candidates.append(candidate)
    return candidates


def _inflate_resume(resume_id: str, meta: dict, document: str) -> dict:
    m = dict(meta)
    m["resume_id"] = resume_id
    m["resume_text"] = document
    m["skills_explicit"] = _csv_to_list(m.get("skills_explicit", ""))
    m["skills_inferred"] = json.loads(m.pop("skills_inferred_json", "[]") or "[]")
    m["certifications"] = _csv_to_list(m.get("certifications", ""))
    m["projects"] = _csv_to_list(m.get("projects", ""))
    m["education_records"] = json.loads(m.pop("education_records_json", "[]") or "[]")
    m["work_history"] = json.loads(m.pop("work_history_json", "[]") or "[]")
    # Last persisted /search score for this candidate (None if never scored yet).
    m["last_score"] = m.get("last_score")
    m["last_score_job_id"] = m.get("last_score_job_id")
    m["last_matched_skills"] = json.loads(m.pop("last_matched_skills_json", "[]") or "[]")
    m["last_missing_skills"] = json.loads(m.pop("last_missing_skills_json", "[]") or "[]")
    m["last_relevant_experience"] = m.get("last_relevant_experience")
    m["last_education_match"] = m.get("last_education_match")
    m["last_role_alignment"] = m.get("last_role_alignment")
    m["last_explanation"] = m.get("last_explanation")
    m["last_scored_at"] = m.get("last_scored_at")
    return m
