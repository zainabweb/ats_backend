"""
The RAG workflow described in the design doc, Section 6:
  1. Embed the HR query.
  2. Vector-search ChromaDB for the top-K most similar resumes only
     (unless show_all=True, in which case every screened candidate is
     scored and returned - useful when HR wants the full ranked list, not
     just a shortlist, no matter if that's 20 or 200 candidates).
  3. Fetch the one target job description.
  4. For EACH retrieved candidate, send only that candidate's metadata +
     full resume text + the JD + the query to the LLM for scoring - UNLESS
     an identical resume (same extracted text, i.e. the same CV uploaded
     more than once) already has a persisted score for this exact job, in
     which case that score is reused so duplicates always show the same
     number instead of a fresh, possibly slightly different LLM call.
  5. Rank by score, best first - every one of these results is persisted to
     ChromaDB (chroma_service.update_resume_score) exactly as before.
  6. Ask the LLM ONE more time for a natural-language answer to the
     recruiter's actual query, grounded in the scores/matched/missing skills
     just computed in step 4-5 - this replaces what used to be a separate
     /ask endpoint. Off-topic questions get a fixed refusal string instead
     of whatever the model would have said.
"""
from app.config import get_settings
from app.core.logging import get_logger
from app.services import chroma_service, embedding_service, extraction_service, llm_service

settings = get_settings()
logger = get_logger(__name__)


def search(job_id: str, query: str, top_k: int | None = None, show_all: bool = False) -> dict:
    job = chroma_service.get_job(job_id)  # raises JobNotFoundError if missing

    if show_all:
        # Every screened candidate, no cap - could be 20 or 2000, all of them
        # get scored and returned, ranked best first.
        candidates = chroma_service.list_resumes()
        logger.info("rag_show_all_candidates", extra={"job_id": job_id, "count": len(candidates)})
    else:
        top_k = min(top_k or settings.DEFAULT_TOP_K, settings.MAX_TOP_K)
        query_embedding = embedding_service.embed_text(query)
        candidates = chroma_service.query_resumes(query_embedding, top_k=top_k)
        logger.info("rag_retrieved_candidates", extra={"job_id": job_id, "count": len(candidates)})

    required_skills = job.get("required_skills", [])
    total_required = len(required_skills)

    results = []
    for candidate in candidates:
        # Deterministic checks - never left to the LLM, same principle as experience math.
        education_match = extraction_service.education_meets_requirement(
            candidate.get("highest_education_level", ""), job.get("education_requirement", "")
        )
        role_status, role_note = extraction_service.check_role_alignment(
            candidate.get("detected_job_title", ""), job.get("job_title", "")
        )

        # Duplicate-CV short-circuit: if a resume with this EXACT extracted
        # text was already scored against this exact job (e.g. the same file
        # was uploaded twice, or the same candidate applied under a second
        # file name), reuse that score verbatim instead of calling the LLM
        # again - guarantees the same CV always shows the same number.
        cached = chroma_service.find_cached_score(candidate.get("content_hash", ""), job_id)
        if cached is not None:
            result_entry = {
                "resume_id": candidate["resume_id"],
                "full_name": candidate.get("full_name", ""),
                "score": cached["score"],
                "matched_skills": cached["matched_skills"],
                "missing_skills": cached["missing_skills"],
                "total_experience": candidate.get("total_experience", 0.0),
                "relevant_experience": cached["relevant_experience"],
                "education_match": education_match,
                "highest_education_level": candidate.get("highest_education_level", ""),
                "detected_job_title": candidate.get("detected_job_title", ""),
                "role_alignment": role_status,
                "explanation": cached["explanation"],
            }
            results.append(result_entry)
            try:
                chroma_service.update_resume_score(candidate["resume_id"], job_id, result_entry)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "score_persist_failed", extra={"resume_id": candidate["resume_id"], "error": str(exc)}
                )
            continue

        try:
            # ONE LLM call per candidate - judges every JD-required skill against
            # THIS resume (explicit / inferred / missing), plus relevant experience
            # and a fit explanation, all in the same round trip. Two separate calls
            # here used to double the API load and was hitting rate limits for no
            # benefit - the numeric score was already always computed in Python.
            scored = llm_service.score_candidate(
                hr_query=query,
                job=job,
                candidate_meta=candidate,
                resume_text=candidate["resume_text"],
            )
            skills_analysis = scored.get("skills_analysis", [])
            matched_skills = [
                {
                    "skill": s.get("skill", ""),
                    "match_type": "exact" if s.get("status") == "explicit" else "related",
                    "evidence": s.get("evidence", ""),
                }
                for s in skills_analysis
                if s.get("status") in ("explicit", "inferred")
            ]
            missing_skills = [s.get("skill", "") for s in skills_analysis if s.get("status") == "missing"]

            # The score is ALWAYS matched / total-required-in-the-JD, computed in
            # Python - never a fraction of everything listed on the candidate's CV,
            # and never a free-form number picked by the LLM.
            score = round((len(matched_skills) / total_required) * 100) if total_required else 0

            explanation = scored.get("explanation", "")
            if role_status == "different_field":
                explanation = f"{role_note} {explanation}".strip()

            result_entry = {
                "resume_id": candidate["resume_id"],
                "full_name": candidate.get("full_name", ""),
                "score": score,
                "matched_skills": matched_skills,  # [{skill, match_type, evidence}] - JD skills only
                "missing_skills": missing_skills,
                "total_experience": candidate.get("total_experience", 0.0),
                "relevant_experience": scored.get("relevant_experience", 0.0),
                "education_match": education_match,
                "highest_education_level": candidate.get("highest_education_level", ""),
                "detected_job_title": candidate.get("detected_job_title", ""),
                "role_alignment": role_status,
                "explanation": explanation,
            }
            results.append(result_entry)

            # Persist this score onto the candidate's own record so it survives
            # a page refresh - GET /resumes no longer has to wait for a fresh
            # /search + LLM call to show the last known score - and so future
            # duplicate uploads of this same CV can reuse it via content_hash.
            # A failure here (e.g. a transient Chroma write hiccup) must never
            # break the search response itself, so it's logged and swallowed.
            try:
                chroma_service.update_resume_score(candidate["resume_id"], job_id, result_entry)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "score_persist_failed", extra={"resume_id": candidate["resume_id"], "error": str(exc)}
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("scoring_failed", extra={"resume_id": candidate["resume_id"], "error": str(exc)})
            results.append(
                {
                    "resume_id": candidate["resume_id"],
                    "full_name": candidate.get("full_name", ""),
                    "score": None,
                    "matched_skills": [],
                    "missing_skills": [],
                    "total_experience": candidate.get("total_experience", 0.0),
                    "relevant_experience": 0.0,
                    "education_match": education_match,
                    "highest_education_level": candidate.get("highest_education_level", ""),
                    "detected_job_title": candidate.get("detected_job_title", ""),
                    "role_alignment": role_status,
                    "explanation": "Scoring failed for this candidate.",
                }
            )

    # Rank by skill-match score, best first. When two candidates tie on score,
    # the one with more total experience wins the tie-break.
    results.sort(
        key=lambda r: (r["score"] is None, -(r["score"] or 0), -(r.get("total_experience") or 0.0))
    )

    answer_text, in_scope = _generate_answer(query, job, candidates, results)
    return {"results": results, "answer": answer_text, "in_scope": in_scope}


# Fixed wording for out-of-scope questions - always this exact text,
# regardless of how the LLM itself would have phrased a refusal, so it can
# never be talked into a different answer for an off-topic question.
SEARCH_REFUSAL_TEXT = (
    "I'm only able to help with CV and candidate-related queries. Please ask something "
    "related to resumes, candidates, or hiring."
)


def _generate_answer(query: str, job: dict, candidates: list[dict], results: list[dict]) -> tuple[str, bool]:
    """One natural-language answer to the recruiter's query, grounded in the
    SAME score/matched/missing/explanation just computed above for every
    candidate - never a separate, looser judgment call. Caps how many
    candidates go into the LLM context (highest-scored first) so a very
    large candidate pool never blows the context window."""
    results_by_id = {r["resume_id"]: r for r in results}
    merged = [{**c, **results_by_id.get(c["resume_id"], {})} for c in candidates]
    merged.sort(key=lambda c: (c.get("score") is None, -(c.get("score") or 0)))
    merged = merged[: settings.SEARCH_ANSWER_MAX_CANDIDATES]

    try:
        answer_result = llm_service.answer_search_query(query, job, merged)
    except Exception as exc:  # noqa: BLE001
        logger.error("search_answer_failed", extra={"job_id": job.get("job_id"), "error": str(exc)})
        return "", True

    if not answer_result.get("in_scope", True):
        return SEARCH_REFUSAL_TEXT, False
    return answer_result.get("answer", ""), True
