"""
Custom exceptions + a single JSON envelope for every error response:

    { "success": false, "error": { "code": "...", "message": "..." } }
"""
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


class ATSException(Exception):
    code = "INTERNAL_ERROR"
    status_code = 500

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class UnsupportedFileType(ATSException):
    code = "UNSUPPORTED_FILE_TYPE"
    status_code = 400


class FileTooLarge(ATSException):
    code = "FILE_TOO_LARGE"
    status_code = 400


class ResumeParseError(ATSException):
    code = "RESUME_PARSE_ERROR"
    status_code = 422


class EmbeddingGenerationError(ATSException):
    code = "EMBEDDING_GENERATION_ERROR"
    status_code = 502


class LLMResponseError(ATSException):
    code = "LLM_RESPONSE_ERROR"
    status_code = 502


class LLMQuotaExhaustedError(LLMResponseError):
    """Every configured API key, across every provider, is currently
    rate-limited/quota-exhausted - as opposed to a one-off malformed
    response. Callers processing a batch (e.g. /resumes/screen) can catch
    this specifically to stop early instead of burning time retrying every
    remaining file against a quota that's still exhausted."""
    code = "LLM_QUOTA_EXHAUSTED"


class ChromaDBError(ATSException):
    code = "CHROMADB_ERROR"
    status_code = 500


class JobNotFoundError(ATSException):
    code = "JOB_NOT_FOUND"
    status_code = 404


class ResumeNotFoundError(ATSException):
    code = "RESUME_NOT_FOUND"
    status_code = 404


def _envelope(exc: ATSException) -> dict:
    return {"success": False, "error": {"code": exc.code, "message": exc.message}}


async def ats_exception_handler(request: Request, exc: ATSException) -> JSONResponse:
    logger.warning("ats_exception", extra={"path": request.url.path, "code": exc.code, "error_message": exc.message})
    return JSONResponse(status_code=exc.status_code, content=_envelope(exc))


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the full traceback to the terminal - previously this was silently
    # swallowed, showing nothing in the console for a 500 response.
    logger.exception("unhandled_exception", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": {"code": "INTERNAL_ERROR", "message": str(exc)}},
    )


def register_exception_handlers(app) -> None:
    app.add_exception_handler(ATSException, ats_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
