import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import routes_jobs, routes_resumes, routes_search
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, new_request_id, setup_logging

settings = get_settings()
setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

app = FastAPI(title=settings.APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    rid = new_request_id()
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)
    logger.info(
        "request_completed",
        extra={"method": request.method, "path": request.url.path,
               "status": response.status_code, "duration_ms": duration_ms},
    )
    response.headers["X-Request-ID"] = rid
    return response


app.include_router(routes_jobs.router)
app.include_router(routes_resumes.router)
app.include_router(routes_search.router)


@app.get("/health")
def health():
    try:
        from app.services import chroma_service
        job_count = len(chroma_service.list_jobs())
        resume_count = len(chroma_service.list_resumes())
        db_status = "connected"
    except Exception as exc:  # noqa: BLE001
        job_count, resume_count, db_status = None, None, f"error: {exc}"

    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "chroma_mode": settings.CHROMA_MODE,
        "chroma_status": db_status,
        "jobs_stored": job_count,
        "resumes_stored": resume_count,
    }


# Optional: serve the plain HTML/JS frontend at /
app.mount("/", StaticFiles(directory="frontend/static", html=True), name="frontend")
