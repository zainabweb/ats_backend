"""
Minimal API-key auth for HR-facing routes (job CRUD, search, delete, screen).
Candidate-facing upload can stay open but should sit behind a rate limiter
at the reverse-proxy layer in production.
"""
from fastapi import Header, HTTPException, status

from app.config import get_settings

settings = get_settings()


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    if not x_api_key or x_api_key != settings.ATS_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
        )
