from fastapi import APIRouter, Depends

from app.core.security import require_api_key
from app.models.schemas import SearchRequest, SearchResponse
from app.services import rag_service

router = APIRouter(prefix="/search", tags=["search"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=SearchResponse)
def search(payload: SearchRequest):
    return rag_service.search(payload.job_id, payload.query, payload.top_k, show_all=payload.show_all)
