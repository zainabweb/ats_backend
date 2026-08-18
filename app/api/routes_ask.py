from fastapi import APIRouter, Depends

from app.core.security import require_api_key
from app.models.schemas import AskRequest, AskResponse
from app.services import rag_service

router = APIRouter(prefix="/ask", tags=["ask"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=AskResponse)
def ask(payload: AskRequest):
    return rag_service.ask(payload.question, job_id=payload.job_id)
