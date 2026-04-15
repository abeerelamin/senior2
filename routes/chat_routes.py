from fastapi import APIRouter

from schemas.requests import ChatRequest
from services.chat_service import chat as chat_service


router = APIRouter()


@router.post("/chat")
def chat(req: ChatRequest):
    return chat_service(req)
