from fastapi import APIRouter
from fastapi.responses import FileResponse

import services.ee_runtime as ee_runtime


router = APIRouter()


@router.get("/")
def serve_frontend():
    return FileResponse("index.html")


@router.get("/health")
def health():
    return {
        "status": "ok",
        "ee_ready": ee_runtime.EE_READY,
        "ee_error": ee_runtime.EE_ERROR,
    }
