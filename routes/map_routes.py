from fastapi import APIRouter, Query

from schemas.requests import MapRequest, VideoRequest
from services.map_service import map_config as map_config_service
from services.place_suggest_service import fetch_place_suggestions
from services.video_service import timeseries_video as timeseries_video_service


router = APIRouter()


@router.post("/map-config")
def map_config(req: MapRequest):
    return map_config_service(req)


@router.post("/timeseries-video")
def timeseries_video(req: VideoRequest):
    return timeseries_video_service(req)


@router.get("/api/place-suggestions")
def place_suggestions(
    q: str = Query("", max_length=200),
    limit: int = Query(8, ge=1, le=10),
):
    return fetch_place_suggestions(q, limit)
