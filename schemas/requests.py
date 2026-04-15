from typing import Optional

from pydantic import BaseModel


class MapRequest(BaseModel):
    mode: str
    date_a: Optional[str] = None
    date_b: Optional[str] = None
    city: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    mode: str
    date_a: Optional[str] = None
    date_b: Optional[str] = None
    city: Optional[str] = None


class VideoRequest(BaseModel):
    year_a: int
    year_b: int
    city: Optional[str] = None
    fps: int = 2
    size: int = 768
    radius_m: int = 5000


class ChangeBody(BaseModel):
    date1: str
    date2: str
    region: Optional[str] = None
    region_name: Optional[str] = None
    window_days: int = 30


class ReportBody(BaseModel):
    region: str
    date_range: dict
    change_stats: dict
