# main.py
import json
import os
from typing import Optional

import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from pydantic import BaseModel

from config import YEARS, LOCATION_LAT, LOCATION_LON, LOCATION_NAME
from gee_utils import get_dw_tile_urls
from chat_utils import ask_chatbot

# ---------------------------------------------------------------------
# FastAPI app + static frontend
# ---------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


# ---------------------------------------------------------------------
# Earth Engine init
# ---------------------------------------------------------------------
EE_READY = False
EE_ERROR = None


def init_ee():
    """Initialize Earth Engine once. Do not crash the whole app if it fails."""
    global EE_READY, EE_ERROR
    if EE_READY:
        return

    try:
        service_account_json = os.environ.get("EE_SERVICE_ACCOUNT_JSON")
        if not service_account_json:
            raise RuntimeError(
                "EE_SERVICE_ACCOUNT_JSON is missing. "
                "Set it in Render → Environment."
            )

        info = json.loads(service_account_json)
        email = info["client_email"]
        project_id = info.get("project_id")

        credentials = ee.ServiceAccountCredentials(email, key_data=service_account_json)
        if project_id:
            ee.Initialize(credentials, project=project_id)
        else:
            ee.Initialize(credentials)

        EE_READY = True
        EE_ERROR = None
        print("✅ Earth Engine initialized successfully.")
    except Exception as e:
        EE_READY = False
        EE_ERROR = str(e)
        print("❌ Failed to initialize Earth Engine:", EE_ERROR)


@app.on_event("startup")
async def startup_event():
    init_ee()


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class MapRequest(BaseModel):
    mode: str
    year_a: int
    year_b: int
    city: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    mode: str
    year_a: int
    year_b: int
    city: Optional[str] = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
geolocator = Nominatim(user_agent="dw-change-app")


def resolve_city(city: Optional[str]):
    """Return (name, lat, lon). Fallback to defaults from config.py."""
    if not city or not city.strip():
        return LOCATION_NAME, LOCATION_LAT, LOCATION_LON

    try:
        loc = geolocator.geocode(city.strip())
        if loc:
            return city.strip(), loc.latitude, loc.longitude
    except (GeocoderTimedOut, GeocoderUnavailable):
        pass

    return LOCATION_NAME, LOCATION_LAT, LOCATION_LON


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ee_ready": EE_READY,
        "ee_error": EE_ERROR,
    }


# ---------------------------------------------------------------------
# Map config endpoint
# ---------------------------------------------------------------------
@app.post("/map-config")
def map_config(req: MapRequest):
    if not EE_READY:
        raise HTTPException(
            status_code=500,
            detail=f"Earth Engine is not ready: {EE_ERROR}",
        )

    mode = req.mode
    year_a = req.year_a
    year_b = req.year_b

    # clamp years
    if year_a not in YEARS:
        year_a = YEARS[0]
    if year_b not in YEARS:
        year_b = YEARS[-1]

    city_name, lat, lon = resolve_city(req.city)
    location_point = ee.Geometry.Point([lon, lat])

    if mode == "single_year":
        tiles = get_dw_tile_urls(location_point, year_a, year_a)
    else:
        tiles = get_dw_tile_urls(location_point, year_a, year_b)

    return {
        "city": city_name,
        "center_lat": lat,
        "center_lon": lon,
        "year_a": year_a,
        "year_b": year_b,
        "mode": mode,
        "tiles": tiles,
    }


# ---------------------------------------------------------------------
# Chat endpoint – returns explanation + short summary (≤ ~2 sentences)
# ---------------------------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    """
    Returns:
      reply   -> full explanation for the chat window
      summary -> very short summary (1–2 sentences) for the bottom bar
    """
    city_name, lat, lon = resolve_city(req.city)

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that explains Dynamic World land cover "
            "maps and changes over time in SIMPLE language. "
            "The app has three analysis modes: change_detection, single_year, timeseries. "
            f"Current mode: {req.mode}, years: {req.year_a}–{req.year_b}. "
            f"Current region: {city_name} at ({lat:.3f}, {lon:.3f}). "
            "First, form a clear explanation for the user. "
            "Then return your answer STRICTLY as JSON with TWO keys: "
            "'explanation' and 'summary'. "
            "'explanation' is a friendly paragraph (up to ~8 sentences). "
            "'summary' is at most TWO short sentences that highlight the main "
            "changes or what the current map likely shows. "
            "Do not include any extra text outside the JSON."
        ),
    }

    messages_for_api = [
        system_msg,
        {"role": "user", "content": req.message},
    ]

    raw = ask_chatbot(messages_for_api)

    explanation = raw
    summary = raw

    # Try to parse JSON from model
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            explanation = data.get("explanation", raw)
            summary = data.get("summary", explanation)
    except Exception:
        # Fallback: cut summary to ~2 sentences
        parts = explanation.split(".")
        summary = ".".join(parts[:2]).strip()
        if summary and not summary.endswith("."):
            summary += "."

    return {
        "reply": explanation,
        "summary": summary,
    }
