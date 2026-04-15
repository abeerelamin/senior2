from datetime import date, timedelta
from typing import Optional

import ee
from fastapi import HTTPException
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim

from config import (
    DW_MIN_DATE,
    LOCATION_LAT,
    LOCATION_LON,
    LOCATION_NAME,
    YEARS,
)
from gee_utils import get_dw_tile_urls, tile_url_at_point, tile_url_global_year
from schemas.requests import MapRequest
from services.ee_runtime import init_ee
import services.ee_runtime as ee_runtime


geolocator = Nominatim(user_agent="dw-change-app")


def parse_iso_date(s: Optional[str]) -> date:
    if not s or not str(s).strip():
        return date.today() - timedelta(days=2)
    try:
        return date.fromisoformat(str(s).strip()[:10])
    except ValueError:
        return date.today() - timedelta(days=2)


def clamp_map_date(d: date) -> date:
    today = date.today()
    if d < DW_MIN_DATE:
        return DW_MIN_DATE
    if d > today:
        return today
    return d


def display_date(d: date) -> str:
    return d.strftime("%d %b %Y")


def resolve_city(city: Optional[str]):
    if not city or not city.strip():
        return LOCATION_NAME, LOCATION_LAT, LOCATION_LON

    try:
        loc = geolocator.geocode(city.strip())
        if loc:
            return city.strip(), loc.latitude, loc.longitude
    except (GeocoderTimedOut, GeocoderUnavailable):
        pass

    return LOCATION_NAME, LOCATION_LAT, LOCATION_LON


def map_config(req: MapRequest):
    init_ee()
    if not ee_runtime.EE_READY:
        raise HTTPException(
            status_code=500,
            detail=f"Earth Engine is not ready: {ee_runtime.EE_ERROR}",
        )

    mode = req.mode
    da = clamp_map_date(parse_iso_date(req.date_a))
    db = clamp_map_date(parse_iso_date(req.date_b))
    if db < da:
        da, db = db, da

    # Dynamic World uses annual composites (original app): year from each selected date
    year_a = da.year
    year_b = db.year
    if year_a not in YEARS:
        year_a = YEARS[0]
    if year_b not in YEARS:
        year_b = YEARS[-1]

    if mode == "home":
        if not req.city or not str(req.city).strip():
            url = tile_url_global_year(year_a)
            return {
                "city": "World",
                "center_lat": 15.0,
                "center_lon": 0.0,
                "date_a": da.isoformat(),
                "date_b": da.isoformat(),
                "date_a_display": display_date(da),
                "date_b_display": display_date(da),
                "dw_year_a": year_a,
                "dw_year_b": year_a,
                "mode": mode,
                "tiles": {"a": url, "b": None, "change": None},
                "map_zoom": 2,
            }

        city_name, lat, lon = resolve_city(req.city)
        point = ee.Geometry.Point([lon, lat])
        url = tile_url_at_point(point, year_a)
        return {
            "city": city_name,
            "center_lat": lat,
            "center_lon": lon,
            "date_a": da.isoformat(),
            "date_b": da.isoformat(),
            "date_a_display": display_date(da),
            "date_b_display": display_date(da),
            "dw_year_a": year_a,
            "dw_year_b": year_a,
            "mode": mode,
            "tiles": {"a": url, "b": None, "change": None},
            "map_zoom": 11,
        }

    city_name, lat, lon = resolve_city(req.city)
    point = ee.Geometry.Point([lon, lat])
    tiles = get_dw_tile_urls(point, year_a, year_b)

    return {
        "city": city_name,
        "center_lat": lat,
        "center_lon": lon,
        "date_a": da.isoformat(),
        "date_b": db.isoformat(),
        "date_a_display": display_date(da),
        "date_b_display": display_date(db),
        "dw_year_a": year_a,
        "dw_year_b": year_b,
        "mode": mode,
        "tiles": tiles,
    }
