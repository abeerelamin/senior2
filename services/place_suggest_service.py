"""Lightweight place search for the region field (Nominatim)."""

from typing import Any, Dict, List

import requests

NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
# https://operations.osmfoundation.org/policies/nominatim/
NOMINATIM_HEADERS = {
    "User-Agent": "EarthMonitor/1.0 (region autocomplete; local dev)",
    "Accept-Language": "en",
}


def fetch_place_suggestions(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if len(q) < 2:
        return []
    lim = max(1, min(int(limit or 8), 10))
    try:
        r = requests.get(
            NOMINATIM_SEARCH,
            params={
                "q": q,
                "format": "json",
                "limit": lim,
                "addressdetails": "0",
            },
            headers=NOMINATIM_HEADERS,
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    out: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        name = row.get("display_name")
        if not name:
            continue
        try:
            lat = float(row.get("lat"))
            lon = float(row.get("lon"))
        except (TypeError, ValueError):
            continue
        out.append({"label": name, "lat": lat, "lon": lon})
    return out
