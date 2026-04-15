import json

from chat_utils import ask_chatbot
from schemas.requests import ChatRequest
from services.map_service import resolve_city, clamp_map_date, parse_iso_date, display_date


def chat(req: ChatRequest):
    city_name, lat, lon = resolve_city(req.city)
    da = clamp_map_date(parse_iso_date(req.date_a))
    db = clamp_map_date(parse_iso_date(req.date_b))

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that explains Dynamic World land cover "
            "maps and changes over time in SIMPLE language. "
            "The app uses dates to pick calendar years; Dynamic World layers are annual composites "
            "(mode of daily labels over each year), matching the standard Earth Engine recipe. "
            "Modes: home (single map; world view if no region, else zoomed region), "
            "change_detection (two maps, date A vs B), timeseries, prediction. "
            "In prediction mode the map shows the same historical A/B/change layers as time series; "
            "your job is to discuss plausible future land-cover outcomes qualitatively, "
            "not as a numerical forecast, unless the user asks for general education. "
            f"Current mode: {req.mode}, date A: {display_date(da)}, date B: {display_date(db)}. "
            f"Current region: {city_name} at ({lat:.3f}, {lon:.3f}). "
            "Return your answer STRICTLY as JSON with two keys: "
            "'explanation' and 'summary'. "
            "'explanation' can be a normal short paragraph. "
            "'summary' must be at most two short sentences."
        ),
    }

    messages_for_api = [
        system_msg,
        {"role": "user", "content": req.message},
    ]

    raw = ask_chatbot(messages_for_api)

    explanation = raw
    summary = raw

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            explanation = data.get("explanation", raw)
            summary = data.get("summary", explanation)
    except Exception:
        parts = explanation.split(".")
        summary = ".".join(parts[:2]).strip()
        if summary and not summary.endswith("."):
            summary += "."

    return {
        "reply": explanation,
        "summary": summary,
    }
