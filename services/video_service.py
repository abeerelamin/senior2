import io
import tempfile

import ee
import imageio.v2 as imageio
import numpy as np
import requests
from fastapi import HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw

from config import CLASS_PALETTE
from schemas.requests import VideoRequest
from services.ee_runtime import init_ee
import services.ee_runtime as ee_runtime
from services.map_service import resolve_city


def month_sequence(year_a: int, year_b: int):
    months = []
    for y in range(year_a, year_b + 1):
        for m in range(1, 13):
            months.append((y, m))
    return months


def next_month(y: int, m: int):
    if m == 12:
        return y + 1, 1
    return y, m + 1


def monthly_dw_visual(region: ee.Geometry, y: int, m: int) -> ee.Image:
    start = f"{y:04d}-{m:02d}-01"
    ny, nm = next_month(y, m)
    end = f"{ny:04d}-{nm:02d}-01"

    palette = ["#" + c for c in CLASS_PALETTE]

    img = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start, end)
        .select("label")
        .mode()
    )

    return img.visualize(min=0, max=8, palette=palette).clip(region)


def add_frame_label(img: Image.Image, label: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    pad = 12
    box_h = 34
    draw.rounded_rectangle(
        [pad, pad, 220, pad + box_h],
        radius=10,
        fill=(15, 23, 42, 210),
        outline=(100, 116, 139, 255),
        width=1,
    )
    draw.text((pad + 10, pad + 9), label, fill=(229, 231, 235))
    return img


def ee_region_bbox(region: ee.Geometry):
    coords = region.bounds().coordinates().getInfo()[0]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return [min(xs), min(ys), max(xs), max(ys)]


def download_month_frame(region: ee.Geometry, y: int, m: int, size: int) -> np.ndarray:
    vis = monthly_dw_visual(region, y, m)
    bbox = ee_region_bbox(region)

    url = vis.getThumbURL({
        "region": bbox,
        "dimensions": size,
        "format": "png",
    })

    r = requests.get(url, timeout=120)
    r.raise_for_status()

    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img = add_frame_label(img, f"{y}-{m:02d}")
    return np.array(img)


def timeseries_video(req: VideoRequest):
    init_ee()
    if not ee_runtime.EE_READY:
        raise HTTPException(
            status_code=500,
            detail=f"Earth Engine is not ready: {ee_runtime.EE_ERROR}",
        )

    if req.year_a > req.year_b:
        raise HTTPException(status_code=400, detail="year_a must be <= year_b")

    city_name, lat, lon = resolve_city(req.city)

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(req.radius_m).bounds()

    months = month_sequence(req.year_a, req.year_b)
    frames = []

    for y, m in months:
        try:
            frame = download_month_frame(region, y, m, req.size)
            frames.append(frame)
        except Exception as e:
            print(f"Skipping frame {y}-{m:02d}: {e}")

    if not frames:
        raise HTTPException(status_code=500, detail="Could not generate any monthly frames.")

    safe_city = city_name.replace(" ", "_").replace("/", "_")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name
    tmp.close()

    writer = imageio.get_writer(
        tmp_path,
        fps=max(1, req.fps),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    filename = f"timeseries_{safe_city}_{req.year_a}_{req.year_b}.mp4"
    return FileResponse(
        tmp_path,
        media_type="video/mp4",
        filename=filename,
    )
