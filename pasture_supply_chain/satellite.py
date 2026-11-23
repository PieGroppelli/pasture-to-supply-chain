"""Google Earth Engine layer: Sentinel-2 NDVI, ERA5-Land weather, AlphaEarth embeddings, thumbnails."""
from __future__ import annotations

import io
import json

import ee
import numpy as np
import pandas as pd
import requests
from PIL import Image

S2_SR = "COPERNICUS/S2_SR_HARMONIZED"     # surface reflectance (preferred)
S2_TOA = "COPERNICUS/S2_HARMONIZED"       # top of atmosphere (fallback for early years)
ERA5 = "ECMWF/ERA5_LAND/DAILY_AGGR"
EMBEDDINGS = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


def init(project_id: str) -> None:
    """Connect to Earth Engine (asks for login the first time)."""
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def load_fields(path: str) -> list[dict]:
    """Read a GeoJSON FeatureCollection of field polygons. Each feature needs a 'name' property."""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    fields = []
    for feat in gj["features"]:
        geom = ee.Geometry(feat["geometry"])
        fields.append({
            "name": feat["properties"]["name"],
            "geometry": geom,
            "area_ha": geom.area(maxError=1).getInfo() / 10_000,
        })
    return fields


# ---------------------------------------------------------------------------
# NDVI time series
# ---------------------------------------------------------------------------
def _ndvi_features(collection: str, geom: ee.Geometry, start: str, end: str) -> pd.DataFrame:
    col = (ee.ImageCollection(collection).filterBounds(geom).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))

    def per_image(img):
        if collection == S2_SR:
            scl = img.select("SCL")
            clear = scl.eq(4).Or(scl.eq(5))            # 4 = vegetation, 5 = bare soil
        else:
            qa = img.select("QA60")
            clear = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(clear)
        n = ndvi.reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e8).get("ndvi")
        c = clear.rename("clear").reduceRegion(ee.Reducer.mean(), geom, 20, maxPixels=1e8).get("clear")
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "ndvi": n, "clear": c})

    fc = col.map(per_image).filter(ee.Filter.notNull(["ndvi", "clear"]))
    rows = fc.reduceColumns(ee.Reducer.toList(3), ["date", "ndvi", "clear"]).get("list").getInfo()
    return pd.DataFrame(rows, columns=["date", "ndvi", "clear"])


def ndvi_observations(geom: ee.Geometry, start: str, end: str,
                      min_clear: float = 0.6, min_obs: int = 6) -> pd.DataFrame:
    """Cloud-free NDVI observations over a field. Falls back to TOA data if SR is too sparse."""
    df = _ndvi_features(S2_SR, geom, start, end)
    df = df[df["clear"] >= min_clear]
    if len(df) < min_obs:
        toa = _ndvi_features(S2_TOA, geom, start, end)
        df = toa[toa["clear"] >= min_clear]
    return df.groupby("date", as_index=False)["ndvi"].mean()


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def weather(geom: ee.Geometry, start: str, end: str) -> pd.DataFrame:
    """Daily solar radiation (J/m2) and mean temperature (deg C) at the field centroid."""
    point = geom.centroid(maxError=1)
    col = (ee.ImageCollection(ERA5).filterDate(start, end)
           .select(["surface_solar_radiation_downwards_sum", "temperature_2m"]))

    def per_image(img):
        v = img.reduceRegion(ee.Reducer.first(), point, 11132)
        return ee.Feature(None, {
            "date": img.date().format("YYYY-MM-dd"),
            "ssrd": v.get("surface_solar_radiation_downwards_sum"),
            "t2m": v.get("temperature_2m"),
        })

    fc = ee.FeatureCollection(col.map(per_image)).filter(ee.Filter.notNull(["ssrd", "t2m"]))
    rows = fc.reduceColumns(ee.Reducer.toList(3), ["date", "ssrd", "t2m"]).get("list").getInfo()
    df = pd.DataFrame(rows, columns=["date", "ssrd_j_m2", "t2m_k"])
    df["t_mean_c"] = df["t2m_k"] - 273.15
    return df[["date", "ssrd_j_m2", "t_mean_c"]]


# ---------------------------------------------------------------------------
# AlphaEarth embeddings: did the field change character between two years?
# ---------------------------------------------------------------------------
def embedding_similarity(geom: ee.Geometry, year_a: int, year_b: int) -> float | None:
    """Cosine similarity of the 64-dim AlphaEarth embedding of the field in two years.

    Close to 1 = same land cover and condition; a clear drop can mean ploughing,
    reseeding or a change of use. Returns None if the data is not available.
    """
    bands = [f"A{i:02d}" for i in range(64)]

    def vector(year):
        img = (ee.ImageCollection(EMBEDDINGS).filterBounds(geom)
               .filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic())
        d = img.reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e8).getInfo() or {}
        if any(d.get(b) is None for b in bands):
            return None
        return np.array([d[b] for b in bands], dtype=float)

    va, vb = vector(year_a), vector(year_b)
    if va is None or vb is None:
        return None
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------
def thumbnail(geom: ee.Geometry, year: int, kind: str = "rgb") -> Image.Image:
    """Summer composite (May-September) of the field: true colour or NDVI."""
    region = geom.buffer(150).bounds()
    col = (ee.ImageCollection(S2_SR).filterBounds(geom)
           .filterDate(f"{year}-05-01", f"{year}-09-30")
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))
    img = col.median()
    if kind == "rgb":
        vis = img.visualize(bands=["B4", "B3", "B2"], min=0, max=2500)
    else:
        vis = img.normalizedDifference(["B8", "B4"]).visualize(
            min=0.2, max=0.9, palette=["a50026", "fdae61", "ffffbf", "a6d96a", "006837"])
    outline = (ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(geom)]), 1, 2)
               .visualize(min=0, max=1, palette=["00ffff", "00ffff"]))
    url = vis.blend(outline).getThumbURL({"region": region, "dimensions": 512, "format": "png"})
    return Image.open(io.BytesIO(requests.get(url, timeout=60).content))
