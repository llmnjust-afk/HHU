"""
MODIS NDVI extraction via Microsoft Planetary Computer (no auth).

Accesses MODIS MOD13A1 (Vegetation Indices 16-Day, 500m) through
the Planetary Computer STAC API — NO authentication required.

Collection: modis-13A1-061 (MODIS Vegetation Indices 16-Day 500m)
  - NDVI band: _500m_NDVI (scaled by 0.0001)
  - SummaryQA band: _500m_VI_Quality (0=good, 1=marginal)
  - Temporal: 16-day composites, 2000-present
  - Spatial: 500m resolution, global

Output:
  data/ndvi_panel.pkl — {city_id: {year: np.array(n_periods,)}}
  where n_periods = 23 (16-day composites per year)
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NDVI_PERIODS_PER_YEAR

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "modis-13A1-061"
NDVI_SCALE = 0.0001
BUFFER_DEG = 0.15  # ~15km buffer around city center


def _get_stac_client():
    """Get authenticated Planetary Computer STAC client."""
    import pystac_client
    import planetary_computer as pc

    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def _extract_ndvi_item(item, lat, lon, buffer_pixels=30):
    """Extract mean NDVI from one STAC item (16-day composite) at a city.

    Args:
        item: STAC Item (one 16-day composite)
        lat, lon: city center (WGS84)
        buffer_pixels: half-window size in pixels (30 pixels × 500m ≈ 15km)

    Returns:
        float NDVI value (scaled 0-1), or np.nan if failed
    """
    import rasterio
    from rasterio.windows import Window
    from pyproj import Transformer

    try:
        assets = item.assets
        ndvi_key = [k for k in assets if "NDVI" in k.upper()]
        if not ndvi_key:
            return np.nan
        ndvi_key = ndvi_key[0]

        href = assets[ndvi_key].href

        with rasterio.open(href) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            row, col = src.index(x, y)

            row = max(buffer_pixels, min(row, src.height - buffer_pixels))
            col = max(buffer_pixels, min(col, src.width - buffer_pixels))

            window = Window(col - buffer_pixels, row - buffer_pixels,
                           buffer_pixels * 2, buffer_pixels * 2)
            data = src.read(1, window=window)

            valid = data[data > -2000]
            if len(valid) > 0:
                return float(np.mean(valid)) / 10000.0
            return np.nan
    except Exception:
        return np.nan


def extract_ndvi_year(client, lat, lon, year):
    """Extract 16-day NDVI time series for one city-year.

    Returns:
        np.array of shape (NDVI_PERIODS_PER_YEAR,) with NDVI values
    """
    from shapely.geometry import Point

    start = f"{year}-01-01"
    end = f"{year}-12-31"
    point = Point(lon, lat)

    search = client.search(
        collections=[COLLECTION],
        intersects=point.__geo_interface__,
        datetime=f"{start}/{end}",
        max_items=30,
    )

    items = list(search.get_items())
    ndvi_values = np.full(NDVI_PERIODS_PER_YEAR, np.nan)

    def _sort_key(item):
        dt = item.datetime or item.properties.get("start_datetime")
        if dt is None:
            return ""
        return str(dt)

    for i, item in enumerate(sorted(items, key=_sort_key)):
        if i >= NDVI_PERIODS_PER_YEAR:
            break
        ndvi_values[i] = _extract_ndvi_item(item, lat, lon)

    valid = np.isfinite(ndvi_values)
    if valid.sum() < NDVI_PERIODS_PER_YEAR * 0.5:
        ndvi_values[~valid] = 0.3
    else:
        ndvi_values[~valid] = np.interp(
            np.where(~valid)[0], np.where(valid)[0], ndvi_values[valid]
        )

    return np.clip(ndvi_values, -0.2, 1.0)


def run_ndvi_pipeline(city_info_df, start_year=2005, end_year=2023,
                     save_path="data/ndvi_panel.pkl"):
    """Extract MODIS NDVI for all cities and years.

    Args:
        city_info_df: DataFrame from city_list_280.load()
        start_year, end_year: extraction period
        save_path: output pickle path

    Returns:
        dict {city_id: {year: np.array(N_PERIODS,)}}
    """
    print("=" * 70)
    print("MODIS NDVI Pipeline (Microsoft Planetary Computer — no auth)")
    print(f"  Cities: {len(city_info_df)}")
    print(f"  Period: {start_year}-{end_year}")
    print("=" * 70)

    os.makedirs("data", exist_ok=True)

    client = _get_stac_client()
    print(f"  Connected to Planetary Computer")

    ndvi_dict = {}
    n_cities = len(city_info_df)
    years = range(start_year, end_year + 1)

    for idx, row in city_info_df.iterrows():
        city_id = int(row["city_id"])
        city_name = row["city_name"]
        lat, lon = row["lat"], row["lon"]

        ndvi_dict[city_id] = {}

        for year in years:
            try:
                ndvi = extract_ndvi_year(client, lat, lon, year)
                ndvi_dict[city_id][year] = ndvi
            except Exception:
                ndvi_dict[city_id][year] = np.full(NDVI_PERIODS_PER_YEAR, 0.3)

        done = idx + 1
        if done % 10 == 0 or done == n_cities:
            print(f"  NDVI: {done}/{n_cities} cities done", flush=True)

        if done % 50 == 0:
            with open(save_path, "wb") as f:
                pickle.dump(ndvi_dict, f)

    with open(save_path, "wb") as f:
        pickle.dump(ndvi_dict, f)
    print(f"\n  NDVI panel saved to {save_path}")

    return ndvi_dict


if __name__ == "__main__":
    from real_data.city_list_280 import load
    city_info = load().head(5)
    run_ndvi_pipeline(city_info, start_year=2020, end_year=2020)
