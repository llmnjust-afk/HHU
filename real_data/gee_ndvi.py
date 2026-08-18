"""
Google Earth Engine MODIS NDVI extraction.

Extracts 16-day NDVI time series for Chinese cities from MODIS MOD13A2
(1km resolution, 2000-present) via Google Earth Engine API.

Authentication:
  Option A (service account):
    1. Create a GEE service account at https://code.earthengine.google.com/
    2. Download the JSON key file
    3. Set environment variable: export EE_SERVICE_ACCOUNT_KEY=/path/to/key.json
    4. Call initialize_gee(service_account=True)

  Option B (interactive OAuth):
    Run ee.Authenticate() once in a browser-enabled environment.
    Credentials are cached in ~/.config/earthengine/

Output format:
  ndvi_dict = {city_id: {year: np.array(n_periods,)}}
  where n_periods = 23 (16-day composites per year)
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

EE_AVAILABLE = False
try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    pass

MODIS_COLLECTION = "MODIS/061/MOD13A2"
NDVI_SCALE = 0.0001
N_PERIODS = 23  # 16-day composites per year
QA_GOOD = {0, 1}  # SummaryQA: 0=good, 1=marginal


def initialize_gee(service_account=False, key_path=None):
    """Initialize Google Earth Engine.

    Args:
        service_account: if True, use service account auth
        key_path: path to service account JSON key file
                  (default: EE_SERVICE_ACCOUNT_KEY env var)
    """
    if not EE_AVAILABLE:
        raise ImportError(
            "earthengine-api not installed. Run: pip install earthengine-api"
        )

    if service_account:
        if key_path is None:
            key_path = os.environ.get("EE_SERVICE_ACCOUNT_KEY")
        if key_path is None:
            raise ValueError(
                "Provide key_path or set EE_SERVICE_ACCOUNT_KEY env var"
            )
        service_account_email = _extract_service_account_email(key_path)
        ee.Initialize(ee.ServiceAccountCredentials(service_account_email, key_path))
    else:
        ee.Initialize()

    print("  Google Earth Engine initialized.")


def _extract_service_account_email(key_path):
    import json
    with open(key_path) as f:
        info = json.load(f)
    return info.get("client_email", "")


def _city_roi(lat, lon, buffer_km=15):
    """Create a circular buffer ROI around city center."""
    if not EE_AVAILABLE:
        raise ImportError("earthengine-api not installed")
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(buffer_km * 1000)


def _extract_ndvi_year(lat, lon, year, buffer_km=15):
    """Extract 16-day NDVI time series for one city-year.

    Returns:
        np.array of shape (N_PERIODS,) with NDVI values (scaled 0-1),
        or None if extraction fails.
    """
    if not EE_AVAILABLE:
        raise ImportError("earthengine-api not installed")

    roi = _city_roi(lat, lon, buffer_km)
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    collection = (ee.ImageCollection(MODIS_COLLECTION)
                  .filterDate(start, end)
                  .select(["NDVI", "SummaryQA"]))

    n_images = collection.size().getInfo()
    if n_images == 0:
        return None

    ndvi_values = np.full(N_PERIODS, np.nan)

    image_list = collection.toList(collection.size())
    for i in range(min(n_images, N_PERIODS)):
        try:
            image = ee.Image(image_list.get(i))
            qa = image.select("SummaryQA")
            ndvi = image.select("NDVI").multiply(NDVI_SCALE)

            masked = ndvi.updateMask(qa.lt(2))
            stats = masked.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=1000,
                maxPixels=1e8,
            ).getInfo()

            val = stats.get("NDVI")
            if val is not None:
                ndvi_values[i] = val
        except Exception:
            continue

    valid = np.isfinite(ndvi_values)
    if valid.sum() < N_PERIODS * 0.5:
        ndvi_values[~valid] = np.nanmean(ndvi_values)
    else:
        ndvi_values[~valid] = np.interp(
            np.where(~valid)[0], np.where(valid)[0], ndvi_values[valid]
        )

    ndvi_values = np.clip(ndvi_values, -0.2, 1.0)
    return ndvi_values


def extract_ndvi_panel(city_info_df, start_year=2005, end_year=2023,
                       buffer_km=15, save_path="data/ndvi_panel.pkl"):
    """Extract NDVI time series for all cities and years.

    Args:
        city_info_df: DataFrame from city_info.load_city_info()
        start_year, end_year: extraction period
        buffer_km: buffer radius around city center
        save_path: where to save the result

    Returns:
        dict {city_id: {year: np.array(N_PERIODS,)}}
    """
    import pickle

    ndvi_dict = {}
    n_cities = len(city_info_df)
    years = range(start_year, end_year + 1)

    for idx, row in city_info_df.iterrows():
        city_id = int(row["city_id"])
        city_name = row["city_name"]
        lat, lon = row["lat"], row["lon"]

        ndvi_dict[city_id] = {}
        for year in years:
            ndvi = _extract_ndvi_year(lat, lon, year, buffer_km)
            if ndvi is not None:
                ndvi_dict[city_id][year] = ndvi

        done = idx + 1
        if done % 10 == 0 or done == n_cities:
            print(f"  NDVI extraction: {done}/{n_cities} cities done", flush=True)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(ndvi_dict, f)
    print(f"  NDVI panel saved to {save_path}")

    return ndvi_dict


def load_ndvi_panel(path="data/ndvi_panel.pkl"):
    """Load pre-extracted NDVI panel from disk."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    if not EE_AVAILABLE:
        print("earthengine-api not installed. Install with:")
        print("  pip install earthengine-api")
        print("\nThen authenticate with:")
        print("  python3 -c \"import ee; ee.Authenticate()\"")
        print("\nOr use a service account key file:")
        print("  export EE_SERVICE_ACCOUNT_KEY=/path/to/key.json")
    else:
        print("earthengine-api available.")
        print("Initialize with: initialize_gee(service_account=True)")
