"""
ERA5 extreme weather event extraction.

Downloads ERA5 daily meteorological data via the Copernicus CDS API and
identifies extreme weather events (heat waves, heavy rain, drought) for
each city-year.

Authentication:
  1. Register at https://cds.climate.copernicus.eu/
  2. Get your API key from your profile page
  3. Create ~/.cdsapirc with:
       url: https://cds.climate.copernicus.eu/api
       key: {your-uid}:{your-api-key}
  4. Accept the ERA5 license terms on the CDS website

Output format:
  events_dict = {city_id: {year: [(event_type, start_day, end_day, intensity)]}}
  event_type: "heat" | "rain" | "drought"
  start_day, end_day: day-of-year (1-365)
  intensity: float (normalized 0-1+)
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

CDS_AVAILABLE = False
try:
    import cdsapi
    CDS_AVAILABLE = True
except ImportError:
    pass

# Extreme weather thresholds (China Meteorological Administration standards)
HEAT_THRESHOLD = 35.0          # °C, daily max temperature
HEAT_MIN_DAYS = 3              # consecutive days for heat wave
HEAVY_RAIN_THRESHOLD = 50.0    # mm/day, 暴雨 standard
DROUGHT_DAYS = 30              # consecutive dry days
DROUGHT_RAIN_THRESHOLD = 1.0   # mm/day, below this = dry day


def download_era5_daily(lat, lon, year, save_dir="data/era5_raw"):
    """Download ERA5 daily data for one city-year via CDS API.

    Downloads: max/min/mean 2m temperature, total precipitation.
    Saves to NetCDF and returns the file path.

    Args:
        lat, lon: city center coordinates
        year: year to download
        save_dir: directory for raw NetCDF files

    Returns:
        path to downloaded NetCDF file, or None if download fails
    """
    if not CDS_AVAILABLE:
        raise ImportError("cdsapi not installed. Run: pip install cdsapi")

    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, f"era5_{lat:.2f}_{lon:.2f}_{year}.nc")

    if os.path.exists(out_file):
        return out_file

    buffer = 0.25  # ~25km box around city center
    area = [lat + buffer, lon - buffer, lat - buffer, lon + buffer]

    c = cdsapi.Client(quiet=True)

    try:
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": [
                    "maximum_2m_temperature",
                    "minimum_2m_temperature",
                    "2m_temperature",
                    "total_precipitation",
                ],
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": "13:00",
                "area": area,
            },
            out_file,
        )
        return out_file
    except Exception as e:
        print(f"  Warning: CDS download failed for ({lat},{lon}) {year}: {e}")
        return None


def parse_era5_netcdf(nc_path):
    """Parse ERA5 NetCDF file into a daily DataFrame.

    Returns:
        DataFrame with columns: date, tmax, tmin, tmean, precip
        (temperatures in °C, precipitation in mm/day)
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    df = ds.to_dataframe().reset_index()

    col_map = {
        "mx2t": "tmax", "mx2t27": "tmax",
        "mn2t": "tmin", "mn2t27": "tmin",
        "t2m": "tmean",
        "tp": "precip",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    for col in ["tmax", "tmin", "tmean"]:
        if col in df.columns:
            df[col] = df[col] - 273.15  # K → °C

    if "precip" in df.columns:
        df["precip"] = df["precip"] * 1000  # m → mm

    ds.close()
    return df


def identify_heat_waves(tmax_series, dates):
    """Identify heat wave events from daily max temperature.

    Heat wave: ≥3 consecutive days with tmax > 35°C.

    Returns:
        list of (start_day, end_day, intensity) tuples
    """
    events = []
    hot = (tmax_series > HEAT_THRESHOLD).values
    n = len(hot)
    i = 0

    while i < n:
        if hot[i]:
            j = i
            while j < n and hot[j]:
                j += 1
            duration = j - i
            if duration >= HEAT_MIN_DAYS:
                start_day = dates.iloc[i].timetuple().tm_yday
                end_day = dates.iloc[j - 1].timetuple().tm_yday
                intensity = float(np.mean(tmax_series.iloc[i:j] - HEAT_THRESHOLD))
                intensity = intensity / 10.0  # normalize: 10°C above threshold = 1.0
                events.append(("heat", start_day, end_day, intensity))
            i = j
        else:
            i += 1

    return events


def identify_heavy_rain(precip_series, dates):
    """Identify heavy rain events from daily precipitation.

    Heavy rain: daily precipitation > 50mm (暴雨 standard).

    Returns:
        list of (start_day, end_day, intensity) tuples
    """
    events = []
    heavy = (precip_series > HEAVY_RAIN_THRESHOLD).values
    n = len(heavy)
    i = 0

    while i < n:
        if heavy[i]:
            start_day = dates.iloc[i].timetuple().tm_yday
            j = i
            while j < n and heavy[j]:
                j += 1
            end_day = dates.iloc[j - 1].timetuple().tm_yday
            total_precip = float(np.sum(precip_series.iloc[i:j]))
            intensity = (total_precip / HEAVY_RAIN_THRESHOLD) / len(precip_series.iloc[i:j])
            events.append(("rain", start_day, end_day, intensity))
            i = j
        else:
            i += 1

    return events


def identify_drought(precip_series, dates):
    """Identify drought events from daily precipitation.

    Drought: ≥30 consecutive days with precipitation < 1mm/day.

    Returns:
        list of (start_day, end_day, intensity) tuples
    """
    events = []
    dry = (precip_series < DROUGHT_RAIN_THRESHOLD).values
    n = len(dry)
    i = 0

    while i < n:
        if dry[i]:
            j = i
            while j < n and dry[j]:
                j += 1
            duration = j - i
            if duration >= DROUGHT_DAYS:
                start_day = dates.iloc[i].timetuple().tm_yday
                end_day = dates.iloc[j - 1].timetuple().tm_yday
                rain_deficit = float(np.sum(DROUGHT_RAIN_THRESHOLD - precip_series.iloc[i:j]))
                intensity = rain_deficit / 100.0  # normalize: 100mm deficit = 1.0
                events.append(("drought", start_day, end_day, intensity))
            i = j
        else:
            i += 1

    return events


def extract_events_for_city_year(nc_path):
    """Parse ERA5 data and identify all extreme weather events for one city-year.

    Returns:
        list of (event_type, start_day, end_day, intensity) tuples
    """
    df = parse_era5_netcdf(nc_path)

    date_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if date_col:
        dates = pd.to_datetime(df[date_col[0]])
    else:
        dates = pd.date_range(start=f"{df.index[0]}", periods=len(df), freq="D")

    events = []

    if "tmax" in df.columns:
        events.extend(identify_heat_waves(df["tmax"], dates))

    if "precip" in df.columns:
        events.extend(identify_heavy_rain(df["precip"], dates))
        events.extend(identify_drought(df["precip"], dates))

    return events


def extract_weather_panel(city_info_df, start_year=2005, end_year=2023,
                          save_path="data/weather_events.pkl"):
    """Extract extreme weather events for all cities and years.

    Args:
        city_info_df: DataFrame from city_info.load_city_info()
        start_year, end_year: extraction period
        save_path: where to save the result

    Returns:
        dict {city_id: {year: [(event_type, start_day, end_day, intensity)]}}
    """
    import pickle

    events_dict = {}
    n_cities = len(city_info_df)

    for idx, row in city_info_df.iterrows():
        city_id = int(row["city_id"])
        city_name = row["city_name"]
        lat, lon = row["lat"], row["lon"]

        events_dict[city_id] = {}

        for year in range(start_year, end_year + 1):
            nc_path = download_era5_daily(lat, lon, year)
            if nc_path and os.path.exists(nc_path):
                events = extract_events_for_city_year(nc_path)
                events_dict[city_id][year] = events
            else:
                events_dict[city_id][year] = []

        done = idx + 1
        if done % 10 == 0 or done == n_cities:
            print(f"  Weather extraction: {done}/{n_cities} cities done", flush=True)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(events_dict, f)
    print(f"  Weather events saved to {save_path}")

    return events_dict


def load_weather_panel(path="data/weather_events.pkl"):
    """Load pre-extracted weather events from disk."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    if not CDS_AVAILABLE:
        print("cdsapi not installed. Install with:")
        print("  pip install cdsapi")
        print("\nThen create ~/.cdsapirc with your CDS API key.")
    else:
        print("cdsapi available.")
        print("Create ~/.cdsapirc with your CDS API key.")
