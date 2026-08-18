"""
Download ERA5 China-wide daily data and extract city-level weather.

Strategy: download one NetCDF per year covering all of China,
then extract daily time series at each city coordinate.
This is ~100x faster than downloading per-city.

Variables:
  - 2m_temperature (4 time steps: 00,06,12,18 UTC → daily max/min)
  - total_precipitation (4 time steps → daily total)

Output:
  data/era5_raw/china_{year}.nc   — raw NetCDF files
  data/era5_daily_{city_id}.pkl   — per-city daily DataFrames
  data/weather_events.pkl          — identified extreme events
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ERA5_DIR = "data/era5_raw"
DAILY_DIR = "data/era5_daily"
CHINA_AREA = [54, 73, 15, 136]

VARIABLES = ["2m_temperature", "total_precipitation"]
TIMES = ["00:00", "06:00", "12:00", "18:00"]


def download_year(year, save_dir=ERA5_DIR):
    """Download ERA5 China-wide data for one year."""
    import cdsapi

    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, f"china_{year}.nc")

    if os.path.exists(out_file):
        print(f"  {year}: already downloaded ({os.path.getsize(out_file) / 1e6:.1f} MB)")
        return out_file

    c = cdsapi.Client(quiet=True)

    try:
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": VARIABLES,
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": TIMES,
                "area": CHINA_AREA,
            },
            out_file,
        )
        size_mb = os.path.getsize(out_file) / 1e6
        print(f"  {year}: downloaded ({size_mb:.1f} MB)")
        return out_file
    except Exception as e:
        print(f"  {year}: FAILED - {e}")
        return None


def extract_daily_city(nc_path, city_info_df, save_dir=DAILY_DIR):
    """Extract daily weather for all cities from a China-wide NetCDF.

    For each city, finds the nearest grid point and extracts:
      - tmax: daily max temperature (°C)
      - tmin: daily min temperature (°C)
      - tmean: daily mean temperature (°C)
      - precip: daily total precipitation (mm)
    """
    import xarray as xr

    os.makedirs(save_dir, exist_ok=True)
    ds = xr.open_dataset(nc_path)

    t2m = ds["t2m"]
    tp = ds["tp"]

    lats = t2m.coords["latitude"].values
    lons = t2m.coords["longitude"].values

    results = {}

    for _, city in city_info_df.iterrows():
        city_id = int(city["city_id"])
        lat, lon = city["lat"], city["lon"]

        lat_idx = int(np.argmin(np.abs(lats - lat)))
        lon_idx = int(np.argmin(np.abs(lons - lon)))

        city_t2m = t2m.isel(latitude=lat_idx, longitude=lon_idx)
        city_tp = tp.isel(latitude=lat_idx, longitude=lon_idx)

        t2m_vals = city_t2m.values - 273.15
        tp_vals = city_tp.values * 1000

        dates = pd.to_datetime(city_t2m.coords["valid_time"].values)

        n_days = len(dates)
        tmax = np.full(n_days, np.nan)
        tmin = np.full(n_days, np.nan)
        tmean = np.full(n_days, np.nan)
        precip = np.full(n_days, np.nan)

        n_times = len(TIMES)
        for d in range(n_days):
            start = d * n_times
            end = start + n_times
            if end <= len(t2m_vals):
                tmax[d] = np.nanmax(t2m_vals[start:end])
                tmin[d] = np.nanmin(t2m_vals[start:end])
                tmean[d] = np.nanmean(t2m_vals[start:end])
                precip[d] = np.nansum(tp_vals[start:end])

        results[city_id] = pd.DataFrame({
            "date": dates,
            "tmax": tmax,
            "tmin": tmin,
            "tmean": tmean,
            "precip": precip,
        })

    ds.close()
    return results


def identify_events_from_daily(daily_df):
    """Identify extreme weather events from daily DataFrame.

    Returns:
        list of (event_type, start_day, end_day, intensity)
    """
    from real_data.era5_weather import (
        identify_heat_waves, identify_heavy_rain, identify_drought
    )

    events = []
    dates = daily_df["date"]

    events.extend(identify_heat_waves(daily_df["tmax"], dates))
    events.extend(identify_heavy_rain(daily_df["precip"], dates))
    events.extend(identify_drought(daily_df["precip"], dates))

    return events


def run_era5_pipeline(city_info_df, start_year=2005, end_year=2023):
    """Full ERA5 pipeline: download → extract → identify events.

    Args:
        city_info_df: DataFrame from city_list_280.load()
        start_year, end_year: download period

    Returns:
        events_dict: {city_id: {year: [(event_type, start_day, end_day, intensity)]}}
    """
    print("=" * 70)
    print("ERA5 Weather Data Pipeline")
    print(f"  Cities: {len(city_info_df)}")
    print(f"  Period: {start_year}-{end_year}")
    print("=" * 70)

    events_dict = {int(cid): {} for cid in city_info_df["city_id"]}

    for year in range(start_year, end_year + 1):
        print(f"\n--- Year {year} ---")

        nc_path = download_year(year)
        if nc_path is None:
            for cid in events_dict:
                events_dict[cid][year] = []
            continue

        print(f"  Extracting city-level data...")
        daily_data = extract_daily_city(nc_path, city_info_df)

        print(f"  Identifying extreme weather events...")
        for city_id, daily_df in daily_data.items():
            events = identify_events_from_daily(daily_df)
            events_dict[city_id][year] = events

        n_total = sum(len(events_dict[cid][year]) for cid in events_dict)
        print(f"  Total events identified: {n_total}")

    os.makedirs("data", exist_ok=True)
    with open("data/weather_events.pkl", "wb") as f:
        pickle.dump(events_dict, f)
    print(f"\nWeather events saved to data/weather_events.pkl")

    return events_dict


if __name__ == "__main__":
    from real_data.city_list_280 import load
    city_info = load()
    run_era5_pipeline(city_info, start_year=2020, end_year=2020)
