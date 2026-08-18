"""
ERA5 daily weather data extraction via AWS Open Data (Planet OS).

Accesses ERA5 reanalysis data from the public AWS S3 bucket:
  s3://era5-pds/
  https://registry.opendata.aws/era5-pds/

NO AUTHENTICATION REQUIRED — anonymous S3 access.

Data structure:
  s3://era5-pds/{variable}/{year}/{month:02d}/data.nc

Variables used:
  - air_temperature_at_2_metres_1_hour_Maximum  → tmax (°C)
  - air_temperature_at_2_metres_1_hour_Minimum  → tmin (°C)
  - precipitation_amount_1_hour_Accumulation    → precip (mm)

Coverage: 1979-present, global, 0.25° grid, hourly → daily aggregation.

Output:
  data/weather_events.pkl — {city_id: {year: [(event_type, start_day, end_day, intensity)]}}
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

S3_BASE = "s3://era5-pds"

VARIABLES = {
    "tmax": "air_temperature_at_2_metres_1_hour_Maximum",
    "tmin": "air_temperature_at_2_metres_1_hour_Minimum",
    "precip": "precipitation_amount_1_hour_Accumulation",
}

CACHE_DIR = "data/era5_cache"


def _open_s3_nc(variable_key, year, month, anon_fs):
    """Open one NetCDF file from S3 as xarray dataset.

    Args:
        variable_key: short key like "tmax", "tmin", "precip"
        year, month: integers
        anon_fs: s3fs.S3FileSystem (anonymous)

    Returns:
        xarray.DataArray for the variable, or None if not found
    """
    var_name = VARIABLES[variable_key]
    path = f"{S3_BASE}/{var_name}/{year}/{month:02d}/data.nc"

    try:
        import xarray as xr
        ds = xr.open_dataset(anon_fs.open(path), engine="h5netcdf")
        return ds[var_name]
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"    Warning: {var_name} {year}-{month:02d}: {e}")
        return None


def _get_nearest_grid(da, lat, lon):
    """Extract time series at nearest grid point to (lat, lon).

    Args:
        da: xarray.DataArray with coords (time, lat, lon) or (time0, lat, lon)
        lat, lon: target coordinates

    Returns:
        np.array of hourly values
    """
    lats = da.coords["lat"].values
    lons = da.coords["lon"].values

    lat_idx = int(np.argmin(np.abs(lats - lat)))
    lon_idx = int(np.argmin(np.abs(lons - lon)))

    time_dim = [d for d in da.dims if "time" in d][0]
    point = da.isel(lat=lat_idx, lon=lon_idx)
    return point.values


def extract_month_city(variable_key, year, month, city_info_df, anon_fs):
    """Extract one month of one variable for all cities.

    Returns:
        dict {city_id: np.array of hourly values}, or None if download fails
    """
    da = _open_s3_nc(variable_key, year, month, anon_fs)
    if da is None:
        return None

    results = {}
    for _, city in city_info_df.iterrows():
        city_id = int(city["city_id"])
        vals = _get_nearest_grid(da, city["lat"], city["lon"])
        results[city_id] = vals

    da.close()
    return results


def aggregate_month_to_daily(month_data_dict, variable_key, year, month):
    """Aggregate hourly data to daily stats for one month.

    Args:
        month_data_dict: {city_id: np.array of hourly values}
        variable_key: "tmax", "tmin", or "precip"
        year, month: for date generation

    Returns:
        dict {city_id: pd.Series indexed by date}
    """
    days_in_month = pd.Timestamp(year, month, 1).days_in_month
    hours_per_day = 24

    dates = pd.date_range(
        start=f"{year}-{month:02d}-01",
        periods=days_in_month,
        freq="D",
    )

    results = {}
    for city_id, vals in month_data_dict.items():
        n_hours = min(len(vals), days_in_month * hours_per_day)
        vals = vals[:n_hours].astype(float)

        if variable_key == "tmax":
            vals = vals - 273.15
            daily = np.array([
                np.nanmax(vals[d*hours_per_day:(d+1)*hours_per_day])
                if (d+1)*hours_per_day <= len(vals) else np.nan
                for d in range(days_in_month)
            ])
        elif variable_key == "tmin":
            vals = vals - 273.15
            daily = np.array([
                np.nanmin(vals[d*hours_per_day:(d+1)*hours_per_day])
                if (d+1)*hours_per_day <= len(vals) else np.nan
                for d in range(days_in_month)
            ])
        elif variable_key == "precip":
            vals = vals * 1000
            daily = np.array([
                np.nansum(vals[d*hours_per_day:(d+1)*hours_per_day])
                if (d+1)*hours_per_day <= len(vals) else np.nan
                for d in range(days_in_month)
            ])
        else:
            daily = np.full(days_in_month, np.nan)

        results[city_id] = pd.Series(daily, index=dates)

    return results


def run_era5_aws_pipeline(city_info_df, start_year=2005, end_year=2023,
                         save_path="data/weather_events.pkl"):
    """Full ERA5 pipeline via AWS S3: download → aggregate → identify events.

    Args:
        city_info_df: DataFrame from city_list_280.load()
        start_year, end_year: period
        save_path: output pickle path

    Returns:
        events_dict: {city_id: {year: [(event_type, start_day, end_day, intensity)]}}
    """
    import s3fs
    from real_data.era5_weather import (
        identify_heat_waves, identify_heavy_rain, identify_drought
    )

    print("=" * 70)
    print("ERA5 Weather Data Pipeline (AWS S3 — Planet OS)")
    print(f"  Cities: {len(city_info_df)}")
    print(f"  Period: {start_year}-{end_year}")
    print(f"  Source: {S3_BASE} (anonymous, no auth required)")
    print("=" * 70)

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    fs = s3fs.S3FileSystem(anon=True)

    all_daily = {int(cid): {} for cid in city_info_df["city_id"]}
    events_dict = {int(cid): {} for cid in city_info_df["city_id"]}

    for year in range(start_year, end_year + 1):
        year_daily = {int(cid): {} for cid in city_info_df["city_id"]}

        for month in range(1, 13):
            tag = f"{year}-{month:02d}"
            print(f"\n  [{tag}] Downloading...", end="", flush=True)

            month_daily = {int(cid): [] for cid in city_info_df["city_id"]}

            for var_key in ["tmax", "tmin", "precip"]:
                data = extract_month_city(var_key, year, month, city_info_df, fs)
                if data is not None:
                    daily = aggregate_month_to_daily(data, var_key, year, month)
                    for cid in daily:
                        month_daily[cid].append(daily[cid])
                    print(f" {var_key}✓", end="", flush=True)
                else:
                    print(f" {var_key}✗", end="", flush=True)
                    dummy = pd.Series(
                        np.full(pd.Timestamp(year, month, 1).days_in_month, np.nan),
                        index=pd.date_range(f"{year}-{month:02d}-01",
                            periods=pd.Timestamp(year, month, 1).days_in_month, freq="D")
                    )
                    for cid in month_daily:
                        month_daily[cid].append(dummy)

            for cid in month_daily:
                year_daily[cid].append(pd.concat(month_daily[cid]))

        print(f"\n  [{year}] Aggregating and identifying events...")

        for cid in year_daily:
            city_df = pd.concat([
                d.reset_index().rename(columns={"index": "date", "index": "date"})
                if "date" not in d.reset_index().columns
                else d.reset_index()
                for d in year_daily[cid]
            ], ignore_index=True) if year_daily[cid] else pd.DataFrame()

            if len(city_df) == 0:
                events_dict[cid][year] = []
                continue

            city_df.columns = ["date", "tmax", "tmin", "precip"][:len(city_df.columns)]
            city_df = city_df.sort_values("date").reset_index(drop=True)
            city_df = city_df.dropna(subset=["date"])

            dates = city_df["date"]
            events = []
            events.extend(identify_heat_waves(city_df["tmax"], dates))
            events.extend(identify_heavy_rain(city_df["precip"], dates))
            events.extend(identify_drought(city_df["precip"], dates))
            events_dict[cid][year] = events

        n_events = sum(len(events_dict[cid][year]) for cid in events_dict)
        print(f"  [{year}] Total events: {n_events}")

        with open(save_path, "wb") as f:
            pickle.dump(events_dict, f)
        print(f"  [{year}] Saved to {save_path}")

    print("\n" + "=" * 70)
    total = sum(len(events_dict[cid][y]) for cid in events_dict for y in events_dict[cid])
    print(f"Total events across all cities/years: {total}")
    print("=" * 70)

    return events_dict


if __name__ == "__main__":
    from real_data.city_list_280 import load
    city_info = load()
    run_era5_aws_pipeline(city_info, start_year=2020, end_year=2020)
