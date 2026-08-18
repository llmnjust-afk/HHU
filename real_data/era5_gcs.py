"""
ERA5 daily weather extraction via ARCO-ERA5 on Google Cloud Storage.

Accesses the Analysis-Ready, Cloud-Optimized ERA5 Zarr store:
  gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3

NO AUTHENTICATION REQUIRED — anonymous GCS access.
Covers 1900-2050, global, 0.25° grid, hourly resolution.

Variables:
  - 2m_temperature          → tmax (daily max), tmin (daily min), tmean
  - total_precipitation     → precip (daily total, mm)

Output:
  data/weather_events.pkl — {city_id: {year: [(event_type, start_day, end_day, intensity)]}}
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARCO_ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

VAR_T2M = "2m_temperature"
VAR_TP = "total_precipitation"


def _open_arco_era5():
    """Open ARCO-ERA5 Zarr store with anonymous GCS access."""
    import gcsfs
    import xarray as xr

    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(ARCO_ERA5_URL)
    ds = xr.open_zarr(mapper, consolidated=True, chunks={"time": 720})
    return ds


def extract_all_cities_year(ds, city_info_df, year):
    """Extract daily weather for ALL cities for ONE year in a single batch.

    Uses xarray nearest-point selection to fetch all city time series
    in one cloud read, then aggregates hourly → daily locally.

    Returns:
        dict {city_id: pd.DataFrame with date, tmax, tmin, tmean, precip}
    """
    import xarray as xr

    time_slice = slice(f"{year}-01-01", f"{year}-12-31T23:59:59")

    city_lats = xr.DataArray(city_info_df["lat"].values, dims="city")
    city_lons = xr.DataArray(city_info_df["lon"].values, dims="city")

    print(f"    Fetching t2m for {len(city_info_df)} cities...", flush=True)
    t2m_all = ds[VAR_T2M].sel(time=time_slice).sel(
        latitude=city_lats, longitude=city_lons, method="nearest"
    ).compute()

    print(f"    Fetching precip...", flush=True)
    tp_all = ds[VAR_TP].sel(time=time_slice).sel(
        latitude=city_lats, longitude=city_lons, method="nearest"
    ).compute()

    t2m_vals = t2m_all.values.astype(float) - 273.15  # K → °C
    tp_vals = tp_all.values.astype(float) * 1000      # m → mm
    times = pd.to_datetime(t2m_all.time.values)

    n_cities = t2m_vals.shape[1]
    city_ids = city_info_df["city_id"].astype(int).values

    results = {}
    for i in range(n_cities):
        cid = int(city_ids[i])
        city_t2m = t2m_vals[:, i]
        city_tp = tp_vals[:, i]

        df = pd.DataFrame({"datetime": times, "t2m": city_t2m, "precip": city_tp})
        df["date"] = df["datetime"].dt.date

        daily = df.groupby("date").agg(
            tmax=("t2m", "max"),
            tmin=("t2m", "min"),
            tmean=("t2m", "mean"),
            precip=("precip", "sum"),
        ).reset_index()
        daily["date"] = pd.to_datetime(daily["date"])

        results[cid] = daily

    del t2m_all, tp_all, t2m_vals, tp_vals

    return results


def run_era5_gcs_pipeline(city_info_df, start_year=2005, end_year=2023,
                         save_path="data/weather_events.pkl"):
    """Full ERA5 pipeline via ARCO-ERA5: batch extract → aggregate → events.

    Args:
        city_info_df: DataFrame from city_list_280.load()
        start_year, end_year: period
        save_path: output pickle path

    Returns:
        events_dict: {city_id: {year: [(event_type, start_day, end_day, intensity)]}}
    """
    from real_data.era5_weather import (
        identify_heat_waves, identify_heavy_rain, identify_drought
    )

    n_cities = len(city_info_df)
    print("=" * 70)
    print("ERA5 Weather Pipeline (ARCO-ERA5 — Google Cloud, no auth)")
    print(f"  Cities: {n_cities}")
    print(f"  Period: {start_year}-{end_year}")
    print("=" * 70)

    os.makedirs("data", exist_ok=True)

    print("\n  Opening ARCO-ERA5 Zarr store...")
    ds = _open_arco_era5()
    print(f"  Store opened. Time range: {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}")

    events_dict = {int(cid): {} for cid in city_info_df["city_id"]}

    for year in range(start_year, end_year + 1):
        print(f"\n  --- Year {year} ---")

        try:
            daily_all = extract_all_cities_year(ds, city_info_df, year)
        except Exception as e:
            print(f"    ERROR extracting {year}: {e}")
            for cid in events_dict:
                events_dict[cid][year] = []
            continue

        all_events_year = 0
        for cid in daily_all:
            daily = daily_all[cid]
            if daily.empty or daily["tmax"].isna().all():
                events_dict[cid][year] = []
                continue

            events = []
            events.extend(identify_heat_waves(daily["tmax"], daily["date"]))
            events.extend(identify_heavy_rain(daily["precip"], daily["date"]))
            events.extend(identify_drought(daily["precip"], daily["date"]))
            events_dict[cid][year] = events
            all_events_year += len(events)

        print(f"  [{year}] Total events: {all_events_year}")

        with open(save_path, "wb") as f:
            pickle.dump(events_dict, f)
        print(f"  [{year}] Saved to {save_path}")

    ds.close()

    total = sum(len(events_dict[c][y]) for c in events_dict for y in events_dict[c])
    print("\n" + "=" * 70)
    print(f"Total events: {total}")
    print("=" * 70)

    return events_dict


if __name__ == "__main__":
    from real_data.city_list_280 import load
    city_info = load()
    run_era5_gcs_pipeline(city_info, start_year=2020, end_year=2020)
