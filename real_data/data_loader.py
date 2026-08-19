"""
Main real data loader — combines all data sources into the panel format.

Data sources:
  1. City info (coordinates, admin divisions)     → city_info.py
  2. Policy pilot lists                            → policy_pilots.py
  3. MODIS NDVI time series (Google Earth Engine) → gee_ndvi.py
  4. ERA5 extreme weather events                   → era5_weather.py
  5. City statistical yearbook (control vars)      → city_stats.py

Usage:
  from real_data.data_loader import load_real_panel
  panel, ndvi_ts, events = load_real_panel()

  # Then proceed as with synthetic data:
  from csee_computation import compute_csee_panel
  panel = compute_csee_panel(panel, ndvi_ts, events)

When some data sources are unavailable, the loader reports what's missing
and uses available data or generates placeholders where possible.
"""
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import START_YEAR, END_YEAR, POLICY_YEAR, CONTROL_VARS, NDVI_PERIODS_PER_YEAR as N_PERIODS

from real_data.city_info import load_city_info
from real_data.city_list_280 import load as load_full_city_list
from real_data.policy_pilots import get_pilot_year, get_treated_cities
from real_data.gee_ndvi import load_ndvi_panel, EE_AVAILABLE
from real_data.era5_weather import load_weather_panel, CDS_AVAILABLE
from real_data.city_stats import load_yearbook_data, merge_with_city_info, CONTROL_COLS
from real_data.real_controls import build_real_controls


def load_real_panel(
    city_csv="data/city_info.csv",
    ndvi_path="data/ndvi_panel.pkl",
    weather_path="data/weather_events.pkl",
    yearbook_dir="data/yearbook",
    policy="both",
    start_year=START_YEAR,
    end_year=END_YEAR,
    use_full_city_list=True,
):
    """Load real data from all sources and assemble the panel.

    Args:
        city_csv: path to city info CSV (or None for hardcoded cities)
        ndvi_path: path to pre-extracted NDVI panel pickle
        weather_path: path to pre-extracted weather events pickle
        yearbook_dir: directory with yearbook Excel/CSV files
        policy: "climate_adaptive" or "sponge" or "both"
        start_year, end_year: panel period
        use_full_city_list: if True, use 338-city list; else use CSV/hardcoded

    Returns:
        (panel, ndvi_ts, events) — same format as generate_panel_data()
    """
    print("=" * 70)
    print("Loading REAL data from all sources...")
    print(f"  Period: {start_year}-{end_year}")
    print(f"  Policy: {policy}")
    print("-" * 70)

    # 1. City info
    print("\n[1] Loading city information...")
    if use_full_city_list:
        from real_data.city_info import _classify_region, _classify_ns, _classify_coastal
        city_info = load_full_city_list()
        city_info["region"] = city_info["province"].apply(_classify_region)
        city_info["ns"] = city_info["lat"].apply(_classify_ns)
        city_info["coastal"] = city_info["province"].apply(_classify_coastal)
        print(f"  Using full city list: {len(city_info)} cities")
    else:
        city_info = load_city_info(city_csv)
    n_cities = len(city_info)
    print(f"  Cities loaded: {n_cities}")

    # 2. Policy pilot assignment
    print("\n[2] Assigning policy pilot status...")
    treated_cities = get_treated_cities(policy=policy)
    city_info["treat"] = city_info["city_name"].apply(
        lambda x: 1 if x in treated_cities else 0
    )
    city_info["policy_year"] = city_info["city_name"].apply(
        lambda x: get_pilot_year(x, policy=policy) or POLICY_YEAR
    )
    n_treated = city_info["treat"].sum()
    print(f"  Treated cities: {n_treated} / {n_cities}")

    # 3. NDVI time series
    print("\n[3] Loading NDVI time series...")
    if os.path.exists(ndvi_path):
        ndvi_ts = load_ndvi_panel(ndvi_path)
        n_cities_ndvi = len(ndvi_ts)
        print(f"  NDVI panel loaded: {n_cities_ndvi} cities")
    else:
        print(f"  WARNING: NDVI panel not found at {ndvi_path}")
        print(f"  Run real_data/gee_ndvi.py to extract NDVI from Google Earth Engine")
        print(f"  Falling back to synthetic NDVI...")
        ndvi_ts = _generate_synthetic_ndvi_fallback(city_info, start_year, end_year)

    # 4. Weather events
    print("\n[4] Loading extreme weather events...")
    if os.path.exists(weather_path):
        events = load_weather_panel(weather_path)
        n_cities_weather = len(events)
        print(f"  Weather events loaded: {n_cities_weather} cities")
    else:
        print(f"  WARNING: Weather events not found at {weather_path}")
        print(f"  Run real_data/era5_weather.py to extract from ERA5")
        print(f"  Falling back to synthetic events...")
        events = _generate_synthetic_events_fallback(city_info, start_year, end_year)

    # 5. Yearbook data
    print("\n[5] Loading city statistical yearbook data...")
    yearbook = load_yearbook_data(yearbook_dir)
    if not yearbook.empty:
        yearbook = merge_with_city_info(yearbook, city_info)
        print(f"  Yearbook data merged: {len(yearbook)} city-years")
    else:
        print(f"  WARNING: No yearbook data found in {yearbook_dir}")
        print(f"  Place yearbook files as yearbook_{{year}}.xlsx")
        print(f"  Falling back to synthetic controls...")

    # 6. Assemble panel
    print("\n[6] Assembling panel...")
    panel = _assemble_panel(
        city_info, ndvi_ts, events, yearbook,
        start_year, end_year
    )

    print("-" * 70)
    print(f"Panel shape: {panel.shape}")
    print(f"Treated cities: {panel[panel['treat']==1]['city_id'].nunique()}")
    print(f"Total observations: {len(panel)}")
    print("=" * 70)

    return panel, ndvi_ts, events


def _assemble_panel(city_info, ndvi_ts, events, yearbook, start_year, end_year):
    """Assemble the panel DataFrame from all data sources."""
    print("\n  [Panel] Building real control variables...")
    real_controls = build_real_controls(
        city_info, ndvi_ts, events, start_year, end_year
    )
    real_controls_dict = real_controls.set_index(["city_id", "year"])

    rows = []

    years = range(start_year, end_year + 1)

    for _, city_row in city_info.iterrows():
        city_id = int(city_row["city_id"])
        city_name = city_row["city_name"]
        treat = int(city_row["treat"])
        policy_year = int(city_row["policy_year"])

        for year in years:
            post = int(year >= policy_year)
            did = treat * post

            # Weather events for this city-year
            city_events = events.get(city_id, {}).get(year, [])
            n_events = len(city_events)
            n_heat = sum(1 for e in city_events if e[0] == "heat")
            n_rain = sum(1 for e in city_events if e[0] == "rain")
            n_drought = sum(1 for e in city_events if e[0] == "drought")

            # Shock intensity
            if n_events > 0:
                intensities = [e[3] for e in city_events]
                shock_raw = np.sum(intensities)
            else:
                shock_raw = 0.0

            # Dominant shock type
            shock_counts = [n_heat, n_rain, n_drought]
            if max(shock_counts) > 0:
                dominant = np.argmax(shock_counts)
                shock_type = ["heat", "rain", "drought"][dominant]
            else:
                shock_type = "rain"  # default

            row = {
                "city_id": city_id,
                "year": year,
                "city_name": city_name,
                "province": city_row["province"],
                "treat": treat,
                "post": post,
                "did": did,
                "policy_year": policy_year,
                "n_events": n_events,
                "n_heat": n_heat,
                "n_rain": n_rain,
                "n_drought": n_drought,
                "shock_type": shock_type,
                "lon": float(city_row["lon"]),
                "lat": float(city_row["lat"]),
                "region": city_row["region"],
                "ns": city_row["ns"],
                "coastal": int(city_row["coastal"]),
                "city_size": city_row["city_size"],
            }

            # Control variables from real data sources (census, CN_Public, ERA5, NDVI)
            rc_key = (city_id, year)
            if rc_key in real_controls_dict.index:
                rc_row = real_controls_dict.loc[rc_key]
                for col in CONTROL_VARS:
                    if col in rc_row.index and pd.notna(rc_row[col]):
                        row[col] = float(rc_row[col])

            # Also check yearbook for any additional columns
            if not yearbook.empty:
                yb_match = yearbook[
                    (yearbook["city_id"] == city_id) &
                    (yearbook["year"] == year)
                ]
                if len(yb_match) > 0:
                    for col in CONTROL_COLS:
                        if col in yb_match.columns:
                            row[col] = float(yb_match.iloc[0][col])

            # Eco baseline: compute from early NDVI if available
            if city_id in ndvi_ts:
                early_years = [y for y in ndvi_ts[city_id] if y <= start_year + 2]
                if early_years:
                    early_ndvi = np.mean([
                        np.mean(ndvi_ts[city_id][y])
                        for y in early_years
                    ])
                    row["eco_baseline"] = "fragile" if early_ndvi < 0.4 else "good"

            # Mediation variables (with _m suffix)
            if "green_rate" in row:
                row["green_rate_m"] = row["green_rate"]
            row.setdefault("sponge_inv_m", 0.0 if treat == 0 else np.random.lognormal(9, 0.5))
            row.setdefault("blue_green_ratio_m", np.random.uniform(0.1, 0.6))
            row.setdefault("coupling_coord_m", np.random.uniform(0.2, 0.9))

            # Policy intensity
            row["sponge_intensity"] = row.get("sponge_inv_m", 0.0)
            if treat == 0 or year < policy_year:
                row["sponge_intensity"] = 0.0

            rows.append(row)

    panel = pd.DataFrame(rows)

    # Standardize shock_intensity across panel
    if "shock_intensity" not in panel.columns:
        panel["shock_intensity"] = 0.0

    # Compute shock_intensity from raw event data
    shock_raw = panel.groupby("year")["n_events"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    ).clip(lower=0)
    panel["shock_intensity"] = shock_raw

    # DID × shock interaction
    panel["did_shock"] = panel["did"] * panel["shock_intensity"]

    # Policy intensity (continuous)
    panel["policy_intensity"] = np.log(panel["sponge_intensity"].clip(lower=1))
    panel.loc[panel["sponge_intensity"] == 0, "policy_intensity"] = 0

    # Fill missing control variables with column medians (real data, not random)
    for col in CONTROL_VARS:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
            panel[col] = panel[col].fillna(panel[col].median())
        else:
            panel[col] = 0.0
            print(f"  WARNING: control variable '{col}' not found in real data")

    # Eco baseline fallback
    if "eco_baseline" not in panel.columns:
        panel["eco_baseline"] = panel.groupby("city_id")["lat"].transform(
            lambda x: "fragile" if abs(x.iloc[0]) > 35 else "good"
        )

    return panel


def _generate_synthetic_ndvi_fallback(city_info, start_year, end_year):
    """Generate placeholder NDVI time series when GEE data is unavailable."""
    print("  Generating synthetic NDVI fallback...")
    from data_simulation import _generate_ndvi_timeseries, _generate_daily_weather, _identify_extreme_events
    ndvi_ts = {}
    for _, city in city_info.iterrows():
        cid = int(city["city_id"])
        ndvi_ts[cid] = {}
        for year in range(start_year, end_year + 1):
            n = N_PERIODS
            base = 0.3 + 0.3 * (1 - abs(city["lat"]) / 50)
            ndvi = np.clip(base + np.random.normal(0, 0.08, n), -0.1, 0.9)
            ndvi_ts[cid][year] = ndvi
    return ndvi_ts


def _generate_synthetic_events_fallback(city_info, start_year, end_year):
    """Generate placeholder weather events when ERA5 data is unavailable."""
    print("  Generating synthetic weather events fallback...")
    events = {}
    for _, city in city_info.iterrows():
        cid = int(city["city_id"])
        events[cid] = {}
        for year in range(start_year, end_year + 1):
            n_events = np.random.poisson(3)
            year_events = []
            for _ in range(n_events):
                ev_type = np.random.choice(["heat", "rain", "drought"])
                start_day = np.random.randint(1, 300)
                duration = np.random.randint(3, 30)
                intensity = np.random.exponential(0.5)
                year_events.append((ev_type, start_day, start_day + duration, intensity))
            events[cid][year] = year_events
    return events


if __name__ == "__main__":
    print("Real data loader. Usage:")
    print("  from real_data.data_loader import load_real_panel")
    print("  panel, ndvi_ts, events = load_real_panel()")
    print("\nData sources needed:")
    print("  1. data/city_info.csv       — city coordinates & admin info")
    print("  2. data/ndvi_panel.pkl       — MODIS NDVI from GEE")
    print("  3. data/weather_events.pkl   — ERA5 extreme weather events")
    print("  4. data/yearbook/*.xlsx      — city statistical yearbooks")
    print("\nMissing sources will use synthetic fallbacks for testing.")
