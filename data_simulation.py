"""
Synthetic data generation module.

Generates realistic city-panel data mimicking:
  - MODIS NDVI 16-day time series (2005-2023, 280 cities)
  - Daily meteorological data (temperature, precipitation)
  - Climate-resilient city pilot policy assignment (2017, 28 cities)
  - High-dimensional control variables (economic, demographic, geographic)
  - Mediation variables (green rate, sponge city investment, etc.)
  - Spatial coordinates for spatial spillover analysis

The data-generating process (DGP) embeds a known treatment effect so that
DML estimates can be validated against ground truth.
"""
import numpy as np
import pandas as pd
from config import (
    N_CITIES, START_YEAR, END_YEAR, POLICY_YEAR, N_PILOT_CITIES,
    NDVI_PERIODS_PER_YEAR, NORMAL_LOOKBACK_YEARS, TRUE_BUFFER_EFFECT,
    TRUE_RESISTANCE_EFFECT, TRUE_RECOVERY_EFFECT, RANDOM_SEED,
    CONTROL_VARS, MEDIATION_VARS,
)

np.random.seed(RANDOM_SEED)


def _generate_city_characteristics(n_cities):
    """Generate time-invariant city characteristics."""
    cities = pd.DataFrame({
        "city_id": np.arange(n_cities),
        "lon": np.random.uniform(73, 135, n_cities),
        "lat": np.random.uniform(18, 53, n_cities),
        "elevation": np.random.lognormal(5, 1.2, n_cities).clip(1, 5000),
    })
    cities["region"] = pd.cut(cities["lon"], bins=[0, 105, 115, 200],
                             labels=["west", "central", "east"])
    cities["ns"] = pd.cut(cities["lat"], bins=[0, 33, 90],
                          labels=["south", "north"])
    cities["coastal"] = ((cities["lon"] > 112) & (cities["lat"] < 40)).astype(int)
    # city size: 20% large, 80% small-medium
    cities["city_size"] = np.where(np.random.rand(n_cities) < 0.2, "large", "small")
    # ecological baseline: 30% fragile
    cities["eco_baseline"] = np.where(np.random.rand(n_cities) < 0.3, "fragile", "good")
    return cities


def _assign_policy(cities):
    """Assign climate-resilient city pilot status with selection on observables."""
    n = len(cities)
    # Selection probability: larger, eastern, higher GDP cities more likely
    logit = (
        -2.5
        + 0.8 * (cities["city_size"] == "large").astype(float)
        + 0.5 * (cities["region"] == "east").astype(float)
        + 0.3 * (cities["coastal"] == 1).astype(float)
        + 0.2 * (cities["eco_baseline"] == "fragile").astype(float)
        + np.random.logistic(0, 1, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    cities["treat"] = 0
    pilot_idx = np.argsort(prob)[-N_PILOT_CITIES:]
    cities.loc[pilot_idx, "treat"] = 1
    return cities


def _generate_daily_weather(cities, year):
    """Generate daily temperature and precipitation for one year."""
    n = len(cities)
    n_days = 365
    days = np.arange(1, n_days + 1)

    # Seasonal temperature cycle
    temp_seasonal = 15 + 12 * np.sin(2 * np.pi * (days - 100) / 365)
    # City-specific base temperature (latitude effect)
    lat_effect = 30 - 0.5 * cities["lat"].values
    # Daily temperature: city × day matrix
    daily_temp = (temp_seasonal[np.newaxis, :]
                  + lat_effect[:, np.newaxis]
                  + np.random.normal(0, 3, (n, n_days)))

    # Precipitation: seasonal + random
    rain_seasonal = 3 + 4 * np.sin(2 * np.pi * (days - 130) / 365)
    rain_seasonal = np.clip(rain_seasonal, 0, None)
    lat_factor = np.where(cities["lat"].values < 33, 1.5, 1.0).reshape(n, 1)
    daily_rain = (rain_seasonal[np.newaxis, :] * lat_factor
                  + np.random.exponential(2, (n, n_days)))

    return daily_temp, daily_rain, days


def _identify_extreme_events(daily_temp, daily_rain, days):
    """Identify extreme weather events from daily data.

    Returns list of (event_type, start_day, end_day, intensity) per city.
    """
    n_cities, n_days = daily_temp.shape
    events = [[] for _ in range(n_cities)]

    for i in range(n_cities):
        temps = daily_temp[i]
        rains = daily_rain[i]

        # Extreme heat: daily max temp > 90th percentile, >=3 consecutive days
        heat_threshold = np.percentile(temps, 90)
        is_heat = temps > heat_threshold
        j = 0
        while j < n_days:
            if is_heat[j]:
                run = 1
                while j + run < n_days and is_heat[j + run]:
                    run += 1
                if run >= 3:
                    intensity = float(np.mean(temps[j:j+run]) - heat_threshold)
                    events[i].append(("heat", int(days[j]), int(days[j+run-1]), intensity))
                j += run
            else:
                j += 1

        # Extreme rain: daily precip > 95th percentile
        rain_threshold = np.percentile(rains, 95)
        extreme_rain_days = np.where(rains > rain_threshold)[0]
        for d in extreme_rain_days:
            events[i].append(("rain", int(days[d]), int(days[d]), float(rains[d])))

        # Drought: >=30 consecutive days with <1mm precipitation
        is_dry = rains < 1.0
        j = 0
        while j < n_days:
            if is_dry[j]:
                run = 1
                while j + run < n_days and is_dry[j + run]:
                    run += 1
                if run >= 30:
                    intensity = float(run)
                    events[i].append(("drought", int(days[j]), int(days[j+run-1]), intensity))
                j += run
            else:
                j += 1

    return events


def _generate_ndvi_timeseries(cities, year, daily_temp, daily_rain, events, policy_active):
    """Generate 16-day NDVI time series for each city-year.

    NDVI follows a seasonal cycle, is reduced by extreme weather events,
    and is buffered by policy if active.
    """
    n_cities = len(cities)
    n_periods = NDVI_PERIODS_PER_YEAR
    ndvi = np.zeros((n_cities, n_periods))

    period_days = np.linspace(1, 365, n_periods)

    for i in range(n_cities):
        # Base NDVI seasonal cycle (peaks in summer)
        base_ndvi = 0.45 + 0.25 * np.sin(2 * np.pi * (period_days - 90) / 365)

        # City-specific NDVI level (ecological baseline)
        if cities.iloc[i]["eco_baseline"] == "fragile":
            base_ndvi *= 0.8
        if cities.iloc[i]["city_size"] == "large":
            base_ndvi *= 0.85  # urbanization reduces NDVI

        # Random variation
        ndvi_series = base_ndvi + np.random.normal(0, 0.03, n_periods)

        # Apply extreme weather shocks (reduce NDVI during event windows)
        for ev_type, start_day, end_day, intensity in events[i]:
            for p in range(n_periods):
                period_center = period_days[p]
                if start_day <= period_center <= end_day + 30:
                    # NDVI reduction depends on event type and intensity
                    if ev_type == "heat":
                        reduction = 0.05 + 0.01 * intensity
                    elif ev_type == "rain":
                        reduction = 0.03 + 0.001 * intensity
                    else:  # drought
                        reduction = 0.08 + 0.002 * intensity
                    ndvi_series[p] -= reduction

                    # Policy buffer effect: if policy active, reduce the shock impact
                    if policy_active[i]:
                        buffer = TRUE_BUFFER_EFFECT * (1 + np.random.normal(0, 0.1))
                        ndvi_series[p] += min(buffer, reduction * 0.6)

        # Policy also has a direct effect on NDVI level (greening)
        if policy_active[i]:
            ndvi_series += np.random.uniform(0.01, 0.03, n_periods)

        ndvi[i] = np.clip(ndvi_series, 0.05, 0.95)

    return ndvi


def _generate_control_variables(cities, year):
    """Generate high-dimensional control variables for each city-year."""
    n = len(cities)
    t = year - START_YEAR

    # Log GDP per capita: grows over time, varies by city
    base_gdp = np.random.lognormal(10.5, 0.6, n)
    gdppc = base_gdp * (1 + 0.07) ** t
    if year >= POLICY_YEAR:
        gdppc[cities["treat"].values == 1] *= 1.03  # policy also boosts economy slightly

    data = pd.DataFrame({
        "ln_gdppc": np.log(gdppc),
        "gdp_growth": np.random.normal(0.08, 0.03, n),
        "ind_share": np.clip(np.random.normal(0.42, 0.1, n), 0.15, 0.65),
        "ter_share": np.clip(np.random.normal(0.45, 0.12, n), 0.2, 0.75),
        "pop_density": np.random.lognormal(6, 0.8, n),
        "urban_rate": np.clip(0.4 + 0.02 * t + np.random.normal(0, 0.1, n), 0.2, 0.95),
        "annual_temp": 15 - 0.4 * cities["lat"].values + np.random.normal(0, 1, n),
        "annual_precip": np.clip(800 + 20 * (40 - cities["lat"].values)
                                 + np.random.normal(0, 200, n), 100, 2500),
        "elevation": cities["elevation"].values,
        "built_area": np.random.lognormal(4 + 0.05 * t, 0.7, n),
        "road_density": np.random.lognormal(2 + 0.03 * t, 0.5, n),
        "green_rate": np.clip(0.35 + 0.01 * t + np.random.normal(0, 0.08, n), 0.1, 0.65),
        "env_exp_share": np.clip(0.03 + 0.002 * t + np.random.normal(0, 0.01, n), 0.01, 0.1),
        "edu_level": np.clip(np.random.normal(0.2 + 0.005 * t, 0.05, n), 0.05, 0.5),
        "tech_exp": np.random.lognormal(8 + 0.08 * t, 0.9, n),
    })

    return data


def _generate_mediation_variables(cities, year, policy_active):
    """Generate mediation variables for mechanism analysis."""
    n = len(cities)
    t = year - START_YEAR

    # Green rate: policy increases greening
    green_rate = np.clip(0.35 + 0.01 * t + np.random.normal(0, 0.08, n), 0.1, 0.65)
    if year >= POLICY_YEAR:
        green_rate[policy_active == 1] += np.random.uniform(0.02, 0.06, n).clip(0)[policy_active == 1]

    # Sponge city investment (log): only after policy
    sponge_inv = np.zeros(n)
    if year >= POLICY_YEAR:
        sponge_inv[policy_active == 1] = np.random.lognormal(9, 0.5, n)[policy_active == 1]
        sponge_inv[policy_active == 0] = np.random.lognormal(6, 0.8, n)[policy_active == 0]
    else:
        sponge_inv = np.random.lognormal(5.5, 0.7, n)

    # Blue-green space ratio
    blue_green_ratio = np.clip(0.3 + 0.005 * t + np.random.normal(0, 0.08, n), 0.1, 0.6)
    if year >= POLICY_YEAR:
        blue_green_ratio[policy_active == 1] += np.random.uniform(0.01, 0.04, n).clip(0)[policy_active == 1]

    # Coupling coordination degree (0-1)
    coupling_coord = np.clip(0.5 + 0.01 * t + np.random.normal(0, 0.1, n), 0.2, 0.9)
    if year >= POLICY_YEAR:
        coupling_coord[policy_active == 1] += np.random.uniform(0.02, 0.06, n).clip(0)[policy_active == 1]

    return pd.DataFrame({
        "green_rate_m": green_rate,
        "sponge_inv": sponge_inv,
        "blue_green_ratio": blue_green_ratio,
        "coupling_coord": coupling_coord,
    })


def generate_panel_data():
    """Generate the full city-year panel dataset.

    Returns:
        panel: DataFrame with city-year observations
        ndvi_ts: dict of {city_id: {year: NDVI array}}
        weather_events: dict of {city_id: {year: events list}}
    """
    print("=" * 70)
    print("Generating synthetic city panel data...")
    print(f"  Cities: {N_CITIES}, Years: {START_YEAR}-{END_YEAR}")
    print("-" * 70)

    # Step 1: City characteristics
    cities = _generate_city_characteristics(N_CITIES)
    cities = _assign_policy(cities)
    n_pilot = cities["treat"].sum()
    print(f"  Pilot cities assigned: {n_pilot} (target: {N_PILOT_CITIES})")

    # Step 2: Generate yearly data
    panel_rows = []
    ndvi_ts = {}
    weather_events = {}

    for year in range(START_YEAR, END_YEAR + 1):
        policy_active = (cities["treat"].values.astype(int) * int(year >= POLICY_YEAR))

        # Generate daily weather
        daily_temp, daily_rain, days = _generate_daily_weather(cities, year)

        # Identify extreme weather events
        events = _identify_extreme_events(daily_temp, daily_rain, days)

        # Generate NDVI time series
        ndvi = _generate_ndvi_timeseries(cities, year, daily_temp, daily_rain,
                                         events, policy_active)

        # Store NDVI time series and events
        for i in range(N_CITIES):
            cid = int(cities.iloc[i]["city_id"])
            if cid not in ndvi_ts:
                ndvi_ts[cid] = {}
                weather_events[cid] = {}
            ndvi_ts[cid][year] = ndvi[i]
            weather_events[cid][year] = events[i]

        # Count extreme weather events per city
        n_events = np.array([len(events[i]) for i in range(N_CITIES)])
        n_heat = np.array([sum(1 for e in events[i] if e[0] == "heat") for i in range(N_CITIES)])
        n_rain = np.array([sum(1 for e in events[i] if e[0] == "rain") for i in range(N_CITIES)])
        n_drought = np.array([sum(1 for e in events[i] if e[0] == "drought") for i in range(N_CITIES)])

        # Shock intensity: standardized event count × intensity
        shock_intensity = np.zeros(N_CITIES)
        for i in range(N_CITIES):
            if len(events[i]) > 0:
                intensities = [e[3] for e in events[i]]
                shock_intensity[i] = np.sum(intensities)
        shock_intensity = (shock_intensity - shock_intensity.mean()) / (shock_intensity.std() + 1e-8)
        shock_intensity = np.clip(shock_intensity, 0, None)  # non-negative

        # Dominant shock type
        shock_counts = np.stack([n_heat, n_rain, n_drought], axis=1)
        dominant_type = np.argmax(shock_counts, axis=1)
        shock_type_labels = np.where(dominant_type == 0, "heat",
                              np.where(dominant_type == 1, "rain", "drought"))

        # Control variables
        controls = _generate_control_variables(cities, year)

        # Mediation variables
        med_vars = _generate_mediation_variables(cities, year, policy_active)

        # Assemble row
        for i in range(N_CITIES):
            row = {
                "city_id": int(cities.iloc[i]["city_id"]),
                "year": year,
                "treat": int(cities.iloc[i]["treat"]),
                "post": int(year >= POLICY_YEAR),
                "did": int(cities.iloc[i]["treat"]) * int(year >= POLICY_YEAR),
                "n_events": int(n_events[i]),
                "n_heat": int(n_heat[i]),
                "n_rain": int(n_rain[i]),
                "n_drought": int(n_drought[i]),
                "shock_intensity": float(shock_intensity[i]),
                "shock_type": shock_type_labels[i],
                "lon": float(cities.iloc[i]["lon"]),
                "lat": float(cities.iloc[i]["lat"]),
                "region": cities.iloc[i]["region"],
                "ns": cities.iloc[i]["ns"],
                "coastal": int(cities.iloc[i]["coastal"]),
                "city_size": cities.iloc[i]["city_size"],
                "eco_baseline": cities.iloc[i]["eco_baseline"],
                "sponge_intensity": float(med_vars.iloc[i]["sponge_inv"]),
            }
            # Add control variables
            for col in CONTROL_VARS:
                row[col] = float(controls.iloc[i][col])
            # Add mediation variables (with _m suffix to avoid name collision)
            for col in med_vars.columns:
                row[col + "_m"] = float(med_vars.iloc[i][col])

            panel_rows.append(row)

        if year % 5 == 0 or year == START_YEAR or year == END_YEAR:
            print(f"  Year {year}: avg events={n_events.mean():.1f}, "
                  f"shock_intensity={shock_intensity.mean():.2f}")

    panel = pd.DataFrame(panel_rows)

    # Create interaction term: DID × shock_intensity
    panel["did_shock"] = panel["did"] * panel["shock_intensity"]

    # Policy intensity (continuous): sponge city investment (log, standardized)
    panel["policy_intensity"] = np.log(panel["sponge_intensity"].clip(1))
    panel.loc[panel["sponge_intensity"] == 0, "policy_intensity"] = 0

    print("-" * 70)
    print(f"Panel shape: {panel.shape}")
    n_pilot_obs = len(panel[panel["treat"] == 1])
    n_post_obs = len(panel[panel["post"] == 1])
    print(f"Pilot cities: {panel[panel['treat']==1]['city_id'].nunique()}")
    print(f"Total observations: {len(panel)}")
    print(f"Years with policy: {panel[panel['post']==1]['year'].nunique()}")
    print("=" * 70)

    return panel, ndvi_ts, weather_events


if __name__ == "__main__":
    panel, ndvi_ts, events = generate_panel_data()
    print(panel.head())
    print(f"\nNDVI time series example (city 0, year 2015): {ndvi_ts[0][2015][:5]}...")
    print(f"Events example (city 0, year 2015): {events[0][2015][:3]}")
