"""
Climate-Stress Ecological Elasticity (CSEE) computation module.

Computes the core dependent variable from NDVI time series and
extreme weather events:
  - CR  (Climate Resistance):  1 - |NDVI_event - NDVI_normal| / NDVI_normal
  - RC  (Climate Recovery):    (NDVI_post - NDVI_event) / (NDVI_normal - NDVI_event)
  - CSEE (composite):           entropy-weighted combination of CR and RC

Also implements:
  - Entropy weight method for CSEE aggregation
  - RSEI (Remote Sensing Ecological Index) as robustness alternative
  - PSR-based composite resilience index as robustness alternative
"""
import numpy as np
import pandas as pd
from config import (
    NDVI_PERIODS_PER_YEAR, NORMAL_LOOKBACK_YEARS, RECOVERY_WINDOW_MONTHS,
    START_YEAR,
)


def compute_cr_rc(ndvi_event, ndvi_normal, ndvi_post):
    """Compute Climate Resistance and Climate Recovery.

    Args:
        ndvi_event:  NDVI mean during extreme weather event window
        ndvi_normal: NDVI mean of normal years (same period)
        ndvi_post:   NDVI mean during recovery window after event

    Returns:
        (cr, rc): tuple of resistance and recovery indices
    """
    # Resistance: how well the ecosystem maintained its function during shock
    if ndvi_normal > 0.01:
        cr = 1.0 - abs(ndvi_event - ndvi_normal) / ndvi_normal
    else:
        cr = 0.5  # default when normal NDVI is too low
    cr = np.clip(cr, 0.0, 1.0)

    # Recovery: how fast the ecosystem bounced back
    denom = ndvi_normal - ndvi_event
    if abs(denom) > 0.005:
        rc = (ndvi_post - ndvi_event) / denom
    else:
        rc = 1.0  # full recovery if barely any damage
    rc = np.clip(rc, 0.0, 1.0)

    return float(cr), float(rc)


def _get_period_index_for_day(day, n_periods=NDVI_PERIODS_PER_YEAR):
    """Map a day-of-year to the closest 16-day NDVI period index."""
    period_days = np.linspace(1, 365, n_periods)
    return int(np.argmin(np.abs(period_days - day)))


def compute_city_year_csee(ndvi_series, events, year, ndvi_history, n_periods=NDVI_PERIODS_PER_YEAR):
    """Compute CR, RC, and CSEE for a single city-year.

    Args:
        ndvi_series:  NDVI array of shape (n_periods,) for this city-year
        events:       list of (event_type, start_day, end_day, intensity)
        year:         the year
        ndvi_history: dict {year: ndvi_array} for this city (for normal baseline)
        n_periods:    number of NDVI periods per year

    Returns:
        dict with cr, rc, csee, n_events, mean_intensity
    """
    if len(events) == 0:
        # No extreme weather events: use NDVI inter-annual variability as proxy
        # Higher stability (lower CV) → higher resilience
        if len(ndvi_history) >= 2:
            historical_means = [np.mean(ndvi_history[y]) for y in sorted(ndvi_history.keys()) if y < year]
            if len(historical_means) >= 2:
                cv = np.std(historical_means) / (np.mean(historical_means) + 1e-8)
                proxy = 1.0 / (1.0 + cv)  # higher stability → higher proxy
            else:
                proxy = 0.5
        else:
            proxy = 0.5

        return {
            "cr": float(np.clip(proxy, 0, 1)),
            "rc": float(np.clip(proxy, 0, 1)),
            "csee": float(np.clip(proxy, 0, 1)),
            "n_events": 0,
            "mean_intensity": 0.0,
        }

    # Compute normal-year NDVI baseline (average of previous N years, same periods)
    normal_years = [y for y in sorted(ndvi_history.keys())
                    if y < year and year - y <= NORMAL_LOOKBACK_YEARS]

    if normal_years:
        ndvi_normal_all = np.array([ndvi_history[y] for y in normal_years])
        ndvi_normal = np.mean(ndvi_normal_all, axis=0)  # shape (n_periods,)
    else:
        ndvi_normal = ndvi_series.copy()  # fallback: use current year

    crs, rcs = [], []

    for ev_type, start_day, end_day, intensity in events:
        # Event period NDVI
        event_start_idx = _get_period_index_for_day(start_day, n_periods)
        event_end_idx = _get_period_index_for_day(end_day, n_periods)
        event_end_idx = min(event_end_idx + 1, n_periods)  # inclusive

        ndvi_event = float(np.mean(ndvi_series[event_start_idx:event_end_idx]))
        ndvi_normal_period = float(np.mean(ndvi_normal[event_start_idx:event_end_idx]))

        # Recovery period NDVI (RECOVERY_WINDOW_MONTHS after event end)
        recovery_start_day = end_day + RECOVERY_WINDOW_MONTHS * 30
        recovery_end_day = recovery_start_day + 30  # 1-month recovery window

        if recovery_end_day <= 365:
            rec_start_idx = _get_period_index_for_day(recovery_start_day, n_periods)
            rec_end_idx = _get_period_index_for_day(recovery_end_day, n_periods)
            rec_end_idx = min(rec_end_idx + 1, n_periods)
            if rec_start_idx < rec_end_idx:
                ndvi_post = float(np.mean(ndvi_series[rec_start_idx:rec_end_idx]))
            else:
                ndvi_post = ndvi_event  # can't measure recovery
        else:
            ndvi_post = ndvi_event  # event too late in year

        cr, rc = compute_cr_rc(ndvi_event, ndvi_normal_period, ndvi_post)
        crs.append(cr)
        rcs.append(rc)

    cr_mean = float(np.mean(crs))
    rc_mean = float(np.mean(rcs))
    mean_intensity = float(np.mean([e[3] for e in events]))

    # CSEE: equal-weight composite (entropy weights computed separately in batch)
    csee = 0.5 * cr_mean + 0.5 * rc_mean

    return {
        "cr": cr_mean,
        "rc": rc_mean,
        "csee": csee,
        "n_events": len(events),
        "mean_intensity": mean_intensity,
    }


def entropy_weight_method(data, positive_cols=None):
    """Compute entropy weights for composite index construction.

    Args:
        data:           DataFrame with indicator columns
        positive_cols:  list of column names (all treated as positive indicators)

    Returns:
        weights: Series of entropy weights
        composite: Series of composite index values
    """
    if positive_cols is None:
        positive_cols = data.columns.tolist()

    df = data[positive_cols].copy().astype(float)

    # Min-max normalization to [0.01, 1] (avoid log(0))
    for col in positive_cols:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max - col_min > 1e-8:
            df[col] = 0.01 + 0.99 * (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.5

    n = len(df)

    # Compute entropy per indicator
    weights = {}
    entropies = {}
    for col in positive_cols:
        p = df[col] / df[col].sum()
        p = p.clip(1e-10, None)
        e = -1.0 / np.log(n) * np.sum(p * np.log(p))
        entropies[col] = e

    # Weight = (1 - e) / sum(1 - e)
    total_d = sum(1 - e for e in entropies.values())
    for col in positive_cols:
        weights[col] = (1 - entropies[col]) / total_d if total_d > 1e-8 else 1.0 / len(positive_cols)

    weights = pd.Series(weights)
    composite = (df[positive_cols] * weights).sum(axis=1)

    return weights, composite


def compute_csee_panel(panel, ndvi_ts, weather_events):
    """Compute CSEE for the full panel.

    Args:
        panel:          DataFrame with city-year observations
        ndvi_ts:        dict {city_id: {year: NDVI array}}
        weather_events: dict {city_id: {year: events list}}

    Returns:
        panel with cr, rc, csee columns added
    """
    print("=" * 70)
    print("Computing Climate-Stress Ecological Elasticity (CSEE)...")
    print("-" * 70)

    cr_list, rc_list, csee_list = [], [], []
    n_events_list, mean_int_list = [], []

    for _, row in panel.iterrows():
        cid = int(row["city_id"])
        year = int(row["year"])

        ndvi_series = ndvi_ts[cid][year]
        events = weather_events[cid][year]

        result = compute_city_year_csee(
            ndvi_series, events, year, ndvi_ts[cid]
        )

        cr_list.append(result["cr"])
        rc_list.append(result["rc"])
        csee_list.append(result["csee"])
        n_events_list.append(result["n_events"])
        mean_int_list.append(result["mean_intensity"])

    panel = panel.copy()
    panel["cr"] = cr_list
    panel["rc"] = rc_list
    panel["csee_raw"] = csee_list  # equal-weight version

    # Recompute CSEE using entropy weight method (batch)
    print("  Computing entropy weights for CSEE aggregation...")
    weights, csee_entropy = entropy_weight_method(
        panel[["cr", "rc"]].dropna(),
        positive_cols=["cr", "rc"]
    )
    panel["csee"] = csee_entropy.reindex(panel.index)
    print(f"  Entropy weights: CR={weights.get('cr', 0.5):.4f}, RC={weights.get('rc', 0.5):.4f}")

    # Also compute an RSEI-like proxy for robustness
    # RSEI integrates greenness (NDVI), wetness, heat, dryness
    # Here we simulate with available variables
    panel["rsei"] = compute_rsei_proxy(panel)

    # PSR-based composite resilience index (for robustness)
    panel["psr_resilience"] = compute_psr_resilience(panel)

    print("-" * 70)
    print(f"  CSEE stats: mean={panel['csee'].mean():.4f}, std={panel['csee'].std():.4f}")
    print(f"  CR stats:   mean={panel['cr'].mean():.4f}, std={panel['cr'].std():.4f}")
    print(f"  RC stats:   mean={panel['rc'].mean():.4f}, std={panel['rc'].std():.4f}")
    print(f"  RSEI stats: mean={panel['rsei'].mean():.4f}, std={panel['rsei'].std():.4f}")
    print("=" * 70)

    return panel


def compute_rsei_proxy(panel):
    """Compute a simplified RSEI (Remote Sensing Ecological Index) proxy.

    RSEI integrates four components:
      - Greenness (NDVI proxy via green_rate)
      - Wetness (annual_precip)
      - Heat (annual_temp, inverted)
      - Dryness (built_area / green space)

    All normalized to [0,1] and averaged.
    """
    green = panel["green_rate"].clip(0, 1)
    wet = (panel["annual_precip"] / panel["annual_precip"].max()).clip(0, 1)
    heat = (1 - (panel["annual_temp"] - panel["annual_temp"].min()) /
            (panel["annual_temp"].max() - panel["annual_temp"].min() + 1e-8)).clip(0, 1)
    dry = (1 - (panel["built_area"] / (panel["built_area"].max() + 1e-8))).clip(0, 1)

    rsei = (green + wet + heat + dry) / 4
    return rsei


def compute_psr_resilience(panel):
    """Compute PSR-framework composite resilience index (for robustness check).

    Pressure:   pollution intensity, extreme weather (inverted)
    State:      green rate, NDVI proxy
    Response:   env expenditure, tech expenditure

    Uses entropy weight method for aggregation.
    """
    psr_data = pd.DataFrame({
        "p_pressure": 1 - (panel["shock_intensity"] / (panel["shock_intensity"].max() + 1e-8)),
        "p_pollution": 1 - (panel["ind_share"] / (panel["ind_share"].max() + 1e-8)),
        "s_green": panel["green_rate"].clip(0, 1),
        "s_eco": panel["edu_level"].clip(0, 1),
        "r_env": panel["env_exp_share"] / (panel["env_exp_share"].max() + 1e-8),
        "r_tech": panel["tech_exp"] / (panel["tech_exp"].max() + 1e-8),
    })

    _, composite = entropy_weight_method(psr_data)
    return composite


if __name__ == "__main__":
    from data_simulation import generate_panel_data

    panel, ndvi_ts, events = generate_panel_data()
    panel = compute_csee_panel(panel, ndvi_ts, events)

    print("\nSample CSEE values by treatment status:")
    print(panel.groupby(["treat", "post"])[["cr", "rc", "csee"]].mean().round(4))
