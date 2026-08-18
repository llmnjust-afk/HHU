"""
Spatial spillover analysis module.

Tests whether climate-resilient city construction generates spatial
spillover effects on neighboring cities:

  - Positive spillover (demonstration effect)
  - Negative spillover (resource siphoning)

Implements Spatial DML by incorporating spatially-lagged treatment
variable from neighboring cities.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from dml_estimator import dml_plr
from config import CONTROL_VARS


def build_spatial_weights(panel, method="inverse_distance", k_neighbors=5):
    """Build spatial weight matrix.

    Args:
        panel:   DataFrame with city_id, lon, lat
        method:  "inverse_distance", "knn", or "contiguity"
        k_neighbors: number of nearest neighbors for KNN

    Returns:
        W: spatial weight matrix (n_cities × n_cities)
        city_ids: array of city_ids corresponding to matrix rows/cols
    """
    # Get unique city coordinates
    cities = panel[["city_id", "lon", "lat"]].drop_duplicates().sort_values("city_id")
    coords = cities[["lon", "lat"]].values
    city_ids = cities["city_id"].values
    n = len(city_ids)

    # Distance matrix
    dist = cdist(coords, coords, metric="euclidean")
    np.fill_diagonal(dist, np.inf)  # exclude self

    if method == "inverse_distance":
        W = 1.0 / (dist + 1e-8)
        np.fill_diagonal(W, 0)  # no self-influence
        # Row-standardize
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

    elif method == "knn":
        W = np.zeros((n, n))
        for i in range(n):
            neighbors = np.argsort(dist[i])[:k_neighbors]
            W[i, neighbors] = 1.0
        # Row-standardize
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

    else:  # contiguity (threshold-based)
        threshold = np.median(dist) * 0.3
        W = (dist < threshold).astype(float)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

    return W, city_ids


def compute_spatial_lag(panel, var_name, W, city_ids):
    """Compute spatially-lagged variable: W·X.

    For each city, compute the average of var_name among neighboring cities.

    Returns:
        Series with spatially-lagged values
    """
    # Pivot to city-year matrix
    pivot = panel.pivot_table(index="city_id", columns="year", values=var_name)

    # Align with weight matrix
    pivot = pivot.reindex(city_ids)

    # Spatial lag: W × X
    spatial_lag = pd.DataFrame(W @ pivot.values, index=city_ids, columns=pivot.columns)
    spatial_lag.index.name = "city_id"

    # Convert back to long format
    spatial_lag_long = spatial_lag.reset_index().melt(
        id_vars="city_id", var_name="year", value_name=f"spatial_{var_name}"
    )
    spatial_lag_long["year"] = spatial_lag_long["year"].astype(int)

    return spatial_lag_long


def spatial_dml(panel, y_col="csee", x_cols=None, weight_method="inverse_distance"):
    """Spatial DML: incorporate spatial spillover into DML framework.

    Model:
        Y = θ·D + ρ·W·D + g(X, W·X) + ε

    Tests whether policy in neighboring cities affects local CSEE.

    Returns:
        dict with direct effect, spatial spillover effect, total effect
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n  [Spatial Spillover] Method: " + weight_method)
    print("  " + "-" * 60)

    # Build spatial weights
    W, city_ids = build_spatial_weights(panel, method=weight_method)

    # Compute spatial lag of treatment
    spatial_did = compute_spatial_lag(panel, "did_shock", W, city_ids)
    panel_sp = panel.merge(spatial_did, on=["city_id", "year"], how="left")

    # Also compute spatial lag of key control variables
    for var in ["shock_intensity", "ln_gdppc"]:
        if var in panel.columns:
            sp_var = compute_spatial_lag(panel, var, W, city_ids)
            panel_sp = panel_sp.merge(sp_var, on=["city_id", "year"], how="left")

    # Extended controls with spatial lags
    x_extended = x_cols + [c for c in panel_sp.columns if c.startswith("spatial_")]

    # DML with direct effect only (baseline)
    direct_res = dml_plr(panel_sp, y_col, "did_shock", x_cols, learner="random_forest")
    print(f"    Direct effect:       θ = {direct_res['theta']:+.6f} (SE = {direct_res['se']:.6f})")

    # DML with spatial spillover
    spillover_res = dml_plr(panel_sp, y_col, "spatial_did_shock", x_cols,
                            learner="random_forest")
    print(f"    Spatial spillover:    ρ = {spillover_res['theta']:+.6f} (SE = {spillover_res['se']:.6f})")

    # Total effect (direct + spatial)
    total_effect = direct_res["theta"] + spillover_res["theta"]
    print(f"    Total effect (direct + spillover): {total_effect:+.6f}")

    # Interpretation
    if spillover_res["theta"] > 0 and spillover_res.get("p_value", 1) < 0.1:
        print("    → Positive spatial spillover (demonstration effect)")
    elif spillover_res["theta"] < 0 and spillover_res.get("p_value", 1) < 0.1:
        print("    → Negative spatial spillover (resource siphoning)")
    else:
        print("    → No significant spatial spillover")

    return {
        "direct": direct_res,
        "spillover": spillover_res,
        "total": total_effect,
        "weight_method": weight_method,
    }


def run_spatial_analysis(panel, x_cols=None):
    """Run spatial spillover analysis with multiple weight matrices.

    Returns:
        dict with results for each weight matrix type
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n" + "=" * 70)
    print("SPATIAL SPILLOVER ANALYSIS")
    print("=" * 70)

    all_results = {}

    for method in ["inverse_distance", "knn", "contiguity"]:
        all_results[method] = spatial_dml(panel, "csee", x_cols, weight_method=method)

    return all_results


if __name__ == "__main__":
    from data_simulation import generate_panel_data
    from csee_computation import compute_csee_panel

    panel, ndvi_ts, events = generate_panel_data()
    panel = compute_csee_panel(panel, ndvi_ts, events)

    results = run_spatial_analysis(panel)
