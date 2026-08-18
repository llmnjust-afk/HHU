"""
Heterogeneity analysis module.

Tests which cities benefit more from the climate-resilient city policy
across five dimensions:

  1. City size (large vs small)
  2. Geographic region (east/central/west, north/south, coastal/inland)
  3. Ecological baseline (fragile vs good)
  4. Extreme weather type (heat/rain/drought dominant)
  5. Policy implementation intensity (high/low investment)
"""
import numpy as np
import pandas as pd
from dml_estimator import dml_plr
from config import CONTROL_VARS, HETERO_VARS


def run_heterogeneity(panel, y_col="csee", x_cols=None):
    """Run heterogeneity analysis across all dimensions.

    Args:
        y_col: outcome variable
        x_cols: control variables

    Returns:
        dict with subgroup DML results
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n" + "=" * 70)
    print("HETEROGENEITY ANALYSIS: Which cities benefit more?")
    print("=" * 70)

    all_results = {}

    # 1. City size
    print("\n  [1] City Size:")
    size_results = _subgroup_analysis(panel, y_col, "city_size", x_cols)
    all_results["city_size"] = size_results
    _print_subgroup("City Size", size_results)

    # 2. Geographic region
    print("\n  [2] Geographic Region (East/Central/West):")
    region_results = _subgroup_analysis(panel, y_col, "region", x_cols)
    all_results["region"] = region_results
    _print_subgroup("Region", region_results)

    # 3. North/South
    print("\n  [3] North/South:")
    ns_results = _subgroup_analysis(panel, y_col, "ns", x_cols)
    all_results["ns"] = ns_results
    _print_subgroup("North/South", ns_results)

    # 4. Coastal/Inland
    print("\n  [4] Coastal/Inland:")
    coastal_results = _subgroup_analysis(panel, y_col, "coastal", x_cols)
    all_results["coastal"] = coastal_results
    _print_subgroup("Coastal", coastal_results)

    # 5. Ecological baseline
    print("\n  [5] Ecological Baseline:")
    eco_results = _subgroup_analysis(panel, y_col, "eco_baseline", x_cols)
    all_results["eco_baseline"] = eco_results
    _print_subgroup("Ecological Baseline", eco_results)

    # 6. Shock type
    print("\n  [6] Dominant Shock Type:")
    shock_results = _subgroup_analysis(panel, y_col, "shock_type", x_cols)
    all_results["shock_type"] = shock_results
    _print_subgroup("Shock Type", shock_results)

    # 7. Policy intensity (continuous, split at median)
    print("\n  [7] Policy Implementation Intensity:")
    panel_temp = panel.copy()
    if "policy_intensity" in panel_temp.columns:
        median_intensity = panel_temp.loc[panel_temp["policy_intensity"] > 0, "policy_intensity"].median()
        panel_temp["intensity_group"] = np.where(
            panel_temp["policy_intensity"] > median_intensity, "high", "low"
        )
        intensity_results = _subgroup_analysis(panel_temp, y_col, "intensity_group", x_cols)
        all_results["policy_intensity"] = intensity_results
        _print_subgroup("Policy Intensity", intensity_results)

    return all_results


def _subgroup_analysis(panel, y_col, group_var, x_cols):
    """Run DML on each subgroup defined by group_var.

    Returns:
        dict {group_value: DML result}
    """
    groups = panel[group_var].dropna().unique()
    results = {}

    for g in sorted(groups, key=str):
        sub = panel[panel[group_var] == g].copy()
        if len(sub) > 50:
            res = dml_plr(sub, y_col, "did_shock", x_cols,
                          learner="random_forest", bootstrap=True)
            results[str(g)] = res

    return results


def _print_subgroup(label, results):
    """Print subgroup results."""
    print(f"    {label}:")
    for group, res in results.items():
        sig = "***" if res.get("p_value", 1) < 0.01 else \
              "**" if res.get("p_value", 1) < 0.05 else \
              "*" if res.get("p_value", 1) < 0.1 else ""
        print(f"      {group:15s}: θ = {res['theta']:+.6f} (SE = {res['se']:.6f}) n={res['n']} {sig}")


def heterogeneity_summary_table(all_results):
    """Compile all heterogeneity results into a summary table.

    Returns:
        DataFrame with all subgroup comparisons
    """
    rows = []
    for dim, results in all_results.items():
        for group, res in results.items():
            rows.append({
                "dimension": dim,
                "subgroup": group,
                "theta": res.get("theta", np.nan),
                "se": res.get("se", np.nan),
                "ci_lower": res.get("ci_lower", np.nan),
                "ci_upper": res.get("ci_upper", np.nan),
                "n": res.get("n", np.nan),
                "p_value": res.get("p_value", np.nan),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_simulation import generate_panel_data
    from csee_computation import compute_csee_panel

    panel, ndvi_ts, events = generate_panel_data()
    panel = compute_csee_panel(panel, ndvi_ts, events)

    all_results = run_heterogeneity(panel)
    summary = heterogeneity_summary_table(all_results)
    print("\n\nHeterogeneity Summary Table:")
    print(summary.to_string(index=False))
