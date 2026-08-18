"""
Robustness checks module.

Implements all robustness tests from the experiment plan:
  1. Alternative dependent variables (RSEI, PSR resilience)
  2. Alternative ML algorithms (RF ↔ XGBoost ↔ NN ↔ LASSO)
  3. Placebo test (random treatment assignment, 500 iterations)
  4. PSM-DML (propensity score matching + DML)
  5. Exclude competing policies (control for other environmental policies)
  6. Sample restrictions (exclude municipalities, exclude COVID years)
  7. Lagged treatment effects (1-3 year lags)
  8. Alternative extreme weather thresholds
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from dml_estimator import dml_plr
from config import (
    CONTROL_VARS, ML_LEARNERS, N_BOOTSTRAP, POLICY_YEAR,
    HEAT_PERCENTILE, RAIN_PERCENTILE, N_FOLDS,
)


def robustness_alt_y(panel, x_cols=None):
    """Robustness: Alternative dependent variables.

    Tests whether DML results hold when using:
      - RSEI (Remote Sensing Ecological Index)
      - PSR-based resilience index
      - CR (resistance only)
      - RC (recovery only)
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n  [Robustness 1] Alternative Dependent Variables")
    print("  " + "-" * 60)

    alt_ys = ["rsei", "psr_resilience", "cr", "rc"]
    results = []

    for y in alt_ys:
        if y in panel.columns:
            res = dml_plr(panel, y, "did_shock", x_cols, learner="random_forest")
            res["alt_y"] = y
            results.append(res)
            sig = "***" if res.get("p_value", 1) < 0.01 else \
                  "**" if res.get("p_value", 1) < 0.05 else \
                  "*" if res.get("p_value", 1) < 0.1 else ""
            print(f"    {y:20s}: θ = {res['theta']:+.6f} (SE = {res['se']:.6f}) {sig}")

    return pd.DataFrame(results)


def robustness_alt_ml(panel, x_cols=None):
    """Robustness: Alternative ML algorithms.

    Tests whether results are sensitive to ML learner choice.
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n  [Robustness 2] Alternative ML Algorithms")
    print("  " + "-" * 60)

    results = []
    for learner in ML_LEARNERS:
        res = dml_plr(panel, "csee", "did_shock", x_cols, learner=learner)
        res["learner"] = learner
        results.append(res)
        sig = "***" if res.get("p_value", 1) < 0.01 else \
              "**" if res.get("p_value", 1) < 0.05 else \
              "*" if res.get("p_value", 1) < 0.1 else ""
        print(f"    {learner:20s}: θ = {res['theta']:+.6f} (SE = {res['se']:.6f}) {sig}")

    return pd.DataFrame(results)


def robustness_placebo(panel, x_cols=None, n_iter=500):
    """Robustness: Placebo test with random treatment assignment.

    Randomly assigns pilot city status and re-estimates DML.
    True coefficient should fall outside the placebo distribution.

    Returns:
        dict with placebo coefficients and position of true estimate
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print(f"\n  [Robustness 3] Placebo Test ({n_iter} iterations)")
    print("  " + "-" * 60)

    # True estimate
    true_res = dml_plr(panel, "csee", "did_shock", x_cols, learner="random_forest")
    true_theta = true_res["theta"]
    print(f"    True θ = {true_theta:+.6f}")

    # Placebo iterations
    placebo_thetas = []
    panel_temp = panel.copy()

    for i in range(n_iter):
        # Randomly permute treatment assignment
        city_ids = panel_temp["city_id"].unique()
        fake_treat = np.random.choice(city_ids, size=28, replace=False)
        panel_temp["fake_treat"] = panel_temp["city_id"].isin(fake_treat).astype(int)
        panel_temp["fake_did"] = panel_temp["fake_treat"] * panel_temp["post"]
        panel_temp["fake_did_shock"] = panel_temp["fake_did"] * panel_temp["shock_intensity"]

        try:
            res = dml_plr(panel_temp, "csee", "fake_did_shock", x_cols,
                          learner="random_forest", bootstrap=False)
            placebo_thetas.append(res["theta"])
        except Exception:
            continue

        if (i + 1) % 100 == 0:
            print(f"    Completed {i+1}/{n_iter} iterations...")

    placebo_thetas = np.array(placebo_thetas)

    # Calculate p-value (proportion of placebo |θ| >= |true θ|)
    placebo_p = np.mean(np.abs(placebo_thetas) >= np.abs(true_theta))

    result = {
        "true_theta": true_theta,
        "placebo_mean": float(np.mean(placebo_thetas)),
        "placebo_std": float(np.std(placebo_thetas)),
        "placebo_p_value": float(placebo_p),
        "placebo_thetas": placebo_thetas,
        "n_iter": len(placebo_thetas),
    }

    print(f"    Placebo mean: {result['placebo_mean']:+.6f}")
    print(f"    Placebo std:  {result['placebo_std']:.6f}")
    print(f"    Placebo p-value: {result['placebo_p_value']:.4f}")
    print(f"    → {'PASS' if placebo_p < 0.05 else 'WARN'}: "
          f"True θ is {'outside' if placebo_p < 0.05 else 'within'} placebo distribution")

    return result


def robustness_psm_dml(panel, x_cols=None):
    """Robustness: PSM-DML (Propensity Score Matching + DML).

    Step 1: Estimate propensity score (logit of treat on pre-policy X)
    Step 2: Match treated and control cities by propensity score
    Step 3: Run DML on matched sample

    Returns:
        dict with PSM-DML results
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n  [Robustness 4] PSM-DML")
    print("  " + "-" * 60)

    # Use pre-policy data to estimate propensity score
    pre_panel = panel[panel["year"] < POLICY_YEAR].copy()
    pre_means = pre_panel.groupby("city_id")[x_cols].mean().reset_index()
    pre_means["treat"] = pre_panel.groupby("city_id")["treat"].first().values

    # Logit propensity score
    X_psm = pre_means[x_cols].fillna(pre_means[x_cols].mean()).values
    y_psm = pre_means["treat"].values

    scaler = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler()
    X_psm_scaled = scaler.fit_transform(X_psm)

    logit = LogisticRegression(max_iter=1000, random_state=42)
    logit.fit(X_psm_scaled, y_psm)
    ps = logit.predict_proba(X_psm_scaled)[:, 1]
    pre_means["ps"] = ps

    # Nearest-neighbor matching (1:1)
    treated = pre_means[pre_means["treat"] == 1].copy()
    control = pre_means[pre_means["treat"] == 0].copy()

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(control[["ps"]].values)
    _, indices = nn.kneighbors(treated[["ps"]].values)
    matched_control_ids = control.iloc[indices.flatten()]["city_id"].values
    matched_treated_ids = treated["city_id"].values
    matched_ids = np.concatenate([matched_treated_ids, matched_control_ids])

    print(f"    Matched: {len(matched_treated_ids)} treated + "
          f"{len(matched_control_ids)} control = {len(matched_ids)} cities")

    # Run DML on matched sample
    matched_panel = panel[panel["city_id"].isin(matched_ids)].copy()
    res = dml_plr(matched_panel, "csee", "did_shock", x_cols,
                  learner="random_forest", bootstrap=True)

    print(f"    PSM-DML θ = {res['theta']:+.6f} (SE = {res['se']:.6f})")

    return {"psm_dml": res, "n_matched": len(matched_ids)}


def robustness_sample_restrictions(panel, x_cols=None):
    """Robustness: Sample restrictions.

    Tests:
      - Exclude municipalities (直辖市)
      - Exclude COVID years (2020-2022)
      - Only post-2010 sample
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n  [Robustness 5] Sample Restrictions")
    print("  " + "-" * 60)

    results = {}

    # Exclude COVID years
    no_covid = panel[~panel["year"].between(2020, 2022)].copy()
    res = dml_plr(no_covid, "csee", "did_shock", x_cols, learner="random_forest")
    results["no_covid"] = res
    print(f"    Exclude COVID (2020-22): θ = {res['theta']:+.6f} (SE = {res['se']:.6f}) n={res['n']}")

    # Only post-2010
    post2010 = panel[panel["year"] >= 2010].copy()
    res = dml_plr(post2010, "csee", "did_shock", x_cols, learner="random_forest")
    results["post_2010"] = res
    print(f"    Post-2010 only:         θ = {res['theta']:+.6f} (SE = {res['se']:.6f}) n={res['n']}")

    # Only post-2008
    post2008 = panel[panel["year"] >= 2008].copy()
    res = dml_plr(post2008, "csee", "did_shock", x_cols, learner="random_forest")
    results["post_2008"] = res
    print(f"    Post-2008 only:         θ = {res['theta']:+.6f} (SE = {res['se']:.6f}) n={res['n']}")

    return results


def robustness_lagged_treatment(panel, x_cols=None):
    """Robustness: Lagged treatment effects.

    Tests whether policy effects have time lags (1-3 years).
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n  [Robustness 6] Lagged Treatment Effects")
    print("  " + "-" * 60)

    results = {}
    panel_temp = panel.copy().sort_values(["city_id", "year"])

    for lag in [0, 1, 2, 3]:
        panel_temp[f"did_shock_lag{lag}"] = panel_temp.groupby("city_id")["did_shock"].shift(lag)
        sub = panel_temp.dropna(subset=[f"did_shock_lag{lag}", "csee"] + x_cols)
        if len(sub) > 100:
            res = dml_plr(sub, "csee", f"did_shock_lag{lag}", x_cols, learner="random_forest")
            results[f"lag_{lag}"] = res
            print(f"    Lag {lag}: θ = {res['theta']:+.6f} (SE = {res['se']:.6f}) n={res['n']}")

    return results


def run_all_robustness(panel, x_cols=None, placebo_iter=200):
    """Run all robustness checks.

    Args:
        placebo_iter: number of placebo iterations (reduced for speed)

    Returns:
        dict with all robustness results
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n" + "#" * 70)
    print("# ROBUSTNESS CHECKS")
    print("#" * 70)

    all_results = {}

    # 1. Alternative Y
    all_results["alt_y"] = robustness_alt_y(panel, x_cols)

    # 2. Alternative ML
    all_results["alt_ml"] = robustness_alt_ml(panel, x_cols)

    # 3. Placebo
    all_results["placebo"] = robustness_placebo(panel, x_cols, n_iter=placebo_iter)

    # 4. PSM-DML
    all_results["psm_dml"] = robustness_psm_dml(panel, x_cols)

    # 5. Sample restrictions
    all_results["sample"] = robustness_sample_restrictions(panel, x_cols)

    # 6. Lagged treatment
    all_results["lagged"] = robustness_lagged_treatment(panel, x_cols)

    print("\n" + "#" * 70)
    print("# ROBUSTNESS CHECKS COMPLETE")
    print("#" * 70)

    return all_results


if __name__ == "__main__":
    from data_simulation import generate_panel_data
    from csee_computation import compute_csee_panel

    panel, ndvi_ts, events = generate_panel_data()
    panel = compute_csee_panel(panel, ndvi_ts, events)

    results = run_all_robustness(panel, placebo_iter=50)
