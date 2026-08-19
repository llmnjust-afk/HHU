"""
Traditional Two-Way Fixed Effects (TWFE) DID model for comparison with DML.

Implements:
  - TWFE DID with interaction term (DID × Shock)
  - Event study / parallel trends test
  - Staggered DID diagnostics
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import CONTROL_VARS, POLICY_YEAR, START_YEAR


def twfe_did(panel, y_col, d_col="did_shock", x_cols=None,
             cluster="city_id"):
    """Traditional Two-Way Fixed Effects DID.

    Y_it = β·DID×Shock + γ·X_it + α_i + μ_t + ε_it

    Uses within-transformation (demeaning by city and year FE).

    Args:
        panel:    DataFrame
        y_col:    outcome variable name
        d_col:    treatment variable name
        x_cols:   control variable names
        cluster:  clustering variable for robust SE

    Returns:
        dict with beta, se, t_stat, p_value, ci, n
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    needed = [y_col, d_col] + x_cols + ["city_id", "year"]
    df = panel[needed].dropna().reset_index(drop=True)

    # Two-way FE via demeaning
    df_dm = df.copy()
    for col in [y_col, d_col] + x_cols:
        city_mean = df_dm.groupby("city_id")[col].transform("mean")
        year_mean = df_dm.groupby("year")[col].transform("mean")
        grand_mean = df_dm[col].mean()
        df_dm[col + "_dm"] = df_dm[col] - city_mean - year_mean + grand_mean

    Y = df_dm[y_col + "_dm"].values
    D = df_dm[d_col + "_dm"].values
    X = df_dm[[c + "_dm" for c in x_cols]].values

    # OLS
    exog = np.column_stack([D, X])
    model = sm.OLS(Y, exog)

    # Clustered standard errors
    clusters = df["city_id"].values
    try:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
    except Exception:
        result = model.fit(cov_type="HC1")

    beta = float(result.params[0])
    se = float(result.bse[0])
    t_stat = float(result.tvalues[0])
    p_value = float(result.pvalues[0])

    return {
        "beta": beta,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": beta - 1.96 * se,
        "ci_upper": beta + 1.96 * se,
        "n": len(df),
        "r_squared": float(result.rsquared),
    }


def event_study(panel, y_col="csee", x_cols=None, policy_year=POLICY_YEAR,
                pre_window=6, post_window=6):
    """Event study for parallel trends test with staggered treatment.

    Estimates leads and lags of treatment to test pre-trends.

    Y_it = Σ_k β_k · 1(year - policy_year_i = k) + γ·X + α_i + μ_t + ε_it

    For staggered treatment, each treated city uses its own policy_year.
    Non-treated cities serve as the control group throughout.

    Args:
        panel:       DataFrame with city-year observations
        y_col:       outcome variable name
        x_cols:      control variable names
        policy_year: fallback single policy year (used only if
                     panel lacks per-city policy_year)
        pre_window:  number of pre-treatment periods to estimate
        post_window: number of post-treatment periods to estimate

    Returns:
        DataFrame with event-time coefficients + parallel trends test stats
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    df = panel.copy()

    # ── Staggered event time ──────────────────────────────────────────
    # Each city uses its own policy_year if available
    if "policy_year" in df.columns:
        df["event_time"] = df.apply(
            lambda r: int(r["year"]) - int(r["policy_year"])
            if int(r["treat"]) == 1 else -999,
            axis=1
        )
    else:
        df["event_time"] = df.apply(
            lambda r: int(r["year"]) - policy_year
            if int(r["treat"]) == 1 else -999,
            axis=1
        )

    # ── Event-time dummies (omit k=-1 as reference) ───────────────────
    event_window = range(-pre_window, post_window + 1)
    et_cols = []
    for k in event_window:
        if k == -1:
            continue  # reference period
        col_name = f"et_{k:+d}".replace("+", "p").replace("-", "m")
        df[col_name] = ((df["event_time"] == k) & (df["treat"] == 1)).astype(int)
        et_cols.append(col_name)

    needed = [y_col, "city_id", "year", "treat"] + et_cols + x_cols
    df_clean = df[needed].dropna().reset_index(drop=True)

    # Two-way FE via demeaning
    for col in [y_col] + et_cols + x_cols:
        cm = df_clean.groupby("city_id")[col].transform("mean")
        ym = df_clean.groupby("year")[col].transform("mean")
        gm = df_clean[col].mean()
        df_clean[col + "_dm"] = df_clean[col] - cm - ym + gm

    Y = df_clean[y_col + "_dm"].values
    X = df_clean[[c + "_dm" for c in et_cols + x_cols]].values

    model = sm.OLS(Y, X)
    try:
        clusters = df_clean["city_id"].values
        result = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
    except Exception:
        result = model.fit(cov_type="HC1")

    # ── Build results table ───────────────────────────────────────────
    results_list = []
    pre_indices = []
    for idx, k in enumerate([k for k in event_window if k != -1]):
        coef = float(result.params[idx])
        se_val = float(result.bse[idx])
        p_val = float(result.pvalues[idx])
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        results_list.append({
            "event_time": k,
            "coefficient": coef,
            "se": se_val,
            "ci_lower": coef - 1.96 * se_val,
            "ci_upper": coef + 1.96 * se_val,
            "p_value": p_val,
            "significance": sig,
        })
        if k < -1:
            pre_indices.append(idx)

    es_df = pd.DataFrame(results_list)

    # ── Parallel trends test: joint F-test on pre-treatment leads ─────
    if len(pre_indices) > 0:
        # Wald test: all pre-treatment coefficients = 0
        R_matrix = np.zeros((len(pre_indices), len(result.params)))
        for i, pi in enumerate(pre_indices):
            R_matrix[i, pi] = 1.0
        try:
            f_test = result.wald_test(R_matrix, use_f=True)
            f_stat = float(f_test.fvalue)
            f_pval = float(f_test.pvalue)
            f_df = (len(pre_indices), int(result.df_resid))
        except Exception:
            f_stat = np.nan
            f_pval = np.nan
            f_df = (len(pre_indices), np.nan)

        print(f"\n  Parallel Trends Test ({y_col}):")
        print(f"  H0: all pre-treatment leads (k={-pre_window}..-2) = 0")
        print(f"  F({f_df[0]}, {f_df[1]}) = {f_stat:.4f}")
        print(f"  p-value = {f_pval:.4f}")
        if f_pval > 0.10:
            print(f"  → PASS: Cannot reject parallel trends (p > 0.10)")
        elif f_pval > 0.05:
            print(f"  → MARGINAL: Weak evidence against parallel trends (0.05 < p ≤ 0.10)")
        else:
            print(f"  → FAIL: Parallel trends rejected (p ≤ 0.05)")
    else:
        f_stat = np.nan
        f_pval = np.nan

    # ── Print event study table ───────────────────────────────────────
    print(f"\n  Event Study Results ({y_col}):")
    print(f"  {'Event Time':>12s} {'Coefficient':>14s} {'SE':>10s} {'p-value':>10s} {'Sig':>5s}")
    print(f"  {'-'*55}")
    for _, r in es_df.iterrows():
        print(f"  {int(r['event_time']):>+12d} {r['coefficient']:>+14.6f} "
              f"{r['se']:>10.6f} {r['p_value']:>10.4f} {r['significance']:>5s}")
    print(f"  {'(ref=-1)':>12s}")

    # Attach test stats as attributes
    es_df.attrs["f_stat"] = f_stat
    es_df.attrs["f_pvalue"] = f_pval
    es_df.attrs["y_col"] = y_col
    es_df.attrs["n_pre_leads"] = len(pre_indices)
    es_df.attrs["n_obs"] = len(df_clean)

    return es_df


def compare_dml_vs_did(panel, y_cols=None, x_cols=None):
    """Compare DML and traditional TWFE-DID results side by side.

    Returns:
        DataFrame comparing the two methods across multiple outcomes
    """
    from dml_estimator import dml_plr

    if y_cols is None:
        y_cols = ["csee", "cr", "rc", "rsei", "psr_resilience"]
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n" + "=" * 70)
    print("Comparison: DML vs Traditional TWFE-DID")
    print("=" * 70)
    print(f"{'Outcome':20s} | {'DML θ':>12s} | {'DML SE':>10s} | {'DID β':>12s} | {'DID SE':>10s} | {'Diff':>10s}")
    print("-" * 80)

    results = []
    for y in y_cols:
        dml_res = dml_plr(panel, y, "did_shock", x_cols, learner="random_forest")
        did_res = twfe_did(panel, y, "did_shack" if "did_shack" in panel.columns else "did_shock", x_cols)

        diff = dml_res["theta"] - did_res["beta"]
        results.append({
            "outcome": y,
            "dml_theta": dml_res["theta"],
            "dml_se": dml_res["se"],
            "dml_p": dml_res.get("p_value", np.nan),
            "did_beta": did_res["beta"],
            "did_se": did_res["se"],
            "did_p": did_res["p_value"],
            "difference": diff,
        })

        print(f"{y:20s} | {dml_res['theta']:+12.6f} | {dml_res['se']:10.6f} | "
              f"{did_res['beta']:+12.6f} | {did_res['se']:10.6f} | {diff:+10.6f}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    from data_simulation import generate_panel_data
    from csee_computation import compute_csee_panel

    panel, ndvi_ts, events = generate_panel_data()
    panel = compute_csee_panel(panel, ndvi_ts, events)

    # TWFE DID
    print("\nTWFE-DID result (CSEE):")
    res = twfe_did(panel, "csee")
    print(f"  β={res['beta']:+.6f}, SE={res['se']:.6f}, t={res['t_stat']:.3f}, p={res['p_value']:.4f}")

    # Event study
    print("\nEvent Study:")
    es = event_study(panel)
    print(es.to_string(index=False))

    # Comparison
    comp = compare_dml_vs_did(panel)
