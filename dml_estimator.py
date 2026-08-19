"""
Double/Debiased Machine Learning (DML) estimator module.

Implements the partially linear model (PLR) and interactive model (IRM)
with cross-fitting, following Chernozhukov et al. (2018, Econometrica).

Key features:
  - K-fold cross-fitting with multiple ML learners
  - Partially Linear Regression (PLR) model
  - Interactive Regression Model (IRM) for heterogeneous treatment effects
  - Bootstrap inference
  - Multiple ML algorithm comparison (RF, XGBoost, NN, LASSO)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import (
    N_FOLDS, N_BOOTSTRAP, RANDOM_SEED,
    RF_PARAMS, XGB_PARAMS, NN_PARAMS, LASSO_PARAMS,
    ML_LEARNERS, CONTROL_VARS, TRUE_BUFFER_EFFECT,
)


def get_ml_learner(learner_name):
    """Return an ML learner instance by name."""
    if learner_name == "random_forest":
        return RandomForestRegressor(**RF_PARAMS)
    elif learner_name == "xgboost":
        return xgb.XGBRegressor(**XGB_PARAMS)
    elif learner_name == "neural_network":
        return MLPRegressor(**NN_PARAMS)
    elif learner_name == "lasso":
        return LassoCV(cv=5, max_iter=10000, random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown learner: {learner_name}")


def _demean_panel(df, y_col, d_col, x_cols):
    """Within-transform: demean Y, D, X by city and year fixed effects.

    This implements the two-way fixed effect transformation before DML,
    so that the DML residual regression captures within-city-year variation.
    """
    df = df.copy()
    cols_to_transform = [y_col, d_col] + x_cols

    # City fixed effects (demean by city)
    for col in cols_to_transform:
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        city_means = df.groupby("city_id")[col].transform("mean")
        if isinstance(city_means, pd.DataFrame):
            city_means = city_means.iloc[:, 0]
        df[col + "_dm"] = series - city_means

    # Year fixed effects (demean by year on already city-demeaned)
    for col in cols_to_transform:
        dm_col = col + "_dm"
        series = df[dm_col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        year_means = df.groupby("year")[dm_col].transform("mean")
        if isinstance(year_means, pd.DataFrame):
            year_means = year_means.iloc[:, 0]
        df[dm_col] = series - year_means

    return df


def dml_plr(panel, y_col, d_col, x_cols=None, learner="random_forest",
            n_folds=N_FOLDS, bootstrap=False, n_bootstrap=N_BOOTSTRAP):
    """Double Machine Learning - Partially Linear Regression.

    Model:
        Y = θ·D + g(X) + ε
        D = m(X) + v

    Args:
        panel:      DataFrame with all variables
        y_col:      name of outcome variable
        d_col:      name of treatment variable
        x_cols:     list of control variable names
        learner:    ML learner name
        n_folds:    number of cross-fitting folds
        bootstrap:  whether to compute bootstrap SE
        n_bootstrap: number of bootstrap iterations

    Returns:
        dict with theta, se, ci_lower, ci_upper, n, learner
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    # Drop missing values
    needed = [y_col, d_col] + x_cols + ["city_id", "year"]
    df = panel[needed].dropna().reset_index(drop=True)

    Y = df[y_col].values
    D = df[d_col].values
    X = df[x_cols].values
    n = len(df)

    if n < 50:
        return {"theta": np.nan, "se": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": n, "learner": learner}

    # Within-transform (two-way FE demeaning)
    df_dm = _demean_panel(df, y_col, d_col, x_cols)
    Y_dm = df_dm[y_col + "_dm"].values
    D_dm = df_dm[d_col + "_dm"].values
    X_dm = df_dm[[c + "_dm" for c in x_cols]].values

    # Standardize X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dm)

    # Cross-fitting
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    Y_resid = np.zeros(n)
    D_resid = np.zeros(n)

    for train_idx, test_idx in kf.split(X_scaled):
        # nuisance function g(X): predict Y from X
        ml_g = get_ml_learner(learner)
        ml_g.fit(X_scaled[train_idx], Y_dm[train_idx])
        Y_resid[test_idx] = Y_dm[test_idx] - ml_g.predict(X_scaled[test_idx])

        # nuisance function m(X): predict D from X
        ml_m = get_ml_learner(learner)
        ml_m.fit(X_scaled[train_idx], D_dm[train_idx])
        D_resid[test_idx] = D_dm[test_idx] - ml_m.predict(X_scaled[test_idx])

    # Final OLS on residuals: θ = (D~'D~)^(-1) D~'Y~
    DTD = np.dot(D_resid, D_resid)
    if abs(DTD) < 1e-10:
        theta = 0.0
    else:
        theta = np.dot(D_resid, Y_resid) / DTD

    # Standard error
    residuals = Y_resid - theta * D_resid
    sigma2 = np.dot(residuals, residuals) / (n - 1)
    se = np.sqrt(sigma2 / DTD) if DTD > 1e-10 else np.nan

    # Bootstrap inference
    if bootstrap:
        boot_thetas = []
        for b in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            Db = D_resid[idx]
            Yb = Y_resid[idx]
            DTDb = np.dot(Db, Db)
            if DTDb > 1e-10:
                boot_thetas.append(np.dot(Db, Yb) / DTDb)
        if len(boot_thetas) > 10:
            boot_thetas = np.array(boot_thetas)
            se = np.std(boot_thetas, ddof=1)

    ci_lower = theta - 1.96 * se
    ci_upper = theta + 1.96 * se

    return {
        "theta": float(theta),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n": n,
        "learner": learner,
        "t_stat": float(theta / se) if se and se > 0 else np.nan,
        "p_value": float(2 * (1 - __import__("scipy").stats.norm.cdf(abs(theta / se)))) if se and se > 0 else np.nan,
    }


def dml_irm_cate(panel, y_col, d_col, x_cols=None, learner="random_forest",
                 n_folds=N_FOLDS, hetero_var=None):
    """DML Interactive Regression Model - CATE estimation.

    Estimates Conditional Average Treatment Effect (CATE) for heterogeneity analysis.

    Model:
        Y = g₀(X) + θ(X)·D + ε

    Args:
        hetero_var: column name to group by for CATE estimation

    Returns:
        dict with CATEs by group
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    if hetero_var is None:
        return dml_plr(panel, y_col, d_col, x_cols, learner, n_folds)

    # Group-wise CATE
    results = {}
    groups = panel[hetero_var].dropna().unique()

    for g in sorted(groups):
        sub = panel[panel[hetero_var] == g].copy()
        if len(sub) > 50:
            res = dml_plr(sub, y_col, d_col, x_cols, learner, n_folds)
            results[str(g)] = res

    return {"hetero_var": hetero_var, "cate": results, "learner": learner}


def run_dml_comparison(panel, y_col, d_col, x_cols=None):
    """Run DML with all ML learners for algorithm comparison.

    Returns DataFrame with results from each learner.
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n" + "=" * 70)
    print(f"DML Estimation Comparison (Y={y_col}, D={d_col})")
    print("=" * 70)

    results = []
    for learner in ML_LEARNERS:
        print(f"  Running DML with {learner}...")
        res = dml_plr(panel, y_col, d_col, x_cols, learner=learner, bootstrap=True)
        res["y"] = y_col
        res["d"] = d_col
        results.append(res)
        print(f"    θ={res['theta']:.6f}, SE={res['se']:.6f}, "
              f"t={res.get('t_stat', 0):.3f}, n={res['n']}")

    df_results = pd.DataFrame(results)
    return df_results


def run_full_dml_analysis(panel, y_cols=None, d_col="did_shock",
                          x_cols=None, best_learner="random_forest"):
    """Run the complete DML analysis pipeline.

    Args:
        y_cols:       list of outcome variables (csee, cr, rc, rsei, psr_resilience)
        d_col:        treatment variable
        x_cols:       control variables
        best_learner: ML learner to use for main results

    Returns:
        dict of results for each outcome
    """
    if y_cols is None:
        y_cols = ["csee", "cr", "rc", "rsei", "psr_resilience"]
    if x_cols is None:
        x_cols = CONTROL_VARS

    all_results = {}

    # 1. Main DML results with best learner
    print("\n" + "=" * 70)
    print(f"Main DML Results (learner={best_learner})")
    print("=" * 70)

    for y in y_cols:
        res = dml_plr(panel, y, d_col, x_cols, learner=best_learner, bootstrap=True)
        all_results[f"main_{y}"] = res
        sig = "***" if res.get("p_value", 1) < 0.01 else \
              "**" if res.get("p_value", 1) < 0.05 else \
              "*" if res.get("p_value", 1) < 0.1 else ""
        print(f"  {y:20s}: θ={res['theta']:+.6f} (SE={res['se']:.6f}) {sig}")

    # 2. ML algorithm comparison for main outcome
    comparison = run_dml_comparison(panel, "csee", d_col, x_cols)
    all_results["ml_comparison"] = comparison

    # 3. DML with alternative treatment specifications
    print("\n  Alternative treatment specifications:")
    alt_specs = [
        ("did", "DID (policy only, no shock interaction)"),
        ("shock_intensity", "Shock intensity only"),
        ("did_shock", "DID × Shock (main specification)"),
    ]
    for d, label in alt_specs:
        if d in panel.columns:
            res = dml_plr(panel, "csee", d, x_cols, learner=best_learner)
            all_results[f"alt_{d}"] = res
            print(f"    {label:45s}: θ={res['theta']:+.6f} (SE={res['se']:.6f})")

    # 4. Fold sensitivity
    print("\n  Cross-fitting fold sensitivity:")
    fold_results = []
    for k in [3, 5, 7, 10]:
        res = dml_plr(panel, "csee", d_col, x_cols, learner=best_learner, n_folds=k)
        fold_results.append({"n_folds": k, "theta": res["theta"], "se": res["se"]})
        print(f"    K={k}: θ={res['theta']:+.6f} (SE={res['se']:.6f})")
    all_results["fold_sensitivity"] = pd.DataFrame(fold_results)

    return all_results


if __name__ == "__main__":
    from data_simulation import generate_panel_data
    from csee_computation import compute_csee_panel

    panel, ndvi_ts, events = generate_panel_data()
    panel = compute_csee_panel(panel, ndvi_ts, events)

    results = run_full_dml_analysis(panel)
    print("\n\nML Algorithm Comparison:")
    print(results["ml_comparison"][["learner", "theta", "se", "t_stat"]].to_string(index=False))
