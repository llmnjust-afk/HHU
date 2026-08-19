"""Fast placebo test: 200 iterations with lighter RF."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
np.random.seed(42)
from config import CONTROL_VARS
from real_data.data_loader import load_real_panel
from csee_computation import compute_csee_panel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

panel, ndvi_ts, events = load_real_panel()
panel = compute_csee_panel(panel, ndvi_ts, events)
print(f"Panel: {len(panel)} obs, {panel['city_id'].nunique()} cities")

# Use lighter RF for speed
def dml_plr_fast(df, y_col, d_col, x_cols, n_folds=5):
    needed = [y_col, d_col] + x_cols + ["city_id", "year"]
    d = df[needed].dropna().reset_index(drop=True)
    Y = d[y_col].values; D = d[d_col].values
    X = d[x_cols].values; n = len(d)
    # Two-way FE demean
    for col in [y_col, d_col] + x_cols:
        d[col+"_dm"] = d[col] - d.groupby("city_id")[col].transform("mean")
    for col in [y_col, d_col] + x_cols:
        d[col+"_dm"] = d[col+"_dm"] - d.groupby("year")[col+"_dm"].transform("mean")
    Y_dm = d[y_col+"_dm"].values; D_dm = d[d_col+"_dm"].values
    X_dm = d[[c+"_dm" for c in x_cols]].values
    X_s = StandardScaler().fit_transform(X_dm)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    Y_r = np.zeros(n); D_r = np.zeros(n)
    rf_params = {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 20,
                 "max_features": "sqrt", "random_state": 42, "n_jobs": -1}
    for tr, te in kf.split(X_s):
        m = RandomForestRegressor(**rf_params); m.fit(X_s[tr], Y_dm[tr])
        Y_r[te] = Y_dm[te] - m.predict(X_s[te])
        m2 = RandomForestRegressor(**rf_params); m2.fit(X_s[tr], D_dm[tr])
        D_r[te] = D_dm[te] - m2.predict(X_s[te])
    DTD = np.dot(D_r, D_r)
    if abs(DTD) < 1e-10: return 0.0
    theta = np.dot(D_r, Y_r) / DTD
    return theta

true_theta = dml_plr_fast(panel, "csee", "did_shock", CONTROL_VARS)
print(f"True θ = {true_theta:.6f}")

placebo_thetas = []
for i in range(200):
    p = panel.copy()
    # Permute treatment within year
    p["did_shock"] = p.groupby("year")["did_shock"].transform(
        lambda x: x.sample(frac=1, random_state=i).values)
    th = dml_plr_fast(p, "csee", "did_shock", CONTROL_VARS)
    placebo_thetas.append(th)
    if (i+1) % 20 == 0:
        print(f"  Completed {i+1}/200 iterations")

placebo_thetas = np.array(placebo_thetas)
p_val = np.mean(np.abs(placebo_thetas) >= abs(true_theta))
print(f"\n  Placebo mean: {placebo_thetas.mean():.6f}")
print(f"  Placebo std:  {placebo_thetas.std():.6f}")
print(f"  Placebo p-value: {p_val:.4f}")
print(f"  → {'PASS' if p_val < 0.10 else 'WARN: True θ within placebo distribution'}")

pd.DataFrame({"placebo_theta": placebo_thetas}).to_csv(
    "results/tables/placebo_200iter.csv", index=False)
print("Saved: results/tables/placebo_200iter.csv")
