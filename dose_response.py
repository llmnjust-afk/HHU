"""
Dose-response analysis module.

Estimates the non-linear dose-response curve of policy implementation
intensity on CSEE, replacing the binary treatment with continuous
policy investment intensity.

Uses DML with continuous treatment to estimate the Average Dose-Response
Function (ADRF).
"""
import numpy as np
import pandas as pd
from dml_estimator import dml_plr, get_ml_learner, _demean_panel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from config import CONTROL_VARS, N_FOLDS, RANDOM_SEED


def dose_response_dml(panel, y_col="csee", d_col="policy_intensity",
                      x_cols=None, n_bins=10, learner="random_forest"):
    """Estimate dose-response function using DML with continuous treatment.

    For each bin of treatment intensity, estimate the conditional average
    treatment effect using DML residualization.

    Args:
        n_bins: number of intensity bins for ADRF estimation

    Returns:
        dict with dose-response curve points
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n  [Dose-Response Analysis]")
    print("  " + "-" * 60)

    needed = [y_col, d_col] + x_cols + ["city_id", "year"]
    df = panel[needed].dropna().reset_index(drop=True)
    n = len(df)

    # Only use observations with non-zero policy intensity for clean estimation
    df_active = df[df[d_col] > 0].copy()

    if len(df_active) < 100:
        print("    Warning: Too few observations with non-zero policy intensity")
        return None

    # Demean by city and year FE
    df_dm = _demean_panel(df, y_col, d_col, x_cols)
    Y_dm = df_dm[y_col + "_dm"].values
    D_dm = df_dm[d_col + "_dm"].values
    X_dm = df_dm[[c + "_dm" for c in x_cols]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dm)

    # Cross-fitting to get residuals
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    Y_resid = np.zeros(n)
    D_resid = np.zeros(n)

    for train_idx, test_idx in kf.split(X_scaled):
        ml_g = get_ml_learner(learner)
        ml_g.fit(X_scaled[train_idx], Y_dm[train_idx])
        Y_resid[test_idx] = Y_dm[test_idx] - ml_g.predict(X_scaled[test_idx])

        ml_m = get_ml_learner(learner)
        ml_m.fit(X_scaled[train_idx], D_dm[train_idx])
        D_resid[test_idx] = D_dm[test_idx] - ml_m.predict(X_scaled[test_idx])

    # Estimate ADRF via binning on D_resid
    # Divide D_resid into bins and compute local average of Y_resid / D_resid
    D_orig = df[d_col].values

    # Create bins based on original treatment intensity (among active cities)
    active_mask = D_orig > 0
    D_active = D_orig[active_mask]
    Y_resid_active = Y_resid[active_mask]
    D_resid_active = D_resid[active_mask]

    bins = np.linspace(D_active.min(), D_active.max(), n_bins + 1)
    bin_centers = []
    bin_effects = []
    bin_ses = []
    bin_ns = []

    for i in range(n_bins):
        mask = (D_active >= bins[i]) & (D_active < bins[i + 1] + 1e-8)
        if mask.sum() < 10:
            continue

        Y_bin = Y_resid_active[mask]
        D_bin = D_resid_active[mask]

        DTD = np.dot(D_bin, D_bin)
        if DTD > 1e-10:
            theta = np.dot(D_bin, Y_bin) / DTD
            residuals = Y_bin - theta * D_bin
            sigma2 = np.dot(residuals, residuals) / (len(Y_bin) - 1)
            se = np.sqrt(sigma2 / DTD) if DTD > 1e-10 else np.nan
        else:
            theta = np.nan
            se = np.nan

        bin_centers.append(float(np.mean(D_active[mask])))
        bin_effects.append(float(theta))
        bin_ses.append(float(se))
        bin_ns.append(int(mask.sum()))

    # Also estimate overall linear effect
    DTD_all = np.dot(D_resid, D_resid)
    if DTD_all > 1e-10:
        linear_theta = np.dot(D_resid, Y_resid) / DTD_all
        residuals = Y_resid - linear_theta * D_resid
        sigma2 = np.dot(residuals, residuals) / (n - 1)
        linear_se = np.sqrt(sigma2 / DTD_all)
    else:
        linear_theta = np.nan
        linear_se = np.nan

    print(f"    Linear dose-response: θ = {linear_theta:+.6f} (SE = {linear_se:.6f})")
    print(f"    Non-linear ADRF ({len(bin_centers)} bins):")
    for c, e, s, nn in zip(bin_centers, bin_effects, bin_ses, bin_ns):
        print(f"      Intensity={c:.2f}: effect={e:+.6f} (SE={s:.6f}) n={nn}")

    return {
        "linear_theta": float(linear_theta),
        "linear_se": float(linear_se),
        "bin_centers": bin_centers,
        "bin_effects": bin_effects,
        "bin_ses": bin_ses,
        "bin_ns": bin_ns,
        "n": n,
    }


def run_dose_response(panel, x_cols=None):
    """Run dose-response analysis.

    Returns:
        dict with dose-response results
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n" + "=" * 70)
    print("DOSE-RESPONSE ANALYSIS")
    print("=" * 70)

    # Main dose-response
    results = dose_response_dml(panel, "csee", "policy_intensity", x_cols)

    # Also for CR and RC
    print("\n  Dose-response for CR (resistance):")
    cr_results = dose_response_dml(panel, "cr", "policy_intensity", x_cols)

    print("\n  Dose-response for RC (recovery):")
    rc_results = dose_response_dml(panel, "rc", "policy_intensity", x_cols)

    return {
        "csee": results,
        "cr": cr_results,
        "rc": rc_results,
    }


if __name__ == "__main__":
    from data_simulation import generate_panel_data
    from csee_computation import compute_csee_panel

    panel, ndvi_ts, events = generate_panel_data()
    panel = compute_csee_panel(panel, ndvi_ts, events)

    results = run_dose_response(panel)
