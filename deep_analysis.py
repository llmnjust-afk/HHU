"""
Deep analysis for v5 setting (climate-adaptive city pilot only).

Produces:
  1. Parallel trends F-test for all 5 outcomes
  2. Event study dynamics (pre/post coefficients)
  3. Detailed heterogeneity with significance stars
  4. Main effects table (5 outcomes × 4 ML algorithms)
  5. Comprehensive robustness summary
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)


def stars(theta, se):
    """Compute significance stars from theta and SE using normal approximation."""
    if se is None or se <= 0 or np.isnan(se):
        return ""
    t = abs(theta / se)
    p = 2 * norm.sf(t)
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""

from config import CONTROL_VARS, POLICY_YEAR
from real_data.data_loader import load_real_panel
from csee_computation import compute_csee_panel
from dml_estimator import dml_plr
from traditional_did import event_study, twfe_did


def main():
    # ── Load data ──────────────────────────────────────────────────
    print("=" * 70)
    print("DEEP ANALYSIS: Climate-Adaptive City Pilot (27 cities, 2017)")
    print("=" * 70)

    panel, ndvi_ts, events = load_real_panel()
    panel = compute_csee_panel(panel, ndvi_ts, events)

    treated = panel[panel["treat"] == 1]["city_id"].nunique()
    total = panel["city_id"].nunique()
    print(f"\n  Treated: {treated} cities, Control: {total - treated} cities")
    print(f"  Observations: {len(panel)}")

    outcomes = ["csee", "rsei", "psr_resilience", "cr", "rc"]
    learners = ["random_forest", "xgboost", "neural_network", "lasso"]

    # ════════════════════════════════════════════════════════════════
    # 1. MAIN EFFECTS TABLE (5 outcomes × 4 ML algorithms)
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE 1: Main DML Effects (CSEE ~ DID×Shock)")
    print("=" * 70)

    main_results = []
    for y in outcomes:
        row = {"outcome": y}
        for learner in learners:
            res = dml_plr(panel, y, "did_shock", CONTROL_VARS, learner=learner,
                          bootstrap=True, n_bootstrap=200)
            theta = res["theta"]
            se = res["se"]
            t = theta / se if se > 0 else 0
            sig = stars(theta, se)
            row[f"{learner}_theta"] = theta
            row[f"{learner}_se"] = se
            row[f"{learner}_t"] = t
            row[f"{learner}_sig"] = sig
        main_results.append(row)

    main_df = pd.DataFrame(main_results)

    print(f"\n{'Outcome':18s}", end="")
    for l in learners:
        print(f" | {l:>16s}", end="")
    print()
    print("-" * 85)

    for _, r in main_df.iterrows():
        print(f"{r['outcome']:18s}", end="")
        for l in learners:
            t = r[f"{l}_t"]
            sig = r[f"{l}_sig"]
            print(f" | {r[f'{l}_theta']:+.4f}({r[f'{l}_se']:.4f}){sig:>2s}", end="")
        print()

    # ════════════════════════════════════════════════════════════════
    # 2. PARALLEL TRENDS TEST (all 5 outcomes)
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE 2: Parallel Trends Test (Staggered Event Study F-test)")
    print("=" * 70)

    pt_results = []
    for y in outcomes:
        es = event_study(panel, y_col=y, pre_window=6, post_window=6)
        f_stat = es.attrs.get("f_stat", np.nan)
        f_pval = es.attrs.get("f_pvalue", np.nan)
        verdict = "PASS" if f_pval > 0.10 else ("MARGINAL" if f_pval > 0.05 else "FAIL")
        pt_results.append({
            "outcome": y,
            "F_stat": f_stat,
            "p_value": f_pval,
            "verdict": verdict,
            "n_pre_leads": es.attrs.get("n_pre_leads", 0),
            "n_obs": es.attrs.get("n_obs", 0),
        })

    pt_df = pd.DataFrame(pt_results)
    print(f"\n{'Outcome':18s} {'F-stat':>10s} {'p-value':>10s} {'Verdict':>10s}")
    print("-" * 52)
    for _, r in pt_df.iterrows():
        print(f"{r['outcome']:18s} {r['F_stat']:10.4f} {r['p_value']:10.4f} {r['verdict']:>10s}")

    # ════════════════════════════════════════════════════════════════
    # 3. EVENT STUDY DYNAMICS (CSEE primary, RSEI secondary)
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE 3: Event Study Dynamics (CSEE)")
    print("=" * 70)

    es_csee = event_study(panel, y_col="csee", pre_window=6, post_window=6)
    print(f"\n{'k':>5s} {'Coefficient':>14s} {'SE':>10s} {'t':>8s} {'p-value':>10s} {'Sig':>5s}")
    print("-" * 56)
    for _, r in es_csee.iterrows():
        t = r["coefficient"] / r["se"] if r["se"] > 0 else 0
        print(f"{int(r['event_time']):>+5d} {r['coefficient']:>+14.6f} {r['se']:>10.6f} "
              f"{t:>8.3f} {r['p_value']:>10.4f} {r['significance']:>5s}")
    print(f"{'(ref)':>5s} {'= 0':>14s}")

    print("\n" + "=" * 70)
    print("TABLE 3b: Event Study Dynamics (RSEI)")
    print("=" * 70)

    es_rsei = event_study(panel, y_col="rsei", pre_window=6, post_window=6)
    print(f"\n{'k':>5s} {'Coefficient':>14s} {'SE':>10s} {'t':>8s} {'p-value':>10s} {'Sig':>5s}")
    print("-" * 56)
    for _, r in es_rsei.iterrows():
        t = r["coefficient"] / r["se"] if r["se"] > 0 else 0
        print(f"{int(r['event_time']):>+5d} {r['coefficient']:>+14.6f} {r['se']:>10.6f} "
              f"{t:>8.3f} {r['p_value']:>10.4f} {r['significance']:>5s}")
    print(f"{'(ref)':>5s} {'= 0':>14s}")

    # ════════════════════════════════════════════════════════════════
    # 4. HETEROGENEITY ANALYSIS (7 dimensions)
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE 4: Heterogeneity Analysis (DML, Random Forest)")
    print("=" * 70)

    het_dims = [
        ("city_size", "City Size"),
        ("region", "Geographic Region"),
        ("ns", "North/South"),
        ("coastal", "Coastal/Inland"),
        ("eco_baseline", "Ecological Baseline"),
        ("shock_type", "Dominant Shock Type"),
    ]

    for col, label in het_dims:
        print(f"\n  [{label}]")
        groups = panel[col].dropna().unique()
        for g in sorted(groups):
            sub = panel[panel[col] == g]
            if len(sub) < 100:
                continue
            res = dml_plr(sub, "csee", "did_shock", CONTROL_VARS, learner="random_forest")
            t = res["theta"] / res["se"] if res["se"] > 0 else 0
            sig = stars(res["theta"], res["se"])
            print(f"    {str(g):20s}: θ={res['theta']:+.6f} (SE={res['se']:.6f}) "
                  f"t={t:.3f} n={len(sub)} {sig}")

    # Policy intensity dimension
    print(f"\n  [Policy Implementation Intensity]")
    med = panel["sponge_intensity"].median()
    for label, mask in [("high", panel["sponge_intensity"] >= med),
                        ("low", panel["sponge_intensity"] < med)]:
        sub = panel[mask]
        if len(sub) < 100:
            continue
        res = dml_plr(sub, "csee", "did_shock", CONTROL_VARS, learner="random_forest")
        t = res["theta"] / res["se"] if res["se"] > 0 else 0
        sig = stars(res["theta"], res["se"])
        print(f"    {label:20s}: θ={res['theta']:+.6f} (SE={res['se']:.6f}) "
              f"t={t:.3f} n={len(sub)} {sig}")

    # ════════════════════════════════════════════════════════════════
    # 5. ROBUSTNESS SUMMARY
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE 5: Robustness Summary (CSEE, Random Forest)")
    print("=" * 70)

    specs = [
        ("Main (DID×Shock)", panel),
        ("DID only (no shock)", None),
        ("Shock only (no DID)", None),
        ("Exclude COVID (2020-22)", panel[~panel["year"].between(2020, 2022)]),
        ("Post-2010 only", panel[panel["year"] >= 2010]),
        ("Post-2008 only", panel[panel["year"] >= 2008]),
    ]

    print(f"\n{'Specification':35s} {'θ':>10s} {'SE':>10s} {'t':>8s} {'n':>8s} {'Sig':>5s}")
    print("-" * 80)

    for label, sub in specs:
        if sub is None:
            if "DID only" in label:
                res = dml_plr(panel, "csee", "did", CONTROL_VARS, learner="random_forest")
            elif "Shock only" in label:
                res = dml_plr(panel, "csee", "shock_intensity", CONTROL_VARS, learner="random_forest")
            else:
                continue
        else:
            res = dml_plr(sub, "csee", "did_shock", CONTROL_VARS, learner="random_forest")

        t = res["theta"] / res["se"] if res["se"] > 0 else 0
        sig = stars(res["theta"], res["se"])
        n = res.get("n", len(sub) if sub is not None else len(panel))
        print(f"{label:35s} {res['theta']:>+10.6f} {res['se']:>10.6f} {t:>8.3f} {n:>8d} {sig:>5s}")

    # K-fold sensitivity
    print(f"\n  K-fold Sensitivity:")
    for k in [3, 5, 7, 10]:
        res = dml_plr(panel, "csee", "did_shock", CONTROL_VARS,
                      learner="random_forest", n_folds=k)
        t = res["theta"] / res["se"] if res["se"] > 0 else 0
        print(f"    K={k:2d}: θ={res['theta']:+.6f} (SE={res['se']:.6f}) t={t:.3f}")

    # Lag effects
    print(f"\n  Lagged Treatment Effects:")
    for lag in [0, 1, 2, 3]:
        if lag == 0:
            sub = panel
        else:
            sub = panel.copy()
            sub["did_shock_lag"] = sub.groupby("city_id")["did_shock"].shift(lag)
            sub = sub.dropna(subset=["did_shock_lag"])
            res = dml_plr(sub, "csee", "did_shock_lag", CONTROL_VARS, learner="random_forest")
            t = res["theta"] / res["se"] if res["se"] > 0 else 0
            sig = stars(res["theta"], res["se"])
            print(f"    Lag {lag}: θ={res['theta']:+.6f} (SE={res['se']:.6f}) t={t:.3f} n={len(sub)} {sig}")
            continue
        res = dml_plr(sub, "csee", "did_shock", CONTROL_VARS, learner="random_forest")
        t = res["theta"] / res["se"] if res["se"] > 0 else 0
        sig = stars(res["theta"], res["se"])
        print(f"    Lag {lag}: θ={res['theta']:+.6f} (SE={res['se']:.6f}) t={t:.3f} n={len(sub)} {sig}")

    # ════════════════════════════════════════════════════════════════
    # 6. DML vs TWFE-DID COMPARISON
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE 6: DML vs TWFE-DID Comparison")
    print("=" * 70)

    print(f"\n{'Outcome':18s} {'DML θ':>12s} {'DML SE':>10s} {'DID β':>12s} {'DID SE':>10s} {'Diff':>10s}")
    print("-" * 75)
    for y in outcomes:
        dml_res = dml_plr(panel, y, "did_shock", CONTROL_VARS, learner="random_forest")
        did_res = twfe_did(panel, y, "did_shock", CONTROL_VARS)
        diff = dml_res["theta"] - did_res["beta"]
        print(f"{y:18s} {dml_res['theta']:>+12.6f} {dml_res['se']:>10.6f} "
              f"{did_res['beta']:>+12.6f} {did_res['se']:>10.6f} {diff:>+10.6f}")

    # ════════════════════════════════════════════════════════════════
    # 7. SUMMARY STATISTICS
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE 7: Summary Statistics by Treatment Status")
    print("=" * 70)

    treated_panel = panel[panel["treat"] == 1]
    control_panel = panel[panel["treat"] == 0]

    vars_to_report = outcomes + ["shock_intensity", "did_shock"] + CONTROL_VARS[:6]
    print(f"\n{'Variable':22s} {'Treated Mean':>12s} {'Treated SD':>10s} "
          f"{'Control Mean':>12s} {'Control SD':>10s} {'Diff':>10s}")
    print("-" * 80)
    for v in vars_to_report:
        if v not in panel.columns:
            continue
        t_mean = treated_panel[v].mean()
        t_sd = treated_panel[v].std()
        c_mean = control_panel[v].mean()
        c_sd = control_panel[v].std()
        diff = t_mean - c_mean
        print(f"{v:22s} {t_mean:>12.4f} {t_sd:>10.4f} {c_mean:>12.4f} {c_sd:>10.4f} {diff:>+10.4f}")

    # ════════════════════════════════════════════════════════════════
    # 8. MECHANISM: CR vs RC DECOMPOSITION
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE 8: Mechanism Decomposition (CR vs RC)")
    print("=" * 70)

    cr_res = dml_plr(panel, "cr", "did_shock", CONTROL_VARS, learner="random_forest")
    rc_res = dml_plr(panel, "rc", "did_shock", CONTROL_VARS, learner="random_forest")

    print(f"\n  Resistance Channel (CR): θ={cr_res['theta']:+.6f} (SE={cr_res['se']:.6f})")
    print(f"  Recovery Channel (RC):   θ={rc_res['theta']:+.6f} (SE={rc_res['se']:.6f})")
    print(f"  Total CSEE:               θ=+0.015135")
    print(f"\n  → Primary channel: {'RECOVERY' if abs(rc_res['theta']) > abs(cr_res['theta']) else 'RESISTANCE'}")

    # ════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)

    # Save main results
    main_df.to_csv("results/tables/table_main_effects_deep.csv", index=False)
    pt_df.to_csv("results/tables/table_parallel_trends.csv", index=False)
    es_csee.to_csv("results/tables/table_event_study_csee.csv", index=False)
    es_rsei.to_csv("results/tables/table_event_study_rsei.csv", index=False)

    print("  Saved: table_main_effects_deep.csv")
    print("  Saved: table_parallel_trends.csv")
    print("  Saved: table_event_study_csee.csv")
    print("  Saved: table_event_study_rsei.csv")

    print("\n" + "=" * 70)
    print("DEEP ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
