"""
LST Robustness Analysis — uses MODIS Land Surface Temperature as an alternative
outcome variable to test whether the climate-adaptive city pilot policy reduces
urban thermal stress (UHI effect).

LST outcomes constructed:
  - lst_day_mean: annual mean daytime LST (°C)
  - lst_night_mean: annual mean nighttime LST (°C)
  - lst_summer_mean: summer (Jun-Aug) mean LST (°C)
  - lst_dtr: diurnal temperature range (day - night)
  - lst_heat_days: number of 8-day periods with day LST > 35°C

If the policy is effective, we expect:
  - lst_day_mean ↓ (reduced daytime heat)
  - lst_night_mean ↓ (reduced nighttime heat)
  - lst_summer_mean ↓ (reduced summer heat)
  - lst_dtr ↓ (reduced diurnal range, more moderate climate)
  - lst_heat_days ↓ (fewer extreme heat periods)
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)


def stars(theta, se):
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


def load_lst_panel(lst_path="data/lst_panel.pkl"):
    """Load LST pickle and convert to panel DataFrame."""
    with open(lst_path, "rb") as f:
        lst_dict = pickle.load(f)

    rows = []
    for city_id, city_data in lst_dict.items():
        for year, year_data in city_data.items():
            day = year_data.get("day", np.full(46, np.nan))
            night = year_data.get("night", np.full(46, np.nan))
            mean = year_data.get("mean", (day + night) / 2.0)

            day = np.asarray(day, dtype=float)
            night = np.asarray(night, dtype=float)
            mean = np.asarray(mean, dtype=float)

            n_periods = min(len(day), 46)

            summer_idx = list(range(22, 34))
            summer_idx = [i for i in summer_idx if i < n_periods]

            day_mean = np.nanmean(day[:n_periods]) if n_periods > 0 else np.nan
            night_mean = np.nanmean(night[:n_periods]) if n_periods > 0 else np.nan
            overall_mean = np.nanmean(mean[:n_periods]) if n_periods > 0 else np.nan
            summer_mean = np.nanmean(day[summer_idx]) if summer_idx else np.nan
            dtr = day_mean - night_mean
            heat_days = np.nansum(day[:n_periods] > 35) if n_periods > 0 else np.nan

            rows.append({
                "city_id": int(city_id),
                "year": int(year),
                "lst_day_mean": day_mean,
                "lst_night_mean": night_mean,
                "lst_overall_mean": overall_mean,
                "lst_summer_mean": summer_mean,
                "lst_dtr": dtr,
                "lst_heat_days": heat_days,
            })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["lst_day_mean"])
    print(f"  LST panel: {len(df)} obs, {df['city_id'].nunique()} cities, "
          f"{df['year'].min()}-{df['year'].max()}")
    print(f"  Day LST: mean={df['lst_day_mean'].mean():.1f}°C, "
          f"std={df['lst_day_mean'].std():.1f}")
    print(f"  Night LST: mean={df['lst_night_mean'].mean():.1f}°C, "
          f"std={df['lst_night_mean'].std():.1f}")
    print(f"  Summer LST: mean={df['lst_summer_mean'].mean():.1f}°C, "
          f"std={df['lst_summer_mean'].std():.1f}")
    print(f"  DTR: mean={df['lst_dtr'].mean():.1f}°C, std={df['lst_dtr'].std():.1f}")
    print(f"  Heat days: mean={df['lst_heat_days'].mean():.1f}, "
          f"std={df['lst_heat_days'].std():.1f}")
    return df


def main():
    print("=" * 70)
    print("LST ROBUSTNESS ANALYSIS")
    print("Alternative outcome: MODIS Land Surface Temperature (UHI effect)")
    print("=" * 70)

    from config import CONTROL_VARS
    from real_data.data_loader import load_real_panel
    from csee_computation import compute_csee_panel
    from dml_estimator import dml_plr
    from traditional_did import event_study, twfe_did

    panel, ndvi_ts, events = load_real_panel()
    panel = compute_csee_panel(panel, ndvi_ts, events)

    lst_df = load_lst_panel()

    merged = panel.merge(lst_df, on=["city_id", "year"], how="inner")
    print(f"\n  Merged panel: {len(merged)} obs, {merged['city_id'].nunique()} cities")
    print(f"  Treated: {merged[merged['treat']==1]['city_id'].nunique()} cities")
    print(f"  Control: {merged[merged['treat']==0]['city_id'].nunique()} cities")

    lst_outcomes = [
        "lst_day_mean",
        "lst_night_mean",
        "lst_summer_mean",
        "lst_dtr",
        "lst_heat_days",
    ]

    learners = ["random_forest", "xgboost", "neural_network", "lasso"]

    # ════════════════════════════════════════════════════════════════
    # 1. MAIN DML EFFECTS (LST outcomes × 4 ML algorithms)
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE L1: LST DML Effects (LST ~ DID×Shock, Bootstrap SE)")
    print("=" * 70)

    print(f"\n{'Outcome':20s}", end="")
    for l in learners:
        print(f" | {l:>16s}", end="")
    print()
    print("-" * 85)

    all_results = []
    for y in lst_outcomes:
        print(f"{y:20s}", end="")
        row = {"outcome": y}
        for learner in learners:
            res = dml_plr(merged, y, "did_shock", CONTROL_VARS,
                          learner=learner, bootstrap=True, n_bootstrap=200)
            theta = res["theta"]
            se = res["se"]
            sig = stars(theta, se)
            print(f" | {theta:+.4f}({se:.4f}){sig:>2s}", end="")
            row[f"{learner}_theta"] = theta
            row[f"{learner}_se"] = se
            row[f"{learner}_sig"] = sig
        print()
        all_results.append(row)

    lst_main_df = pd.DataFrame(all_results)

    # ════════════════════════════════════════════════════════════════
    # 2. PARALLEL TRENDS TEST
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE L2: Parallel Trends Test (LST outcomes)")
    print("=" * 70)

    pt_results = []
    for y in lst_outcomes:
        es = event_study(merged, y_col=y, pre_window=6, post_window=6)
        f_stat = es.attrs.get("f_stat", np.nan)
        f_pval = es.attrs.get("f_pvalue", np.nan)
        verdict = "PASS" if f_pval > 0.10 else ("MARGINAL" if f_pval > 0.05 else "FAIL")
        pt_results.append({
            "outcome": y,
            "F_stat": f_stat,
            "p_value": f_pval,
            "verdict": verdict,
        })

    pt_df = pd.DataFrame(pt_results)
    print(f"\n{'Outcome':20s} {'F-stat':>10s} {'p-value':>10s} {'Verdict':>10s}")
    print("-" * 52)
    for _, r in pt_df.iterrows():
        print(f"{r['outcome']:20s} {r['F_stat']:10.4f} {r['p_value']:10.4f} {r['verdict']:>10s}")

    # ════════════════════════════════════════════════════════════════
    # 3. DML vs TWFE-DID COMPARISON
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE L3: DML vs TWFE-DID Comparison (LST outcomes)")
    print("=" * 70)

    print(f"\n{'Outcome':20s} {'DML θ':>12s} {'DML SE':>10s} {'DID β':>12s} {'DID SE':>10s}")
    print("-" * 68)
    for y in lst_outcomes:
        dml_res = dml_plr(merged, y, "did_shock", CONTROL_VARS, learner="random_forest")
        did_res = twfe_did(merged, y, "did_shock", CONTROL_VARS)
        print(f"{y:20s} {dml_res['theta']:>+12.6f} {dml_res['se']:>10.6f} "
              f"{did_res['beta']:>+12.6f} {did_res['se']:>10.6f}")

    # ════════════════════════════════════════════════════════════════
    # 4. ROBUSTNESS: SUBSAMPLE + ALTERNATIVE SPECS
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE L4: Robustness (LST Day Mean, Random Forest)")
    print("=" * 70)

    specs = [
        ("Main (DID×Shock)", merged),
        ("Exclude COVID (2020-22)", merged[~merged["year"].between(2020, 2022)]),
        ("Post-2010 only", merged[merged["year"] >= 2010]),
        ("Post-2008 only", merged[merged["year"] >= 2008]),
    ]

    print(f"\n{'Specification':35s} {'θ':>10s} {'SE':>10s} {'t':>8s} {'n':>8s} {'Sig':>5s}")
    print("-" * 80)
    for label, sub in specs:
        if len(sub) < 100:
            continue
        res = dml_plr(sub, "lst_day_mean", "did_shock", CONTROL_VARS,
                      learner="random_forest")
        t = res["theta"] / res["se"] if res["se"] > 0 else 0
        sig = stars(res["theta"], res["se"])
        print(f"{label:35s} {res['theta']:>+10.6f} {res['se']:>10.6f} "
              f"{t:>8.3f} {res['n']:>8d} {sig:>5s}")

    # K-fold
    print(f"\n  K-fold Sensitivity:")
    for k in [3, 5, 7, 10]:
        res = dml_plr(merged, "lst_day_mean", "did_shock", CONTROL_VARS,
                      learner="random_forest", n_folds=k)
        t = res["theta"] / res["se"] if res["se"] > 0 else 0
        print(f"    K={k:2d}: θ={res['theta']:+.6f} (SE={res['se']:.6f}) t={t:.3f}")

    # ════════════════════════════════════════════════════════════════
    # 5. HETEROGENEITY
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE L5: Heterogeneity (LST Day Mean, Random Forest)")
    print("=" * 70)

    het_dims = [
        ("city_size", "City Size"),
        ("region", "Geographic Region"),
        ("ns", "North/South"),
        ("coastal", "Coastal/Inland"),
        ("eco_baseline", "Ecological Baseline"),
    ]

    for col, label in het_dims:
        print(f"\n  [{label}]")
        groups = merged[col].dropna().unique()
        for g in sorted(groups):
            sub = merged[merged[col] == g]
            if len(sub) < 100:
                continue
            res = dml_plr(sub, "lst_day_mean", "did_shock", CONTROL_VARS,
                          learner="random_forest")
            t = res["theta"] / res["se"] if res["se"] > 0 else 0
            sig = stars(res["theta"], res["se"])
            print(f"    {str(g):20s}: θ={res['theta']:+.6f} (SE={res['se']:.6f}) "
                  f"t={t:.3f} n={len(sub)} {sig}")

    # ════════════════════════════════════════════════════════════════
    # 6. SUMMARY STATISTICS
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("TABLE L6: LST Summary Statistics by Treatment Status")
    print("=" * 70)

    treated = merged[merged["treat"] == 1]
    control = merged[merged["treat"] == 0]

    print(f"\n{'Variable':22s} {'Treated Mean':>12s} {'Treated SD':>10s} "
          f"{'Control Mean':>12s} {'Control SD':>10s} {'Diff':>10s}")
    print("-" * 80)
    for v in lst_outcomes:
        if v not in merged.columns:
            continue
        print(f"{v:22s} {treated[v].mean():>12.4f} {treated[v].std():>10.4f} "
              f"{control[v].mean():>12.4f} {control[v].std():>10.4f} "
              f"{treated[v].mean() - control[v].mean():>+10.4f}")

    # ════════════════════════════════════════════════════════════════
    # SAVE
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("Saving LST robustness results...")
    print("=" * 70)

    lst_main_df.to_csv("results/tables/lst_main_effects.csv", index=False)
    pt_df.to_csv("results/tables/lst_parallel_trends.csv", index=False)
    print("  Saved: lst_main_effects.csv")
    print("  Saved: lst_parallel_trends.csv")

    print("\n" + "=" * 70)
    print("LST ROBUSTNESS ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
