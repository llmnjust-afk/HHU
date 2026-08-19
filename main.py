"""
Main orchestration script for the CSEE-DML research pipeline.

Runs the complete analysis:
  Phase 1: Data generation / loading
  Phase 2: CSEE computation (CR, RC, CSEE, RSEI, PSR resilience)
  Phase 3: DML estimation (main results + ML comparison + fold sensitivity)
  Phase 4: Traditional TWFE-DID comparison + event study
  Phase 5: Mechanism analysis (CR/RC decomposition + mediation)
  Phase 6: Heterogeneity analysis (7 dimensions)
  Phase 7: Robustness checks (placebo, PSM-DML, sample restrictions, lags)
  Phase 8: Spatial spillover analysis (optional)
  Phase 9: Dose-response analysis (optional)
  Phase 10: Visualization (all figures and tables)

Usage:
    python main.py                    # Run full pipeline (synthetic data)
    python main.py --real-data        # Use real data (GEE NDVI, ERA5, yearbook)
    python main.py --quick            # Quick run (reduced iterations)
    python main.py --no-spatial       # Skip spatial analysis
    python main.py --no-dose           # Skip dose-response
"""
import argparse
import sys
import time
import numpy as np
import pandas as pd

from config import ensure_dirs, RANDOM_SEED, ML_LEARNERS

ensure_dirs()
np.random.seed(RANDOM_SEED)


def main():
    parser = argparse.ArgumentParser(description="CSEE-DML Research Pipeline")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real data instead of synthetic simulation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with reduced iterations")
    parser.add_argument("--no-spatial", action="store_true",
                        help="Skip spatial spillover analysis")
    parser.add_argument("--no-dose", action="store_true",
                        help="Skip dose-response analysis")
    args = parser.parse_args()

    t0 = time.time()

    print("\n" + "#" * 70)
    print("#  CSEE-DML Research Pipeline")
    print("#  Climate-Stress Ecological Elasticity × Double Machine Learning")
    print("#" * 70)
    print(f"#  Random seed: {RANDOM_SEED}")
    print(f"#  Quick mode: {args.quick}")
    print(f"#  Data source: {'REAL' if args.real_data else 'SYNTHETIC'}")
    print("#" * 70)

    all_results = {}

    # ── Phase 1: Data Generation / Loading ────────────────────────────────
    print("\n\n" + "=" * 70)
    print("PHASE 1: Data Generation" + (" (REAL DATA)" if args.real_data else ""))
    print("=" * 70)

    if args.real_data:
        from real_data.data_loader import load_real_panel
        panel, ndvi_ts, weather_events = load_real_panel()
    else:
        from data_simulation import generate_panel_data
        panel, ndvi_ts, weather_events = generate_panel_data()
    all_results["panel"] = panel

    # ── Phase 2: CSEE Computation ─────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("PHASE 2: CSEE Computation")
    print("=" * 70)

    from csee_computation import compute_csee_panel
    panel = compute_csee_panel(panel, ndvi_ts, weather_events)
    all_results["panel"] = panel

    # ── Phase 3: DML Estimation ───────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("PHASE 3: DML Estimation")
    print("=" * 70)

    from dml_estimator import run_full_dml_analysis
    dml_results = run_full_dml_analysis(panel, best_learner="random_forest")
    all_results["dml"] = dml_results

    # ── Phase 4: Traditional DID Comparison ───────────────────────────────
    print("\n\n" + "=" * 70)
    print("PHASE 4: Traditional TWFE-DID + Event Study")
    print("=" * 70)

    from traditional_did import compare_dml_vs_did, event_study
    all_results["dml_vs_did"] = compare_dml_vs_did(panel)
    all_results["event_study"] = event_study(panel, y_col="csee")
    all_results["event_study_rsei"] = event_study(panel, y_col="rsei")

    # ── Phase 5: Mechanism Analysis ───────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("PHASE 5: Mechanism Analysis")
    print("=" * 70)

    from mechanism_analysis import run_all_mechanisms
    all_results["mechanism"] = run_all_mechanisms(panel)

    # ── Phase 6: Heterogeneity Analysis ───────────────────────────────────
    print("\n\n" + "=" * 70)
    print("PHASE 6: Heterogeneity Analysis")
    print("=" * 70)

    from heterogeneity import run_heterogeneity
    all_results["heterogeneity"] = run_heterogeneity(panel)

    # ── Phase 7: Robustness Checks ────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("PHASE 7: Robustness Checks")
    print("=" * 70)

    from robustness import run_all_robustness
    placebo_iter = 50 if args.quick else 200
    all_results["robustness"] = run_all_robustness(panel, placebo_iter=placebo_iter)

    # ── Phase 8: Spatial Spillover ────────────────────────────────────────
    if not args.no_spatial:
        print("\n\n" + "=" * 70)
        print("PHASE 8: Spatial Spillover Analysis")
        print("=" * 70)

        from spatial_spillover import run_spatial_analysis
        all_results["spatial"] = run_spatial_analysis(panel)

    # ── Phase 9: Dose-Response ────────────────────────────────────────────
    if not args.no_dose:
        print("\n\n" + "=" * 70)
        print("PHASE 9: Dose-Response Analysis")
        print("=" * 70)

        from dose_response import run_dose_response
        all_results["dose_response"] = run_dose_response(panel)

    # ── Phase 10: Visualization ──────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("PHASE 10: Visualization & Tables")
    print("=" * 70)

    from visualization import generate_all_figures_and_tables
    all_results["outputs"] = generate_all_figures_and_tables(panel, all_results)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n\n" + "#" * 70)
    print(f"#  PIPELINE COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("#" * 70)

    # Print key results summary
    print("\n" + "=" * 70)
    print("KEY RESULTS SUMMARY")
    print("=" * 70)

    # Main DML result
    if "dml" in all_results and "main_csee" in all_results["dml"]:
        r = all_results["dml"]["main_csee"]
        sig = "***" if r.get("p_value", 1) < 0.01 else \
              "**" if r.get("p_value", 1) < 0.05 else \
              "*" if r.get("p_value", 1) < 0.1 else ""
        print(f"\n  Main DML (CSEE ~ DID×Shock):")
        print(f"    θ = {r['theta']:+.6f} (SE = {r['se']:.6f}) {sig}")
        print(f"    95% CI: [{r['ci_lower']:+.6f}, {r['ci_upper']:+.6f}]")

    # Mechanism
    if "mechanism" in all_results:
        decomp = all_results["mechanism"].get("decomposition", {})
        cr = decomp.get("cr", {})
        rc = decomp.get("rc", {})
        print(f"\n  Mechanism Decomposition:")
        print(f"    CR (resistance):  θ = {cr.get('theta', 0):+.6f}")
        print(f"    RC (recovery):    θ = {rc.get('theta', 0):+.6f}")

    # Placebo
    if "robustness" in all_results and "placebo" in all_results["robustness"]:
        p = all_results["robustness"]["placebo"]
        print(f"\n  Placebo Test:")
        print(f"    Placebo p-value = {p['placebo_p_value']:.4f}")

    print("\n" + "=" * 70)
    print(f"Results saved to: results/")
    print(f"  Figures: results/figures/")
    print(f"  Tables:  results/tables/")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = main()
