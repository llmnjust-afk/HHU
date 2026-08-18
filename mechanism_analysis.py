"""
Mechanism analysis module.

Tests four mechanism channels for "why climate-resilient city policy
buffers extreme weather ecological impacts":

  Path 1: Resistance enhancement (green infrastructure → higher CR)
  Path 2: Recovery acceleration (ecological restoration → faster RC)
  Path 3: Blue-green infrastructure (blue-green ratio → buffering)
  Path 4: Coupling coordination (system synergy → resilience)
"""
import numpy as np
import pandas as pd
from dml_estimator import dml_plr
from config import CONTROL_VARS, MEDIATION_VARS


def mechanism_decomposition(panel, x_cols=None):
    """Decompose CSEE into CR and RC, run DML separately.

    Tests whether the policy buffer operates through:
      - Resistance (CR): reducing damage during shock
      - Recovery (RC): accelerating post-shock recovery

    Returns:
        dict with DML results for CR and RC separately
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n" + "=" * 70)
    print("Mechanism Decomposition: Resistance vs Recovery")
    print("=" * 70)

    results = {}

    # CR (resistance) channel
    print("\n  [Path 1] Resistance Channel (CR):")
    cr_res = dml_plr(panel, "cr", "did_shock", x_cols, learner="random_forest",
                     bootstrap=True)
    results["cr"] = cr_res
    _print_result("CR", cr_res)

    # RC (recovery) channel
    print("\n  [Path 2] Recovery Channel (RC):")
    rc_res = dml_plr(panel, "rc", "did_shock", x_cols, learner="random_forest",
                     bootstrap=True)
    results["rc"] = rc_res
    _print_result("RC", rc_res)

    # Summary
    print("\n  Summary:")
    print(f"    CSEE total effect:   via main DML (see main results)")
    print(f"    CR (resistance):     θ = {cr_res['theta']:+.6f} (SE={cr_res['se']:.6f})")
    print(f"    RC (recovery):       θ = {rc_res['theta']:+.6f} (SE={rc_res['se']:.6f})")

    if abs(cr_res["theta"]) > abs(rc_res["theta"]):
        print(f"    → Primary channel: RESISTANCE (damage reduction)")
    else:
        print(f"    → Primary channel: RECOVERY (faster bounce-back)")

    return results


def mediation_analysis(panel, mediator, x_cols=None):
    """DML-based mediation analysis.

    Tests whether policy effect on CSEE is mediated through a specific variable.

    Step 1: DML for total effect (policy → CSEE)
    Step 2: DML for mediator effect (policy → M)
    Step 3: DML for controlled direct effect (policy + M → CSEE)
    Step 4: Indirect effect = Total - Direct

    Args:
        mediator: name of mediation variable (with _m suffix)

    Returns:
        dict with total, direct, indirect effects
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    med_col = mediator + "_m" if mediator + "_m" in panel.columns else mediator

    print(f"\n  Mediation Analysis: {mediator}")

    # Step 1: Total effect (policy → CSEE)
    total_res = dml_plr(panel, "csee", "did_shock", x_cols, learner="random_forest")
    total_effect = total_res["theta"]

    # Step 2: Effect of policy on mediator (policy → M)
    med_res = dml_plr(panel, med_col, "did_shock", x_cols, learner="random_forest")
    mediator_effect = med_res["theta"]

    # Step 3: Controlled direct effect (policy → CSEE | M)
    # Add mediator to controls
    x_with_med = x_cols + [med_col]
    direct_res = dml_plr(panel, "csee", "did_shock", x_with_med, learner="random_forest")
    direct_effect = direct_res["theta"]

    # Step 4: Indirect effect
    indirect_effect = total_effect - direct_effect
    mediation_ratio = indirect_effect / total_effect if abs(total_effect) > 1e-8 else 0

    result = {
        "mediator": mediator,
        "total_effect": total_effect,
        "total_se": total_res["se"],
        "mediator_effect": mediator_effect,
        "direct_effect": direct_effect,
        "direct_se": direct_res["se"],
        "indirect_effect": indirect_effect,
        "mediation_ratio": mediation_ratio,
    }

    print(f"    Total effect:   {total_effect:+.6f} (SE={total_res['se']:.6f})")
    print(f"    Mediator effect: {mediator_effect:+.6f} (SE={med_res['se']:.6f})")
    print(f"    Direct effect:  {direct_effect:+.6f} (SE={direct_res['se']:.6f})")
    print(f"    Indirect effect: {indirect_effect:+.6f}")
    print(f"    Mediation ratio: {mediation_ratio:.1%}")

    return result


def run_all_mechanisms(panel, x_cols=None):
    """Run all mechanism analyses.

    Returns:
        dict with all mechanism results
    """
    if x_cols is None:
        x_cols = CONTROL_VARS

    print("\n" + "=" * 70)
    print("MECHANISM ANALYSIS: Why does the policy buffer ecological impacts?")
    print("=" * 70)

    all_results = {}

    # 1. CR/RC decomposition
    all_results["decomposition"] = mechanism_decomposition(panel, x_cols)

    # 2. Mediation analyses for each mediator
    mediators = ["green_rate", "sponge_inv", "blue_green_ratio", "coupling_coord"]
    med_results = []
    for m in mediators:
        if m + "_m" in panel.columns or m in panel.columns:
            res = mediation_analysis(panel, m, x_cols)
            med_results.append(res)
    all_results["mediation"] = pd.DataFrame(med_results)

    # Summary table
    print("\n" + "=" * 70)
    print("Mediation Summary")
    print("=" * 70)
    if med_results:
        df = pd.DataFrame(med_results)
        print(df[["mediator", "total_effect", "direct_effect", "indirect_effect",
                  "mediation_ratio"]].to_string(index=False))

    return all_results


def _print_result(label, res):
    """Print DML result in formatted way."""
    sig = "***" if res.get("p_value", 1) < 0.01 else \
          "**" if res.get("p_value", 1) < 0.05 else \
          "*" if res.get("p_value", 1) < 0.1 else ""
    print(f"    {label}: θ = {res['theta']:+.6f} (SE = {res['se']:.6f}) {sig}")
    print(f"    95% CI: [{res['ci_lower']:+.6f}, {res['ci_upper']:+.6f}]")


if __name__ == "__main__":
    from data_simulation import generate_panel_data
    from csee_computation import compute_csee_panel

    panel, ndvi_ts, events = generate_panel_data()
    panel = compute_csee_panel(panel, ndvi_ts, events)

    results = run_all_mechanisms(panel)
