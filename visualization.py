"""
Visualization module for generating publication-quality figures and tables.

Generates:
  1. Event study plot (parallel trends)
  2. DML vs DID comparison bar chart
  3. ML algorithm comparison plot
  4. Heterogeneity forest plot
  5. Placebo test density plot
  6. Dose-response curve
  7. Mechanism decomposition chart
  8. Summary statistics table
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIGURES_DIR, TABLES_DIR, FIG_DPI, FIG_FORMAT, COLOR_PALETTE, ensure_dirs

ensure_dirs()
sns.set_palette(COLOR_PALETTE)
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "savefig.bbox": "tight",
})


def _save_fig(fig, name):
    """Save figure to figures directory."""
    path = f"{FIGURES_DIR}/{name}.{FIG_FORMAT}"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _save_table(df, name):
    """Save table to tables directory."""
    path = f"{TABLES_DIR}/{name}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


def plot_event_study(event_study_df, name="fig_event_study"):
    """Plot event study coefficients with confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 5))

    df = event_study_df.copy()
    ax.errorbar(df["event_time"], df["coefficient"], yerr=1.96 * df["se"],
                fmt="o-", capsize=3, capthick=1, linewidth=1.5, markersize=5,
                color="#2c7fb8", ecolor="#7fcdbb")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(x=-1, color="red", linestyle=":", linewidth=1, label="Policy year (ref)")

    ax.set_xlabel("Event Time (years relative to policy)")
    ax.set_ylabel("Coefficient (CSEE)")
    ax.set_title("Event Study: Parallel Trends Test")
    ax.legend()

    return _save_fig(fig, name)


def plot_dml_vs_did(comparison_df, name="fig_dml_vs_did"):
    """Bar chart comparing DML and TWFE-DID estimates."""
    fig, ax = plt.subplots(figsize=(10, 5))

    df = comparison_df.copy()
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df["dml_theta"], width, yerr=1.96 * df["dml_se"],
           label="DML", color="#2c7fb8", capsize=3, alpha=0.85)
    ax.bar(x + width/2, df["did_beta"], width, yerr=1.96 * df["did_se"],
           label="TWFE-DID", color="#fdae61", capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["outcome"], rotation=30, ha="right")
    ax.set_ylabel("Estimated Effect")
    ax.set_title("DML vs Traditional TWFE-DID Comparison")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()

    return _save_fig(fig, name)


def plot_ml_comparison(ml_df, name="fig_ml_comparison"):
    """Plot DML estimates across different ML algorithms."""
    fig, ax = plt.subplots(figsize=(8, 5))

    df = ml_df.copy()
    colors = sns.color_palette(COLOR_PALETTE, len(df))

    ax.barh(df["learner"], df["theta"], xerr=1.96 * df["se"],
            color=colors, capsize=3, alpha=0.85)

    ax.set_xlabel("Estimated Effect (CSEE)")
    ax.set_title("DML Estimates by ML Algorithm")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)

    return _save_fig(fig, name)


def plot_heterogeneity(hetero_summary, name="fig_heterogeneity"):
    """Forest plot of heterogeneity analysis results."""
    fig, ax = plt.subplots(figsize=(10, 8))

    df = hetero_summary.copy()
    y_pos = np.arange(len(df))

    colors = ["#2c7fb8" if t > 0 else "#fdae61" for t in df["theta"]]

    ax.errorbar(df["theta"], y_pos, xerr=1.96 * df["se"],
                fmt="o", capsize=3, capthick=1, markersize=6,
                ecolor="gray", linestyle="none")

    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row["theta"], i, color=colors[i], s=60, zorder=5)

    labels = [f"{r['dimension']}={r['subgroup']}" for _, r in df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Estimated Effect (CSEE)")
    ax.set_title("Heterogeneity Analysis: Subgroup DML Estimates")

    return _save_fig(fig, name)


def plot_placebo(placebo_result, name="fig_placebo"):
    """Density plot of placebo test coefficients."""
    fig, ax = plt.subplots(figsize=(9, 5))

    thetas = placebo_result["placebo_thetas"]
    true_theta = placebo_result["true_theta"]

    ax.hist(thetas, bins=40, density=True, alpha=0.6, color="#7fcdbb",
            edgecolor="white", label=f"Placebo (n={len(thetas)})")

    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(thetas)
    x_range = np.linspace(thetas.min(), thetas.max(), 200)
    ax.plot(x_range, kde(x_range), color="#2c7fb8", linewidth=2)

    ax.axvline(x=true_theta, color="red", linewidth=2, linestyle="--",
               label=f"True θ = {true_theta:+.4f}")
    ax.axvline(x=0, color="gray", linewidth=0.8)

    ax.set_xlabel("Estimated Coefficient")
    ax.set_ylabel("Density")
    ax.set_title(f"Placebo Test (p-value = {placebo_result['placebo_p_value']:.4f})")
    ax.legend()

    return _save_fig(fig, name)


def plot_dose_response(dose_result, name="fig_dose_response"):
    """Plot dose-response curve."""
    if dose_result is None or "bin_centers" not in dose_result:
        print("  No dose-response data to plot")
        return None

    fig, ax = plt.subplots(figsize=(9, 5))

    centers = dose_result["bin_centers"]
    effects = dose_result["bin_effects"]
    ses = dose_result["bin_ses"]

    ax.errorbar(centers, effects, yerr=1.96 * np.array(ses),
                fmt="o-", capsize=4, linewidth=2, markersize=8,
                color="#2c7fb8", ecolor="#7fcdbb")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Policy Implementation Intensity (log investment)")
    ax.set_ylabel("Effect on CSEE")
    ax.set_title("Dose-Response Curve: Policy Intensity → CSEE")

    return _save_fig(fig, name)


def plot_mechanism(mechanism_results, name="fig_mechanism"):
    """Bar chart of mechanism decomposition (CR vs RC)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    decomp = mechanism_results.get("decomposition", {})
    cr_res = decomp.get("cr", {})
    rc_res = decomp.get("rc", {})

    labels = ["CR\n(Resistance)", "RC\n(Recovery)"]
    thetas = [cr_res.get("theta", 0), rc_res.get("theta", 0)]
    ses = [cr_res.get("se", 0), rc_res.get("se", 0)]

    colors = ["#2c7fb8", "#fdae61"]
    ax.bar(labels, thetas, yerr=1.96 * np.array(ses),
           color=colors, capsize=5, alpha=0.85, width=0.5)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("DML Estimated Effect")
    ax.set_title("Mechanism Decomposition: Resistance vs Recovery Channel")

    return _save_fig(fig, name)


def table_summary_stats(panel, name="table_summary_stats"):
    """Generate summary statistics table."""
    y_vars = ["csee", "cr", "rc", "rsei", "psr_resilience"]
    x_vars = ["did_shock", "shock_intensity", "n_events"]
    all_vars = [v for v in y_vars + x_vars if v in panel.columns]

    stats = []
    for col in all_vars:
        s = panel[col].dropna()
        stats.append({
            "Variable": col,
            "N": len(s),
            "Mean": s.mean(),
            "Std": s.std(),
            "Min": s.min(),
            "p25": s.quantile(0.25),
            "Median": s.median(),
            "p75": s.quantile(0.75),
            "Max": s.max(),
        })

    df = pd.DataFrame(stats)
    _save_table(df, name)
    return df


def table_main_results(dml_results, name="table_main_results"):
    """Generate main regression results table."""
    rows = []
    for key, res in dml_results.items():
        if isinstance(res, dict) and "theta" in res:
            y_var = key.replace("main_", "")
            sig = "***" if res.get("p_value", 1) < 0.01 else \
                  "**" if res.get("p_value", 1) < 0.05 else \
                  "*" if res.get("p_value", 1) < 0.1 else ""
            rows.append({
                "Outcome": y_var,
                "DML Estimate": f"{res['theta']:+.4f}{sig}",
                "Std Error": f"({res['se']:.4f})",
                "95% CI": f"[{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}]",
                "N": res["n"],
                "ML Learner": res.get("learner", ""),
            })

    df = pd.DataFrame(rows)
    _save_table(df, name)
    return df


def table_robustness_summary(robustness_results, name="table_robustness"):
    """Generate robustness checks summary table."""
    rows = []

    # Alternative Y
    if "alt_y" in robustness_results:
        for _, r in robustness_results["alt_y"].iterrows():
            rows.append({
                "Check": "Alternative Y",
                "Specification": r.get("alt_y", ""),
                "Estimate": f"{r['theta']:+.4f}",
                "SE": f"({r['se']:.4f})",
            })

    # Alternative ML
    if "alt_ml" in robustness_results:
        for _, r in robustness_results["alt_ml"].iterrows():
            rows.append({
                "Check": "Alternative ML",
                "Specification": r.get("learner", ""),
                "Estimate": f"{r['theta']:+.4f}",
                "SE": f"({r['se']:.4f})",
            })

    # Sample restrictions
    if "sample" in robustness_results:
        for key, r in robustness_results["sample"].items():
            rows.append({
                "Check": "Sample Restriction",
                "Specification": key,
                "Estimate": f"{r['theta']:+.4f}",
                "SE": f"({r['se']:.4f})",
            })

    # Lagged
    if "lagged" in robustness_results:
        for key, r in robustness_results["lagged"].items():
            rows.append({
                "Check": "Lagged Treatment",
                "Specification": key,
                "Estimate": f"{r['theta']:+.4f}",
                "SE": f"({r['se']:.4f})",
            })

    df = pd.DataFrame(rows)
    _save_table(df, name)
    return df


def generate_all_figures_and_tables(panel, all_results):
    """Generate all figures and tables from results.

    Args:
        panel:       main panel DataFrame
        all_results: dict with all analysis results

    Returns:
        dict with paths to all outputs
    """
    print("\n" + "=" * 70)
    print("Generating Figures and Tables...")
    print("=" * 70)

    outputs = {}

    # Summary statistics
    outputs["summary_stats"] = table_summary_stats(panel)

    # Main results table
    if "dml" in all_results:
        outputs["main_results"] = table_main_results(all_results["dml"])

    # Event study
    if "event_study" in all_results:
        outputs["event_study"] = plot_event_study(all_results["event_study"])

    # DML vs DID
    if "dml_vs_did" in all_results:
        outputs["dml_vs_did"] = plot_dml_vs_did(all_results["dml_vs_did"])

    # ML comparison
    if "dml" in all_results and "ml_comparison" in all_results["dml"]:
        outputs["ml_comparison"] = plot_ml_comparison(all_results["dml"]["ml_comparison"])

    # Heterogeneity
    if "heterogeneity" in all_results:
        from heterogeneity import heterogeneity_summary_table
        het_df = heterogeneity_summary_table(all_results["heterogeneity"])
        outputs["heterogeneity"] = plot_heterogeneity(het_df)

    # Placebo
    if "robustness" in all_results and "placebo" in all_results["robustness"]:
        outputs["placebo"] = plot_placebo(all_results["robustness"]["placebo"])

    # Dose-response
    if "dose_response" in all_results and "csee" in all_results["dose_response"]:
        outputs["dose_response"] = plot_dose_response(all_results["dose_response"]["csee"])

    # Mechanism
    if "mechanism" in all_results:
        outputs["mechanism"] = plot_mechanism(all_results["mechanism"])

    # Robustness table
    if "robustness" in all_results:
        outputs["robustness_table"] = table_robustness_summary(all_results["robustness"])

    print(f"\n  Generated {len(outputs)} outputs")
    print("=" * 70)

    return outputs
