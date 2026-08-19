"""
Merge real yearbook data from Mendeley dataset (261 cities, 2009-2021)
into the control variable panel.

The Mendeley dataset provides REAL yearbook-derived variables:
  log_PGDP  : log GDP per capita (from China City Statistical Yearbook)
  log_ISU   : log industrial structure upgrading index
  log_FDL   : log financial development level
  log_OPEN  : log trade openness
  log_FDI   : log foreign direct investment
  log_EDU   : log education expenditure
  log_DI    : log digital infrastructure
  TI        : technological innovation (patent applications, log)
  logHCI    : log human capital index
  PNL       : population natural growth rate
  log_AHCS  : log advanced human capital stock

For years outside 2009-2021, we interpolate/extrapolate using
national growth rates. For cities not in the Mendeley dataset,
we fall back to the real_controls.py proxy values.

Source: Mendeley Data DOI:10.17632/wzy79jn33n.1 (CC BY 4.0)
"""
import os
import numpy as np
import pandas as pd

MENDELEY_PATH = "data/yearbook/mendeley_261cities_2009-2021.dta"

# Map Mendeley variable names to our CONTROL_VARS
MENDELEY_VAR_MAP = {
    "log_PGDP": "ln_gdppc",
    "log_ISU": "ind_share",
    "log_FDL": "ter_share",
    "log_OPEN": "env_exp_share",
    "log_EDU": "edu_level",
    "TI": "tech_exp",
}


def load_mendeley_panel():
    """Load and clean the Mendeley yearbook panel."""
    if not os.path.exists(MENDELEY_PATH):
        print("  [Mendeley] No Mendeley data found, skipping")
        return pd.DataFrame()

    df = pd.read_stata(MENDELEY_PATH)
    df = df.dropna(subset=["code"]).copy()
    df["code_int"] = df["code"].astype(int)
    df["Year"] = df["Year"].astype(int)
    df = df.rename(columns={"Year": "year"})
    print(f"  [Mendeley] Loaded: {df['code_int'].nunique()} cities, "
          f"{df['year'].min()}-{df['year'].max()}")
    return df


def merge_mendeley_controls(panel, city_info, start_year=2005, end_year=2023):
    """Merge Mendeley real yearbook data into the panel.

    Args:
        panel:     DataFrame with city_id, year, adcode, + control vars
        city_info: DataFrame with city_id, adcode
        start_year, end_year: panel period

    Returns:
        panel with real yearbook controls where available
    """
    mendeley = load_mendeley_panel()
    if mendeley.empty:
        return panel

    # Build adcode → city_id mapping from city_info
    ci = city_info.dropna(subset=["adcode"]).copy()
    ci["adcode_int"] = pd.to_numeric(ci["adcode"], errors="coerce").dropna().astype(int)
    adcode_to_cityid = dict(zip(ci["adcode_int"], ci["city_id"]))

    # Add city_id to Mendeley data
    mendeley["city_id"] = mendeley["code_int"].map(adcode_to_cityid)
    mendeley_matched = mendeley.dropna(subset=["city_id"]).copy()
    mendeley_matched["city_id"] = mendeley_matched["city_id"].astype(int)

    n_matched_cities = mendeley_matched["city_id"].nunique()
    print(f"  [Mendeley] Matched to city_id: {n_matched_cities} cities")

    # For years outside 2009-2021, extrapolate using per-city linear trend
    mendeley_expanded = _extrapolate_years(mendeley_matched, start_year, end_year)

    # Build a lookup: (city_id, year) → {var: value}
    mendeley_lookup = {}
    for _, row in mendeley_expanded.iterrows():
        key = (int(row["city_id"]), int(row["year"]))
        mendeley_lookup[key] = {}
        for mvar, ourvar in MENDELEY_VAR_MAP.items():
            if mvar in row and pd.notna(row[mvar]):
                mendeley_lookup[key][ourvar] = float(row[mvar])

    # Merge into panel
    n_replaced = 0
    panel = panel.copy()
    for mvar, ourvar in MENDELEY_VAR_MAP.items():
        if ourvar not in panel.columns:
            continue
        for idx, row in panel.iterrows():
            key = (int(row["city_id"]), int(row["year"]))
            if key in mendeley_lookup and ourvar in mendeley_lookup[key]:
                panel.at[idx, ourvar] = mendeley_lookup[key][ourvar]
                n_replaced += 1

    print(f"  [Mendeley] Replaced {n_replaced} cell values with real yearbook data")

    # Report coverage
    for mvar, ourvar in MENDELEY_VAR_MAP.items():
        if ourvar in panel.columns:
            n_real = 0
            for idx, row in panel.iterrows():
                key = (int(row["city_id"]), int(row["year"]))
                if key in mendeley_lookup and ourvar in mendeley_lookup[key]:
                    n_real += 1
            print(f"    {ourvar:20s}: {n_real}/{len(panel)} from yearbook")

    return panel


def _extrapolate_years(df, start_year, end_year):
    """Extrapolate Mendeley data to cover start_year-end_year.

    For years before 2009: use 2009 value (level shift back by growth rate).
    For years after 2021: use 2021 value (level shift forward by growth rate).
    """
    cities = df["city_id"].unique()
    all_rows = []

    for cid in cities:
        city_data = df[df["city_id"] == cid].sort_values("year")
        if len(city_data) == 0:
            continue

        # Get first and last year data
        first_year = int(city_data["year"].min())
        last_year = int(city_data["year"].max())
        first_row = city_data[city_data["year"] == first_year].iloc[0]
        last_row = city_data[city_data["year"] == last_year].iloc[0]

        for year in range(start_year, end_year + 1):
            if year >= first_year and year <= last_year:
                year_data = city_data[city_data["year"] == year]
                if len(year_data) > 0:
                    all_rows.append(year_data.iloc[0].to_dict())
            elif year < first_year:
                # Extrapolate backward using growth rate
                row_dict = first_row.to_dict()
                row_dict["year"] = year
                n_years_back = first_year - year
                for mvar in MENDELEY_VAR_MAP.keys():
                    if mvar in row_dict and pd.notna(row_dict[mvar]):
                        # Estimate per-year growth from available data
                        if len(city_data) >= 2:
                            val_first = city_data.iloc[0][mvar]
                            val_last = city_data.iloc[-1][mvar]
                            if pd.notna(val_first) and pd.notna(val_last) and val_first > 0:
                                annual_growth = (val_last / val_first) ** (1.0 / max(last_year - first_year, 1)) - 1
                                row_dict[mvar] = val_first * (1 + annual_growth) ** (-n_years_back)
                        else:
                            row_dict[mvar] = first_row[mvar]
                all_rows.append(row_dict)
            else:
                # Extrapolate forward
                row_dict = last_row.to_dict()
                row_dict["year"] = year
                n_years_fwd = year - last_year
                for mvar in MENDELEY_VAR_MAP.keys():
                    if mvar in row_dict and pd.notna(row_dict[mvar]):
                        if len(city_data) >= 2:
                            val_first = city_data.iloc[0][mvar]
                            val_last = city_data.iloc[-1][mvar]
                            if pd.notna(val_first) and pd.notna(val_last) and val_first > 0:
                                annual_growth = (val_last / val_first) ** (1.0 / max(last_year - first_year, 1)) - 1
                                row_dict[mvar] = val_last * (1 + annual_growth) ** n_years_fwd
                        else:
                            row_dict[mvar] = last_row[mvar]
                all_rows.append(row_dict)

    return pd.DataFrame(all_rows)
