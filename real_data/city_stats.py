"""
City statistical yearbook data loader.

Loads socio-economic control variables from Chinese city statistical yearbooks:
  - 中国城市统计年鉴 (China City Statistical Yearbook)
  - 中国城市建设统计年鉴 (China Urban Construction Statistical Yearbook)

Expected input: Excel/CSV files with city-year observations.

Variables extracted:
  - ln_gdppc:       log GDP per capita (元)
  - ln_pop:         log population (万人)
  - ln_fixinv:      log fixed asset investment (万元)
  - industry_share: secondary industry share of GDP (%)
  - urban_rate:     urbanization rate (%)
  - green_rate:     green coverage rate (%)
  - fiscal_pressure: fiscal revenue / GDP
  - ln_road:        log per capita road area (m²)
  - ln_patent:      log patent applications (件)

Data files should be placed in data/yearbook/ with naming convention:
  yearbook_{year}.xlsx or yearbook_{year}.csv
"""
import os
import numpy as np
import pandas as pd

# Column name mappings (Chinese → English)
COLUMN_MAP = {
    # GDP per capita
    "人均GDP": "gdppc", "人均gdp": "gdppc", "人均地区生产总值": "gdppc",
    "人均地区生产总值(元)": "gdppc", "per_capita_gdp": "gdppc",
    # Population
    "年末总人口": "pop", "总人口": "pop", "人口": "pop",
    "总人口(万人)": "pop", "population": "pop",
    # Fixed asset investment
    "固定资产投资": "fixinv", "固定资产投资额": "fixinv",
    "全社会固定资产投资": "fixinv", "fixed_investment": "fixinv",
    # Industrial structure
    "第二产业比重": "industry_share", "第二产业占GDP比重": "industry_share",
    "second_industry_share": "industry_share",
    # Urbanization
    "城镇化率": "urban_rate", "城市化率": "urban_rate",
    "urbanization_rate": "urban_rate",
    # Green coverage
    "建成区绿化覆盖率": "green_rate", "绿化覆盖率": "green_rate",
    "green_coverage_rate": "green_rate",
    # Fiscal
    "财政收入": "fiscal_rev", "一般公共预算收入": "fiscal_rev",
    "fiscal_revenue": "fiscal_rev",
    # Road
    "人均道路面积": "road_area", "道路面积": "road_area",
    "per_capita_road_area": "road_area",
    # Patents
    "专利申请数": "patent", "专利申请量": "patent",
    "patent_applications": "patent",
    # City name
    "城市": "city_name", "城市名称": "city_name", "地区": "city_name",
    "city": "city_name",
}


def _rename_columns(df):
    """Rename Chinese column names to standardized English names."""
    rename_dict = {}
    for col in df.columns:
        col_lower = str(col).strip()
        if col_lower in COLUMN_MAP:
            rename_dict[col] = COLUMN_MAP[col_lower]
        else:
            for cn, en in COLUMN_MAP.items():
                if cn in col_lower:
                    rename_dict[col] = en
                    break
    return df.rename(columns=rename_dict)


def _compute_derived(df):
    """Compute derived variables from raw data."""
    if "gdppc" in df.columns and df["gdppc"].dtype != object:
        df["ln_gdppc"] = np.log(df["gdppc"].clip(lower=1))

    if "pop" in df.columns and df["pop"].dtype != object:
        df["ln_pop"] = np.log(df["pop"].clip(lower=0.1))

    if "fixinv" in df.columns and df["fixinv"].dtype != object:
        df["ln_fixinv"] = np.log(df["fixinv"].clip(lower=1))

    if "road_area" in df.columns and df["road_area"].dtype != object:
        df["ln_road"] = np.log(df["road_area"].clip(lower=0.1))

    if "patent" in df.columns and df["patent"].dtype != object:
        df["ln_patent"] = np.log(df["patent"].clip(lower=1))

    if "fiscal_rev" in df.columns and "gdppc" in df.columns:
        if df["fiscal_rev"].dtype != object and df["gdppc"].dtype != object:
            df["fiscal_pressure"] = df["fiscal_rev"] / (df["gdppc"] * df.get("pop", 1))

    if "industry_share" in df.columns and df["industry_share"].max() > 1:
        df["industry_share"] = df["industry_share"] / 100

    if "urban_rate" in df.columns and df["urban_rate"].max() > 1:
        df["urban_rate"] = df["urban_rate"] / 100

    if "green_rate" in df.columns and df["green_rate"].max() > 1:
        df["green_rate"] = df["green_rate"] / 100

    return df


CONTROL_COLS = [
    "ln_gdppc", "ln_pop", "ln_fixinv", "industry_share",
    "urban_rate", "green_rate", "fiscal_pressure", "ln_road", "ln_patent",
]


def load_yearbook_data(data_dir="data/yearbook", city_name_col="city_name"):
    """Load all yearbook data from Excel/CSV files in data_dir.

    Files should be named: yearbook_{year}.xlsx or yearbook_{year}.csv

    Returns:
        DataFrame with city_name, year, and all control variables
    """
    all_dfs = []

    if not os.path.exists(data_dir):
        print(f"  Warning: {data_dir} does not exist.")
        print(f"  Place yearbook Excel/CSV files in {data_dir}/")
        print(f"  Naming convention: yearbook_{{year}}.xlsx or yearbook_{{year}}.csv")
        return pd.DataFrame()

    for fname in sorted(os.listdir(data_dir)):
        if not (fname.endswith(".xlsx") or fname.endswith(".csv")):
            continue

        parts = fname.replace(".xlsx", "").replace(".csv", "").split("_")
        year = None
        for p in parts:
            if p.isdigit() and len(p) == 4:
                year = int(p)
                break

        if year is None:
            continue

        filepath = os.path.join(data_dir, fname)
        try:
            if fname.endswith(".xlsx"):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath, encoding="utf-8-sig")

            df = _rename_columns(df)
            df["year"] = year
            all_dfs.append(df)
            print(f"  Loaded {fname}: {len(df)} cities, {year}")
        except Exception as e:
            print(f"  Warning: Failed to load {fname}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    panel = pd.concat(all_dfs, ignore_index=True)
    panel = _compute_derived(panel)

    available = [c for c in CONTROL_COLS if c in panel.columns]
    if available:
        panel[available] = panel[available].fillna(panel[available].median())

    print(f"  Total yearbook panel: {len(panel)} city-years, {len(available)} controls")
    return panel


def merge_with_city_info(yearbook_df, city_info_df):
    """Merge yearbook data with city info to get city_id.

    Matches by city_name (fuzzy match if needed).

    Returns:
        DataFrame with city_id added
    """
    if yearbook_df.empty:
        return yearbook_df

    name_to_id = dict(zip(city_info_df["city_name"], city_info_df["city_id"]))

    yearbook_df["city_id"] = yearbook_df["city_name"].map(name_to_id)

    unmatched = yearbook_df["city_id"].isna()
    if unmatched.any():
        unmatched_names = yearbook_df.loc[unmatched, "city_name"].unique()
        print(f"  Warning: {len(unmatched_names)} cities not matched to city_info:")
        for name in unmatched_names[:10]:
            print(f"    - {name}")
        if len(unmatched_names) > 10:
            print(f"    ... and {len(unmatched_names) - 10} more")

    return yearbook_df.dropna(subset=["city_id"])


if __name__ == "__main__":
    print("City statistics loader.")
    print("Place yearbook files in data/yearbook/ as yearbook_{year}.xlsx")
    print("Required columns (Chinese names auto-mapped):")
    for cn, en in COLUMN_MAP.items():
        if en != "city_name":
            print(f"  {en:20s} ← {cn}")
