"""
City information: codes, coordinates, administrative divisions.

Provides metadata for Chinese prefecture-level cities needed for:
  - GEE NDVI extraction (lat/lon bounding boxes)
  - Spatial weight matrices (coordinates)
  - Heterogeneity analysis (region, north/south, coastal, city size)
  - Policy matching (city name → pilot status)

Data source: user-provided CSV at data/city_info.csv (template auto-generated).
Fallback: hardcoded major cities for testing.
"""
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Region classification (East / Central / West)
# ---------------------------------------------------------------------------
EAST_PROVINCES = {
    "北京市", "天津市", "河北省", "辽宁省", "上海市", "江苏省", "浙江省",
    "福建省", "山东省", "广东省", "海南省",
}
CENTRAL_PROVINCES = {
    "山西省", "吉林省", "黑龙江省", "安徽省", "江西省", "河南省", "湖北省",
    "湖南省",
}
WEST_PROVINCES = {
    "内蒙古自治区", "广西壮族自治区", "重庆市", "四川省", "贵州省", "云南省",
    "西藏自治区", "陕西省", "甘肃省", "青海省", "宁夏回族自治区",
    "新疆维吾尔自治区", "新疆生产建设兵团",
}

# Qinling-Huaihe line (~33°N) for North/South classification
NORTH_SOUTH_LINE = 33.0

# Coastal provinces
COASTAL_PROVINCES = {
    "辽宁省", "河北省", "天津市", "山东省", "江苏省", "上海市", "浙江省",
    "福建省", "广东省", "广西壮族自治区", "海南省",
}

# ---------------------------------------------------------------------------
# Hardcoded major cities for testing (subset)
# Format: (city_name, province, adcode, lat, lon, city_size)
# ---------------------------------------------------------------------------

MAJOR_CITIES = [
    ("北京市", "北京市", "110000", 39.90, 116.40, "large"),
    ("天津市", "天津市", "120000", 39.13, 117.20, "large"),
    ("上海市", "上海市", "310000", 31.23, 121.47, "large"),
    ("重庆市", "重庆市", "500000", 29.56, 106.55, "large"),
    ("南京市", "江苏省", "320100", 32.06, 118.80, "large"),
    ("武汉市", "湖北省", "420100", 30.59, 114.31, "large"),
    ("成都市", "四川省", "510100", 30.67, 104.07, "large"),
    ("广州市", "广东省", "440100", 23.13, 113.27, "large"),
    ("深圳市", "广东省", "440300", 22.54, 114.06, "large"),
    ("杭州市", "浙江省", "330100", 30.27, 120.16, "large"),
    ("西安市", "陕西省", "610100", 34.27, 108.95, "large"),
    ("济南市", "山东省", "370100", 36.65, 117.00, "large"),
    ("合肥市", "安徽省", "340100", 31.82, 117.23, "large"),
    ("福州市", "福建省", "350100", 26.07, 119.30, "large"),
    ("郑州市", "河南省", "410100", 34.75, 113.62, "large"),
    ("长沙市", "湖南省", "430100", 28.23, 112.94, "large"),
    ("南宁市", "广西壮族自治区", "450100", 22.82, 108.37, "large"),
    ("海口市", "海南省", "460100", 20.04, 110.20, "large"),
    ("昆明市", "云南省", "530100", 25.04, 102.71, "large"),
    ("贵阳市", "贵州省", "520100", 26.65, 106.71, "large"),
    ("乌鲁木齐市", "新疆维吾尔自治区", "650100", 43.83, 87.62, "large"),
    ("呼和浩特市", "内蒙古自治区", "150100", 40.81, 111.75, "large"),
    ("大连市", "辽宁省", "210200", 38.91, 121.60, "large"),
    ("青岛市", "山东省", "370200", 36.07, 120.38, "large"),
    ("宁波市", "浙江省", "330200", 29.87, 121.55, "large"),
    ("厦门市", "福建省", "350200", 24.48, 118.09, "large"),
    ("苏州市", "江苏省", "320500", 31.30, 120.62, "large"),
    ("无锡市", "江苏省", "320200", 31.49, 120.31, "large"),
    ("常州市", "江苏省", "320400", 31.77, 119.95, "large"),
    ("南通市", "江苏省", "320600", 31.98, 120.89, "large"),
    ("扬州市", "江苏省", "321000", 32.39, 119.42, "large"),
    ("温州市", "浙江省", "330300", 28.00, 120.66, "large"),
    ("嘉兴市", "浙江省", "330400", 30.75, 120.76, "small"),
    ("丽水市", "浙江省", "331100", 28.45, 119.92, "small"),
    ("镇江市", "江苏省", "321100", 32.20, 119.42, "small"),
    ("池州市", "安徽省", "341700", 30.66, 117.49, "small"),
    ("萍乡市", "江西省", "360300", 27.62, 113.85, "small"),
    ("九江市", "江西省", "360400", 29.71, 115.99, "small"),
    ("鹤壁市", "河南省", "410600", 35.75, 114.30, "small"),
    ("安阳市", "河南省", "410500", 36.10, 114.39, "small"),
    ("常德市", "湖南省", "430700", 29.04, 111.69, "small"),
    ("岳阳市", "湖南省", "430600", 29.36, 113.13, "small"),
    ("十堰市", "湖北省", "420300", 32.63, 110.80, "small"),
    ("遂宁市", "四川省", "510900", 30.53, 105.57, "small"),
    ("广元市", "四川省", "510800", 32.44, 105.84, "small"),
    ("六盘水市", "贵州省", "520200", 26.59, 104.83, "small"),
    ("百色市", "广西壮族自治区", "451000", 23.90, 106.62, "small"),
    ("白银市", "甘肃省", "620400", 36.54, 104.14, "small"),
    ("庆阳市", "甘肃省", "621000", 35.73, 107.64, "small"),
    ("西宁市", "青海省", "630100", 36.62, 101.78, "small"),
    ("固原市", "宁夏回族自治区", "640400", 36.02, 106.24, "small"),
    ("商洛市", "陕西省", "611000", 33.87, 109.94, "small"),
    ("朝阳市", "辽宁省", "211300", 41.57, 120.45, "small"),
    ("淮北市", "安徽省", "340600", 33.97, 116.80, "small"),
    ("玉溪市", "云南省", "530400", 24.35, 102.55, "small"),
    ("珠海市", "广东省", "440400", 22.27, 113.58, "small"),
    ("三亚市", "海南省", "460200", 18.25, 109.51, "small"),
    ("库尔勒市", "新疆维吾尔自治区", "652801", 41.76, 86.15, "small"),
    ("石河子市", "新疆生产建设兵团", "659001", 44.30, 86.03, "small"),
    ("毕节市", "贵州省", "520500", 27.28, 105.29, "small"),
    ("迁安市", "河北省", "130283", 40.00, 118.70, "small"),
    ("白城市", "吉林省", "220800", 45.62, 122.84, "small"),
]


def _classify_region(province):
    if province in EAST_PROVINCES:
        return "east"
    elif province in CENTRAL_PROVINCES:
        return "central"
    elif province in WEST_PROVINCES:
        return "west"
    return "west"


def _classify_ns(lat):
    return "north" if lat >= NORTH_SOUTH_LINE else "south"


def _classify_coastal(province):
    return 1 if province in COASTAL_PROVINCES else 0


def load_city_info(csv_path="data/city_info.csv"):
    """Load city information from CSV or fall back to hardcoded data.

    CSV columns: city_name, province, adcode, lat, lon, city_size

    Returns:
        DataFrame with: city_id, city_name, province, adcode, lat, lon,
                        city_size, region, ns, coastal
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  Loaded city info from {csv_path}: {len(df)} cities")
    else:
        df = pd.DataFrame(MAJOR_CITIES, columns=[
            "city_name", "province", "adcode", "lat", "lon", "city_size"
        ])
        print(f"  Using hardcoded city info: {len(df)} cities")
        print(f"  (Provide data/city_info.csv for full 280-city panel)")

    df["city_id"] = range(1, len(df) + 1)
    df["region"] = df["province"].apply(_classify_region)
    df["ns"] = df["lat"].apply(_classify_ns)
    df["coastal"] = df["province"].apply(_classify_coastal)

    return df


def generate_template_csv(path="data/city_info_template.csv"):
    """Generate a template CSV for the user to fill with full city list."""
    df = pd.DataFrame(MAJOR_CITIES, columns=[
        "city_name", "province", "adcode", "lat", "lon", "city_size"
    ])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Template saved to {path}")
    print(f"  Fill with all 280 prefecture cities and rename to city_info.csv")


if __name__ == "__main__":
    df = load_city_info()
    print(df[["city_id", "city_name", "province", "region", "ns", "coastal"]].to_string(index=False))
    print(f"\nTotal: {len(df)} cities")
    print(f"  East: {len(df[df['region']=='east'])}")
    print(f"  Central: {len(df[df['region']=='central'])}")
    print(f"  West: {len(df[df['region']=='west'])}")
