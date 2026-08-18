"""
Policy pilot city lists — verified from official government sources.

Sources:
  - 海绵城市试点: 财政部/住建部/水利部 (财建〔2014〕838号, 财办建〔2015〕4号, 财办建〔2016〕25号)
  - 气候适应型城市试点: 国家发改委/住建部 (发改气候〔2017〕343号)
    https://www.ndrc.gov.cn/xxgk/zcfb/tz/201702/t20170224_962916.html
"""
import pandas as pd

# ---------------------------------------------------------------------------
# 海绵城市建设试点 (Sponge City Construction Pilots)
# ---------------------------------------------------------------------------

SPONGE_CITY_BATCH1 = {
    "year": 2015,
    "cities": [
        ("迁安市", "河北省"),
        ("白城市", "吉林省"),
        ("镇江市", "江苏省"),
        ("嘉兴市", "浙江省"),
        ("池州市", "安徽省"),
        ("厦门市", "福建省"),
        ("萍乡市", "江西省"),
        ("济南市", "山东省"),
        ("鹤壁市", "河南省"),
        ("武汉市", "湖北省"),
        ("常德市", "湖南省"),
        ("南宁市", "广西壮族自治区"),
        ("重庆市", "重庆市"),
        ("遂宁市", "四川省"),
        ("贵安新区", "贵州省"),
        ("西咸新区", "陕西省"),
    ],
}

SPONGE_CITY_BATCH2 = {
    "year": 2016,
    "cities": [
        ("北京市", "北京市"),
        ("天津市", "天津市"),
        ("大连市", "辽宁省"),
        ("上海市", "上海市"),
        ("宁波市", "浙江省"),
        ("福州市", "福建省"),
        ("青岛市", "山东省"),
        ("珠海市", "广东省"),
        ("深圳市", "广东省"),
        ("三亚市", "海南省"),
        ("玉溪市", "云南省"),
        ("庆阳市", "甘肃省"),
        ("西宁市", "青海省"),
        ("固原市", "宁夏回族自治区"),
    ],
}

# ---------------------------------------------------------------------------
# 气候适应型城市建设试点 (Climate-Adaptive City Construction Pilots)
# 发改气候〔2017〕343号, 2017-02-24
# ---------------------------------------------------------------------------

CLIMATE_ADAPTIVE_PILOTS = {
    "year": 2017,
    "document": "发改气候〔2017〕343号",
    "cities": [
        ("呼和浩特市", "内蒙古自治区"),
        ("大连市", "辽宁省"),
        ("朝阳市", "辽宁省"),
        ("丽水市", "浙江省"),
        ("合肥市", "安徽省"),
        ("淮北市", "安徽省"),
        ("九江市", "江西省"),
        ("济南市", "山东省"),
        ("安阳市", "河南省"),
        ("武汉市", "湖北省"),
        ("十堰市", "湖北省"),
        ("常德市", "湖南省"),
        ("岳阳市", "湖南省"),
        ("百色市", "广西壮族自治区"),
        ("海口市", "海南省"),
        ("璧山区", "重庆市"),
        ("潼南区", "重庆市"),
        ("广元市", "四川省"),
        ("六盘水市", "贵州省"),
        ("毕节市", "贵州省"),
        ("商洛市", "陕西省"),
        ("西咸新区", "陕西省"),
        ("白银市", "甘肃省"),
        ("庆阳市", "甘肃省"),
        ("西宁市", "青海省"),
        ("库尔勒市", "新疆维吾尔自治区"),
        ("阿克苏市", "新疆维吾尔自治区"),
        ("石河子市", "新疆生产建设兵团"),
    ],
}


def get_pilot_dataframe(policy="climate_adaptive"):
    """Return a DataFrame of pilot cities.

    Args:
        policy: "climate_adaptive" (default) or "sponge" or "both"

    Returns:
        DataFrame with columns: city_name, province, policy_type, pilot_year
    """
    rows = []

    if policy in ("climate_adaptive", "both"):
        for city, prov in CLIMATE_ADAPTIVE_PILOTS["cities"]:
            rows.append({
                "city_name": city,
                "province": prov,
                "policy_type": "climate_adaptive",
                "pilot_year": CLIMATE_ADAPTIVE_PILOTS["year"],
            })

    if policy in ("sponge", "both"):
        for city, prov in SPONGE_CITY_BATCH1["cities"]:
            rows.append({
                "city_name": city,
                "province": prov,
                "policy_type": "sponge",
                "pilot_year": SPONGE_CITY_BATCH1["year"],
            })
        for city, prov in SPONGE_CITY_BATCH2["cities"]:
            rows.append({
                "city_name": city,
                "province": prov,
                "policy_type": "sponge",
                "pilot_year": SPONGE_CITY_BATCH2["year"],
            })

    return pd.DataFrame(rows)


def get_treated_cities(policy="climate_adaptive"):
    """Return set of treated city names.

    Args:
        policy: "climate_adaptive", "sponge", or "both"

    Returns:
        set of city names (Chinese)
    """
    df = get_pilot_dataframe(policy=policy)
    return set(df["city_name"].tolist())


def get_pilot_year(city_name, policy="climate_adaptive"):
    """Get pilot year for a given city.

    Returns:
        int pilot year, or None if city is not a pilot
    """
    df = get_pilot_dataframe(policy=policy)
    match = df[df["city_name"] == city_name]
    if len(match) > 0:
        return int(match.iloc[0]["pilot_year"])
    return None


if __name__ == "__main__":
    print("=== 气候适应型城市建设试点 (28 cities, 2017) ===")
    df_ca = get_pilot_dataframe("climate_adaptive")
    print(f"Total: {len(df_ca)} cities")
    print(df_ca[["city_name", "province"]].to_string(index=False))

    print("\n=== 海绵城市建设试点 (30 cities, 2015-2016) ===")
    df_sp = get_pilot_dataframe("sponge")
    print(f"Total: {len(df_sp)} cities")
    print(df_sp[["city_name", "province", "pilot_year"]].to_string(index=False))

    print("\n=== Overlap (cities in both programs) ===")
    ca_cities = set(df_ca["city_name"])
    sp_cities = set(df_sp["city_name"])
    overlap = ca_cities & sp_cities
    print(f"Overlap: {overlap}")
