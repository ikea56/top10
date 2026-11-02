"""業種ランキング・テーマランキング。"""


def _agg(sub):
    return {
        "score": float(sub["tech_scaled"].mean()),
        "n": int(len(sub)),
        "n_high": int(sub["hi60"].sum()),
        "ret20": float(sub["ret20d"].mean()) if sub["ret20d"].notna().any() else 0.0,
        "leaders": list(sub.sort_values(["tech_raw", "vol_ratio"], ascending=False)["name"].head(3)),
    }


def rank_sectors(liquid, top=5, min_members=5):
    rows = []
    for name, sub in liquid.groupby("sector"):
        if len(sub) >= min_members and name not in ("-", "nan"):
            rows.append({"name": name, **_agg(sub)})
    return sorted(rows, key=lambda r: -r["score"])[:top]


def rank_themes(liquid, themes, top=5, min_members=3):
    rows = []
    for name, codes in themes.items():
        sub = liquid[liquid["code"].isin(codes)]
        if len(sub) >= min_members:
            rows.append({"name": name, **_agg(sub)})
    return sorted(rows, key=lambda r: -r["score"])[:top]


def themes_of(code, themes):
    return [n for n, codes in themes.items() if code in codes]
