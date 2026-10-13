"""Exports docs/data.json for the dashboard and a Tableau-ready summary CSV.

Run from the repo root:  python scripts/build_dashboard_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARMS = {"Mens E-Mail": "mens", "Womens E-Mail": "womens"}

df = pd.read_csv(ROOT / "data/raw/hillstrom.csv.gz")
df["buyer_type"] = np.select([(df["mens"] == 1) & (df["womens"] == 1), df["mens"] == 1], ["both", "mens only"], "womens only")
df["customer_age"] = df["newbie"].map({1: "New", 0: "Existing"})
df["last_purchase"] = pd.cut(df["recency"], [0, 3, 6, 12], labels=["1-3 months", "4-6 months", "7-12 months"]).astype(str)
df["past_spend"] = pd.cut(df["history"], [0, 100, 350, 1e9], labels=["< $100", "$100-350", "> $350"]).astype(str)
df["area"] = df["zip_code"].replace({"Surburban": "Suburban"})
DIMENSIONS = {"buyer_type": "Has bought before", "customer_age": "Customer age", "last_purchase": "Last purchase",
              "past_spend": "Spent in past year", "channel": "Bought via", "area": "Area"}
METRICS = ["visit", "conversion", "spend"]
rng = np.random.default_rng(11)


def lift(t, c):
    d = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return round(float(d), 5), round(float(d - 1.96 * se), 5), round(float(d + 1.96 * se), 5)


def boot(t, c, draws=400):
    t, c = t.to_numpy(), c.to_numpy()
    return [round(float(rng.choice(t, len(t)).mean() - rng.choice(c, len(c)).mean()), 4) for _ in range(draws)]


ctrl = df[df["segment"] == "No E-Mail"]
out = {"arms": {}, "segments": {}, "planner": {}, "dimensions": DIMENSIONS, "n": len(df)}

for arm in ["No E-Mail", *ARMS]:
    g = df[df["segment"] == arm]
    out["arms"][arm] = {m: {"rate": round(float(g[m].mean()), 5), "ci": round(float(1.96 * g[m].std(ddof=1) / np.sqrt(len(g))), 5)}
                        for m in METRICS} | {"n": len(g)}

for dim in DIMENSIONS:
    out["segments"][dim] = []
    for value, g in df.groupby(dim):
        c = g[g["segment"] == "No E-Mail"]
        row = {"value": value, "n": len(g)}
        for arm, key in ARMS.items():
            t = g[g["segment"] == arm]
            row[key] = {m: lift(t[m], c[m]) for m in METRICS}
        out["segments"][dim].append(row)

# Planner: bootstrap draws of the per-customer lift, overall and by buyer type
for arm, key in ARMS.items():
    out["planner"][key] = {}
    for bt in ["all", "both", "mens only", "womens only"]:
        g = df if bt == "all" else df[df["buyer_type"] == bt]
        t, c = g[g["segment"] == arm], g[g["segment"] == "No E-Mail"]
        out["planner"][key][bt] = {
            "share": round(len(g) / len(df), 4),
            "visit": round(float(t["visit"].mean() - c["visit"].mean()), 5),
            "conversion": round(float(t["conversion"].mean() - c["conversion"].mean()), 5),
            "spend_draws": boot(t["spend"], c["spend"]),
        }

(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/data.json").write_text(json.dumps(out, separators=(",", ":")))

# Tableau-ready extract: one row per segment x email, with lifts vs the control group already calculated
LABELS = {"No E-Mail": "No email", "Mens E-Mail": "Men's email", "Womens E-Mail": "Women's email"}
NICE = {"both": "Both lines", "mens only": "Men's only", "womens only": "Women's only"}
rows = []
for dim, label in {"all": "All customers", **DIMENSIONS}.items():
    groups = [("All customers", df)] if dim == "all" else list(df.groupby(dim))
    for value, g in groups:
        c = g[g["segment"] == "No E-Mail"]
        for arm in ["No E-Mail", *ARMS]:
            t = g[g["segment"] == arm]
            row = {"Dimension": label, "Segment": NICE.get(value, value), "Email": LABELS[arm], "Customers": len(t),
                   "Visits": int(t["visit"].sum()), "Orders": int(t["conversion"].sum()), "Revenue": round(float(t["spend"].sum()), 2),
                   "Visit rate": round(float(t["visit"].mean()), 5), "Conversion rate": round(float(t["conversion"].mean()), 5),
                   "Spend per customer": round(float(t["spend"].mean()), 4)}
            for m, name in [("visit", "Visit lift"), ("spend", "Spend lift")]:
                d, lo, hi = lift(t[m], c[m]) if arm != "No E-Mail" else (0.0, 0.0, 0.0)
                row[name], row[f"{name} low"], row[f"{name} high"] = d, lo, hi
            rows.append(row)
(ROOT / "data/tableau").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(ROOT / "data/tableau/campaign_summary.csv", index=False)
print("docs/data.json and data/tableau/campaign_summary.csv written")
