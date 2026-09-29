import math
import numpy as np, pandas as pd
from load import load_all

TOL = 1e-3
pd.set_option("display.width", 160)

df = load_all()
keys = ["family", "n", "p", "K_true", "rep"]

# per-row nonmono flag: compare to the loglik of the SAME dataset's next-smaller
# K_fit that returned a fit at all (status == ok), regardless of guard/threshold.
df = df.sort_values(keys + ["K_fit"]).reset_index(drop=True)

def flag_nonmono(g):
    g = g.sort_values("K_fit").copy()
    llprev = np.nan
    flags = []
    for _, r in g.iterrows():
        if r.ok and not np.isnan(llprev) and r.loglik < llprev - TOL:
            flags.append(True)
        else:
            flags.append(False)
        if r.ok:
            llprev = r.loglik if np.isnan(llprev) else max(llprev, r.loglik)
    g["nonmono"] = flags
    return g

df = df.groupby(keys, group_keys=False).apply(flag_nonmono)
df["over"] = df.K_fit > df.K_true  # over-fitted relative to truth

def describe(sub, col):
    x = sub[col].dropna()
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": len(x), "mean": x.mean(), "median": x.median(),
        "p90": x.quantile(0.90), "p95": x.quantile(0.95),
        "max": x.max(), "share_gt10": (x > 10).mean(), "share_gt50": (x > 50).mean(),
    }

print("=" * 100)
print("Q1a: max_rownorm / relload by family x (K_fit<=K_true vs K_fit>K_true), status==ok rows only")
print("=" * 100)
rows = []
for fam, g in df[df.ok].groupby("family"):
    for label, sub in (("K<=Ktrue", g[~g.over]), ("K>Ktrue", g[g.over])):
        for col in ("max_rownorm", "relload"):
            d = describe(sub, col)
            d.update(family=fam, group=label, stat=col)
            rows.append(d)
Q1a = pd.DataFrame(rows)[["family", "group", "stat", "n", "mean", "median", "p90", "p95", "max", "share_gt10", "share_gt50"]]
print(Q1a.to_string(index=False))

print()
print("=" * 100)
print("Q1b: max_rownorm / relload by family x nonmono flag, status==ok rows only")
print("=" * 100)
rows = []
for fam, g in df[df.ok].groupby("family"):
    for label, sub in (("monotone", g[~g.nonmono]), ("nonmono", g[g.nonmono])):
        for col in ("max_rownorm", "relload"):
            d = describe(sub, col)
            d.update(family=fam, group=label, stat=col)
            rows.append(d)
Q1b = pd.DataFrame(rows)[["family", "group", "stat", "n", "mean", "median", "p90", "p95", "max", "share_gt10", "share_gt50"]]
print(Q1b.to_string(index=False))

print()
print("=" * 100)
print("Nonmono incidence rate by family x (K<=Ktrue vs K>Ktrue)")
print("=" * 100)
print(df[df.ok].groupby(["family", "over"]).nonmono.agg(["mean", "count"]))

df.to_pickle("df_annotated.pkl")
print("\nSaved annotated df to df_annotated.pkl")
