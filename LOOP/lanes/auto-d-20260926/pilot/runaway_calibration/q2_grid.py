import math
import numpy as np, pandas as pd
from load import load_all

TOL = 1e-3
pd.set_option("display.width", 160)

df = load_all()
keys = ["family", "n", "p", "K_true", "rep"]
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

df = df.groupby(keys, group_keys=False).apply(flag_nonmono, include_groups=False).join(df[keys])
df["over"] = df.K_fit > df.K_true
df["broken"] = (df.max_rownorm > 50) | df.nonmono

SDS = [6, 8, 10, 12, 15, 20]
RATIOS = [10, 15, 20, 25, 30, 50]


def flagged(sub, sd, ratio):
    f = sub.max_rownorm > sd
    is_binom = sub.family.eq("binomial")
    f = f | (is_binom & (sub.relload >= ratio))
    return f


print("=" * 100)
print("Q2a: false-alarm rate (share of K_fit<=K_true, status==ok rows flagged) by family x max_latent_sd")
print("(ratio_max fixed at 25 for this slice; binomial only extra-flagged by ratio)")
print("=" * 100)
ok = df[df.ok]
rows = []
for fam, g in ok.groupby("family"):
    healthy = g[~g.over]
    for sd in SDS:
        fl = flagged(healthy, sd, 25)
        rows.append({"family": fam, "max_latent_sd": sd, "n": len(healthy), "false_alarm_rate": fl.mean()})
FA = pd.DataFrame(rows)
print(FA.pivot(index="max_latent_sd", columns="family", values="false_alarm_rate").to_string())

print()
print("=" * 100)
print("Q2a2: false-alarm rate for BINOMIAL, full grid over (max_latent_sd, ratio_max)")
print("=" * 100)
bh = ok[(ok.family == "binomial") & ~ok.over]
grid = pd.DataFrame(index=SDS, columns=RATIOS, dtype=float)
for sd in SDS:
    for r in RATIOS:
        grid.loc[sd, r] = flagged(bh, sd, r).mean()
grid.index.name = "max_latent_sd \\ ratio_max"
print(grid.to_string())

print()
print("=" * 100)
print("Q2b: detection rate of 'clearly broken' fits (max_rownorm>50 or nonmono), by family x max_latent_sd")
print("=" * 100)
rows = []
for fam, g in ok.groupby("family"):
    broken = g[g.broken]
    for sd in SDS:
        fl = flagged(broken, sd, 25)
        rows.append({"family": fam, "max_latent_sd": sd, "n_broken": len(broken), "detection_rate": fl.mean() if len(broken) else np.nan})
DET = pd.DataFrame(rows)
print(DET.pivot(index="max_latent_sd", columns="family", values="detection_rate").to_string())
print()
print("n_broken by family:")
print(DET.pivot(index="max_latent_sd", columns="family", values="n_broken").iloc[0])

print()
print("=" * 100)
print("Q2b2: detection rate for BINOMIAL 'broken' fits, full grid over (max_latent_sd, ratio_max)")
print("=" * 100)
bb = ok[(ok.family == "binomial") & ok.broken]
print(f"n_broken (binomial) = {len(bb)}  (of {len(ok[ok.family=='binomial'])} total ok binomial fits)")
grid2 = pd.DataFrame(index=SDS, columns=RATIOS, dtype=float)
for sd in SDS:
    for r in RATIOS:
        grid2.loc[sd, r] = flagged(bb, sd, r).mean()
grid2.index.name = "max_latent_sd \\ ratio_max"
print(grid2.to_string())

df.to_pickle("df_annotated2.pkl")
