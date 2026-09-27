"""Harvest the auto-d recovery grid: replay each dataset through the old select_lv rule
(argmin over every fit that returned) and the guarded rule (converged, logLik non-decreasing
vs the last accepted K, not runaway), for AIC, BIC log(p*n) and BIC log(n).
Usage: python3 analyze.py <dir with task-*.csv> <out.md>"""
import sys, glob, math
import numpy as np, pandas as pd

TOL, SD_MAX, RATIO_MAX = 1e-3, 10.0, 25.0
files = sorted(glob.glob(f"{sys.argv[1]}/task-*.csv"))
df = pd.concat([pd.read_csv(f) for f in files if sum(1 for _ in open(f)) > 1], ignore_index=True)
for c in ("max_rownorm", "relload"):
    if c not in df: df[c] = np.nan
df["ok"] = df.status.eq("ok")
df["conv"] = df.converged.astype(str).str.lower().eq("true")
keys = ["family", "n", "p", "K_true", "rep"]
crits = {"aic": "aic", "bic_pn": "bic_pn", "bic_n": "bic_n"}

def runaway(r):
    if r.family == "gaussian" or pd.isna(r.max_rownorm): return False
    if r.max_rownorm > SD_MAX: return True
    return r.family == "binomial" and r.relload >= RATIO_MAX

rows, kstat = [], []
for key, g in df.groupby(keys):
    g = g.sort_values("K_fit")
    rec = dict(zip(keys, key)); rec["nfits"] = len(g)
    okg = g[g.ok]
    for name, col in crits.items():
        rec[f"old_{name}"] = int(okg.K_fit[okg[col].idxmin()]) if len(okg) else np.nan
    acc, llprev = [], -math.inf
    for _, r in g.iterrows():
        st = ("failed" if not r.ok else "unconverged" if not r.conv else
              "runaway" if runaway(r) else "nonmonotone" if r.loglik < llprev - TOL else "ok")
        kstat.append({**rec, "K_fit": r.K_fit, "st": st})
        if st == "ok": acc.append(r); llprev = r.loglik
    a = pd.DataFrame(acc)
    for name, col in crits.items():
        rec[f"new_{name}"] = int(a.K_fit[a[col].idxmin()]) if len(a) else np.nan
    rows.append(rec)
R, KS = pd.DataFrame(rows), pd.DataFrame(kstat)

def summ(sub):
    out = {"datasets": len(sub)}
    for rule in ("old", "new"):
        for name in crits:
            k = sub[f"{rule}_{name}"]; t = sub.K_true; m = k.notna()
            e = (k[m] == t[m]).mean() if m.any() else np.nan
            out[f"{rule}_{name}"] = f"{e:.2f}±{math.sqrt(e*(1-e)/max(m.sum(),1)):.2f} (u{(k[m]<t[m]).mean():.2f}/o{(k[m]>t[m]).mean():.2f})" if m.any() else "NA"
    return pd.Series(out)

with open(sys.argv[2], "w") as fh:
    fh.write(f"# Auto-d recovery grid ({len(files)} task files, {len(R)} datasets, {len(df)} fits)\n\n")
    fh.write("Cells: exact-recovery rate ± MCSE (u = too few, o = too many). old = argmin over every fit that returned;\n")
    fh.write(f"new = guarded (converged, logLik non-decreasing, not runaway: row norm > {SD_MAX} or binomial ratio ≥ {RATIO_MAX}).\n\n")
    for fam, gf in R.groupby("family"):
        fh.write(f"## {fam}\n\n" + gf.groupby(["n", "p", "K_true"])[list(gf.columns)].apply(summ).to_markdown() + "\n\n")
    fh.write("## Fit status by family and K_fit (share of attempted fits)\n\n")
    fh.write(pd.crosstab([KS.family, KS.K_fit], KS.st, normalize="index").round(3).to_markdown() + "\n")
print(f"{len(files)} files, {len(R)} datasets -> {sys.argv[2]}")
