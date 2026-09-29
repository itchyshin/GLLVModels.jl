"""Harvest the auto-d recovery grid: replay each dataset through the old select_lv rule
(argmin over every fit that returned) and the guarded rule (converged, logLik non-decreasing
vs the last accepted K, not runaway), for AIC, BIC log(p*n) and BIC log(n).
Usage: python3 analyze.py <dir with task-*.csv> <out.md>"""
import sys, glob, math, os
import numpy as np, pandas as pd

TOL, SD_MAX, RATIO_MAX = 1e-3, 10.0, 25.0
# Usage: python3 analyze.py <dir>[,<dir2>...] <out.md>. Later dirs (e.g. a re-run) fill datasets the
# earlier ones left incomplete; only COMPLETE datasets (every K_fit 1..min(K_true+2, p-1)) are counted.
dirs = sys.argv[1].split(",")
files, parts = [], []
for i, d in enumerate(dirs):
    for f in sorted(glob.glob(f"{d}/task-*.csv")):
        if os.path.getsize(f) > 0 and sum(1 for _ in open(f)) > 1:
            x = pd.read_csv(f); x["src"] = i; parts.append(x); files.append(f)
df = pd.concat(parts, ignore_index=True)
_k = ["family", "n", "p", "K_true", "rep"]
_c = df.groupby(_k + ["src"]).K_fit.nunique().rename("nk").reset_index()
_c["need"] = np.minimum(_c.K_true + 2, _c.p - 1)
_c = _c[_c.nk >= _c.need].sort_values("src").drop_duplicates(_k, keep="first")[_k + ["src"]]
n_incomplete = df[_k].drop_duplicates().shape[0] - len(_c)
df = df.merge(_c, on=_k + ["src"])
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
    # lenient: unconverged fits are allowed if not runaway and monotone
    acc2, llp2 = [], -math.inf
    for _, r in g.iterrows():
        if not r.ok or runaway(r) or r.loglik < llp2 - TOL: continue
        acc2.append(r); llp2 = r.loglik
    a2 = pd.DataFrame(acc2)
    for name, col in crits.items():
        rec[f"len_{name}"] = int(a2.K_fit[a2[col].idxmin()]) if len(a2) else np.nan
    rows.append(rec)
R, KS = pd.DataFrame(rows), pd.DataFrame(kstat)

def summ(sub):
    out = {"datasets": len(sub)}
    for rule in ("old", "new", "len"):
        for name in crits:
            k = sub[f"{rule}_{name}"]; t = sub.K_true; m = k.notna()
            e = (k[m] == t[m]).mean() if m.any() else np.nan
            out[f"{rule}_{name}"] = f"{e:.2f}±{math.sqrt(e*(1-e)/max(m.sum(),1)):.2f} (u{(k[m]<t[m]).mean():.2f}/o{(k[m]>t[m]).mean():.2f})" if m.any() else "NA"
    return pd.Series(out)

with open(sys.argv[2], "w") as fh:
    fh.write(f"# Auto-d recovery grid ({len(files)} task files, {len(R)} complete datasets, {len(df)} fits; {n_incomplete} incomplete datasets dropped)\n\n")
    fh.write("Cells: exact-recovery rate ± MCSE (u = too few, o = too many). old = argmin over every fit that returned;\n")
    fh.write(f"new = guarded (converged, logLik non-decreasing, not runaway: row norm > {SD_MAX} or binomial ratio ≥ {RATIO_MAX});\nlen = lenient guard (unconverged allowed if not runaway and non-decreasing).\n\n")
    for fam, gf in R.groupby("family"):
        fh.write(f"## {fam}\n\n" + gf.groupby(["n", "p", "K_true"])[list(gf.columns)].apply(summ).to_markdown() + "\n\n")
    fh.write("## Mean exact-recovery rate across cells (unweighted), by family and rule\n\n")
    cols = [f"{r}_{c}" for r in ("old", "new", "len") for c in crits]
    tab = R.assign(**{c: (R[c] == R.K_true).astype(float).where(R[c].notna()) for c in cols}).groupby("family")[cols].mean().round(3)
    fh.write(tab.to_markdown() + "\n\n")
    fh.write("## Fit status by family and K_fit (share of attempted fits)\n\n")
    fh.write(pd.crosstab([KS.family, KS.K_fit], KS.st, normalize="index").round(3).to_markdown() + "\n")
print(f"{len(files)} files, {len(R)} datasets -> {sys.argv[2]}")
