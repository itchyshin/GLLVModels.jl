"""Recovery-rate impact of different (max_latent_sd, ratio_max) guard settings,
reimplementing analyze.py's guarded-selection logic exactly, parametrized over thresholds.
"""
import math
import numpy as np, pandas as pd
from load import load_all

TOL = 1e-3
pd.set_option("display.width", 160)

df = load_all()
keys = ["family", "n", "p", "K_true", "rep"]

# --- dataset completeness filter, exactly as analyze.py does (single source dir here) ---
_c = df.groupby(keys).K_fit.nunique().rename("nk").reset_index()
_c["need"] = np.minimum(_c.K_true + 2, _c.p - 1)
complete_keys = _c[_c.nk >= _c.need][keys]
n_incomplete = df[keys].drop_duplicates().shape[0] - len(complete_keys)
df = df.merge(complete_keys, on=keys)
print(f"complete datasets kept: {len(complete_keys)}; incomplete dropped: {n_incomplete}")

df["ok"] = df.status.eq("ok")
df["conv"] = df.converged.astype(str).str.lower().eq("true")
crits = {"aic": "aic", "bic_pn": "bic_pn", "bic_n": "bic_n"}


def make_runaway(max_latent_sd, ratio_max):
    def runaway(r):
        if r.family == "gaussian" or pd.isna(r.max_rownorm):
            return False
        if r.max_rownorm > max_latent_sd:
            return True
        return r.family == "binomial" and r.relload >= ratio_max
    return runaway


def guarded_recovery(max_latent_sd, ratio_max):
    runaway = make_runaway(max_latent_sd, ratio_max)
    rows = []
    for key, g in df.groupby(keys):
        g = g.sort_values("K_fit")
        rec = dict(zip(keys, key))
        acc, llprev = [], -math.inf
        for _, r in g.iterrows():
            st = ("failed" if not r.ok else "unconverged" if not r.conv else
                  "runaway" if runaway(r) else "nonmonotone" if r.loglik < llprev - TOL else "ok")
            if st == "ok":
                acc.append(r)
                llprev = r.loglik
        a = pd.DataFrame(acc)
        for name, col in crits.items():
            rec[f"new_{name}"] = int(a.K_fit[a[col].idxmin()]) if len(a) else np.nan
        rows.append(rec)
    R = pd.DataFrame(rows)
    out = {}
    for fam, gf in R.groupby("family"):
        k = gf["new_bic_n"]
        t = gf.K_true
        m = k.notna()
        exact = (k[m] == t[m]).mean() if m.any() else np.nan
        under = (k[m] < t[m]).mean() if m.any() else np.nan
        overr = (k[m] > t[m]).mean() if m.any() else np.nan
        nodef = 1 - m.mean()
        out[fam] = dict(exact=exact, under=under, over=overr, no_pick=nodef, n=len(gf))
    return out


settings = [
    ("current (10, 25)", 10, 25),
    ("looser rownorm (20, 25)", 20, 25),
    ("tighter rownorm (6, 25)", 6, 25),
    ("looser ratio (10, 50)", 10, 50),
    ("tighter ratio (10, 10)", 10, 10),
    ("both loose (20, 50)", 20, 50),
    ("both tight (6, 10)", 6, 10),
]

results = {}
for label, sd, ratio in settings:
    results[label] = guarded_recovery(sd, ratio)

fams = ["gaussian", "poisson", "binomial", "nb"]
for fam in fams:
    print()
    print("=" * 90)
    print(f"Recovery under bic_sites (bic_n), family = {fam}")
    print("=" * 90)
    tab = pd.DataFrame({label: results[label].get(fam, {}) for label in results}).T
    print(tab.to_string())
