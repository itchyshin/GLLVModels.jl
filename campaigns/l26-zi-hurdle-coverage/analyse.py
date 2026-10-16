#!/usr/bin/env python3
"""Per-cell summaries for the L26 sweep.
Coverage MC SE is rep-clustered: coverage is averaged within each rep (over the species /
entries in the block), then SD across reps / sqrt(R). Flags follow the L26 finding rules."""
import glob, math, sys
import numpy as np, pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/l26/out"
rc = pd.read_csv(f"{OUT}/receipts.csv")
import subprocess
subprocess.run([sys.executable, "/home/user/l26/normalise_params.py", OUT], check=True, capture_output=True)
pr = pd.read_csv(f"{OUT}/params.csv")
pr = pr.drop_duplicates(["cell", "rep", "param"], keep="last")

def block(p):
    if p.startswith("betaz"): return "betaz"
    if p.startswith("betac"): return "betac"
    if p.startswith("LLt"):
        i, j = p[4:-1].split(",")
        return "LLt_diag" if i == j else "LLt_off"
    return p
pr["block"] = pr["param"].map(block)
ok = rc[rc.status == "ok"][["cell", "rep"]]
pr = pr.merge(ok, on=["cell", "rep"])

def cov_stats(d):
    d = d[d.covered >= 0]
    if d.empty:
        return pd.Series(dict(cov=np.nan, cov_mcse=np.nan, n_int=0, R=0))
    per = d.groupby("rep").covered.mean()
    R = len(per)
    return pd.Series(dict(cov=d.covered.mean(), cov_mcse=per.std(ddof=1) / math.sqrt(R) if R > 1 else np.nan,
                          n_int=len(d), R=R))

def bias_stats(d):
    e = d.est - d.truth
    per = e.groupby(d.rep).mean()
    R = len(per)
    se_ok = d.se[np.isfinite(d.se)]
    # SE calibration per parameter: mean SE / empirical SD of estimates (across reps), averaged over params
    g = d[np.isfinite(d.se)].groupby("param")
    ratio = (g.se.mean() / g.est.std(ddof=1)).replace([np.inf, -np.inf], np.nan)
    return pd.Series(dict(bias=e.mean(), bias_mcse=per.std(ddof=1) / math.sqrt(R) if R > 1 else np.nan,
                          rmse=math.sqrt((e ** 2).mean()), se_over_sd=ratio.mean()))

rows = []
for (cell, blk), d in pr.groupby(["cell", "block"]):
    s = pd.concat([cov_stats(d), bias_stats(d)])
    s["cell"] = cell; s["block"] = blk
    rows.append(s)
summ = pd.DataFrame(rows)
meta = rc.groupby("cell").agg(family=("family", "first"), pi=("pi", "first"), n=("n", "first"),
                              reps_run=("rep", "count"),
                              ok=("status", lambda s: (s == "ok").sum()),
                              timeout=("status", lambda s: (s == "timeout").sum()),
                              error=("status", lambda s: s.isin(["error", "crash", "badresult"]).sum()))
okr = rc[rc.status == "ok"].copy()
okr["pd"] = okr.pd_hessian.astype(str).str.lower() == "true"
okr["conv"] = okr.converged.astype(str).str.lower() == "true"
m2 = okr.groupby("cell").agg(nonpd_rate=("pd", lambda s: 1 - s.mean()), nonconv_rate=("conv", lambda s: 1 - s.mean()),
                             med_t=("t_total_s", "median"), max_t=("t_total_s", "max"),
                             med_grad=("grad_maxabs", "median"), max_grad=("grad_maxabs", "max"))
meta = meta.join(m2)
if len(summ):
    summ = summ.merge(meta[["family", "pi", "n"]], left_on="cell", right_index=True)
    summ["flag_cov"] = ((summ["cov"] < 0.92) | (summ["cov"] > 0.98)) & (
        (summ["cov"] - 0.95).abs() > 1.96 * summ["cov_mcse"])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("=== cell status ===")
print(meta.to_string(float_format=lambda x: f"{x:.3g}"))
if len(summ):
    print("\n=== coverage / bias by cell x block ===")
    print(summ[["cell", "family", "pi", "n", "block", "R", "n_int", "cov", "cov_mcse", "flag_cov", "bias",
                "bias_mcse", "rmse", "se_over_sd"]].sort_values(["cell", "block"]).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    summ.to_csv(f"{OUT}/summary_by_cell_block.csv", index=False)
meta.to_csv(f"{OUT}/summary_by_cell.csv")
