# N1b analysis, thresholds fixed by the pre-registration (Sections 6 and 10). Usage: analyze.py <outdir> ds1 ds2 ...
import numpy as np, pandas as pd, sys, json
od = sys.argv[1]; dss = sys.argv[2:]
Z_AGREE, RATIO_AGREE, Z_DIS, RATIO_DIS, ABS_CELL, ABS_COMM, MINFIN, SEMIN = 0.50, (0.80, 1.25), 1.00, (0.67, 1.50), 0.01, 0.05, 950, 1e-3
lt = [(i, j) for j in range(5) for i in range(j, 5)]
idx = [i + 5 * j for (i, j) in lt] + [50 + k for k in range(5)]
names = [f"Sigma{i+1}{j+1}" for (i, j) in lt] + [f"comm{k+1}" for k in range(5)]
Ridx = [25 + i + 5 * j for (i, j) in lt if i != j]; Rnames = [f"R{i+1}{j+1}" for (i, j) in lt if i != j]
lam = np.array([0.9, 0.6, -0.5, 0.7, 0.4]); s = np.array([0.50, 0.40, 0.60, 0.30, 0.50])
Strue = np.outer(lam, lam) + np.diag(s)
truth = np.array([Strue[i, j] for (i, j) in lt] + list(lam**2 / (lam**2 + s)))
Rtrue = Strue / np.sqrt(np.outer(np.diag(Strue), np.diag(Strue)))
Rtruth = np.array([Rtrue[i, j] for (i, j) in lt if i != j])
def ends(M):
    M = M[np.all(np.isfinite(M), axis=1)]
    return np.percentile(M, [2.5, 97.5], axis=0, method="linear"), M.std(0, ddof=1), len(M)
res = {}
for d in dss:
    p = f"{od}/p{d}"
    Rm_all = pd.read_csv(f"{p}/pairedR_{d}.csv").values.astype(float)
    Rinfo = pd.read_csv(f"{p}/pairedR_info_{d}.csv")
    J_all = pd.read_csv(f"{p}/jdraws_{d}.csv", header=None).values.astype(float)
    Jinfo = pd.read_csv(f"{p}/jinfo_{d}.csv")
    Rm = Rm_all[:, idx]; J = J_all[:, idx]
    out = {"ds": d}
    Rrow = np.all(np.isfinite(Rm), axis=1); Jrow = np.all(np.isfinite(J), axis=1)
    out["R_failed_rows"] = int((~Rrow).sum()); out["J_failed_rows"] = int((~Jrow).sum())
    out["R_conv_nonzero"] = int((Rinfo.conv.fillna(-1) != 0).sum()); out["J_thrown"] = int(Jinfo.thrown.sum())
    out["J_not_ok_wrapper"] = int((~Jinfo.ok.astype(bool)).sum()); out["J_driver_default_would_fail"] = int((~Jinfo.conv_driver.astype(bool)).sum())
    nfR = np.isfinite(Rm).sum(0); nfJ = np.isfinite(J).sum(0)
    out["finite_min_R"] = int(nfR.min()); out["finite_min_J"] = int(nfJ.min())
    qR, sR, nR = ends(Rm); qJ, sJ, nJ = ends(J)
    out["n_allfinite_R"], out["n_allfinite_J"] = nR, nJ
    sp = np.sqrt((sR**2 + sJ**2) / 2); dEnd = qJ - qR
    small = sp < SEMIN
    z = np.abs(dEnd) / np.where(small, np.nan, sp)
    out["maxz"] = float(np.nanmax(z)); k = np.unravel_index(np.nanargmax(z), z.shape); out["maxz_at"] = f"{names[k[1]]} {'lo' if k[0]==0 else 'hi'}"
    out["small_cells"] = [names[c] for c in np.where(small)[0]]
    absd = np.abs(dEnd)
    out["abs_rule_small_cells_ok"] = bool(np.all(absd[:, small] <= ABS_CELL)) if small.any() else True
    out["abs_small_max"] = float(absd[:, small].max()) if small.any() else None
    out["abs_comm_max"] = float(absd[:, 20 - 5 + 15:].max()) if False else float(absd[:, 15:20].max())
    out["abs_comm_ok"] = bool(out["abs_comm_max"] <= ABS_COMM)
    # common survivors
    both = Rrow & Jrow
    qR2, sR2, _ = ends(Rm[both]); qJ2, sJ2, _ = ends(J[both]); sp2 = np.sqrt((sR2**2 + sJ2**2) / 2)
    z2 = np.abs(qJ2 - qR2) / np.where(sp2 < SEMIN, np.nan, sp2); out["maxz_common"] = float(np.nanmax(z2)); out["n_common"] = int(both.sum())
    ratio = sJ / sR
    out["ratio_min"], out["ratio_max"] = float(ratio.min()), float(ratio.max())
    out["ratio_argmin"], out["ratio_argmax"] = names[int(ratio.argmin())], names[int(ratio.argmax())]
    rat_ok = (ratio >= RATIO_AGREE[0]) & (ratio <= RATIO_AGREE[1])
    out["ratio_outside_agree"] = [(names[c], float(ratio[c]), bool(small[c])) for c in np.where(~rat_ok)[0]]
    rat_dis = (ratio < RATIO_DIS[0]) | (ratio > RATIO_DIS[1])
    out["ratio_outside_disagree"] = [(names[c], float(ratio[c]), bool(small[c])) for c in np.where(rat_dis)[0]]
    # ratio restricted to non-small cells (sensitivity only)
    ns = ~small
    out["ratio_range_nonsmall"] = (float(ratio[ns].min()), float(ratio[ns].max()))
    # abs rule for z cells is z; verdict
    z_ok = out["maxz"] <= Z_AGREE
    agree = z_ok and bool(rat_ok.all()) and out["abs_rule_small_cells_ok"] and out["abs_comm_ok"]
    if out["finite_min_R"] < MINFIN or out["finite_min_J"] < MINFIN: v = "Inconclusive (rule 1: fewer than 950 finite draws)"
    elif agree: v = "Agree"
    elif out["maxz"] > Z_DIS or bool(rat_dis.any()): v = "Disagree"
    else: v = "Inconclusive"
    out["verdict"] = v
    out["why"] = {"z_ok": bool(z_ok), "ratio_ok": bool(rat_ok.all()), "abs_small_ok": out["abs_rule_small_cells_ok"], "abs_comm_ok": out["abs_comm_ok"],
                  "z_gt_dis": bool(out["maxz"] > Z_DIS), "ratio_dis": bool(rat_dis.any())}
    # descriptive
    sdJ = np.nanstd(J[Jrow], axis=0, ddof=1); sdR = np.nanstd(Rm[Rrow], axis=0, ddof=1)
    sdp = np.sqrt((sdJ**2 + sdR**2) / 2)
    dz = np.abs(Rm[both] - J[both]) / np.where(sdp > SEMIN, sdp, np.nan)
    mx = np.nanmax(dz, axis=1); out["paired_rep_max_median"] = float(np.median(mx)); out["paired_rep_max_overall"] = float(mx.max()); out["n_rep_gt_0.05SD"] = int((mx > 0.05).sum())
    llJ = Jinfo.loglik.values.astype(float); llR = -Rinfo.nll.values.astype(float)
    okb = np.isfinite(llJ) & np.isfinite(llR); dl = llJ - llR
    out["n_ll_both"] = int(okb.sum())
    out["ll_diff_gt_1e-4"] = int((np.abs(dl[okb]) > 1e-4).sum()); out["ll_J_higher"] = int((dl[okb] > 1e-4).sum()); out["ll_R_higher"] = int((dl[okb] < -1e-4).sum())
    out["ll_max_abs_diff"] = float(np.abs(dl[okb]).max())
    # truth containment (2 engines, 20 cells)
    cj = (qJ[0] <= truth) & (truth <= qJ[1]); cr = (qR[0] <= truth) & (truth <= qR[1])
    out["contain_J"] = f"{cj.sum()}/20"; out["contain_R"] = f"{cr.sum()}/20"
    out["contain_miss_J"] = [names[c] for c in np.where(~cj)[0]]; out["contain_miss_R"] = [names[c] for c in np.where(~cr)[0]]
    # R_phy descriptive
    RJ = J_all[:, Ridx]; RR = Rm_all[:, Ridx]
    qa, sa, _ = ends(RR); qb, sb, _ = ends(RJ)
    spr = np.sqrt((sa**2 + sb**2) / 2); zr = np.abs(qb - qa) / np.where(spr < SEMIN, np.nan, spr)
    out["Rphy_maxz"] = float(np.nanmax(zr)); out["Rphy_ratio_range"] = (float((sb/sa).min()), float((sb/sa).max()))
    out["Rphy_nfinite_min"] = (int(np.isfinite(RR).sum(0).min()), int(np.isfinite(RJ).sum(0).min()))
    out["median_endpoints"] = None
    res[d] = out
print(json.dumps(res, indent=1, default=str))
vs = [res[d]["verdict"] for d in dss]
print("OVERALL:", "Agree" if all(v == "Agree" for v in vs) else ("Disagree" if any(v == "Disagree" for v in vs) else "Inconclusive"))
