#!/usr/bin/env python3
"""Markdown n-series table for one family from summary_by_cell_block.csv + receipts.csv."""
import sys
import pandas as pd
fam = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/user/l26/out"
s = pd.read_csv(f"{OUT}/summary_by_cell_block.csv")
rc = pd.read_csv(f"{OUT}/receipts.csv")
pr = pd.read_csv(f"{OUT}/params.csv")
okk = rc[rc.status == "ok"][["cell", "rep"]]
bzp = pr[pr.param.str.startswith("betaz")].merge(okk, on=["cell", "rep"])
med = (bzp.est - bzp.truth).groupby(bzp.cell).median()
s = s[s.family == fam]
blocks = [b for b in ["betaz", "betac", "LLt_diag", "LLt_off", "r"] if b in set(s.block)]
lab = {"betaz": "betaz", "betac": "betac", "LLt_diag": "ΛΛ' diag", "LLt_off": "ΛΛ' off-diag", "r": "r (NB size)"}
rows = []
hdr = "| π | n | reps | " + " | ".join(f"{lab[b]} cov (MC SE) / bias" for b in blocks) + " | betaz median bias | converged | PD Hessian | timeouts | median s/rep |"
sep = "|" + "---|" * (hdr.count("|") - 1)
rows.append(hdr); rows.append(sep)
for (pi, n), d in s.groupby(["pi", "n"]):
    cell = int(d.cell.iloc[0])
    r = rc[rc.cell == cell]; ok = r[r.status == "ok"]
    conv = (ok.converged.astype(str).str.lower() == "true").sum()
    pdh = (ok.pd_hessian.astype(str).str.lower() == "true").sum()
    to = (r.status == "timeout").sum()
    cells = []
    for b in blocks:
        x = d[d.block == b]
        if x.empty: cells.append("–"); continue
        x = x.iloc[0]
        mark = "**" if x.flag_cov else ""
        cells.append(f"{mark}{x['cov']:.3f}{mark} ({x.cov_mcse:.3f}) / {x.bias:+.3f}")
    rows.append(f"| {pi} | {n} | {len(r)} | " + " | ".join(cells) +
                f" | {med.get(cell, float('nan')):+.3f} | {conv}/{len(ok)} | {pdh}/{len(ok)} | {to} | {ok.t_total_s.median():.0f} |")
print("\n".join(rows))
