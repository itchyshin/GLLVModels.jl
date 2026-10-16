#!/usr/bin/env python3
"""Merge worker param files into one well-formed CSV (LLt[i,j] names contain a comma)."""
import csv, glob, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/l26/out"
rows = []
for f in sorted(glob.glob(f"{OUT}/params_w*.csv")):
    for line in open(f):
        v = line.rstrip("\n").split(",")
        if len(v) == 10:
            v = v[:2] + [v[2] + "," + v[3]] + v[4:]
        if len(v) == 9:
            rows.append(v)
with open(f"{OUT}/params.csv", "w", newline="") as g:
    w = csv.writer(g)
    w.writerow(["cell", "rep", "param", "truth", "est", "se", "lower", "upper", "covered"])
    w.writerows(rows)
print(len(rows), "param rows")
