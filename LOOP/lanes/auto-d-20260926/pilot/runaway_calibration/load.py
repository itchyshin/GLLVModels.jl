import glob, os
import numpy as np, pandas as pd

D = os.path.expanduser("~/local-scratch/lanes/GLLVM.jl-auto-d-20260926/LOOP/lanes/auto-d-20260926/pilot/harvest")

def load_all():
    parts = []
    for f in sorted(glob.glob(f"{D}/task-*.csv")):
        if os.path.getsize(f) == 0:
            continue
        try:
            x = pd.read_csv(f)
        except Exception:
            continue
        if len(x) == 0:
            continue
        parts.append(x)
    df = pd.concat(parts, ignore_index=True)
    for c in ("max_rownorm", "relload"):
        if c not in df:
            df[c] = np.nan
    df["ok"] = df.status.eq("ok")
    df["conv"] = df.converged.astype(str).str.lower().eq("true")
    return df

if __name__ == "__main__":
    df = load_all()
    print(df.shape)
    print(df.groupby("family").size())
    print(df.groupby(["family"]).agg(n_files=("rep","count")))
