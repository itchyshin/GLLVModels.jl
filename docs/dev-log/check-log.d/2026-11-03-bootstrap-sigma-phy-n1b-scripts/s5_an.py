import numpy as np, pandas as pd, sys
od=sys.argv[1]
for d in sys.argv[2:]:
    th=pd.read_csv(f"{od}/s5/theory_{d}.csv")
    for eng in ("J","R"):
        S=pd.read_csv(f"{od}/s5/s5{eng}_{d}.csv",header=None).values
        m=S.mean(0); se=S.std(0,ddof=1)/np.sqrt(len(S)); z=(m-th.theory.values)/se
        k=int(np.abs(z).argmax()); r=th.iloc[k]
        bad=[f"{r.stat}{int(r.i)}{int(r.j)}" for r,zz in zip(th.itertuples(),z) if abs(zz)>4]
        print(d,eng,"n",len(S),"max|z| %.2f at %s%d%d"%(abs(z[k]),r.stat,int(r.i),int(r.j)),"fail>4:",bad,"max rel dev %.4f"%np.abs((m-th.theory.values)/np.abs(th.theory.values)).max())
