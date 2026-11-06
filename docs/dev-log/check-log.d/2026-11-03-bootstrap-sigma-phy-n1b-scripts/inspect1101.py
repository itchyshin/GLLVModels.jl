import numpy as np, pandas as pd, sys
od=sys.argv[1]
for d in sys.argv[2:]:
    R=pd.read_csv(f"{od}/p{d}/pairedR_{d}.csv").values[:,50:55]; J=pd.read_csv(f"{od}/p{d}/jdraws_{d}.csv",header=None).values[:,50:55]
    R=R[np.all(np.isfinite(R),axis=1)]; J=J[np.all(np.isfinite(J),axis=1)]
    qR=np.percentile(R,[2.5,97.5],axis=0); qJ=np.percentile(J,[2.5,97.5],axis=0)
    print(d,"comm endpoints R lo/hi",np.round(qR,4).tolist(),"\n   J",np.round(qJ,4).tolist(),"\n  diff",np.round(qJ-qR,4).tolist(),"\n  SD R",np.round(R.std(0,ddof=1),4).tolist(),"SD J",np.round(J.std(0,ddof=1),4).tolist())
