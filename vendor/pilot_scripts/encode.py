import numpy as np, pandas as pd
OPS=['rw','rwz','rf','rfz','rs','rsz','b']; MAXLEN=12
LOGCOLS=['c_n_pi','c_n_po','c_n_and','c_depth','c_n_edges','c_and_per_depth','c_fo_max','c_width_max','c_width_mean','c_lvl_gap_max']
def recipe_features(ops_str):
    ops=ops_str.split(); f={'r_len':len(ops),'r_nz':sum(o.endswith('z') for o in ops)}
    for o in OPS: f[f'r_cnt_{o}']=ops.count(o)
    for p in range(MAXLEN):
        for o in OPS: f[f'r_pos{p}_{o}']=int(p<len(ops) and ops[p]==o)
    for a in OPS:
        for b in OPS: f[f'r_tr_{a}_{b}']=sum(1 for x,y in zip(ops,ops[1:]) if x==a and y==b)
    f['r_first_b']=int(ops[0]=='b'); f['r_last_b']=int(ops[-1]=='b')
    return f
def build(df):
    """Return X (DataFrame), circuit col names, recipe col names."""
    uniq=df.recipe_ops.unique(); R=pd.DataFrame([recipe_features(s) for s in uniq],index=uniq)
    X=df[[c for c in df.columns if c.startswith('c_')]].copy()
    for c in LOGCOLS: X[c]=np.log1p(X[c])
    ccols=list(X.columns)
    Xr=R.loc[df.recipe_ops].reset_index(drop=True); Xr.index=X.index
    X=pd.concat([X,Xr],axis=1)
    return X,ccols,list(R.columns)
