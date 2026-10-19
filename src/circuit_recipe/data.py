"""Dataset checks and leakage-safe feature construction."""
from pathlib import Path
import numpy as np
import pandas as pd
from .recipes import encode_recipe, OPS

LOG_COLUMNS = ['c_n_pi','c_n_po','c_n_and','c_depth','c_n_edges','c_and_per_depth',
               'c_fo_max','c_width_max','c_width_mean','c_lvl_gap_max']

def load_dataset(path):
    d=pd.read_csv(path)
    required=['circuit','family','recipe_id','recipe_ops','status','area','delay','base_area','base_delay','yA','yD']
    if any(c not in d for c in required):
        raise ValueError('Missing required dataset columns.')
    if d.duplicated(['circuit','recipe_id']).any():
        raise ValueError('Duplicate circuit/recipe observations.')
    if not d.status.eq('ok').all():
        raise ValueError('Failed synthesis observations in dataset.')
    counts=d.groupby('circuit').recipe_id.nunique()
    if counts.nunique()!=1 or counts.iloc[0]!=d.recipe_id.nunique():
        raise ValueError('Each circuit must have the same complete recipe library.')
    numeric=d.select_dtypes('number')
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Nonfinite numeric measurements.')
    for y,measure,base in [('yA','area','base_area'),('yD','delay','base_delay')]:
        if (d[[measure,base]]<=0).any().any() or not np.allclose(d[y],d[measure]/d[base],rtol=1e-10,atol=1e-10):
            raise ValueError('Invalid normalized measurements.')
    static=[c for c in d if c.startswith(('c_','p_'))]+['base_area','base_delay']
    if d.groupby('circuit')[static].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Circuit features change between recipes.')
    if d.groupby('recipe_id').recipe_ops.nunique().gt(1).any():
        raise ValueError('Recipe IDs have inconsistent operation strings.')
    return d.sort_values(['circuit','recipe_id']).reset_index(drop=True)

def feature_frame(d, group='full', encoding='full', interactions=False):
    static=[c for c in d if c.startswith('c_')]
    probes=[c for c in d if c.startswith('p_')]
    if group=='probes':
        cols=probes
    elif group in ('static','mapping','full'):
        cols=static+(probes if group=='full' else [])
    else:
        raise ValueError(group)
    X=d[cols].astype(float).copy()
    for col in LOG_COLUMNS:
        if col in X:
            X[col]=np.log1p(X[col])
    if group in ('mapping','full'):
        X['m_log_area']=np.log(d.base_area)
        X['m_log_delay']=np.log(d.base_delay)
        X['m_area_per_node']=d.base_area/d.c_n_and
        X['m_delay_per_level']=d.base_delay/d.c_depth
    cache={s:encode_recipe(s,encoding) for s in d.recipe_ops.unique()}
    R=pd.DataFrame([cache[s] for s in d.recipe_ops],index=d.index,dtype=float)
    X=pd.concat([X,R],axis=1)
    if interactions:
        additions={f'ix_{p}_{o}':d[p].to_numpy()*R[f'r_cnt_{o}'].to_numpy() for p in probes for o in OPS}
        X=pd.concat([X,pd.DataFrame(additions,index=d.index)],axis=1)
    return X.astype(np.float32)

def targets(d, lam=.5, kind='centered'):
    y=lam*d.yA+(1-lam)*d.yD
    if kind=='raw':
        return y.to_numpy()
    if kind=='centered':
        return (y-y.groupby(d.circuit).transform('mean')).to_numpy()
    if kind=='rank':
        counts=y.groupby(d.circuit).transform('count')
        return ((y.groupby(d.circuit).rank(method='average')-1)/(counts-1)).to_numpy()
    raise ValueError(kind)
