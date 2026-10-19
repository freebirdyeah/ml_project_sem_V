"""Reproduces the dataset-level statistics quoted in the methodology document.
usage: python3 dataset_stats.py dataset_pilot_v1.csv"""
import sys, numpy as np, pandas as pd
d=pd.read_csv(sys.argv[1]); d=d[d.status=='ok']
print('runs',len(d),'circuits',d.circuit.nunique(),'recipes',d.recipe_id.nunique())
print('yA median range per circuit %.3f | yD %.3f | J %.3f'%tuple(
  (lambda g:(g.max()-g.min()).median())(d.groupby('circuit')[c]) for c in ('yA','yD','J_0.5')))
J=d.groupby('circuit')['J_0.5']; r=J.max()-J.min()
print('circuits with J spread <1%%: %.1f%%  <3%%: %.1f%%'%(100*(r<.01).mean(),100*(r<.03).mean()))
y=d['J_0.5']; gm=y.mean(); tot=((y-gm)**2).sum()
sc=((d.groupby('circuit')['J_0.5'].transform('mean')-gm)**2).sum()/tot; sr=((d.groupby('recipe_id')['J_0.5'].transform('mean')-gm)**2).sum()/tot
print('variance share: circuit %.3f recipe %.3f interaction %.3f'%(sc,sr,1-sc-sr))
w=y-d.groupby('circuit')['J_0.5'].transform('mean'); rm=w.groupby(d.recipe_id).transform('mean')
print('within-circuit variance explained by recipe main effect %.3f'%((rm**2).sum()/(w**2).sum()))
for lam in (0.0,0.5,1.0):
    P=(lam*d.yA+(1-lam)*d.yD).groupby([d.circuit,d.recipe_id]).first().unstack()
    g=[(P.loc[c,P.drop(index=c).mean().idxmin()]-P.loc[c].min())/P.loc[c,P.drop(index=c).mean().idxmin()]*100 for c in P.index]
    print('lambda %.1f: oracle vs leave-one-circuit-out best fixed recipe: mean %.2f%%, median %.2f%%, distinct best recipes %d'%(lam,np.mean(g),np.median(g),P.idxmin(axis=1).nunique()))
