"""Nested circuit-grouped evaluation, saved out-of-fold rankings and budgets."""
import argparse
import hashlib
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from .data import load_dataset,feature_frame,targets
from .models import GRIDS,make_model
from .synthesis import ROOT,digest

def circuit_splits(d, folds, seed):
    circuits=d[['circuit','family']].drop_duplicates().sort_values('circuit')
    split=StratifiedKFold(folds,shuffle=True,random_state=seed)
    # Fall back to shuffled circuit K-fold if a training family is too small.
    if circuits.family.value_counts().min()<folds:
        split=KFold(folds,shuffle=True,random_state=seed)
    for tr,te in split.split(circuits.circuit,circuits.family):
        train=circuits.circuit.iloc[tr].to_numpy(); test=circuits.circuit.iloc[te].to_numpy()
        yield np.flatnonzero(d.circuit.isin(train)), np.flatnonzero(d.circuit.isin(test))

def inner_gain(train,validation,scores,lam,gate=None):
    means=(lam*train.yA+(1-lam)*train.yD).groupby(train.recipe_id).mean()
    fixed=means.idxmin()
    v=validation[['circuit','recipe_id']].copy()
    v['J']=lam*validation.yA+(1-lam)*validation.yD;v['score']=scores
    selected=v.loc[v.groupby('circuit').score.idxmin()].set_index('circuit')
    chosen=selected.J.copy()
    baseline=v[v.recipe_id==fixed].set_index('circuit').J
    if gate is not None:
        fixed_scores=v[v.recipe_id==fixed].set_index('circuit').score
        chosen=chosen.where(fixed_scores-selected.score>=gate,baseline)
    return float(((baseline-chosen)/baseline).mean())

def tune_fit(train,test,name,lam=.5,target='centered',group='full',encoding='full',seed=0):
    train=train.reset_index(drop=True);test=test.reset_index(drop=True)
    X=feature_frame(train,group,encoding,name=='ridge_interactions')
    Xt=feature_frame(test,group,encoding,name=='ridge_interactions')
    inner=list(circuit_splits(train,3,seed+1000))
    candidate_scores=[];candidate_predictions=[]
    for params in GRIDS[name]:
        gains=[];predictions=[]
        for a,b in inner:
            model=make_model(name,params,seed,jobs=1)
            # Train labels are computed strictly from the inner training rows.
            model.fit(X.iloc[a],targets(train.iloc[a],lam,target))
            pred=model.predict(X.iloc[b]);predictions.append(pred)
            gains.append(inner_gain(train.iloc[a],train.iloc[b],pred,lam))
        candidate_scores.append(float(np.mean(gains)))
        candidate_predictions.append(predictions)
    best=int(np.argmax(candidate_scores));params=GRIDS[name][best]
    gates=[0.,.0025,.005,.01,.02]
    gate_scores=[np.mean([inner_gain(train.iloc[a],train.iloc[b],pred,lam,gate)
                          for (a,b),pred in zip(inner,candidate_predictions[best])]) for gate in gates]
    gate=gates[int(np.argmax(gate_scores))]
    model=make_model(name,params,seed,jobs=1)
    model.fit(X,targets(train,lam,target))
    start=time.perf_counter();scores=model.predict(Xt);elapsed=time.perf_counter()-start
    return model,scores,params,candidate_scores[best],elapsed,gate

def expected_random_min(values,k):
    """Exact expected minimum of k uniformly sampled distinct recipes."""
    from math import comb
    v=np.sort(values);n=len(v)
    return sum(v[i]*comb(n-i-1,k-1)/comb(n,k) for i in range(n-k+1))

def measurements(test,train,scores,lam,name,repeat,fold,inference=0.,ks=(1,2,3,5,8),gate=0.):
    global_order=(lam*train.yA+(1-lam)*train.yD).groupby(train.recipe_id).mean().sort_values(kind='stable').index.tolist()
    records=[]; ranking=[]
    for circuit,g in test.groupby('circuit',sort=True):
        # Reindex scores through row positions, preserving no test-label ordering.
        s=scores[g.index.to_numpy()]
        g=g.copy();g['score']=s;g['J']=lam*g.yA+(1-lam)*g.yD
        order=g.sort_values(['score','recipe_id'],kind='stable')
        measured=g.set_index('recipe_id')
        fixed=float(measured.loc[global_order[0],'J']);oracle=float(g.J.min())
        corr=spearmanr(g.score,g.J).statistic if np.std(s)>1e-12 else 0.
        base=dict(circuit=circuit,family=g.family.iloc[0],model=name,repeat=repeat,fold=fold,
                  lam=lam,fixed_J=fixed,oracle_J=oracle,resyn2_J=float(measured.loc['R001','J']),
                  spearman=float(np.nan_to_num(corr)))
        feature_cost=float(g.iloc[0].get('probe_runtime_s',0)+g.iloc[0].get('baseline_runtime_s',0)+g.iloc[0].get('structural_runtime_s',0))
        for k in ks:
            selected=order.iloc[:k];chosen=selected.loc[selected.J.idxmin()]
            global_rows=measured.loc[global_order[:k]]
            for selector,j,chosen_id,cost in [
                ('learned',float(chosen.J),chosen.recipe_id,feature_cost+inference+float(selected.runtime_s.sum())),
                ('global',float(global_rows.J.min()),global_rows.J.idxmin(),float(global_rows.runtime_s.sum())),
                ('random',float(expected_random_min(g.J.to_numpy(),k)),'expected',float(k*g.runtime_s.mean())),
            ]:
                records.append(dict(**base,selector=selector,k=k,J=j,selected_recipe=chosen_id,
                                    improvement=100*(fixed-j)/fixed,improvement_resyn2=100*(base['resyn2_J']-j)/base['resyn2_J'],
                                    regret=100*(j-oracle)/oracle,cost_s=cost,
                                    oracle_hit=int(np.isclose(j,oracle,atol=1e-12,rtol=0))))
        fixed_score=float(measured.loc[global_order[0],'score'])
        choice=order.iloc[0] if fixed_score-order.iloc[0].score>=gate else measured.loc[global_order[0]]
        choice_id=order.iloc[0].recipe_id if fixed_score-order.iloc[0].score>=gate else global_order[0]
        j=float(choice.J)
        records.append(dict(**base,selector='margin',k=1,J=j,selected_recipe=choice_id,
                            improvement=100*(fixed-j)/fixed,improvement_resyn2=100*(base['resyn2_J']-j)/base['resyn2_J'],
                            regret=100*(j-oracle)/oracle,cost_s=feature_cost+inference+float(choice.runtime_s),
                            oracle_hit=int(np.isclose(j,oracle,atol=1e-12,rtol=0))))
        for rid,pred in zip(g.recipe_id,g.score):
            ranking.append(dict(circuit=circuit,recipe_id=rid,model=name,repeat=repeat,fold=fold,lam=lam,score=float(pred)))
    return records,ranking

def run_protocol(data, out, repeats=3, names=None, protocol='p1',lam=.5,target='centered',group='full',encoding='full',workers=3,reserved=None,train_limit=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    names=names or list(GRIDS)
    code_hash=hashlib.sha256(b''.join((ROOT/'src/circuit_recipe'/name).read_bytes()
                                    for name in ['evaluate.py','data.py','models.py','recipes.py'])).hexdigest()
    data_hash=hashlib.sha256(pd.util.hash_pandas_object(data,index=False).to_numpy().tobytes()).hexdigest()
    specification=dict(protocol=protocol,repeats=repeats,models=names,lam=lam,target=target,group=group,
                       encoding=encoding,reserved=reserved,train_limit=train_limit,code_sha256=code_hash,data_sha256=data_hash)
    spec_path=out/'specification.json'
    if spec_path.exists() and json.loads(spec_path.read_text())!=specification:
        raise ValueError(f'{out}: cached evaluation settings, code, or data differ. Use a new output directory.')
    spec_path.write_text(json.dumps(specification,indent=2)+'\n')
    if protocol=='p1':
        splits=[(rep,fold,a,b) for rep in range(repeats) for fold,(a,b) in enumerate(circuit_splits(data,5,rep))]
    elif protocol=='p2':
        splits=[(0,0,np.flatnonzero(data.family.eq('MCNC')),np.flatnonzero(data.family.ne('MCNC')))]
    elif protocol=='p3':
        splits=[(0,i,np.flatnonzero(data.family.ne(f)),np.flatnonzero(data.family.eq(f))) for i,f in enumerate(sorted(data.family.unique()))]
    else:
        raise ValueError(protocol)
    start=time.perf_counter()
    for rep,fold,a,b in splits:
        tag=f'r{rep}_f{fold}'; dest=out/(tag+'.csv')
        if all(p.exists() for p in [dest,out/(tag+'_rankings.csv.gz'),out/(tag+'_tuning.json'),out/(tag+'_split.json')]):
            print(out,tag,'cached',flush=True);continue
        train=data.iloc[a].reset_index(drop=True);test=data.iloc[b].reset_index(drop=True)
        if train_limit is not None:
            choices=np.random.default_rng(rep*13+fold).permutation(sorted(train.circuit.unique()))[:train_limit]
            train=train[train.circuit.isin(choices)].reset_index(drop=True)
        if reserved:
            train=train[~train.recipe_id.isin(reserved)].reset_index(drop=True)
        assert not set(train.circuit)&set(test.circuit)
        results={}
        def task(name):
            return name,tune_fit(train,test,name,lam,target,group,encoding,rep*13+fold)
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(workers) as ex:
                for future in as_completed([ex.submit(task,name) for name in names]):
                    name,result=future.result();results[name]=result
                    print(f'{out.name} repeat={rep} fold={fold} {name}: inner gain {result[3]*100:.2f}%, elapsed {time.perf_counter()-start:.0f}s',flush=True)
        records=[];rankings=[];tuning=[]
        # Family selection uses the same inner held-out circuits as hyperparameter tuning.
        chosen_name=max(names,key=lambda name:results[name][3])
        for name in names+['adaptive']:
            source=chosen_name if name=='adaptive' else name
            model,scores,params,gain,inference,gate=results[source]
            rec,rank=measurements(test,train,scores,lam,name,rep,fold,inference/test.circuit.nunique(),gate=gate)
            records+=rec;rankings+=rank
            tuning.append(dict(model=name,chosen_family=source,params=params,inner_gain=gain,margin_threshold=gate))
        pd.DataFrame(records).to_csv(dest.with_suffix('.tmp'),index=False)
        pd.DataFrame(rankings).to_csv(out/(tag+'_rankings.csv.gz'),index=False)
        (out/(tag+'_tuning.json')).write_text(json.dumps(tuning,indent=2)+'\n')
        (out/(tag+'_split.json')).write_text(json.dumps(dict(train=sorted(train.circuit.unique()),test=sorted(test.circuit.unique())),indent=2)+'\n')
        dest.with_suffix('.tmp').replace(dest)
    frames=[pd.read_csv(out/f'r{rep}_f{fold}.csv') for rep,fold,_,_ in splits]
    pd.concat(frames,ignore_index=True).to_csv(out/'selections.csv',index=False)
    metadata=dict(protocol=protocol,repeats=repeats,models=names,lambda_=lam,target=target,features=group,encoding=encoding,reserved=reserved,train_limit=train_limit,
                  dataset_sha256=digest(ROOT/'data/processed/dataset.csv'),wall_seconds=time.perf_counter()-start)
    (out/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')

def bootstrap(values,seed=2026,n=2000):
    v=np.asarray(values);rng=np.random.default_rng(seed)
    means=v[rng.integers(0,len(v),size=(n,len(v)))].mean(axis=1)
    return np.quantile(means,[.025,.975]).tolist()

def summarize(path):
    d=pd.read_csv(Path(path)/'selections.csv');rows=[]
    cols=['J','fixed_J','oracle_J','resyn2_J','improvement','improvement_resyn2','regret','cost_s','spearman','oracle_hit']
    avg=d.groupby(['model','selector','k','circuit','family'],as_index=False)[cols].mean()
    for (model,selector,k),g in avg.groupby(['model','selector','k']):
        lo,hi=bootstrap(g.improvement)
        rows.append(dict(model=model,selector=selector,k=k,n=len(g),improvement=g.improvement.mean(),ci_low=lo,ci_high=hi,
                         median=g.improvement.median(),improvement_resyn2=g.improvement_resyn2.mean(),
                         regret=g.regret.mean(),win_pct=100*g.improvement.gt(1e-8).mean(),lose_pct=100*g.improvement.lt(-1e-8).mean(),
                         cost_s=g.cost_s.mean(),spearman=g.spearman.mean(),hit_pct=100*g.oracle_hit.mean(),
                         gap_closed_pct=100*(g.fixed_J-g.J).sum()/(g.fixed_J-g.oracle_J).sum()))
    summary=pd.DataFrame(rows);summary.to_csv(Path(path)/'summary.csv',index=False)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=3);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--protocol',choices=['p1','p2','p3'],default='p1');p.add_argument('--models',nargs='+')
    p.add_argument('--target',default='centered');p.add_argument('--features',default='full');p.add_argument('--encoding',default='full')
    p.add_argument('--lam',type=float,default=.5);p.add_argument('--output',default='results/p1')
    args=p.parse_args();d=load_dataset(ROOT/'data/processed/dataset.csv')
    run_protocol(d,args.output,args.repeats,args.models,args.protocol,args.lam,args.target,args.features,args.encoding,args.workers)
    print(summarize(args.output).to_string(index=False))
