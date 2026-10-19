"""Nested NN evaluation on exactly the original outer circuit splits."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from circuit_recipe.data import load_dataset
from circuit_recipe.evaluate import circuit_splits,inner_gain,measurements,summarize
from circuit_recipe.neural import CircuitMLP,NEURAL_GRID
from circuit_recipe.synthesis import ROOT,digest

OUT=ROOT/'results/neural'

def tune(train,group,seed):
    inner=list(circuit_splits(train,3,seed+1000));trials=[]
    for params in NEURAL_GRID:
        gains=[];seconds=time.perf_counter()
        for a,b in inner:
            model=CircuitMLP(params,group,seed).fit(train.iloc[a])
            gains.append(inner_gain(train.iloc[a],train.iloc[b],model.predict_rows(train.iloc[b]),.5))
        trials.append(dict(params=params,inner_gain=float(np.mean(gains)),
                           fold_gains=gains,seconds=time.perf_counter()-seconds))
    best=max(trials,key=lambda t:t['inner_gain'])
    start=time.perf_counter();model=CircuitMLP(best['params'],group,seed).fit(train)
    return model,dict(chosen=best,trials=trials,refit_seconds=time.perf_counter()-start)

def evaluate(repeats=3):
    OUT.mkdir(parents=True,exist_ok=True)
    data=load_dataset(ROOT/'data/processed/dataset.csv')
    source_files=['src/circuit_recipe/neural.py','scripts/run_neural.py',
                  'src/circuit_recipe/data.py','src/circuit_recipe/evaluate.py']
    spec=dict(dataset_sha256=digest(ROOT/'data/processed/dataset.csv'),grid=NEURAL_GRID,
              repeats=repeats,groups=['full','static'],torch=torch.__version__,
              code_sha256=hashlib.sha256(b''.join((ROOT/f).read_bytes() for f in source_files)).hexdigest())
    p=OUT/'specification.json'
    if p.exists() and json.loads(p.read_text())!=spec:raise ValueError('NN cache specification changed')
    p.write_text(json.dumps(spec,indent=2)+'\n')
    start=time.perf_counter();parts=[]
    for rep in range(repeats):
        for fold,(a,b) in enumerate(circuit_splits(data,5,rep)):
            tag=f'r{rep}_f{fold}'
            train=data.iloc[a].reset_index(drop=True);test=data.iloc[b].reset_index(drop=True)
            original=json.loads((ROOT/'results/p1'/f'{tag}_split.json').read_text())
            split=dict(train=sorted(train.circuit.unique()),test=sorted(test.circuit.unique()))
            assert split==original
            (OUT/f'{tag}_split.json').write_text(json.dumps(split,indent=2)+'\n')
            for group in spec['groups']:
                name='nn_'+group;dest=OUT/f'{tag}_{name}.csv'
                if dest.exists():parts.append(pd.read_csv(dest));continue
                model,tuning=tune(train,group,rep*13+fold)
                predictions=model.predict_rows(test)
                # Include preprocessing and a separate prediction call for each circuit.
                start_pred=time.perf_counter()
                for _,g in test.groupby('circuit'):model.predict_rows(g)
                inference=(time.perf_counter()-start_pred)/test.circuit.nunique()
                timed=test.copy()
                if group=='static':timed['probe_runtime_s']=0.
                records,rankings=measurements(timed,train,predictions,.5,name,rep,fold,inference)
                # Comparing >=2 normalized objectives requires baseline mapping for global lists.
                for r in records:
                    if r['selector'] in ['global','random'] and r['k']>1:
                        r['cost_s']+=float(test.loc[test.circuit.eq(r['circuit']),'baseline_runtime_s'].iloc[0])
                pd.DataFrame(rankings).to_csv(OUT/f'{tag}_{name}_rankings.csv.gz',index=False)
                (OUT/f'{tag}_{name}_tuning.json').write_text(json.dumps(tuning,indent=2)+'\n')
                joblib.dump(model,OUT/f'{tag}_{name}.joblib')
                pd.DataFrame(records).to_csv(dest,index=False);parts.append(pd.DataFrame(records))
                print(tag,name,'inner gain',round(tuning['chosen']['inner_gain']*100,3),
                      'epochs',tuning['chosen']['params']['epochs'],'elapsed',round(time.perf_counter()-start,1),flush=True)
    pd.concat(parts,ignore_index=True).to_csv(OUT/'selections.csv',index=False)
    print(summarize(OUT).to_string(index=False),flush=True)
    (OUT/'COMPLETE').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-start),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    with threadpool_limits(limits=1):evaluate(args.repeats)
