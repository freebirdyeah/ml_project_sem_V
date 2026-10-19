"""Precompute repeat 2 with the same evaluation functions on spare CPU cores.

Run while the main driver is still evaluating repeat 0. Completed outer-fold
files are identical in schema and are reused by the main driver's resume path.
"""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import pandas as pd
from threadpoolctl import threadpool_limits
from circuit_recipe.data import load_dataset
from circuit_recipe.evaluate import circuit_splits,tune_fit,measurements
from circuit_recipe.models import GRIDS
from circuit_recipe.synthesis import ROOT

def main():
    out=ROOT/'results/p1';spec=json.loads((out/'specification.json').read_text())
    code=hashlib.sha256(b''.join((ROOT/'src/circuit_recipe'/name).read_bytes()
                               for name in ['evaluate.py','data.py','models.py','recipes.py'])).hexdigest()
    d=load_dataset(ROOT/'data/processed/dataset.csv')
    data_hash=hashlib.sha256(pd.util.hash_pandas_object(d,index=False).to_numpy().tobytes()).hexdigest()
    assert spec['code_sha256']==code and spec['data_sha256']==data_hash
    assert spec['protocol']=='p1' and spec['repeats']==3 and spec['models']==list(GRIDS)
    assert spec['lam']==.5 and spec['target']=='centered' and spec['group']=='full' and spec['encoding']=='full'
    names=list(GRIDS);repeat=2;splits=list(circuit_splits(d,5,repeat));start=time.perf_counter()
    def fold_task(item):
        fold,(a,b)=item;tag=f'r{repeat}_f{fold}';dest=out/(tag+'.csv')
        if dest.exists():return tag
        train=d.iloc[a].reset_index(drop=True);test=d.iloc[b].reset_index(drop=True);results={}
        def fit(name):return name,tune_fit(train,test,name,seed=repeat*13+fold)
        with ThreadPoolExecutor(3) as ex:
            for future in as_completed([ex.submit(fit,name) for name in names]):
                name,result=future.result();results[name]=result
                print(tag,name,'inner gain',result[3],f'{time.perf_counter()-start:.0f}s',flush=True)
        chosen=max(names,key=lambda name:results[name][3]);records=[];rankings=[];tuning=[]
        for name in names+['adaptive']:
            source=chosen if name=='adaptive' else name
            model,scores,params,gain,inference,gate=results[source]
            rec,rank=measurements(test,train,scores,.5,name,repeat,fold,inference/test.circuit.nunique(),gate=gate)
            records+=rec;rankings+=rank
            tuning.append(dict(model=name,chosen_family=source,params=params,inner_gain=gain,margin_threshold=gate))
        pd.DataFrame(records).to_csv(dest.with_suffix('.tmp'),index=False)
        pd.DataFrame(rankings).to_csv(out/(tag+'_rankings.csv.gz'),index=False)
        (out/(tag+'_tuning.json')).write_text(json.dumps(tuning,indent=2)+'\n')
        (out/(tag+'_split.json')).write_text(json.dumps(dict(train=sorted(train.circuit.unique()),test=sorted(test.circuit.unique())),indent=2)+'\n')
        dest.with_suffix('.tmp').replace(dest)
        return tag
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(2) as ex:
            for future in as_completed([ex.submit(fold_task,item) for item in enumerate(splits)]):
                print('COMPLETED',future.result(),flush=True)
    (out/'parallel_schedule.json').write_text(json.dumps(dict(repeat=2,outer_workers=2,model_workers=3,
        seed_rule='repeat * 13 + fold',code_sha256=code,wall_seconds=time.perf_counter()-start),indent=2)+'\n')

if __name__=='__main__':main()
