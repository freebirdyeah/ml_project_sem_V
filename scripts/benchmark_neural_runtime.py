"""Serial timing audit: rerun all recipes on 12 size-stratified held-out circuits."""
import json
import time
import platform
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from circuit_recipe.data import load_dataset
from circuit_recipe.synthesis import ROOT,ABC,digest

OUT=ROOT/'results/neural_runtime'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=load_dataset(ROOT/'data/processed/dataset.csv')
    circuits=d.drop_duplicates('circuit');sample=[]
    for family,g in circuits.groupby('family'):
        ordered=g.sort_values(['c_n_and','circuit'])
        sample.extend(ordered.iloc[[round(q*(len(g)-1)) for q in [.25,.5,.75]]].circuit.tolist())
    inventory=pd.read_csv(ROOT/'data/processed/circuits.csv').set_index('circuit')
    spec=dict(circuits=sample,quantiles=[.25,.5,.75],seed=0,workers=1,
              dataset_sha256=digest(ROOT/'data/processed/dataset.csv'),
              timing='Warm loaded model; serial ABC calls; sum of independently executed candidate times',
              python=platform.python_version(),processor=platform.machine())
    (OUT/'specification.json').write_text(json.dumps(spec,indent=2)+'\n')
    abc=ABC();records=[]
    for circuit in sample:
        dest=OUT/f'{circuit}.json'
        if dest.exists():records.extend(json.loads(dest.read_text())['records']);continue
        path=ROOT/inventory.loc[circuit,'path'];old=d[d.circuit.eq(circuit)]
        start=time.perf_counter();features=abc.describe(path);feature_wall=time.perf_counter()-start
        rows=pd.DataFrame([dict(circuit=circuit,**features,**r) for r in old[['recipe_id','recipe_ops']].to_dict('records')])
        measured=[]
        for r in rows.to_dict('records'):
            m=abc.measure(path,r['recipe_ops'])
            m.update(recipe_id=r['recipe_id'],J=.5*(m['area']/features['base_area']+m['delay']/features['base_delay']))
            measured.append(m)
        outcomes=pd.DataFrame(measured).set_index('recipe_id');oracle=float(outcomes.J.min())
        assert np.allclose(outcomes.sort_index().area,old.sort_values('recipe_id').area,atol=1e-8,rtol=0)
        assert np.allclose(outcomes.sort_index().delay,old.sort_values('recipe_id').delay,atol=1e-8,rtol=0)
        exhaustive=float(outcomes.runtime_s.sum()+features['baseline_runtime_s'])
        fold=next(f for f in range(5) if circuit in json.loads((ROOT/'results/neural'/f'r0_f{f}_split.json').read_text())['test'])
        train_names=json.loads((ROOT/'results/neural'/f'r0_f{fold}_split.json').read_text())['train']
        train=d[d.circuit.isin(train_names)]
        global_order=train.groupby('recipe_id')['J_0.5'].mean().sort_values(kind='stable').index.tolist()
        local=[]
        def add(name,k,order,cost):
            candidates=outcomes.loc[order[:k]];j=float(candidates.J.min())
            local.append(dict(circuit=circuit,family=inventory.loc[circuit,'family'],model=name,k=k,
                              J=j,oracle_J=oracle,regret=100*(j-oracle)/oracle,cost_s=cost,
                              exhaustive_s=exhaustive,speedup=exhaustive/cost,
                              selected_recipe=candidates.J.idxmin()))
        add('exhaustive',120,outcomes.index.tolist(),exhaustive)
        for k in [1,2,3,5,8]:
            cost=float(outcomes.loc[global_order[:k]].runtime_s.sum())
            if k>1:cost+=features['baseline_runtime_s']
            add('global',k,global_order,cost)
        for group in ['full','static']:
            model=joblib.load(ROOT/'results/neural'/f'r0_f{fold}_nn_{group}.joblib')
            # Median repeated prediction includes feature preprocessing, model already loaded.
            timings=[]
            for _ in range(7):
                t=time.perf_counter();scores=model.predict_rows(rows);timings.append(time.perf_counter()-t)
            prediction=float(np.median(timings))
            order=rows.assign(score=scores).sort_values(['score','recipe_id']).recipe_id.tolist()
            feature_cost=feature_wall if group=='full' else features['baseline_runtime_s']+features['structural_runtime_s']
            for k in [1,2,3,5,8]:
                add('nn_'+group,k,order,float(feature_cost+prediction+outcomes.loc[order[:k]].runtime_s.sum()))
        records.extend(local)
        dest.write_text(json.dumps(dict(records=local,measurements=measured,features=features,
                                       feature_wall_s=feature_wall,fold=fold),indent=2)+'\n')
        print(circuit,'exhaustive',round(exhaustive,2),'s',flush=True)
    frame=pd.DataFrame(records);frame.to_csv(OUT/'measurements.csv',index=False)
    summary=frame.groupby(['model','k']).agg(circuits=('circuit','nunique'),cost_s=('cost_s','mean'),
              exhaustive_s=('exhaustive_s','mean'),regret=('regret','mean'),median_speedup=('speedup','median')).reset_index()
    summary['ratio_of_mean_times']=summary.exhaustive_s/summary.cost_s
    summary['time_saved_pct']=100*(1-summary.cost_s/summary.exhaustive_s)
    summary.to_csv(OUT/'summary.csv',index=False);print(summary.to_string(index=False),flush=True)
    (OUT/'COMPLETE').write_text('All 1,440 recipe runs reproduced their recorded area and delay.\n')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
