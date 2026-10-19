"""Train a deployment artifact or choose recipes for a supplied circuit."""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from .data import load_dataset,feature_frame
from .evaluate import tune_fit,circuit_splits,inner_gain
from .models import GRIDS
from .synthesis import ABC,ROOT,digest,sweep

def train(output,lam=.5,family='classical'):
    if not 0<=lam<=1:
        raise ValueError('Area weight must be between 0 and 1.')
    d=load_dataset(ROOT/'data/processed/dataset.csv');fits={}
    if family.startswith('nn_'):
        from .neural import CircuitMLP,NEURAL_GRID
        import numpy as np
        group=family.removeprefix('nn_');trials=[]
        for params in NEURAL_GRID:
            gains=[]
            for a,b in circuit_splits(d,3,1000):
                model=CircuitMLP(params,group,0,lam).fit(d.iloc[a])
                gains.append(inner_gain(d.iloc[a],d.iloc[b],model.predict_rows(d.iloc[b]),lam))
            trials.append(dict(params=params,inner_gain=float(np.mean(gains))))
        best=max(trials,key=lambda t:t['inner_gain'])
        model=CircuitMLP(best['params'],group,0,lam).fit(d)
        recipes=d[['recipe_id','recipe_ops']].drop_duplicates().sort_values('recipe_id')
        artifact=dict(model=model,family=family,params=best['params'],lambda_=lam,
                      features=model.features_,recipes=recipes.to_dict('records'),
                      best_fixed=(lam*d.yA+(1-lam)*d.yD).groupby(d.recipe_id).mean().idxmin(),
                      library_sha256=digest(ROOT/'data/raw/nangate45.lib'),
                      dataset_sha256=digest(ROOT/'data/processed/dataset.csv'),tuning=trials)
        Path(output).parent.mkdir(parents=True,exist_ok=True);joblib.dump(artifact,output)
        print(json.dumps({k:v for k,v in artifact.items() if k not in ['model','recipes','features']},indent=2))
        return
    with threadpool_limits(limits=1):
        for name in GRIDS:
            fits[name]=tune_fit(d,d.iloc[:120],name,lam=lam)
            print(name,'inner improvement',fits[name][3],flush=True)
    name=max(fits,key=lambda x:fits[x][3]);fit=fits[name]
    recipes=d[['recipe_id','recipe_ops']].drop_duplicates().sort_values('recipe_id')
    artifact=dict(model=fit[0],family=name,params=fit[2],margin_threshold=fit[5],lambda_=lam,
                  features=list(feature_frame(d.iloc[:120],interactions=name=='ridge_interactions')),
                  recipes=recipes.to_dict('records'),
                  best_fixed=(lam*d.yA+(1-lam)*d.yD).groupby(d.recipe_id).mean().idxmin(),
                  library_sha256=digest(ROOT/'data/raw/nangate45.lib'),
                  dataset_sha256=digest(ROOT/'data/processed/dataset.csv'))
    Path(output).parent.mkdir(parents=True,exist_ok=True);joblib.dump(artifact,output)
    print(json.dumps({k:v for k,v in artifact.items() if k not in ['model','recipes','features']},indent=2))

def select(circuit,artifact_path,verify=0,netlist=None):
    if not 0<=verify<=120:
        raise ValueError('Verification budget must be between 0 and 120.')
    start=time.perf_counter();artifact=joblib.load(artifact_path);load_seconds=time.perf_counter()-start
    abc=ABC()
    if digest(abc.library)!=artifact['library_sha256']:
        raise ValueError('The mapping library differs from the trained model library.')
    start=time.perf_counter()
    if artifact['family']=='nn_static':
        from .aig import features as graph_features
        base=abc.measure(circuit)
        with tempfile.TemporaryDirectory(prefix='recipe-static-') as tmp:
            aig=Path(tmp)/'circuit.aig'
            abc.execute(abc.read(circuit)+f'; write_aiger {aig}')
            features={f'c_{k}':v for k,v in graph_features(aig).items()}
        features.update(base_area=base['area'],base_delay=base['delay'])
    else:
        features=abc.describe(circuit)
    feature_seconds=time.perf_counter()-start
    rows=[dict(circuit=Path(circuit).stem,**features,**r) for r in artifact['recipes']]
    frame=pd.DataFrame(rows)
    start=time.perf_counter()
    if artifact['family'].startswith('nn_'):
        scores=artifact['model'].predict_rows(frame)
    else:
        X=feature_frame(frame,interactions=artifact['family']=='ridge_interactions')
        scores=artifact['model'].predict(X[artifact['features']])
    prediction_seconds=time.perf_counter()-start
    frame['predicted_relative_score']=scores
    ordered=frame.sort_values(['predicted_relative_score','recipe_id'])
    result=dict(circuit=str(Path(circuit).resolve()),model=artifact['family'],lambda_=artifact['lambda_'],
                baseline=dict(area=features['base_area'],delay=features['base_delay']),
                recommendation=ordered[['recipe_id','recipe_ops','predicted_relative_score']].head(max(verify,1)).to_dict('records'),
                timing=dict(model_load_s=load_seconds,features_s=feature_seconds,prediction_s=prediction_seconds))
    if verify:
        measured=[];start=time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='recipe-verify-') as tmp:
            for r in ordered.head(verify).to_dict('records'):
                dest=Path(tmp)/(r['recipe_id']+'.blif')
                m=abc.measure(circuit,r['recipe_ops'],dest)
                m.update(recipe_id=r['recipe_id'],recipe_ops=r['recipe_ops'])
                m.update(yA=m['area']/features['base_area'],yD=m['delay']/features['base_delay'])
                m['J']=artifact['lambda_']*m['yA']+(1-artifact['lambda_'])*m['yD']
                measured.append(m)
            best=min(measured,key=lambda m:m['J']);result.update(measurements=measured,best=best)
            if netlist:
                Path(netlist).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(Path(tmp)/(best['recipe_id']+'.blif'),netlist)
                result['netlist']=str(Path(netlist).resolve())
        result['timing']['verification_s']=time.perf_counter()-start
    elif netlist:
        raise ValueError('--netlist requires --verify >= 1.')
    result['timing']['total_warm_s']=sum(result['timing'].get(k,0.) for k in ['features_s','prediction_s','verification_s'])
    print(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('sweep');s.add_argument('--workers',type=int,default=4)
    t=sub.add_parser('train');t.add_argument('--output',default='results/neural_model.joblib');t.add_argument('--lam',type=float,default=.5)
    t.add_argument('--family',choices=['nn_full','nn_static','classical'],default='nn_full')
    r=sub.add_parser('select');r.add_argument('circuit');r.add_argument('--model',default='results/neural_model.joblib')
    r.add_argument('--verify',type=int,default=0);r.add_argument('--netlist')
    a=p.parse_args()
    if a.command=='sweep':sweep(a.workers)
    elif a.command=='train':train(a.output,a.lam,a.family)
    else:select(a.circuit,a.model,a.verify,a.netlist)

if __name__=='__main__':main()
