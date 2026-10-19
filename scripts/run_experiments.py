"""Run the paper's fixed protocols, saving each outer fold for safe resumption."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import importlib.metadata as meta
import numpy as np
import pandas as pd
from circuit_recipe.data import load_dataset
from circuit_recipe.evaluate import run_protocol,summarize
from circuit_recipe.synthesis import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-for-data',action='store_true');p.add_argument('--stage',choices=['all','main','secondary'],default='all')
    args=p.parse_args();path=ROOT/'data/processed/dataset.csv'
    while args.wait_for_data and not (path.exists() and (ROOT/'data/processed/manifest.json').exists()):time.sleep(5)
    d=load_dataset(path)
    environment=dict(python=platform.python_version(),platform=platform.platform(),logical_cpus=os.cpu_count(),
                     packages={n:meta.version(n) for n in ['numpy','pandas','scipy','scikit-learn','lightgbm','matplotlib','joblib']})
    (ROOT/'results/environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    if args.stage in ('main','all'):
        run_protocol(d,ROOT/'results/p1',repeats=3);summarize(ROOT/'results/p1')
        run_protocol(d,ROOT/'results/p2',protocol='p2');summarize(ROOT/'results/p2')
        run_protocol(d,ROOT/'results/p3',protocol='p3');summarize(ROOT/'results/p3')
        (ROOT/'results/MAIN_EXPERIMENTS_COMPLETE').write_text('Primary and transfer protocols completed.\n')
    if args.stage in ('secondary','all'):
        variants=[('features_static',dict(group='static')),('features_mapping',dict(group='mapping')),
                  ('features_probes',dict(group='probes')),('target_raw',dict(target='raw')),
                  ('target_rank',dict(target='rank')),('encoding_counts',dict(encoding='counts')),
                  ('encoding_positions',dict(encoding='positions')),('delay',dict(lam=0.)),('area',dict(lam=1.))]
        for n in [25,50,100]:variants.append((f'train_{n}',dict(train_limit=n)))
        for name,kw in variants:
            out=ROOT/'results'/name
            run_protocol(d,out,repeats=1,names=['boosting'],**kw);summarize(out)
        for size in [30,60]:
            if size==60:
                ids=[f'R{i:03d}' for i in range(60)]
            else:
                lib=d[['recipe_id','recipe_kind']].drop_duplicates();rng=np.random.default_rng(23)
                ids=['R000','R001']
                for kind,n in [('area',6),('delay',6),('mixed',6),('random',10)]:
                    candidates=sorted(lib[(lib.recipe_kind==kind)&lib.recipe_id.isin([f'R{i:03d}' for i in range(60)])].recipe_id)
                    ids+=list(rng.choice(candidates,n,replace=False))
            subset=d[d.recipe_id.isin(ids)].reset_index(drop=True);out=ROOT/f'results/library_{size}'
            run_protocol(subset,out,repeats=1,names=['boosting']);summarize(out)
        lib=d[['recipe_id','recipe_kind']].drop_duplicates();rng=np.random.default_rng(29);reserved=[]
        for kind,n in [('area',5),('delay',5),('mixed',5),('random',9)]:
            reserved+=list(rng.choice(sorted(lib[lib.recipe_kind==kind].recipe_id),n,replace=False))
        out=ROOT/'results/unseen_recipes'
        run_protocol(d,out,repeats=1,names=['boosting'],reserved=sorted(reserved));summarize(out)
        (ROOT/'results/SECONDARY_EXPERIMENTS_COMPLETE').write_text('All secondary protocols completed.\n')
    if all((ROOT/'results'/name).exists() for name in ['MAIN_EXPERIMENTS_COMPLETE','SECONDARY_EXPERIMENTS_COMPLETE']):
        (ROOT/'results/EXPERIMENTS_COMPLETE').write_text('All protocols completed.\n')

if __name__=='__main__':main()
