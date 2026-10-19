"""Validate NN folds, saved rankings, timing labels, and deployment artifacts."""
import json
import joblib
import numpy as np
import pandas as pd
from circuit_recipe.data import load_dataset
from circuit_recipe.synthesis import ROOT,digest

def main():
    d=load_dataset(ROOT/'data/processed/dataset.csv');folds=0;predictions=0
    for rep in range(3):
        seen=[]
        for fold in range(5):
            tag=f'r{rep}_f{fold}';base=ROOT/'results/neural'
            split=json.loads((base/f'{tag}_split.json').read_text())
            assert not set(split['train'])&set(split['test'])
            assert split==json.loads((ROOT/'results/p1'/f'{tag}_split.json').read_text())
            seen.extend(split['test'])
            test=d[d.circuit.isin(split['test'])].reset_index(drop=True)
            for group in ['full','static']:
                name='nn_'+group;model=joblib.load(base/f'{tag}_{name}.joblib')
                saved=pd.read_csv(base/f'{tag}_{name}_rankings.csv.gz').set_index(['circuit','recipe_id'])
                expected=saved.loc[pd.MultiIndex.from_frame(test[['circuit','recipe_id']]),'score'].to_numpy()
                np.testing.assert_allclose(model.predict_rows(test),expected,rtol=0,atol=1e-14)
                predictions+=len(test);folds+=1
        assert len(seen)==len(set(seen))==205
    pairs=0
    for p in (ROOT/'results/neural_runtime').glob('*.json'):
        if p.name=='specification.json':continue
        saved=json.loads(p.read_text());circuit=p.stem
        old=d[d.circuit.eq(circuit)].set_index('recipe_id')
        new=pd.DataFrame(saved['measurements']).set_index('recipe_id').loc[old.index]
        np.testing.assert_allclose(old[['area','delay']],new[['area','delay']],rtol=0,atol=1e-10)
        for key,value in saved['features'].items():
            if key.startswith(('c_','p_')) or key in ['base_area','base_delay']:
                np.testing.assert_allclose(value,old.iloc[0][key],rtol=1e-12,atol=1e-12)
        pairs+=len(new)
    assert pairs==1440
    for name in ['neural_model.joblib','neural_static_model.joblib']:
        model=joblib.load(ROOT/'results'/name)
        assert model['dataset_sha256']==digest(ROOT/'data/processed/dataset.csv')
        assert len(model['recipes'])==120
    out=dict(nn_outer_fits=folds,ranking_values_reproduced=predictions,identical_classical_outer_splits=True,
             test_circuits_per_repeat=205,serial_pairs_reproduced=pairs,serial_features_reproduced=True,
             dataset_sha256=digest(ROOT/'data/processed/dataset.csv'))
    (ROOT/'results/neural_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
