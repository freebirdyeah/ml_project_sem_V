"""Checks for label leakage, feature leakage, recipe order, and budget arithmetic."""
from itertools import combinations
import numpy as np
import pandas as pd
import pytest
from circuit_recipe.data import load_dataset,feature_frame,targets
from circuit_recipe.recipes import encode_recipe,extend
from circuit_recipe.evaluate import circuit_splits,expected_random_min,inner_gain,measurements,run_protocol
from circuit_recipe.synthesis import ROOT,strip_exdc_export

@pytest.fixture(scope='module')
def data():
    return load_dataset(ROOT/'data/raw/dataset_pilot_v1.csv')

def test_grouped_splits_keep_circuits_disjoint(data):
    test_circuits=[]
    for a,b in circuit_splits(data,5,7):
        assert not set(data.iloc[a].circuit)&set(data.iloc[b].circuit)
        assert len(b)%60==0
        test_circuits.extend(data.iloc[b].circuit.unique())
    assert len(set(test_circuits))==len(test_circuits)==205

def test_features_exclude_labels_and_identity(data):
    d=data.iloc[:120]
    X=feature_frame(d)
    assert X.shape[1]==194
    assert not {'area','delay','yA','yD','J_0.5','circuit','recipe_id','aig_nodes','aig_depth','runtime_s'}&set(X)
    assert sum(c.startswith('p_') for c in X)==22
    assert sum(c.startswith('m_') for c in X)==4
    changed=d.copy()
    changed[['area','delay','yA','yD']]=12345
    pd.testing.assert_frame_equal(X,feature_frame(changed))

def test_centering_and_ranks_only_use_given_recipe_rows(data):
    subset=data[data.recipe_id.isin(['R000','R001','R002'])].copy()
    y=targets(subset)
    assert np.allclose(pd.Series(y).groupby(subset.circuit.to_numpy()).mean(),0,atol=1e-12)
    assert np.all((targets(subset,kind='rank')>=0)&(targets(subset,kind='rank')<=1))

def test_sequence_order_is_encoded():
    a=encode_recipe('rw b rf rs rwz rfz');b=encode_recipe('b rw rf rs rwz rfz')
    assert all(a[f'r_cnt_{o}']==b[f'r_cnt_{o}'] for o in ['rw','b','rf','rs','rwz','rfz'])
    assert a!=b and a['r_tr_rw_b']!=b['r_tr_rw_b']
    assert len(a)==144

def test_random_budget_expectation_matches_enumeration():
    values=[.8,.9,1.1,1.2,1.5]
    for k in range(1,6):
        expected=np.mean([min(c) for c in combinations(values,k)])
        assert expected_random_min(values,k)==pytest.approx(expected)

def test_recipe_extension_preserves_pilot():
    pilot=pd.read_csv(ROOT/'data/raw/recipes.csv').fillna('')
    final=extend(ROOT/'data/raw/recipes.csv').fillna('')
    pd.testing.assert_frame_equal(pilot,final.iloc[:60].reset_index(drop=True))
    assert len(final)==final.ops.nunique()==120
    assert final.kind.value_counts().to_dict()=={'random':46,'area':24,'delay':24,'mixed':24,'known':2}

def test_validation_rejects_bad_normalization(data,tmp_path):
    broken=data.iloc[:120].copy();broken.loc[0,'yA']+=.1
    dest=tmp_path/'bad.csv';broken.to_csv(dest,index=False)
    with pytest.raises(ValueError,match='normalized'):
        load_dataset(dest)

def test_inner_fixed_recipe_uses_training_labels():
    train=pd.DataFrame(dict(circuit=['a','a'],recipe_id=['r0','r1'],yA=[1.,2.],yD=[1.,2.]))
    test=pd.DataFrame(dict(circuit=['b','b'],recipe_id=['r0','r1'],yA=[2.,1.],yD=[2.,1.]))
    assert inner_gain(train,test,[1.,0.],.5)==pytest.approx(.5)
    assert inner_gain(train,test,[1.,0.],.5,gate=2.)==pytest.approx(0.)

def test_top_k_uses_predictions_then_only_shortlist_labels(data):
    test=data[data.circuit=='c1908'].reset_index(drop=True)
    train=data[data.circuit!='c1908'].reset_index(drop=True)
    scores=np.full(len(test),100.)
    scores[test.recipe_id.eq('R000')]=0.
    scores[test.recipe_id.eq('R037')]=1.
    records,_=measurements(test,train,scores,.5,'test',0,0,ks=(1,2))
    selected=[r for r in records if r['selector']=='learned']
    assert selected[0]['selected_recipe']=='R000'
    allowed=test[test.recipe_id.isin(['R000','R037'])]
    assert selected[1]['J']==pytest.approx(allowed['J_0.5'].min())

def test_cache_rejects_changed_experiment(data,tmp_path):
    (tmp_path/'specification.json').write_text('{"different": true}')
    with pytest.raises(ValueError,match='cached evaluation'):
        run_protocol(data,tmp_path,repeats=1,names=['boosting'])

def test_exdc_export_retains_main_function(tmp_path):
    p=tmp_path/'mapped.blif'
    main='.model f\n.inputs a b\n.outputs y\n.gate AND2_X1 A1=a A2=b ZN=y\n'
    p.write_text(main+'.exdc\n.inputs a b\n.outputs y\n.names a y\n1 1\n.end\n')
    assert strip_exdc_export(p)
    assert p.read_text()==main+'.end\n'
    assert not strip_exdc_export(p)
