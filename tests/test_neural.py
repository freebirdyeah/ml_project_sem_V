"""NN leakage and deployment invariants."""
import joblib
import numpy as np
import pytest
from circuit_recipe.data import load_dataset
from circuit_recipe.neural import CircuitMLP,circuit_frame,NEURAL_GRID
from circuit_recipe.synthesis import ROOT

@pytest.fixture(scope='module')
def sample():
    data=load_dataset(ROOT/'data/processed/dataset.csv')
    return data[data.circuit.isin(sorted(data.circuit.unique())[:8])].reset_index(drop=True)

def test_nn_inputs_one_row_per_circuit_and_no_recipe_or_labels(sample):
    X=circuit_frame(sample)
    assert X.shape==(8,50)
    assert circuit_frame(sample,'static').shape==(8,24)
    changed=sample.copy();changed[['area','delay','yA','yD']]=1234
    changed['recipe_ops']='b rw rf rs rwz rfz';changed['recipe_id']='arbitrary'
    assert circuit_frame(changed).equals(X)

def test_nn_predictions_ignore_heldout_outcomes_and_survive_save(sample,tmp_path):
    train=sample[sample.circuit.ne(sample.circuit.iloc[-1])]
    test=sample[sample.circuit.eq(sample.circuit.iloc[-1])].copy()
    model=CircuitMLP(dict(NEURAL_GRID[0],epochs=3),seed=8).fit(train)
    before=model.predict_rows(test)
    test[['yA','yD','area','delay']]=np.nan
    np.testing.assert_array_equal(before,model.predict_rows(test))
    assert len(before)==120 and np.isfinite(before).all()
    joblib.dump(model,tmp_path/'model.joblib')
    np.testing.assert_array_equal(before,joblib.load(tmp_path/'model.joblib').predict_rows(test))
    # Row order must not change which output corresponds to a recipe.
    reversed_test=test.iloc[::-1]
    np.testing.assert_array_equal(before[::-1],model.predict_rows(reversed_test))
