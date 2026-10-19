"""Small integration run on supplied pilot data before the full experiment."""
import time
from threadpoolctl import threadpool_limits
from circuit_recipe.data import load_dataset
from circuit_recipe.evaluate import tune_fit
from circuit_recipe.models import GRIDS

d=load_dataset('data/raw/dataset_pilot_v1.csv')
circuits=d.circuit.unique();train=d[d.circuit.isin(circuits[:160])];test=d[d.circuit.isin(circuits[160:])]
with threadpool_limits(limits=1):
    for name in GRIDS:
        t=time.perf_counter();result=tune_fit(train,test,name)
        print(name,'seconds',round(time.perf_counter()-t,2),'inner gain',result[3],flush=True)
