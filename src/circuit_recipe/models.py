"""Small, predeclared model grids; selection is tuned on held-out circuits."""
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor

GRIDS = {
    'ridge': [{'alpha':10.},{'alpha':100.}],
    'ridge_interactions': [{'alpha':10.},{'alpha':100.}],
    'extra_trees': [{'min_samples_leaf':5},{'min_samples_leaf':15}],
    'random_forest': [{'min_samples_leaf':5},{'min_samples_leaf':15}],
    'boosting': [{'num_leaves':15},{'num_leaves':31}],
}

def make_model(name, params, seed=0, jobs=2):
    if name.startswith('ridge'):
        return make_pipeline(StandardScaler(),Ridge(**params))
    if name=='extra_trees':
        return ExtraTreesRegressor(n_estimators=80,max_features=.3,max_depth=16,n_jobs=jobs,random_state=seed,**params)
    if name=='random_forest':
        return RandomForestRegressor(n_estimators=80,max_features=.3,max_depth=16,n_jobs=jobs,random_state=seed,**params)
    if name=='boosting':
        return LGBMRegressor(n_estimators=180,learning_rate=.05,min_child_samples=20,
                            reg_lambda=1.,verbosity=-1,n_jobs=jobs,random_state=seed,**params)
    raise ValueError(name)
