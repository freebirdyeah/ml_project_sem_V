"""A circuit-to-library MLP. One circuit is one training example."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from .data import feature_frame

# Fixed before examining outer-test outcomes. Inner circuit folds choose duration.
NEURAL_GRID = [dict(epochs=e, hidden=[64, 32], lr=.003, weight_decay=.03, dropout=.1)
               for e in [25, 75, 200]]

def circuit_frame(data, group='full'):
    """One row per circuit; recipe encodings and outcome columns are excluded."""
    unique=data.drop_duplicates('circuit').set_index('circuit',drop=False).sort_index()
    frame=feature_frame(unique,group=group)
    return frame[[c for c in frame if not c.startswith('r_')]]

class CircuitMLP:
    def __init__(self, params, group='full', seed=0, lam=.5):
        self.params=params; self.group=group; self.seed=seed; self.lam=lam

    def fit(self, data):
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        X=circuit_frame(data,self.group)
        self.features_=list(X)
        self.scaler_=StandardScaler().fit(X)
        values=data.assign(objective=self.lam*data.yA+(1-self.lam)*data.yD)
        Y=values.pivot(index='circuit',columns='recipe_id',values='objective').loc[X.index]
        if Y.isna().any().any():
            raise ValueError('The NN requires the complete fixed recipe library per training circuit.')
        self.recipes_=list(Y.columns)
        centered=Y.to_numpy()-Y.to_numpy().mean(axis=1,keepdims=True)
        self.mean_=centered.mean(axis=0).astype('float32')
        residual=centered-self.mean_
        self.scale_=max(float(residual.std()),1e-6)
        x=torch.tensor(self.scaler_.transform(X),dtype=torch.float32)
        y=torch.tensor(residual/self.scale_,dtype=torch.float32)
        a,b=self.params['hidden']
        self.network_=nn.Sequential(nn.Linear(x.shape[1],a),nn.ReLU(),nn.Dropout(self.params['dropout']),
                                    nn.Linear(a,b),nn.ReLU(),nn.Dropout(self.params['dropout']),
                                    nn.Linear(b,len(self.recipes_)))
        # Training starts at the global mean ranking, then learns circuit adjustments.
        nn.init.zeros_(self.network_[-1].weight);nn.init.zeros_(self.network_[-1].bias)
        optimizer=torch.optim.AdamW(self.network_.parameters(),lr=self.params['lr'],
                                    weight_decay=self.params['weight_decay'])
        self.network_.train()
        for _ in range(self.params['epochs']):
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(self.network_(x),y)
            loss.backward(); optimizer.step()
        self.network_.eval()
        self.training_loss_=float(loss.detach())
        return self

    def predict_matrix(self, data):
        X=circuit_frame(data,self.group)[self.features_]
        with torch.inference_mode():
            residual=self.network_(torch.tensor(self.scaler_.transform(X),dtype=torch.float32)).numpy()
        return pd.DataFrame(self.mean_+self.scale_*residual,index=X.index,columns=self.recipes_)

    def predict_rows(self, data):
        scores=self.predict_matrix(data)
        row=scores.index.get_indexer(data.circuit)
        col=scores.columns.get_indexer(data.recipe_id)
        if (row<0).any() or (col<0).any():
            raise ValueError('Unknown circuit or recipe in NN prediction.')
        return scores.to_numpy()[row,col]
