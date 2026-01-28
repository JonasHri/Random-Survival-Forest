# %%
import matplotlib.pyplot as plt
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from random import shuffle

# %%


class LocalSurvivalRF(RandomSurvivalForest):
    def __init__(
        self,
        local_features: list,
        all_features: list,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0,
        max_features="sqrt",
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
        low_memory=False
    ):
        super().__init__(
            n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
            low_memory=low_memory,
        )
        self.local_features = local_features
        self.all_features = all_features


def federate_data(
    X: pd.DataFrame, Y: pd.DataFrame, n_clients: int, drop_cols_precentage: float = 0.3
):
    idx = list(X.index)
    shuffle(idx)
    X = X.loc[idx].reset_index(drop=True)
    Y = Y[idx]
    X_splits = []
    y_splits = []
    n_samples = len(X)
    split_size = n_samples // n_clients
    for i in range(n_clients):
        start = i * split_size
        end = (i + 1) * split_size if i != n_clients - 1 else n_samples
        X_cur = X.iloc[start:end]
        cols_to_drop = X_cur.columns.to_series().sample(
            frac=drop_cols_precentage, random_state=42 + i
        )
        # X_cur = X_cur.drop(columns=cols_to_drop)
        X_cur.loc[:, cols_to_drop] = np.nan
        X_splits.append(X_cur)
        y_splits.append(Y[start:end])
    return X_splits, y_splits


X, y = load_veterans_lung_cancer()
client_count = 3
X_splits, Y_splits = federate_data(X, y, client_count)

models = []
Xt_splits = [OneHotEncoder().fit_transform(x_split) for x_split in X_splits]

all_features = OneHotEncoder().fit_transform(X).columns.to_list()


for i in range(client_count):
    X, Y = Xt_splits[i], Y_splits[i]

    local_features = X.columns[~X.isna().all()].to_list()

    model = LocalSurvivalRF(local_features, all_features).fit(X, Y)

    models.append(model)

models

# %%

model.estimators_
