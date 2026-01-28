# %%
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from random import shuffle, sample
from typing import List

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
        self.local_features = set(local_features)
        self.all_features = all_features


class FederatedSurvivalRF(RandomSurvivalForest):
    def __init__(self, local_models: List[LocalSurvivalRF], **kwargs):
        super().__init__(**kwargs)
        self.local_models = local_models
        self.features = local_models[0].all_features
        self.feature_names_in_ = local_models[0].feature_names_in_
        self.estimators_ = []
        for model in local_models:
            self.estimators_.extend(model.estimators_)
        self.n_estimators = len(self.estimators_)


        self.estimator_features = []
        for estimator in self.estimators_:
            tree_features = estimator.tree_.feature
            tree_features_names = set([self.features[i] for i in tree_features if i >= 0])
            self.estimator_features.append(tree_features_names)

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Federated model cannot be fit directly.")

    def update_local_models(self, local_size: int = None):
        for model in self.local_models:
            valid_estimators = []
            for estimator, feat_set in zip(self.estimators_, self.estimator_features):
                # find all estimators that only use features available locally
                if feat_set.issubset(model.local_features):
                    valid_estimators.append(estimator)
            
            if local_size is None:
                model.estimators_ = valid_estimators
            else:
                model.estimators_ = sample(valid_estimators, local_size)
            model.n_estimators = len(model.estimators_)


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

        # this implementation does not work with fully missing columns
        # X_cur = X_cur.drop(columns=cols_to_drop) 


        # nan-columns for missing features in federated setting
        # these have to exist for the trees to work in this framework
        X_cur.loc[:, cols_to_drop] = np.nan
        X_splits.append(X_cur)
        y_splits.append(Y[start:end])
    return X_splits, y_splits


X_base, Y_base = load_veterans_lung_cancer()
client_count = 3
X_splits, Y_splits = federate_data(X_base, Y_base, client_count)

Xt_splits = [OneHotEncoder().fit_transform(x_split) for x_split in X_splits]

all_features = OneHotEncoder().fit_transform(X_base).columns.to_list()

models = []
for i in range(client_count):
    X, Y = Xt_splits[i], Y_splits[i]

    local_features = X.columns[~X.isna().all()].to_list()
    model = LocalSurvivalRF(local_features, all_features).fit(X, Y)
    models.append(model)
    print(f'local prediction of client {i} with {len(model.estimators_)} estimators:')
    print(model.predict(X.iloc[:10]))

# %%

federated_model = FederatedSurvivalRF(models, n_estimators=300)
federated_model.update_local_models()
# example prediction using federated model

print(f'federated prediction with {len(federated_model.estimators_)} estimators on data from client 1:')
print(federated_model.predict(Xt_splits[0].iloc[:10]))

# preditctions on updated local models
for i, (model, data) in enumerate(zip(models, Xt_splits)):
    print(f'local prediction of client {i} with {len(model.estimators_)} estimators:')
    print(model.predict(data.iloc[:10]))