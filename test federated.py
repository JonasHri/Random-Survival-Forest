# %%
import matplotlib.pyplot as plt
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.preprocessing import OneHotEncoder
from fedrf4panod.federated_random_survival_forest import LocalRandomSurvivalForest, FederatedRandomSurvivalForest
import pandas as pd
from random import shuffle
# %%

def federate_data(X: pd.DataFrame, Y: pd.DataFrame, n_clients: int, drop_cols_precentage: float = 0.3):
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
        cols_to_drop = X_cur.columns.to_series().sample(frac=drop_cols_precentage, random_state=42+i)
        X_cur = X_cur.drop(columns=cols_to_drop)
        X_splits.append(X_cur)
        y_splits.append(Y[start:end])
    return X_splits, y_splits

X, y = load_veterans_lung_cancer()
client_count = 3
X_splits, y_splits = federate_data(X, y, client_count)

federator = FederatedRandomSurvivalForest(tree_aggregation_method='add')

local_forests = []

for i in range(client_count):
    local_forest = LocalRandomSurvivalForest(federator)
    Xt = OneHotEncoder().fit_transform(X_splits[i])
    local_forest.fit(Xt, y_splits[i])
    local_forest.commit_local_random_forest()
    local_forests.append(local_forest)


estimator: LocalRandomSurvivalForest = local_forests[-1]
surv_funcs = estimator.predict_survival_function(Xt.iloc[:10])
for fn in surv_funcs:
    plt.step(fn.x, fn(fn.x), where="post")
plt.ylim(0, 1)
plt.show()

previous = estimator.predict(Xt.iloc[:10])

# %%
estimator.get_updated_trees_from_federated_model()
surv_funcs = estimator.predict_survival_function(Xt.iloc[:10])
for fn in surv_funcs:
    plt.step(fn.x, fn(fn.x), where="post")
plt.ylim(0, 1)
plt.show()  

updated = estimator.predict(Xt.iloc[:10], use_updated_federated_model=True)
# %%
estimator.commit_local_random_forest()

# %%

