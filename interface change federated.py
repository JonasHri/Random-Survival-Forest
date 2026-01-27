# %%
import matplotlib.pyplot as plt
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.preprocessing import OneHotEncoder
from fedrf4panod.models import LocalRandomSurvivalForest, FederatedRandomSurvivalForest
from fedrf4panod import LocalRandomSurvivalForest, FederatedRandomSurvivalForest
import pandas as pd
from random import shuffle

# %%


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
        X_cur = X_cur.drop(columns=cols_to_drop)
        X_splits.append(X_cur)
        y_splits.append(Y[start:end])
    return X_splits, y_splits


X, y = load_veterans_lung_cancer()
client_count = 3
X_splits, y_splits = federate_data(X, y, client_count)

# run using filesystem for aggregation instead of in-memory
federator = FederatedRandomSurvivalForest(
    tree_aggregation_method="add", aggregation_path="./.aggregation_data/"
)

local_forests = []

for i in range(client_count):
    # client side
    # no reference to federator needed
    local_forest = LocalRandomSurvivalForest(
        name=f"client_{i}",
        aggregation_path="./.aggregation_data/",
        save_on_fit=False,
    )
    Xt = OneHotEncoder().fit_transform(X_splits[i])
    local_forest.fit(Xt, y_splits[i])
    local_forest.save_trees()
    local_forests.append(local_forest)


estimator: LocalRandomSurvivalForest = local_forests[-1]
surv_funcs = estimator.predict_survival_function(Xt.iloc[:10])
for fn in surv_funcs:
    plt.step(fn.x, fn(fn.x), where="post")
plt.ylim(0, 1)
plt.show()

previous = estimator.predict(Xt.iloc[:10])

# %% 'server' side
federator.aggregate_trees_from_clients()
federator.save_updated_models()

# %% client side same session
estimator.load_updated_trees()
# or for new session:
estimator = LocalRandomSurvivalForest.from_file(
    name="client_2", aggregation_path="./.aggregation_data/"
)


surv_funcs = estimator.predict_survival_function(
    Xt.iloc[:10]
)  # does not use updated trees
for fn in surv_funcs:
    plt.step(fn.x, fn(fn.x), where="post")
plt.ylim(0, 1)
plt.show()

updated = estimator.predict(Xt.iloc[:10])  # run without updated trees automatically
# %%
estimator.commit_local_random_forest()

# %%
# also break up 'helper' file