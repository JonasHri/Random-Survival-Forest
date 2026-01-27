# %%
import matplotlib.pyplot as plt
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
# %%
X, y = load_veterans_lung_cancer()
Xt = OneHotEncoder().fit_transform(X)

estimator = RandomSurvivalForest().fit(Xt, y)
surv_funcs = estimator.predict_survival_function(Xt.iloc[:10])
for fn in surv_funcs:
    plt.step(fn.x, fn(fn.x), where="post")
plt.ylim(0, 1)
plt.show()  