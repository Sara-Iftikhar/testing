"""
===========================
Hyperparameter Optimization
===========================
"""

import site
site.addsitedir(r"E:\AA\AI4Water")

import os
import math
import numpy as np
from skopt.plots import plot_objective
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

SEP = os.sep

from typing import Union

from ai4water import Model
from ai4water.models import MLP
from ai4water.utils.utils import jsonize
from ai4water.utils.utils import dateandtime_now
from ai4water.hyperopt import Categorical, Real, Integer, HyperOpt

from SeqMetrics import RegressionMetrics

from utils import _make_data, get_data, evaluate_model, get_dataset

# %%

ads_df_enc, _, _ = _make_data()

# %%

X_train, y_train, X_test, y_test = get_data()
ds ,  _, _ = get_dataset()

# %%
# Performance with default hyperparameters
# ----------------------------------------

model = Model(
                model=MLP(),
                epochs=400,
                input_features=ds.input_features,
                output_features=ds.output_features
             )

# %%

model.fit(X_train,y_train, validation_data=(X_test, y_test))

# %%
# Training data
# --------------

train_p = model.predict(x=X_train,)

# %%

evaluate_model(y_train, train_p)

# %%
# Test data
# ----------

test_p = model.predict(x=X_test,)

# %%

evaluate_model(y_test, test_p)

# %%
# Hyperparameter Optimization
# ---------------------------

PREFIX = f"hpo_mlp_{dateandtime_now()}"
ITER = 0
num_iterations = 70

MONITOR = {"mse": [], "r2_score": [], "r2": []}

seed = 1575

# %%

def objective_fn(
        prefix: str = None,
        return_model: bool = False,
        epochs:int = 50,
        verbosity: int = -1,
        predict : bool = False,
        seed=seed,
        **suggestions
                )->Union[float, Model]:

    suggestions = jsonize(suggestions)
    global ITER

    # build model
    _model = Model(
        model=MLP(units=suggestions['units'],
                   num_layers=suggestions['num_layers'],
                   activation=suggestions['activation']),
        batch_size=suggestions["batch_size"],
        lr=suggestions["lr"],
        prefix=prefix or PREFIX,
        train_fraction=0.8,
        val_fraction=0.3,
        split_random=True,
        seed=seed,
        epochs=epochs,
        input_features=ads_df_enc.columns.tolist()[0:-1],
        output_features=ads_df_enc.columns.tolist()[-1:],
        verbosity=verbosity)

    # train model
    _model.fit(data=ads_df_enc)

    # evaluate model
    t, p = _model.predict_on_validation_data(data=ads_df_enc, return_true=True,
                                             process_results=False)
    metrics = RegressionMetrics(t, p)
    val_score = metrics.mse()

    for metric in MONITOR.keys():
        val = getattr(metrics, metric)()
        MONITOR[metric].append(val)

    # here we are evaluating model with respect to mse, therefore
    # we don't need to subtract it from 1.0
    if not math.isfinite(val_score):
        val_score = 9999

    print(f"{ITER} {val_score} {seed}")

    ITER += 1

    if predict:
        _model.predict_on_training_data(data=ads_df_enc)
        _model.predict_on_validation_data(data=ads_df_enc)
        _model.predict_on_all_data(data=ads_df_enc)

    if return_model:
        return _model

    return val_score

# %%

param_space = [
    Integer(30, 100, name="units"),
    Integer(1, 4, name="num_layers"),
    Categorical(["relu", "elu", "tanh", "sigmoid"], name="activation"),
    Real(0.00001, 0.01, name="lr"),
    Categorical([4, 8, 12, 16, 24, 32, 48, 64], name="batch_size")
                ]

# %%

x0 = [30, 1, "relu", 0.001, 8]

# %%

optimizer = HyperOpt(
    algorithm="bayes",
    objective_fn=objective_fn,
    param_space=param_space,
    x0=x0,
    num_iterations=num_iterations,
    process_results=False, # we can turn it False if we want post-processing of results
    opt_path=f"results{SEP}{PREFIX}"
)

# %%

results = optimizer.fit()

# %%

best_iteration = optimizer.best_iter()

# %%

print(f"optimized parameters are \n{optimizer.best_paras()} at {best_iteration}")

# %%

for key in ['mse']:
    print(key, np.nanmin(MONITOR[key]), np.nanargmin(MONITOR[key]))

# %%

for key in ['r2', 'r2_score']:
    print(key, np.nanmax(MONITOR[key]), np.nanargmax(MONITOR[key]))

# %%

model = objective_fn(prefix=f"{PREFIX}{SEP}best",
                     seed=seed,
                     return_model=True,
                     epochs=400,
                     verbosity=1,
                     predict=True,
                     **optimizer.best_paras())

# %%

model.evaluate_on_test_data(data=ads_df_enc, metrics=['r2', 'nse'])

# %%

optimizer._plot_convergence(save=False)

# %%

optimizer.plot_importance(save=False)
plt.tight_layout()
plt.show()

# %%

#_ = plot_objective(results)




