"""
================================
4. Hyperparameter Optimization
================================
"""

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
from ai4water.utils.utils import jsonize, TrainTestSplit, dateandtime_now
from ai4water.hyperopt import Categorical, Real, Integer, HyperOpt

from SeqMetrics import RegressionMetrics

from utils import get_dataset, evaluate_model

# %%

dataset ,  _, _ = get_dataset(encoding="ohe")
X_train, y_train = dataset.training_data()
X_test, y_test = dataset.test_data()

# %%
# Performance with default hyperparameters
# ----------------------------------------
# First, we will train the hyperparameters with default parameters

model = Model(
    model=MLP(),
                epochs=400,
                input_features=dataset.input_features,
                output_features=dataset.output_features
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

# %%
# Number of iterations will be 70 when running locally, it will be
# 40 on cloud due to computational constraints.
num_iterations = 70

# %%

MONITOR = {"mse": [], "r2_score": [], "r2": []}

seed = 1575

spliter = TrainTestSplit(seed=seed)
train_x, val_x, train_y, val_y = spliter.split_by_random(X_train, y_train)

# %%

def objective_fn(
        prefix: str = None,
        return_model: bool = False,
        epochs:int = 50,
        fit_on_all_data : bool = False,
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
        split_random=True,
        seed=seed,
        epochs=epochs,
        input_features=dataset.input_features,
        output_features=dataset.output_features,
        verbosity=0)

    # train model
    if fit_on_all_data:
        _model.fit(X_train,y_train, validation_data=(X_test, y_test))
    else:
        _model.fit(train_x, train_y, validation_data=(val_x, val_y))

    # evaluate model
    t, p = _model.predict(val_x, val_y, return_true=True,
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

    if fit_on_all_data:
        _model.predict(X_train,y_train)
        _model.predict(X_test, y_test)

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
    process_results=True, # we can turn it False if we want post-processing of results
    opt_path=f"results{SEP}{PREFIX}"
)

# %%
# We have already optimized the hyperparameters using Bayesian with 100 iterations
# Therefore, we are not running optimizer.fit online. We will, instead, load
# the results of optimization and plot them. If you however want to optimize
# the hyperparameters, you can set ``OPTIMIZE`` to True

OPTIMIZE = False

# path where hpo results are saved.
path = os.path.join(os.getcwd(), 'results', 'hpo_mlp_20221228_132336', 'hpo_results.bin')

if OPTIMIZE:
    results = optimizer.fit()
else:
    optimizer.load_results(path)

# %%

best_iteration = optimizer.best_iter()

# %%

print(f"optimized parameters are \n{optimizer.best_paras()} at {best_iteration}")

# %%

if OPTIMIZE:
    for key in ['mse']:
        print(key, np.nanmin(MONITOR[key]), np.nanargmin(MONITOR[key]))

# %%

if OPTIMIZE:
    for key in ['r2', 'r2_score']:
        print(key, np.nanmax(MONITOR[key]), np.nanargmax(MONITOR[key]))

# %%

model = objective_fn(prefix=f"{PREFIX}{SEP}best",
                     seed=seed,
                     return_model=True,
                     epochs=400,
                     fit_on_all_data=True,
                     **optimizer.best_paras())

# %%

model.evaluate(X_test, y_test, metrics=['r2', 'nse'])

# %%
    
optimizer._plot_convergence()
plt.show()

# %%

optimizer._plot_evaluations()
plt.tight_layout()
plt.show()

# %%

optimizer.plot_importance()
plt.tight_layout()
plt.show()

# %%

optimizer._plot_parallel_coords(figsize=(14, 8))
plt.tight_layout()
plt.show()

# %%

_ = plot_objective(optimizer.gpmin_results)






