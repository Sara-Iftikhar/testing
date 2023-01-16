"""
==================
2. ML Experiments
==================
"""

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
# %%

from utils import make_data
from ai4water.experiments import MLRegressionExperiments

# %%

data, _, _ = make_data(encoding="ohe")

print(data.shape)

# %%

data.head()

# %%
# Initialize the experiment

comparisons = MLRegressionExperiments(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    split_random=True,
    seed=1575,
    verbosity=0,
    show=False
)

# %%
# fit/train all the models

comparisons.fit(
    data=data,
    run_type="dry_run",
    include=['XGBRegressor',
             'AdaBoostRegressor', 'LinearSVR',
             'BaggingRegressor', 'DecisionTreeRegressor',
             'HistGradientBoostingRegressor',
             'ExtraTreesRegressor', 'ExtraTreeRegressor',
             'LinearRegression', 'KNeighborsRegressor']
)

# %%
# Compare R2

_ = comparisons.compare_errors(
    'r2',
    data=data)
plt.tight_layout()
plt.show()


# %%
# Compare MSE

_ = comparisons.compare_errors(
    'mse',
    data=data,
    cutoff_val=1e7,
    cutoff_type="less"
)
plt.tight_layout()
plt.show()

# %%

_ = best_models = comparisons.compare_errors(
    'r2_score',
    cutoff_type='greater',
    cutoff_val=0.01,
    data=data
)
plt.tight_layout()
plt.show()

# %%

comparisons.taylor_plot(data=data)

# %%

comparisons.compare_edf_plots(
    data=data,
    exclude=["SGDRegressor", "KernelRidge", "PoissonRegressor"])

plt.tight_layout()
plt.show()

# %%

_ = comparisons.compare_regression_plots(data=data, figsize=(12, 14))

# %%

_ = comparisons.compare_residual_plots(data=data, figsize=(12, 14))


