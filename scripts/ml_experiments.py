"""
================
ML Experiments
================
"""
import site
site.addsitedir(r"E:\AA\AI4Water")

import matplotlib.pyplot as plt

# %%

from utils import _make_data
from ai4water.experiments import MLRegressionExperiments

# %%

ads_df_enc, _, _ = _make_data()

ads_df_enc.head()

# %%

comparisons = MLRegressionExperiments(
    input_features=ads_df_enc.columns.tolist()[0:-1],
    output_features=ads_df_enc.columns.tolist()[-1:],
    split_random=True,
    seed=1509,
    verbosity=0
)

# %%

comparisons.fit(data=ads_df_enc, run_type="dry_run",
                include=['XGBRegressor',
                          'AdaBoostRegressor', 'LinearSVR',
                         'BaggingRegressor', 'DecisionTreeRegressor',
                         'HistGradientBoostingRegressor',
                         'ExtraTreesRegressor', 'ExtraTreeRegressor',
                         'LinearRegression', 'KNeighborsRegressor']
                )

# %%

_ = comparisons.compare_errors('r2', data=ads_df_enc, show=False)
plt.tight_layout()
plt.show()


# %%

_ = comparisons.compare_errors('mse', data=ads_df_enc,
                               cutoff_val=1e7, cutoff_type="less",
                               show=False)
plt.tight_layout()
plt.show()

# %%

_ = best_models = comparisons.compare_errors('r2_score',
                                         cutoff_type='greater',
                                         cutoff_val=0.01, data=ads_df_enc,
                                             show=False)
plt.tight_layout()
plt.show()

# %%

comparisons.taylor_plot(data=ads_df_enc)

# %%

comparisons.compare_edf_plots(data=ads_df_enc,
                              exclude=["SGDRegressor", "KernelRidge", "PoissonRegressor"],
                              show=False)
plt.tight_layout()
plt.show()

# %%

_ = comparisons.compare_regression_plots(data=ads_df_enc, figsize=(12, 14))

# %%

_ = comparisons.compare_residual_plots(data=ads_df_enc, figsize=(12, 14))


