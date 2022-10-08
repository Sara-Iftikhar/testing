"""
================
ML Experiments
================
"""
import site
site.addsitedir(r"E:\AA\AI4Water")

import matplotlib.pyplot as plt

# %%

from load_data import _make_data
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

comparisons.fit(data=ads_df_enc, run_type="dry_run" )

# %%

_ = comparisons.compare_errors('r2', data=ads_df_enc, show=False)
plt.tight_layout()
plt.show()


# %%

_ = comparisons.compare_errors('mse', data=ads_df_enc, show=False)
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
