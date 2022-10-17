"""
===================
Prediction Analysis
===================
"""
import site
site.addsitedir(r"E:\AA\AI4Water")
site.addsitedir(r"E:\AA\easy_mpl")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ai4water.postprocessing import prediction_distribution_plot

from utils import get_dataset, get_data, get_fitted_model, evaluate_model


# %%

X_train, y_train, X_test, y_test = get_data()

# %%

dataset, _, _ = get_dataset()

# %%

model = get_fitted_model()

# %%

test_p = model.predict(x=X_test)

# %%

evaluate_model(y_test, test_p)

# %%
# Feature Interaction
# --------------------

_ = model.prediction_analysis(
    x = pd.DataFrame(X_test, columns=dataset.input_features),
    features = ['Adsorption_time (min)', 'calcination (min)'],
    feature_names = ['Adsorption_time (min)', 'calcination (min)'],
    grid_types=["percentile", "percentile"],
    num_grid_points=[6,6],
    annotate_kws={'annotate_fontsize':15,
                  'annotate_colors':np.array([['black', 'black', 'black', 'black'],
                                              ['black', 'black', 'black', 'black'],
                                              ['black', 'black', 'black', 'black'],
                                              ['black', 'black', 'black', 'black'],
                                              ['white', 'black', 'black', 'black']])}
    )


# %%
_ = model.prediction_analysis(
    x = pd.DataFrame(X_test, columns=dataset.input_features),
    features = ['Adsorption_time (min)', 'initial concentration'],
    feature_names = ['Adsorption_time (min)', 'initial concentration'],
    grid_types=["percentile", "percentile"],
    num_grid_points=[6, 6],
    annotate_kws={'annotate_fontsize': 15,
                  'annotate_colors': np.array([['black', 'black', 'black', 'black'],
                                               ['black', 'black', 'black', 'black'],
                                               ['black', 'black', 'black', 'black'],
                                               ['black', 'black', 'black', 'black'],
                                               ['white', 'black', 'black', 'black']])}
)

# %%
_ = model.prediction_analysis(
    x = pd.DataFrame(X_test, columns=dataset.input_features),
    features = ['Adsorption_time (min)', 'solution pH'],
    feature_names = ['Adsorption_time (min)', 'solution pH'],
    grid_types=["percentile", "percentile"],
    num_grid_points=[6, 6],
    annotate_kws={'annotate_fontsize': 15,
                  'annotate_colors': np.array([['black', 'black', 'black', 'black'],
                                               ['black', 'black', 'black', 'black'],
                                               ['black', 'black', 'black', 'black'],
                                               ['black', 'black', 'black', 'black'],
                                               ['white', 'black', 'black', 'black']])}
)


# %%
_ = model.prediction_analysis(
    x = pd.DataFrame(X_test, columns=dataset.input_features),
    features = ['Adsorption_time (min)', 'adsorbent loading '],
    feature_names = ['Adsorption_time (min)', 'adsorbent loading '],
    grid_types=["percentile", "percentile"],
    num_grid_points=[6, 6],
    annotate_kws={'annotate_fontsize': 15,
                  'annotate_colors': np.array([['black', 'black', 'black', 'black', 'black'],
                                               ['black', 'black', 'black', 'black', 'black'],
                                               ['black', 'black', 'black', 'black', 'white'],
                                               ['black', 'black', 'black', 'black', 'white'],
                                               ['black', 'black', 'black', 'black', 'black']])
                                 }
)

# %%
# Prediction Distribution
# ------------------------

# ax, df = prediction_distribution_plot(mode='regression',
#                              inputs=pd.DataFrame(X_test, columns=dataset.input_features),
#                              prediction=test_p,
#                              feature='Adsorption_time (min)',
#                              feature_name='Adsorption_time (min)')
#
# preds = {}
# for interval in df['display_column']:
#     st, en = interval.split(',')  # todo
#     df1 = pd.DataFrame(X_test, columns=dataset.input_features)
#     df1['target'] = test_p
#     df1 = df1[['Adsorption_time (min)', 'target']]
#     df1 = df1[df1['Adsorption_time (min)']>0 & df1['Adsorption_time (min)']<20]
#     preds[interval]  = df1['target'].values