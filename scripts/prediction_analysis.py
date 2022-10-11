"""
===================
Prediction Analysis
===================
"""
import site
site.addsitedir(r"E:\AA\AI4Water")

import pandas as pd
import matplotlib.pyplot as plt

from ai4water.postprocessing._info_plots import prediction_distribution_plot, feature_interaction

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
_, _, _ = feature_interaction(
    model.predict ,
    X = pd.DataFrame(X_test, columns=dataset.input_features),
    features = ['Adsorption_time (min)', 'calcination (min)'],
    feature_names = ['Adsorption_time (min)', 'calcination (min)'],
    plot_type="heatmap",
)
plt.show()

# %%
_, _, _ = feature_interaction(
    model.predict ,
    X = pd.DataFrame(X_test, columns=dataset.input_features),
    features = ['Adsorption_time (min)', 'initial concentration'],
    feature_names = ['Adsorption_time (min)', 'initial concentration'],
    plot_type="heatmap",
)
plt.show()

# %%
_, _, _ = feature_interaction(
    model.predict ,
    X = pd.DataFrame(X_test, columns=dataset.input_features),
    features = ['Adsorption_time (min)', 'solution pH'],
    feature_names = ['Adsorption_time (min)', 'solution pH'],
    plot_type="heatmap",
)
plt.show()

# %%
_, _, _ = feature_interaction(
    model.predict ,
    X = pd.DataFrame(X_test, columns=dataset.input_features),
    features = ['Adsorption_time (min)', 'adsorbent loading '],
    feature_names = ['Adsorption_time (min)', 'adsorbent loading '],
    plot_type="heatmap",
)
plt.show()

# %%
# Prediction Distribution
# ------------------------
_, _, _ = out = prediction_distribution_plot(mode='regression',
                             inputs=pd.DataFrame(X_test, columns=dataset.input_features),
                             prediction=test_p,
                             feature='Adsorption_time (min)',
                             feature_name='Adsorption_time (min)')