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
plt.rcParams["font.family"] = "Times New Roman"

from utils import get_dataset, get_data, get_fitted_model, evaluate_model, plot_violin_


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
                  'annotate_colors': np.array([['black', 'black', 'black', 'black'],
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

plot_violin_('Adsorption_time (min)', test_p, 0.4)

# %%
# plot_violin_('Adsorbent', test_p, 0.4)

# %%
plot_violin_('calcination_temperature', test_p, 0.4)

# %%
plot_violin_('calcination (min)', test_p, 0.4)

# %%
# plot_violin_('Dye', test_p, 0.4)

# %%
plot_violin_('initial concentration', test_p, 0.4)

# %%
plot_violin_('solution pH', test_p, 0.4)

# %%
plot_violin_('adsorbent loading ', test_p, 0.4)

# %%
plot_violin_('Volume (L)', test_p, 0.4)

# %%
plot_violin_('adsorption_temperature ', test_p, 0.4)

# %%
plot_violin_('Particle size', test_p, 0.4)

# %%
plot_violin_('Surface area', test_p, 0.4)

# %%
grid = [0.0, 0.18, 0.38, 0.39, 0.72, 1.32]
plot_violin_('Pore volume', test_p, 0.4, grid=grid)
