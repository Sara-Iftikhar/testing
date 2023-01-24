"""
=======================
6. Prediction Analysis
=======================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from utils import get_dataset
from utils import evaluate_model
from utils import get_fitted_model
from utils import prediction_distribution


# %%

dataset ,  _, _ = get_dataset(encoding="ohe")
X_train, y_train = dataset.training_data()
X_test, y_test = dataset.test_data()

# %%

model, _ = get_fitted_model()

# %%

test_p = model.predict(x=X_test)

# %%

evaluate_model(y_test, test_p)

# %%
# Feature Interaction
# --------------------

_ = model.prediction_analysis(
    x = pd.DataFrame(X_test, columns=dataset.input_features),
    features = ['Adsorption Time (min)', 'Pyrolysis Time (min)'],
    feature_names = ['Adsorption Time (min)', 'Pyrolysis Time (min)'],
    grid_types=["percentile", "percentile"],
    num_grid_points=[6,6],
    annotate_kws={'annotate_fontsize':15,
                  'annotate_colors': np.array([['black', 'black', 'black', 'black'],
                                              ['black', 'black', 'black', 'black'],
                                              ['black', 'white', 'black', 'black'],
                                              ['black', 'black', 'black', 'black']])}
    )


# %%
_ = model.prediction_analysis(
    x = pd.DataFrame(X_test, columns=dataset.input_features),
    features = ['Adsorption Time (min)', 'Initial Concentration'],
    feature_names = ['Adsorption Time (min)', 'Initial Concentration'],
    grid_types=["percentile", "percentile"],
    num_grid_points=[6, 6],
    annotate_kws={'annotate_fontsize': 15,
                  'annotate_colors': np.array([['black', 'black', 'black', 'black', 'black'],
                                               ['black', 'black', 'black', 'black', 'black'],
                                               ['black', 'black', 'black', 'black', 'black'],
                                               ['white', 'black', 'black', 'black', 'black']])}
)

# %%
_ = model.prediction_analysis(
    x = pd.DataFrame(X_test, columns=dataset.input_features),
    features = ['Adsorption Time (min)', 'Solution pH'],
    feature_names = ['Adsorption Time (min)', 'Solution pH'],
    grid_types=["percentile", "percentile"],
    num_grid_points=[6, 6],
    annotate_kws={'annotate_fontsize': 15,
                  'annotate_colors': np.array([['black', 'white', 'black', 'black'],
                                               ['black', 'white', 'black', 'black'],
                                               ['black', 'white', 'black', 'black'],
                                               ['black', 'black', 'black', 'black']])}
)


# %%
_ = model.prediction_analysis(
    x = pd.DataFrame(X_test, columns=dataset.input_features),
    features = ['Adsorption Time (min)', 'Adsorbent Loading'],
    feature_names = ['Adsorption Time (min)', 'Adsorbent Loading'],
    grid_types=["percentile", "percentile"],
    num_grid_points=[6, 6],
    annotate_kws={'annotate_fontsize': 15,
                  'annotate_colors': np.array([['black', 'black', 'black', 'black', 'black'],
                                               ['black', 'black', 'black', 'black', 'black'],
                                               ['black', 'white', 'black', 'black', 'black'],
                                               ['black', 'black', 'black', 'black', 'black']])
                                 }
)

# %%
# Prediction Distribution
# ------------------------

prediction_distribution('Adsorption Time (min)', test_p, 0.4)

# %%
prediction_distribution('Pyrolysis Temperature', test_p, 0.4)

# %%
prediction_distribution('Initial Concentration', test_p, 0.4)

# %%
prediction_distribution('Solution pH', test_p, 0.4)

# %%
prediction_distribution('Adsorbent Loading', test_p, 0.4)

# %%
prediction_distribution('Volume (L)', test_p, 0.4)

# %%
prediction_distribution('Adsorption Temperature', test_p, 0.4)

# %%
grid = [2.75, 26.55, 81, 147.2, 495.5, 1085, 1509.11, 2430]
prediction_distribution('Surface Area', test_p, 0.4, grid=grid)

# %%
grid = [0.0, 0.18, 0.38, 0.39, 0.72, 1.32]
prediction_distribution('Pore Volume', test_p, 0.4, grid=grid)
