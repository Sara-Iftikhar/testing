"""
==========================
9. More Interpretations
==========================
For the calculation of integraded gradients we need to enable eager execution in tensorflow.
However, for SHAP value calculation of tensorflow based models, we need to disable v2 behaviour
which means disable eager execution. Therefore, integradted execution is put in separate file
as compared to that of shap values.
"""
all_vars = [var for var in globals() if var[0] != "_"]
for var in all_vars:
    del globals()[var]

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import numpy as np
from alibi.explainers import IntegratedGradients

from easy_mpl import imshow

from utils import get_dataset, get_fitted_model


dataset, adsorbent_enc, dye_enc = get_dataset(encoding="ohe")

X_train, y_train = dataset.training_data()

# %%

X_test, y_test = dataset.test_data()
# %%

model, _ = get_fitted_model(model_type='functional')

# %%
# Integrated Gradients
# =======================
#
# ig  = IntegratedGradients(model._model,
#                           layer=None,
#                           method="gausslegendre",
#                           n_steps=50,
#                           internal_batch_size=100)
#
# # %%
# # Training data
# explanation = ig.explain(X_train,
#                          baselines=None,
#                          target=None)
#
# attributions = explanation.attributions[0]
#
# con = attributions[:, 0:10]
# ads = attributions[:, 10:58]
# dye = attributions[:, 58:]
# attr = np.column_stack([con, ads.sum(axis=1), dye.sum(axis=1)])
#
# imshow(attr, aspect="auto", colorbar=True,
#        xticklabels=model.input_features[0:10] + ["Adsorbent", "Dye"],
#        show=False, ax_kws=dict(ylabel="Samples"), cmap="Greens")
# plt.tight_layout()
# plt.show()
#
# # %%
# # Test data
# explanation = ig.explain(X_test,
#                          baselines=None,
#                          target=None)
#
# attributions = explanation.attributions[0]
#
# con = attributions[:, 0:10]
# ads = attributions[:, 10:58]
# dye = attributions[:, 58:]
# attr = np.column_stack([con, ads.sum(axis=1), dye.sum(axis=1)])
#
# imshow(attr, aspect="auto", colorbar=True,
#        xticklabels=model.input_features[0:10] + ["Adsorbent", "Dye"],
#        show=False, ax_kws=dict(ylabel="Samples"), cmap="Greens")
# plt.tight_layout()
# plt.show()
