"""
==========================
8. More Interpretations
==========================
For the calculation of integraded gradients we need to enable eager execution in tensorflow.
However, for SHAP value calculation of tensorflow based models, we need to disable v2 behaviour
which means disable eager execution. Therefore, integradted execution is put in separate file
as compared to that of shap values.
"""
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from alibi.utils import gen_category_map
from alibi.explainers import plot_pd_variance
from alibi.explainers import PartialDependenceVariance, IntegratedGradients

from easy_mpl import imshow, bar_chart
from ai4water.postprocessing import PermutationImportance
from ai4water.postprocessing import PartialDependencePlot

from utils import get_dataset, get_fitted_model, make_data


dataset, adsorbent_enc, dye_enc = get_dataset(encoding="ohe")

X_train, y_train = dataset.training_data()

# %%

X_test, y_test = dataset.test_data()
# %%

model, _ = get_fitted_model(model_type='functional')

# %%
# Integrated Gradients
# =======================

ig  = IntegratedGradients(model._model,
                          layer=None,
                          method="gausslegendre",
                          n_steps=50,
                          internal_batch_size=100)

# %%
# Training data
explanation = ig.explain(X_train,
                         baselines=None,
                         target=None)

attributions = explanation.attributions[0]

con = attributions[:, 0:10]
ads = attributions[:, 10:58]
dye = attributions[:, 58:]
attr = np.column_stack([con, ads.sum(axis=1), dye.sum(axis=1)])

imshow(attr, aspect="auto", colorbar=True,
       xticklabels=model.input_features[0:10] + ["Adsorbent", "Dye"],
       show=False, ax_kws=dict(ylabel="Samples"), cmap="Greens")
plt.tight_layout()
plt.show()

# %%
# Test data
explanation = ig.explain(X_test,
                         baselines=None,
                         target=None)

attributions = explanation.attributions[0]

con = attributions[:, 0:10]
ads = attributions[:, 10:58]
dye = attributions[:, 58:]
attr = np.column_stack([con, ads.sum(axis=1), dye.sum(axis=1)])

imshow(attr, aspect="auto", colorbar=True,
       xticklabels=model.input_features[0:10] + ["Adsorbent", "Dye"],
       show=False, ax_kws=dict(ylabel="Samples"), cmap="Greens")
plt.tight_layout()
plt.show()


# %%
# partial dependence plots
# ============================
# Now we will calculate partial dependence plot (PDP) and Individual Component
# Elements (ICE) curves using ``PartialDependencePlot`` class of ai4water.


pdp = PartialDependencePlot(
    model.predict,
    X_train,
    num_points=20,
    feature_names=model.input_features,
    show=False,
    save=False
)

# %%
# We calculate pdp plots only once and then plot them with
# different options
feature = [f for f in model.input_features if f.startswith("Dye")]
pdp_vals, ice_vals = pdp.calc_pdp_1dim(X_train, feature)

# %%

ax = pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, feature,
                        pdp_line_kws={'color': 'darkcyan'})
ax.set_xticklabels(dye_enc.categories_[0])
ax.set_xlabel("Dye")
plt.show()

# %%
ax = pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, feature, ice=False,
                        pdp_line_kws={'color': 'darkcyan'})
ax.set_xticklabels(dye_enc.categories_[0])
ax.set_xlabel("Dye")
plt.show()

# %%
# Calculate pdp and ice for Adsorbent

feature = [f for f in model.input_features if f.startswith("Adsorbent")]
pdp_vals, ice_vals = pdp.calc_pdp_1dim(X_train, feature)

# %%

ax = pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, feature,
                        pdp_line_kws={'color': 'darkcyan'})
ax.set_xticklabels(adsorbent_enc.categories_[0])
ax.set_xlabel("Adsorbent")
plt.show()

# %%
ax = pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, feature,
                        ice=False, pdp_line_kws={'color': 'darkcyan'})
ax.set_xticklabels(adsorbent_enc.categories_[0])
ax.set_xlabel("Adsorbent")
plt.show()

# %%
# pdp and ice for Surface Area
pdp_vals, ice_vals = pdp.calc_pdp_1dim(X_train, 'Surface area')

# %%
pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, 'Surface area',
                   pdp_line_kws={'color': 'darkcyan'})
plt.show()

# %%
ax = pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train,
                        'Surface area', ice=False,
                        pdp_line_kws={'color': 'darkcyan'})
plt.tight_layout()
plt.show()


# %%
# pdp and ice for Initial Concentration
feature = 'initial concentration'
pdp_vals, ice_vals = pdp.calc_pdp_1dim(X_train, feature)


# %%
pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, feature,
                   pdp_line_kws={'color': 'darkcyan'})
plt.tight_layout()
plt.show()

# %%
pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train,
                        feature, ice=False,
                        pdp_line_kws={'color': 'darkcyan'})
plt.tight_layout()
plt.show()

# %%
# pdp and ice for Pyrolysis Temperature
feature = 'calcination_temperature'
pdp_vals, ice_vals = pdp.calc_pdp_1dim(X_train, feature)


# %%
pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train, feature,
                   pdp_line_kws={'color': 'darkcyan'})
plt.tight_layout()
plt.show()

# %%
pdp._plot_pdp_1dim(pdp_vals, ice_vals, X_train,
                        feature, ice=False,
                        pdp_line_kws={'color': 'darkcyan'})
plt.tight_layout()
plt.show()

# %%
# Accumulated Local Effects
# ===========================

from alepython import ale_plot

class Model:
    def predict(self, X):
        return model.predict(X).reshape(-1,)

ale_plot(train_set=pd.DataFrame(X_train, columns=model.input_features),
             model=Model(),
                  features=["Surface area"]
             )

# %%
ae_eff = ale_plot(train_set=pd.DataFrame(X_train, columns=model.input_features),
             model=Model(),
                  features=['calcination_temperature']
             )

# %%
ale_plot(train_set=pd.DataFrame(X_train, columns=model.input_features),
             model=Model(),
                  features=['initial concentration']
             )

# %%

ale_plot(train_set=pd.DataFrame(X_train, columns=model.input_features),
             model=Model(),
                  features=["Surface area", 'Pore volume']
             )

# %%
# Permutation importance
# =======================
# Permutation importance quantifies reduction in model performance when
# we corrupt one feature column intentionally. The corruption in one feature
# column is carried out by randomly permuting its values ``n`` number of
# times. Then the average reduction in model performance is recorded
# as permutation feature importance for the feature.

cat_map = {'Catalyst': list(range(10, 58)), 'Anions': list(range(58, 74))}

pimp = PermutationImportance(
    model.predict, X_train, y_train,
    show=False,
    save=False,
    cat_map=cat_map,
    feature_names = model.input_features,
    n_repeats=20)

pimp.plot_1d_pimp()
plt.tight_layout()
plt.show()

# %%
pimp.plot_1d_pimp("barchart")
plt.tight_layout()
plt.show()

# %%
# Partial Dependence Variance
# ============================
# We give the data to ``PartialDependenceVariance`` class label-encoded
# data. When is then one-hot-encoded inside the ``predictor_fn``. Then
# the data is given to model to make a prediction

# first get data without any encoding
data, _, _ = make_data()
ads_ohe_encoder = OneHotEncoder(sparse=False)
ads_ohe_encoder.fit(data.loc[:, 'Adsorbent'].values.reshape(-1,1))

dye_ohe_encoder = OneHotEncoder(sparse=False)
dye_ohe_encoder.fit(data.loc[:, 'Dye'].values.reshape(-1,1))

# now get the label-encoded data. This will be passed to ``PartialDependenceVariance``
# class.
data_le, ads_le_encoder, dye_le_encoder = make_data(encoding="le")

def predictor_fn(X):

    # The X is given/suggested by ``PartialDependenceVariance`` which
    # means it is label-encoded. First inverse transform
    # to get the string columns
    ads_encoded = X[:, -2]
    ads_decoded = ads_le_encoder.inverse_transform(ads_encoded.astype(np.int16))
    ads_ohe_encoded = ads_ohe_encoder.transform(ads_decoded.reshape(-1,1))
    ads_cols = [f'Adsorbent_{i}' for i in range(ads_ohe_encoded.shape[1])]

    dye_encoded = X[:, -1]
    dye_decoded = dye_le_encoder.inverse_transform(dye_encoded.astype(np.int16))
    dye_ohe_encoded = dye_ohe_encoder.transform(dye_decoded.reshape(-1,1))
    dye_cols = [f'Dye_{i}' for i in range(dye_ohe_encoded.shape[1])]

    X = pd.DataFrame(X, columns=data.columns.tolist()[0:-1])
    X.pop('Adsorbent')
    X.pop('Dye')

    X[ads_cols] = ads_ohe_encoded
    X[dye_cols] = dye_ohe_encoded

    return model.predict(X.values).reshape(-1,)


category_map = gen_category_map(data)

pd_variance = PartialDependenceVariance(predictor=predictor_fn,
                                        feature_names=data.columns.tolist()[0:-1],
                                        categorical_names=category_map,
                                        target_names=["Adsorption"])

exp_importance = pd_variance.explain(X=data_le.values[:, 0:-1], method='importance')

# %%

bar_chart(
    exp_importance.feature_importance.reshape(-1,),
    sort=True,
    labels=data.columns.tolist()[0:-1],
    ax_kws=dict(tight_layout=True)
)

# %%
f, ax = plt.subplots(3,4, figsize=(10,10))
plot_pd_variance(exp=exp_importance, summarise=False, ax=ax)
plt.subplots_adjust(hspace=0.1)
plt.show()