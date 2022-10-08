"""
====================
shap
====================
"""
import site
site.addsitedir(r"E:\AA\AI4Water")

import shap
import numpy as np
import matplotlib.pyplot as plt
from shap import DeepExplainer, GradientExplainer, KernelExplainer
from shap import Explanation
from shap.plots import scatter, bar, beeswarm, force, violin, heatmap, waterfall
from sklearn.manifold import TSNE
from easy_mpl import imshow
from umap import UMAP

from load_data import get_dataset, get_fitted_model

# %%

print(shap.__version__)

# %%

dataset, adsorbent_enc, dye_enc = get_dataset()


X_train, y_train = dataset.training_data()

# %%

X_test, y_test = dataset.test_data()

# %%

feature_names = dataset.input_features[0:11] + ['Adsorbent'] + ['Dye']
# %%

model = get_fitted_model()

 # %%

test_p = model.predict(x=X_test)

# %%

train_p = model.predict(x=X_train)
print(f"Average value on prediction on training data {train_p.mean()}")

# %%

exp = DeepExplainer(model, data=X_train)

sv = exp.shap_values(X_test)[0]

dye_sv = sv[:, 54:].sum(axis=1)

adsorbent_sv = sv[:, 11:54].sum(axis=1)

shap_values = np.column_stack((sv[:, 0:11], dye_sv, adsorbent_sv))

print(shap_values.shape)

# %%
adsorbent_ohe = X_test[:, 11:54]
adsorbent_original = adsorbent_enc.inverse_transform(adsorbent_ohe)
print(adsorbent_ohe.shape, adsorbent_original.shape)
# %%

dye_ohe = X_test[:, 54:]
dye_original = dye_enc.inverse_transform(dye_ohe)

print(dye_ohe.shape, dye_original.shape)
# %%

x_test_original = np.column_stack((X_test[:, 0:11], adsorbent_original, dye_original))
# %%


imshow(shap_values, aspect="auto", colorbar=True,
       xticklabels=feature_names, show=False)
plt.tight_layout()
plt.show()

# %%
shap_values_exp = Explanation(
    shap_values,
    data=x_test_original,
    feature_names=feature_names
)

heatmap(shap_values_exp, show=False)
plt.tight_layout()
plt.show()

# %%
heatmap(shap_values_exp, feature_values=shap_values_exp.abs.max(0), show=False)
plt.tight_layout()
plt.show()

# %%
heatmap(shap_values_exp, instance_order=shap_values_exp.sum(1), show=False)
plt.tight_layout()
plt.show()

# %%
beeswarm(shap_values_exp, show=False)
plt.tight_layout()
plt.show()

# %%
violin(shap_values, feature_names=feature_names, show=False)
plt.tight_layout()
plt.show()

# %%
index = test_p.argmax()
e = Explanation(
    shap_values[index],
    base_values=exp.expected_value[0],
    data=x_test_original[index],
    feature_names=feature_names
)

waterfall(e, show=False)
plt.tight_layout()
plt.show()

# %%
index = test_p.argmin()
e = Explanation(
    shap_values[index],
    base_values=exp.expected_value[0],
    data=x_test_original[index],
    feature_names=feature_names
)

waterfall(e, show=False)
plt.tight_layout()
plt.show()

# %%

tsne = TSNE(n_components=2)
sv_2D = tsne.fit_transform(shap_values)

s = plt.scatter(sv_2D[:, 0], sv_2D[:, 1], c=y_test.reshape(-1,), cmap="Spectral",
            s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(s)
plt.title('TSNE projection of shap values', fontsize=18)
plt.show()

# %%

sv_umap = UMAP(n_components=2).fit_transform(shap_values)
s = plt.scatter(sv_umap[:, 0], sv_umap[:, 1], c=y_test.reshape(-1,),
            s=5, cmap="Spectral")
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(s)
plt.title('UMAP projection of shap values', fontsize=18)
plt.show()
