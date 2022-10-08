"""
=====================
confidence intervals
=====================
"""
import site
site.addsitedir(r"E:\AA\AI4Water")

from load_data import get_dataset, get_fitted_model, confidenc_interval

# %%

dataset, adsorbent_enc, dye_enc = get_dataset()


X_train, y_train = dataset.training_data()

# %%

X_test, y_test = dataset.test_data()

# %%

feature_names = dataset.input_features[0:11] + ['Adsorbent'] + ['Dye']
# %%

# model = get_fitted_model()
#
# confidenc_interval(model, X_train, y_train, X_test, alpha=0.05)