"""
=========================
8. confidence intervals
=========================
"""

import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from keras.engine.training import *
from utils import get_dataset, get_fitted_model, \
    confidenc_interval, plot_ci, evaluate_model

# %%

dataset, adsorbent_enc, dye_enc = get_dataset(encoding="ohe")


X_train, y_train = dataset.training_data()

# %%

X_test, y_test = dataset.test_data()

# %%

feature_names = dataset.input_features[0:10] + ['Adsorbent'] + ['Dye']
# %%

model, path, _ = get_fitted_model(return_path=True)

# %%
model._distribution_strategy = None
test_p = model.predict(x=X_test)

# %%

evaluate_model(y_test, test_p)

# %%

alpha = 0.05
df = confidenc_interval(model, X_train[0:685], y_train[0:685],
                        X_test, y_test,
                        alpha=alpha)

# %%

plot_ci(df, alpha)
percent = int((1 - alpha) * 100)
fpath = os.path.join(path, f"{percent}_interval_")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

plot_ci(df.iloc[0:50], alpha)
percent = int((1 - alpha) * 100)
fpath = os.path.join(path, f"{percent}_interval_")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

alpha = 0.1
df = confidenc_interval(model, X_train[0:685], y_train[0:685],
                        X_test, y_test,
                        alpha=alpha)

plot_ci(df, alpha)
percent = int((1 - alpha) * 100)
fpath = os.path.join(path, f"{percent}_interval_")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

plot_ci(df.iloc[0:50], alpha)
percent = int((1 - alpha) * 100)
fpath = os.path.join(path, f"{percent}_interval_")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

alpha = 0.15
df = confidenc_interval(model, X_train[0:685], y_train[0:685],
                        X_test, y_test,
                        alpha=alpha)

plot_ci(df, alpha)
percent = int((1 - alpha) * 100)
fpath = os.path.join(path, f"{percent}_interval_")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

plot_ci(df.iloc[0:50], alpha)
percent = int((1 - alpha) * 100)
fpath = os.path.join(path, f"{percent}_interval_")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

alpha = 0.2
df = confidenc_interval(model, X_train[0:685], y_train[0:685],
                        X_test, y_test,
                        alpha=alpha)

plot_ci(df, alpha)
percent = int((1 - alpha) * 100)
fpath = os.path.join(path, f"{percent}_interval_")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

plot_ci(df.iloc[0:50], alpha)
percent = int((1 - alpha) * 100)
fpath = os.path.join(path, f"{percent}_interval_")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

alpha = 0.25
df = confidenc_interval(model, X_train[0:685], y_train[0:685],
                        X_test, y_test,
                        alpha=alpha)

plot_ci(df, alpha)
percent = int((1 - alpha) * 100)
fpath = os.path.join(path, f"{percent}_interval_")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%

plot_ci(df.iloc[0:50], alpha)
percent = int((1 - alpha) * 100)
fpath = os.path.join(path, f"{percent}_interval_")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()