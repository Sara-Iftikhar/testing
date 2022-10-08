"""
=====================
confidence intervals
=====================
"""
import site
site.addsitedir(r"E:\AA\AI4Water")

import os
import matplotlib.pyplot as plt
from load_data import get_dataset, get_fitted_model, confidenc_interval, plot_ci

# %%

dataset, adsorbent_enc, dye_enc = get_dataset()


X_train, y_train = dataset.training_data()

# %%

X_test, y_test = dataset.test_data()

# %%

feature_names = dataset.input_features[0:11] + ['Adsorbent'] + ['Dye']
# %%

model, path = get_fitted_model(return_path=True)

#
df = confidenc_interval(model, X_train[0:685], y_train[0:685],
                        X_test, y_test,
                        alpha=0.05)

plot_ci(df, 0.05)
percent = int((1 - 0.05) * 100)
fpath = os.path.join(path, f"{percent}_interval_")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
plt.show()

# %%
plot_ci(df.iloc[0:50], 0.05)
percent = int((1 - 0.05) * 100)
fpath = os.path.join(path, f"{percent}_interval_")
plt.savefig(fpath, dpi=300, bbox_inches="tight")
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