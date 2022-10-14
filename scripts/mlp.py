"""
====================
MLP
====================
"""

import site
site.addsitedir(r"E:\AA\AI4Water")

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt

from ai4water.postprocessing import LossCurve, ProcessPredictions
from ai4water.utils.utils import dateandtime_now

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils import get_data, reset_seed, evaluate_model

reset_seed(313)

print(np.__version__, tf.__version__)

# %%

X_train, y_train, X_test, y_test = get_data()

# %%

path = os.path.join(os.getcwd(),'results',f'mlp_{dateandtime_now()}')
os.makedirs(path)

inp = Input(shape=(68,))
dense1 = Dense(units=37, activation="relu")(inp)
dense2 = Dense(units=37, activation="relu")(dense1)
dense3 = Dense(units=37, activation="relu")(dense2)
dense4 = Dense(units=37, activation="relu")(dense3)
dense5 = Dense(units=1)(dense4)

# %%

model = Model(inputs=inp, outputs=dense5)

# %%

model.compile(optimizer=Adam(learning_rate=0.004561316449575947),
              loss='mse')

# %%

print(model.summary())

# %%
_monitor = 'val_loss'
fname = "{val_loss:.5f}.hdf5"

callbacks = list()
callbacks.append(keras.callbacks.ModelCheckpoint(
    filepath=path + f"{os.sep}weights_" + "{epoch:03d}_" + fname,
    save_weights_only=True,
    monitor=_monitor,
    mode='min',
    save_best_only=True))

callbacks.append(keras.callbacks.EarlyStopping(
    monitor=_monitor, min_delta=0.001,
    patience=100, verbose=0, mode='auto'
))

h = model.fit(x=X_train, y=y_train,
          validation_data=(X_test, y_test),
          epochs=400, batch_size=24,
              callbacks=callbacks)

# %%
# find and update best weights
best_weights = [f for f in os.listdir(path) if f.endswith(".hdf5")][-1]
filepath = os.path.join(path, best_weights)
model.load_weights(filepath)

# %%
LossCurve(path=path).plot_loss(h.history)

# %%
# Training data
# --------------
train_p = model.predict(x=X_train,)

# %%

evaluate_model(y_train, train_p)

# %%

pp = ProcessPredictions(mode='regression', forecast_len=1,
                   path=path)

pp.edf_plot(y_train, train_p, 'train', path)

# %%

pp.murphy_plot(y_train,train_p, prefix="train", where=path, inputs=X_train)

# %%
pp.is_multiclass_ = False
pp.errors_plot(y_train, train_p, 'train', path)

# %%
pp.residual_plot(pd.DataFrame(y_train), pd.DataFrame(train_p), 'train', path)

# %%

data_ = [pd.DataFrame(y_train, columns=['observed']), pd.DataFrame(train_p, columns=['predicted'])]
data = pd.concat(data_, axis=1)

sns.set_style("white")
gridobj = sns.lmplot(x="observed", y="predicted",
                     data=data,
                     height=7, aspect=1.3, robust=True,
                     scatter_kws=dict(s=150, linewidths=1.0, edgecolors='black',
                                      marker="3",
                                      alpha=0.5))
gridobj.ax.set_xticklabels(gridobj.ax.get_xticklabels(), fontsize=20)
gridobj.ax.set_yticklabels(gridobj.ax.get_yticklabels(), fontsize=20)
gridobj.ax.set_xlabel(gridobj.ax.get_xlabel(), fontsize=24)
gridobj.ax.set_ylabel(gridobj.ax.get_ylabel(), fontsize=24)
plt.show()

# %%
# Test data
# --------------
test_p = model.predict(x=X_test,)

# %%

evaluate_model(y_test, test_p)

# %%

pp = ProcessPredictions(mode='regression', forecast_len=1, path=path)

pp.edf_plot(y_test, test_p, 'test', path)

# %%

pp.murphy_plot(y_test, test_p, prefix="test", where=path, inputs=X_test)

# %%
pp.is_multiclass_ = False
pp.errors_plot(y_test, test_p, 'test', path)

# %%
pp.residual_plot(pd.DataFrame(y_test), pd.DataFrame(test_p), 'test', path)

# %%

data_ = [pd.DataFrame(y_test, columns=['observed']), pd.DataFrame(test_p, columns=['predicted'])]
data = pd.concat(data_, axis=1)

sns.set_style("white")
gridobj = sns.lmplot(x="observed", y="predicted",
                     data=data,
                     height=7, aspect=1.3, robust=True,
                     scatter_kws=dict(s=150, linewidths=1.0, edgecolors='black',
                                      marker="3",
                                      alpha=0.5))
gridobj.ax.set_xticklabels(gridobj.ax.get_xticklabels(), fontsize=20)
gridobj.ax.set_yticklabels(gridobj.ax.get_yticklabels(), fontsize=20)
gridobj.ax.set_xlabel(gridobj.ax.get_xlabel(), fontsize=24)
gridobj.ax.set_ylabel(gridobj.ax.get_ylabel(), fontsize=24)
plt.tight_layout()
plt.show()
