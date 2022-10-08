
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ai4water.preprocessing import DataSet
from ai4water.utils.utils import dateandtime_now

#read excel
ads_df = pd.read_excel('Adsorption and regeneration data_1007c.xlsx', sheet_name=0)
# regen_df = pd.read_excel('Adsorption and regeneration data.xlsx', sheet_name=1)

#dropping unnecessary columns
ads_df = ads_df.drop(columns=['final concentation', 'Volume (mL)',
                              'Unnamed: 16','Unnamed: 17',
                              'Unnamed: 18', 'Unnamed: 19',
                              'Unnamed: 20', 'Unnamed: 21',
                              'Unnamed: 22', 'Unnamed: 23'
                              ])
# regen_df = regen_df.drop(columns=['Desorption'])

#function for OHE
def _ohe_encoder(df:pd.DataFrame, col_name:str)->tuple:
    assert isinstance(col_name, str)

    encoder = OneHotEncoder(sparse=False)
    ohe_cat = encoder.fit_transform(df[col_name].values.reshape(-1, 1))
    cols_added = [f"{col_name}_{i}" for i in range(ohe_cat.shape[-1])]

    df[cols_added] = ohe_cat

    return df, cols_added, encoder

def _make_data():
    #applying OHE
    ads_df_enc_, _, adsorbent_enc = _ohe_encoder(ads_df, 'Adsorbent')
    ads_df_enc__ = ads_df_enc_.drop(columns='Adsorbent')

    ads_df_enc, _, dye_encoder = _ohe_encoder(ads_df_enc__, 'Dye')
    ads_df_enc = ads_df_enc.drop(columns='Dye')

    df1 = ads_df_enc.pop('qe')
    ads_df_enc['qe'] = df1
    return ads_df_enc, adsorbent_enc, dye_encoder

#ads_df_enc = ads_df_enc.drop([374])

# regen_df_enc, _, _ = _ohe_encoder(regen_df, 'Adsorbent')
# regen_df_enc = regen_df_enc.drop(columns='Adsorbent')
# df1 = regen_df_enc.pop('efficiency_%')
# regen_df_enc['efficiency_%'] = df1


def get_dataset():
    ads_df_enc, adsorbent_encoder, dye_encoder = _make_data()

    dataset = DataSet(data=ads_df_enc,
                      seed=1509,
                      val_fraction=0.0,
                      split_random=True,
                      input_features=ads_df_enc.columns.tolist()[0:-1],
                      output_features=ads_df_enc.columns.tolist()[-1:],
                      )
    return dataset, adsorbent_encoder, dye_encoder


def get_data():

    dataset, _, _ = get_dataset()

    # %%

    X_train, y_train = dataset.training_data()

    # %%

    X_test, y_test = dataset.test_data()
    return X_train, y_train, X_test, y_test


def make_path():
    path = os.path.join(os.getcwd(), 'results', f'mlp_{dateandtime_now()}')
    os.makedirs(path)
    return path

def get_fitted_model(return_path=False):

    X_train, y_train, X_test, y_test = get_data()

    path = make_path()

    reset_seed()

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

    callbacks = get_callbacks(path)

    model.fit(x=X_train, y=y_train,
                  validation_data=(X_test, y_test),
                  epochs=400, batch_size=24,
                  callbacks=callbacks, verbose=0)

    # %%
    update_model_with_best_weights(model, path)

    if return_path:
        return model, path
    return model


def get_callbacks(path):
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
    return callbacks


def update_model_with_best_weights(model, path):
    # find and update best weights
    best_weights = [f for f in os.listdir(path) if f.endswith(".hdf5")][-1]
    filepath = os.path.join(path, best_weights)
    model.load_weights(filepath)
    return


def reset_seed(seed=313):
    import numpy as np
    np.random.seed(seed=313)

    import tensorflow as tf
    tf.random.set_random_seed(313)

    import random
    random.seed(313)

    import os
    os.environ['PYTHONHASHSEED'] = str(313)
    return


def confidenc_interval(model, X_train, y_train, X_test, y_test, alpha,
                    n_splits=5):

    def generate_results_dataset(preds, _ci):
        _df = pd.DataFrame()
        _df['prediction'] = preds
        if _ci >= 0:
            _df['upper'] = preds + _ci
            _df['lower'] = preds - _ci
        else:
            _df['upper'] = preds - _ci
            _df['lower'] = preds + _ci

        return _df

    path = make_path()
    callbacks = get_callbacks(path)
    model.fit(X_train, y_train, batch_size=24, verbose=0,
              validation_data=(X_test, y_test),
              epochs=400, callbacks=callbacks)
    update_model_with_best_weights(model, path)

    residuals = y_train - model.predict(X_train)
    ci = np.quantile(residuals, 1 - alpha)
    preds = model.predict(X_test)
    df = generate_results_dataset(preds.reshape(-1, ), ci)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    res = []
    estimators = []
    for train_index, test_index in kf.split(X_train):
        X_train_, X_test_ = X_train[train_index], X_train[test_index]
        y_train_, y_test_ = y_train[train_index], y_train[test_index]

        path = make_path()
        callbacks = get_callbacks(path)
        model.fit(X_train_, y_train_, validation_data=(X_test_, y_test_),
                  callbacks=callbacks, verbose=0, batch_size=24, epochs=400)
        update_model_with_best_weights(model, path)

        estimators.append(model)
        _pred = model.predict(X_test_)
        res.extend(list(y_test_ - _pred.reshape(-1, )))

    y_pred_multi = np.column_stack([e.predict(X_test) for e in estimators])

    ci = np.quantile(res, 1 - alpha)
    top = []
    bottom = []
    for i in range(y_pred_multi.shape[0]):
        if ci > 0:
            top.append(np.quantile(y_pred_multi[i] + ci, 1 - alpha))
            bottom.append(np.quantile(y_pred_multi[i] - ci, 1 - alpha))
        else:
            top.append(np.quantile(y_pred_multi[i] - ci, 1 - alpha))
            bottom.append(np.quantile(y_pred_multi[i] + ci, 1 - alpha))

    preds = np.median(y_pred_multi, axis=1)
    df = pd.DataFrame()
    df['pred'] = preds
    df['upper'] = top
    df['lower'] = bottom

    return df

def plot_ci(df, alpha):
    # plots the confidence interval

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill_between(np.arange(len(df)), df['upper'], df['lower'], alpha=0.5, color='C1')
    p1 = ax.plot(df['pred'], color="C1", label="Prediction")
    p2 = ax.fill(np.NaN, np.NaN, color="C1", alpha=0.5)
    percent = int((1 - alpha) * 100)
    ax.legend([(p2[0], p1[0]), ], [f'{percent}% Confidence Interval'],
              fontsize=12)
    ax.set_xlabel("Test Samples", fontsize=12)
    ax.set_ylabel("Adsorption Capacity", fontsize=12)

    return ax