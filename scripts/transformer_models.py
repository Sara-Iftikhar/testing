"""
===========================
4. Transformer based models
===========================
"""

from ai4water import Model
from ai4water.utils.utils import TrainTestSplit
from ai4water.models.utils import gen_cat_vocab
from ai4water.models import FTTransformer, TabTransformer

from utils import make_data, evaluate_model

# %%
# Tab Transformer
# ===================

data, *_ = make_data()
data = data.drop(labels=800, axis=0)  # 'GSAC-Ce-1'
data = data.drop(labels=[921, 923], axis=0)  # 'AB25'
data = data.drop(labels=[920, 922], axis=0)  # 'TRAC'
data = data.drop(labels=801, axis=0)  # 'GSAC'

NUMERIC_FEATURES = data.columns.to_list()[0:10]
CAT_FEATURES = ["Adsorbent", "Dye"]
LABEL = "Adsorption"


splitter = TrainTestSplit(seed=313)

data[NUMERIC_FEATURES] = data[NUMERIC_FEATURES].astype(float)
data[CAT_FEATURES] = data[CAT_FEATURES].astype(str)
data['Adsorption'] = data['Adsorption'].astype(float)

train_data, test_data, _, _ = splitter.split_by_random(data)

# create vocabulary of unique values of categorical features
cat_vocabulary = gen_cat_vocab(data)

# %%

# %%
# make a list of input arrays for training data
train_x = [train_data[NUMERIC_FEATURES].values, train_data["Adsorbent"].values,
           train_data["Dye"].values]
test_x = [test_data[NUMERIC_FEATURES].values, test_data["Adsorbent"].values,
          test_data["Dye"].values]

# %%

depth = 3
num_heads = 4
hidden_units = 16
final_mpl_units = [84, 42]
num_numeric_features = len(NUMERIC_FEATURES)

# %%

model = Model(model=TabTransformer(
    cat_vocabulary=cat_vocabulary,
    num_numeric_features=num_numeric_features,
    hidden_units=hidden_units,
    final_mlp_units = final_mpl_units,
    depth=depth,
    num_heads=num_heads,
))


# %%

model.fit(x=train_x, y= train_data[LABEL].values,
              validation_data=(test_x, test_data[LABEL].values),
              epochs=500, verbose=0)

# %%

train_p = model.predict(x=train_x,)

# %%

evaluate_model(train_data[LABEL].values, train_p)

# %%

test_p = model.predict(x=test_x,)

# %%

evaluate_model(test_data[LABEL].values, test_p)

# %%
# FT Transformer
# =================

# build the FTTransformer model
model = Model(model=FTTransformer(cat_vocabulary, len(NUMERIC_FEATURES),
                                  hidden_units=16, num_heads=8))

# %%
model.fit(x=train_x, y= train_data[LABEL].values,
              validation_data=(test_x, test_data[LABEL].values),
              epochs=500, verbose=0)

# %%

train_p = model.predict(x=train_x)

# %%

evaluate_model(train_data[LABEL].values, train_p)

# %%

test_p = model.predict(x=test_x,)

# %%

evaluate_model(test_data[LABEL].values, test_p)
