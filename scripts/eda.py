"""
==============
EDA
==============
"""

import site
site.addsitedir(r"E:\AA\AI4Water")

from ai4water.eda import EDA
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import ads_df

ads_df = ads_df.copy(deep=True)

# %%
print(ads_df.shape)

# %%
ads_df.head()

# %%
ads_df.tail()

# %%

ads_df['Adsorbent'].unique()
# %%

ads_df['Dye'].unique()

# %%

ads_df.pop("Adsorbent")
ads_df.pop("Dye")

eda = EDA(data = ads_df, save=False)

eda.correlation()

# %%
sns.pairplot(ads_df, hue='qe', palette='Spectral')
plt.show()