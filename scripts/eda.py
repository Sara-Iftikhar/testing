"""
==============
EDA
==============
"""

import site
site.addsitedir(r"E:\AA\AI4Water")

from ai4water.eda import EDA

from load_data import ads_df

ads_df.pop("Adsorbent")
ads_df.pop("Dye")

eda = EDA(data = ads_df, save=False)

eda.correlation()