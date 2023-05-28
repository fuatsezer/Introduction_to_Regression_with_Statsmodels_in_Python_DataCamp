import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
fish = pd.read_csv("datasets/fish.csv")

# The fish dataset: bream
bream = fish[fish["species"] == "Bream"]
print(bream.head())
# .params attribute
mdl_mass_vs_length = ols("mass_g ~ length_cm",data=bream).fit()
print(mdl_mass_vs_length.summary())
print(mdl_mass_vs_length.rsquared)

mse = mdl_mass_vs_length.mse_resid
print("mse ",mse)
rse = np.sqrt(mse)
print("rse ",rse)