import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
fish = pd.read_csv("datasets/fish.csv")

# The fish dataset: Roach
roach = fish[fish["species"] == "Roach"]
print(roach.head())

# Which points are outliers?
roach["extreme_l"] = ((roach["length_cm"] < 15) | (roach["length_cm"] > 26))
roach["extreme_m"] = roach["mass_g"] < 1
fig = plt.figure()
sns.regplot(x="length_cm",y="mass_g", data=roach,ci=None)
sns.scatterplot(x="length_cm",y="mass_g", hue="extreme_l",style="extreme_m",data=roach)
plt.show()

# Fitting model
mdl_roach = ols("mass_g ~ length_cm",data=roach).fit()
summary_roach = mdl_roach.get_influence().summary_frame()

roach["leverage"] = summary_roach["hat_diag"]
print(roach.head())
roach["cooks_dist"] = summary_roach["cooks_d"]
print(roach.head())

## Most Influential roaches
print(roach.sort_values("cooks_dist",ascending=False))

## Removing the most influential roach
roach_not_short = roach[roach["length_cm"] != 12.9]

sns.regplot(x="length_cm",y="mass_g", data=roach,ci=None,line_kws={"color":"green"})
sns.regplot(x="length_cm",y="mass_g", data=roach_not_short,ci=None,line_kws={"color":"red"})
plt.show()