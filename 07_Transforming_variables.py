import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
fish = pd.read_csv("datasets/fish.csv")

# The fish dataset: Perch
perch = fish[fish["species"] == "Perch"]
print(perch.head())

# It is not a linear relationship
sns.regplot(x="length_cm",y="mass_g", data=perch, ci=None)
plt.show()

# Plotting mass vs. length cubed
perch["length_cm_cubed"] = perch["length_cm"] ** 3
sns.regplot(x="length_cm_cubed",y="mass_g", data=perch, ci=None)
plt.show()

# Running the model
mdl_perch = ols("mass_g ~ length_cm_cubed",data=perch).fit()
print(mdl_perch.params)