import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
fish = pd.read_csv("datasets/fish.csv")

# The fish dataset: bream
bream = fish[fish["species"] == "Bream"]
print(bream.head())

# Plotting mass vs. length
sns.regplot(x="length_cm", y = "mass_g",data=bream, ci=None)
plt.show()
# Running the model
mdl_mass_vs_length = ols("mass_g ~ length_cm",data=bream).fit()
print(mdl_mass_vs_length.params)

explanatory_data = pd.DataFrame({"length_cm": np.arange(20,41)})
print(mdl_mass_vs_length.predict(explanatory_data))

# Predicting Inside a DataFrame
explanatory_data = pd.DataFrame(
    {"length_cm": np.arange(20,41)}
)

prediction_data = explanatory_data.assign(
    mass_g= mdl_mass_vs_length.predict(explanatory_data)
)
print(prediction_data)

# Showing Prediction
fig = plt.figure()
sns.regplot(x="length_cm",y="mass_g",ci=None,data=bream)
sns.scatterplot(x="length_cm",y="mass_g",data=prediction_data,color="red",marker="s")
plt.show()

# Extrapolating
little_bream = pd.DataFrame({"length_cm": [10]})
pred_little_bream = little_bream.assign(
    mass_g =  mdl_mass_vs_length.predict(little_bream)
)
print(pred_little_bream)