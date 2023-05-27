import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
fish = pd.read_csv("datasets/fish.csv")
## Visualizing 1 numeric and 1 categorical variable

sns.displot(data=fish,x="mass_g",col="species",col_wrap=2,bins=9)
## Summary statistics: mean mass by species
summary_stats = fish.groupby("species")["mass_g"].mean()
print(summary_stats)

## Linear Regression model with intercept
mdl_mass_vs_species = ols("mass_g ~ species",data= fish).fit()
print(mdl_mass_vs_species.params)

## Model without intercept
mdl_mass_vs_species = ols("mass_g ~ species + 0",data= fish).fit()
print(mdl_mass_vs_species.params)
## in case of a single, categorical variable, coefficients are the means.
