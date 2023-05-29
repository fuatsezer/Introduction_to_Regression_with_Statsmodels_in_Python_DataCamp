import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
fish = pd.read_csv("datasets/fish.csv")

# The fish dataset: bream
bream = fish[fish["species"] == "Bream"]
print(bream.head())
# The fish dataset: perch
perch = fish[fish["species"] == "Perch"]
print(perch.head())
# Fitting model
mdl_bream = ols("mass_g ~ length_cm",data=bream).fit()
# Fitting model
mdl_perch = ols("mass_g ~ length_cm",data=perch).fit()

## Residual plot
df_resid_bream = pd.DataFrame({"Residuals":mdl_bream.resid,"Fitted values":mdl_bream.fittedvalues})
df_resid_perch = pd.DataFrame({"Residuals":mdl_perch.resid,"Fitted values":mdl_perch.fittedvalues})
sns.residplot(x="length_cm",y="mass_g",data=bream,lowess=True)
plt.show()

sns.residplot(x="length_cm",y="mass_g",data=perch,lowess=True)
plt.show()

## QQ Plot
sm.qqplot(df_resid_bream["Residuals"], line='45',fit=True)

# Set plot labels
plt.title("QQ Plot")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

sm.qqplot(df_resid_perch["Residuals"], line='45',fit=True)

# Set plot labels
plt.title("QQ Plot")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

## Scale-location plot
model_norm_residuals_bream = mdl_bream.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt_bream = np.sqrt(np.abs(model_norm_residuals_bream))
sns.regplot(x=mdl_bream.fittedvalues, y=model_norm_residuals_abs_sqrt_bream,ci=None,lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Sqrt of abs val of stdized residuals")
plt.show()



model_norm_residuals_perch = mdl_perch.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt_perch = np.sqrt(np.abs(model_norm_residuals_perch))
sns.regplot(x=mdl_perch.fittedvalues, y=model_norm_residuals_abs_sqrt_perch,ci=None,lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Sqrt of abs val of stdized residuals")
plt.show()
