import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
swedish_motor_insurance = pd.read_csv("datasets/SwedishMotorInsurance.csv")
swedish_motor_insurance.head()
mdl_payment_vs_claims = ols("Payment ~ Claims",data= swedish_motor_insurance)

mdl_payment_vs_claims = mdl_payment_vs_claims.fit()
print(mdl_payment_vs_claims.params)