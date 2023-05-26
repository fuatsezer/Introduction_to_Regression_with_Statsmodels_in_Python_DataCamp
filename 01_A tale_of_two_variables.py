import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
swedish_motor_insurance = pd.read_csv("datasets/SwedishMotorInsurance.csv")

# Descriptive Statistics
swedish_motor_insurance.mean()
swedish_motor_insurance["Claims"].corr(swedish_motor_insurance["Payment"])

# Visualizing pairs of variables
sns.scatterplot(x="Claims",y="Payment",data=swedish_motor_insurance)
plt.show()

# Adding a linear trend line
sns.regplot(x="Claims",y="Payment",data=swedish_motor_insurance,ci=None)
plt.show()
#gfgf
