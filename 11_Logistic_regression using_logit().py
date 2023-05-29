import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import logit
import statsmodels.api as sm
churn = pd.read_csv("datasets/churn.csv")

## Running model
mdl_churn_vs_recency_logit = logit("has_churned ~ time_since_last_purchase", data=churn).fit()
print(mdl_churn_vs_recency_logit.params)

## Visualizing the logistic model
sns.regplot(x="time_since_last_purchase",y="has_churned",data=churn,ci=None,logistic=True);


