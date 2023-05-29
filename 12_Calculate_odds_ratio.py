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

## Making predictions
mdl_recency = logit("has_churned ~ time_since_last_purchase", data=churn).fit()

explanatory_data = pd.DataFrame(
    {"time_since_last_purchase": np.arange(-1,6.25,0.25)}
)

prediction_data = explanatory_data.assign(has_churned = mdl_recency.predict(explanatory_data))

## Adding point predictions

sns.regplot(x="time_since_last_purchase",y="has_churned",data=churn,ci=None,logistic=True);
sns.scatterplot(x="time_since_last_purchase",y="has_churned",data=prediction_data,color="red");

## Getting the most likely outcome

prediction_data = explanatory_data.assign(has_churned = mdl_recency.predict(explanatory_data))
prediction_data["most_likely_outcome"] = np.round(prediction_data["has_churned"])

## Visualizing most likely outcome

sns.regplot(x="time_since_last_purchase",y="has_churned",data=churn,ci=None,logistic=True);
sns.scatterplot(x="time_since_last_purchase",y="most_likely_outcome",data=prediction_data,color="red");

## Calculating odds ratio

prediction_data["odds_ratio"] = prediction_data["has_churned"] / (1 - prediction_data["has_churned"])

## Visualizing odds ratio
sns.set_theme()
plt.plot("time_since_last_purchase","odds_ratio",data=prediction_data)
plt.axhline(y=1,linestyle="dotted")
plt.show()

## Calculating log odds ratio
prediction_data["log_odds_ratio"] = np.log(prediction_data["odds_ratio"])


