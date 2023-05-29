import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import logit
from statsmodels.graphics.mosaicplot import mosaic
import statsmodels.api as sm
churn = pd.read_csv("datasets/churn.csv")

mdl_recency = logit("has_churned ~ time_since_last_purchase", data=churn).fit()

## Confusion matrix: counts of outcomes

actual_response = churn["has_churned"]

predicted_response = np.round(mdl_recency.predict())

outcomes = pd.DataFrame({"actual_response": actual_response,
                         "predicted_response": predicted_response})

print(outcomes.value_counts(sort=False))

## Visualizing the confusion matrix

conf_matrix = mdl_recency.pred_table()
print(conf_matrix)

mosaic(conf_matrix)