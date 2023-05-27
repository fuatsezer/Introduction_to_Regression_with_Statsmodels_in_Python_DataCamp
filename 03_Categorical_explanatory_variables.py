import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

fish = pd.read_csv("datasets/fish.csv")
## Visualizing 1 numeric and 1 categorical variable

sns.displot(data=fish,x="mass_g",col="species",col_wrap=2,bins=9)
