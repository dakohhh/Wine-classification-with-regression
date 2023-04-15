import numpy as np
import pandas as pd
import statsmodels.api as sm


data = pd.read_csv("winequality-white.csv", sep=";")

data_to_predict = "quality"

x = data.drop([data_to_predict, "citric acid", "chlorides","total sulfur dioxide"], 1)

y = data[data_to_predict]



x = sm.add_constant(x)

model = sm.OLS(y, x).fit()

summary = model.summary()

print(summary)