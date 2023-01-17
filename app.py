import pandas as pd
import numpy as np
import sklearn 
from sklearn.linear_model import LinearRegression
import  matplotlib.pyplot as pyplot
from matplotlib import style





#Using Regression
data = pd.read_csv("winequality-white.csv", sep=";")

#print(data.head(20))

data_to_predict = "quality"

x = np.array(data.drop([data_to_predict], 1))

y = np.array(data[data_to_predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.6)

linear_model = LinearRegression()

linear_model.fit(x_train, y_train)

#print(data.corr())

print(linear_model.score(x_test, y_test))

attributes = ["fixed acidity", "volatile acidity", 
                "citric acid", "residual sugar", "chlorides", 
                "free sulfur dioxide", "total sulfur dioxide", 
                "density", "pH", "sulphates", "alcohol"
            ]



p = attributes[10]
style.use("ggplot")
pyplot.scatter(data[p], data[data_to_predict])
pyplot.xlabel(p)
pyplot.ylabel(data_to_predict)

pyplot.show()