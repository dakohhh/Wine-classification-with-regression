import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score


#Using Regression
data = pd.read_csv("winequality-white.csv", sep=";")

data_to_predict = "quality"


print(data.corr())


data["good_quality"] = [1 if x >=7 else 0 for x in data["quality"]]



x = np.array(data.drop([data_to_predict, "good_quality"], 1))

y = np.array(data["good_quality"])




x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

model = LogisticRegression(max_iter=5000)

kmodel = KNeighborsClassifier(n_neighbors=12)

model.fit(x_train, y_train)

kmodel.fit(x_train, y_train)



predictions = model.predict(x_test)

k_predictions = kmodel.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

k_accuracy = accuracy_score(y_test, k_predictions)

print(accuracy)

print(k_accuracy)

count = 0
for x in range(len(predictions)):


    if int(predictions[x]) == y_test[x]:
        count += 1

print(f"LOG REG Passes {count} out of {len(predictions)}")

count = 0
for x in range(len(k_predictions)):
    print("Predicted:", int(k_predictions[x]), " ", "Actual:", y_test[x])

    if int(predictions[x]) == y_test[x]:
        count += 1

print(f"KNN Passes {count} out of {len(predictions)}")



# BOTH LOGISTIC REGRESSION AND KNN BOTH GIVE NEAR THE SAME RESULT ABOUT 80% ACCURACY