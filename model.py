import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
import pickle

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# #logistic model
# model=LogisticRegression(max_iter=500)
# model.fit(X_train,y_train)
#
# #save model
# pickle.dump(model,open("model.pkl",'wb'))
#
# y_pred=model.predict(X_test)


# knn model
model_knn = KNeighborsRegressor(n_neighbors=4)
model_knn.fit(X_train, y_train)
predictions = model_knn.predict(X_test)
print(predictions)

#save model
pickle.dump(model_knn,open("model_knn.pkl",'wb'))


# print(accuracy_score(y_test, predictions))
