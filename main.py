from flask import Flask, render_template, request, url_for, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
app = Flask(__name__)



model = pickle.load(open('model.pkl', "rb"))
flowers = ['setosa', 'versicolor',  'virginica']

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



def create_knn(k):
    model_knn = KNeighborsRegressor(n_neighbors=int(k))
    model_knn.fit(X_train, y_train)
    return model_knn




@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=["post"])
def predict():
    ml_input = []
    form_data = request.form
    print(form_data)
    for (key, value) in form_data.items():
        ml_input.append(float(value))
    if form_data.get('ml_alg')=="0":
        prediction = model.predict(np.array(ml_input[:-2]).reshape(1,4))
    else:
        prediction = create_knn(ml_input[-2]).predict(np.array(ml_input[:-2]).reshape(1, 4))



    print(prediction)
    flower_name = flowers[round(prediction[0]) ]
    return render_template('index.html',prediction_text='Species  should be ' + flower_name)


if __name__ == '__main__':
    app.run(debug=True)
