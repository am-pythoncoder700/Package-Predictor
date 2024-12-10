import pickle
from flask import Flask, render_template, request
import numpy as np

# Implementing our Simple Linear Regression class
class Simple_Linear_Regression:
    def __init__(self, coef_=None, intercept_=None):
        self.coef_ = coef_
        self.intercept_= intercept_
    def fit(self, X_train, y_train):
        x_mean, y_mean = np.mean(X_train), np.mean(y_train)
        self.coef_ = np.dot((y_train - y_mean), (X_train - x_mean)) / np.dot((X_train - x_mean), (X_train - x_mean))
        self.intercept_ = y_mean - self.coef_ * x_mean
    def predict(self, X_test):
        return self.coef_ * X_test + self.intercept_
    def get_params(self, deep):
        return {"coef_" : self.coef_, "intercept_" : self.intercept_}

app = Flask(__name__)

# Mock Model for Prediction
def predict_package(cgpa):
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model.predict(cgpa)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cgpa = float(request.form['cgpa'])
        prediction = predict_package(cgpa)
        return render_template('index.html', prediction=f"{prediction:.2f}")
    except ValueError:
        return render_template('index.html', prediction="Invalid Input")

if __name__ == '__main__':
    app.run(debug=True)