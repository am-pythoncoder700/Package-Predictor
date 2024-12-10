import pickle
from flask import Flask, render_template, request
import numpy as np

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
        prediction = predict_package([[cgpa])
        return render_template('index.html', prediction=f"{prediction[0]:.2f}")
    except ValueError:
        return render_template('index.html', prediction="Invalid Input")

if __name__ == '__main__':
    app.run(debug=True)
