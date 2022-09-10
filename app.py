import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
model = pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.json['data']
    print(data)
    output = model.predict(data)
    print(output)
    return jsonify(output[0])

if __name__=='__main__':
    app.run(debug=True)