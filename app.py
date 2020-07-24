# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np


classifier = pickle.load(open('car_price2.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Car_Age = int(request.form['Car_Age'])
        Present_Price = int(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Fuel_Type = int(request.form['Fuel_Type'])
        Seller_Type = int(request.form['Seller_Type'])
        Transmission = int(request.form['Transmission'])
        Owner = int(request.form['Owner'])



        data = np.array([[Car_Age,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner]])
        prediction = classifier.predict(data)

        output = round(prediction[0], 2)


        return render_template('home.html',prediction_texts=output)





if __name__ == '__main__':
    app.run(debug=True)