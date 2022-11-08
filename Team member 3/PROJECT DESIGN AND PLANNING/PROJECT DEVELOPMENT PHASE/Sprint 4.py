# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:25:58 2022

@author: gurun
"""
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
app = Flask('_name_',template_folder='template')
filename ='resale_model.sav' 
model_rand = pickle.load(open('price prediction.pkl','rb'))
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict')
def predict():
    return render_template('index.html')
@app.route('/y_predict', methods=['GET', 'POST'])
def y_predict():
    Selling_Price= int(request.form[ 'Selling_Price'])
    Present_Price = float(request.form['Present_Price']) 
    Kms_Driven = float(request.form['Kms_Driven'])
    Years_old = int(request.fors.get('Years_old')) 
    Transmission = request.form['Transmission']
    Owner = request.form.get("Owner") 
    Fuel_Type = request.form.get('Fuel_Type')
    Seller_Type = request.form.get('Seller_Type')
    new_row = {'yearOfRegistration':Selling_Price, 'powerPS':Present_Price, 'kilometer': Kms_Driven,
       'monthofRegistration':Years_old, 'gearbox':Transmission,
        'brand':Owner, 'fuel Type':Fuel_Type, 
       'vehicleType':Seller_Type}
    print(new_row)
    new_df = pd.DataFrame(columns = ['vehicleType', 'yearOfRegistration', 'gearbox'
                            'powerPS', 'kilometer', 'monthofRegistration", "fuelType',
                            'brand' ])
    new_df = new_df.append(new_row,ignore_index = True)
    labels = ['gearbox','brand', 'fuel Type', 'vehicleType']
    mapper = ()
    for i in labels:
        mapper[i] = LabelEncoder()
        mapper[i].classes_ = np.load(str('classes'+i+'.npy'))
        tr = mapper[i].fit_transform(new_df[i])
        new_df.loc[:,i + '_ Labels'] = pd.Series (tr, index-new_df. Index)
        labeled = new_df[ ['yearOfRegistration'
                           ,'powerPS'
                           ,'kilometer' 
                           ,'monthofRegistration'
                           ]
                         + [x+'_Labels' for x in labels]]
        X = labeled.values
        print(X)
        y_prediction = model_rand.predict(X)
        print(y_prediction)
        return render_template('index.html',ypred = 'The resale value predicted is (:.2f)s'.format(y_prediction[0]))
    if __name__ == '_main_':
        app.debug = True
        app.run(host = '0.0.0.0', port = 5000)
        
        
        