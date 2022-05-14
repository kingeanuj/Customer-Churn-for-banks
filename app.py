import numpy as np
from flask import Flask, request, render_template,send_file, url_for, redirect
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import werkzeug

app = Flask(__name__)
model = pickle.load(open('XGBoostCLassifier.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('frontpage.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    temp_array1 =list()
    temp_array2 =list()
    temp_array3 =list()
    temp_array4 =list()
    
    HasCrCard = request.form["HasCrCard"]
    if HasCrCard == "No":
        temp_array1 = temp_array1+ [0]
    else :
        temp_array1 = temp_array1+ [1]
        
    IsActiveMember = request.form["IsActiveMember"]
    if IsActiveMember ==  "No":
        temp_array2 =temp_array2+ [0]
    else :
        temp_array2 = temp_array2+ [1]
        
    Geography = request.form["Geography"]
    if Geography == "France":
        temp_array3 = temp_array3 +[1,0]
    elif Geography == "Germany" :
        temp_array3 = temp_array3 + [0,1]
    else :
        temp_array3 = temp_array3 + [0,0]
            
    Gender = request.form["Gender"]
    if Gender == "Female":
        temp_array4 = temp_array4 + [0]
    else :
        temp_array4 = temp_array4 + [1]
    
    features = [x for x in request.form.values()]
    first6 = features[:6]
    
    first6 = [float(x) for x in first6] 
    
    continuous_features = np.array([first6])
    
    categorical_features = np.array(temp_array1+temp_array2+temp_array3+temp_array4).reshape(1,5)
    
    to_be_predicted = np.concatenate((continuous_features[0],categorical_features[0])).reshape(1,11)
    
    pred = model.predict(to_be_predicted)

    output = pred[0]
    
    if output == 0 :
        prediction_text="Don't worry, customer won't churn.".format(output)
        
    else :
        prediction_text="Customer is most likely to churn.".format(output)
        
    return render_template('prediction.html', prediction_text=prediction_text)



@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/predictioninfile',methods=['GET','POST'])
def predictioninfile():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST' :
        f = request.files['myfile']  
        f.save(f.filename)
        name = f.filename
        pred_dataset = pd.read_csv("{}".format(name))
        pred_dataset2 = pred_dataset.iloc[:,3:]
        pred_dataset2= pd.get_dummies(data = pred_dataset2, columns=["Geography","Gender"])
        pred_dataset2 = pred_dataset2.drop(columns=["Geography_Spain", "Gender_Female"])
        pred_dataset2 = pd.DataFrame(data= pred_dataset2, columns= ["CreditScore",
                                                "Age",
                                                "Tenure",
                                                "Balance",
                                                "NumOfProducts", 
                                                "EstimatedSalary",
                                                "HasCrCard",
                                               "IsActiveMember",
                                               "Geography_France",
                                               "Geography_Germany",
                                               "Gender_Male"])
        X = pred_dataset2.iloc[:,:].values
        pred_dataset["Churn"] = model.predict(X)
        pred_dataset.to_csv("Final Prediction File.csv", index=False)
        
    
    return render_template('predictioninfile.html')

@app.route('/download')
def download():
    return send_file("Final Prediction File.csv", as_attachment=True)

if __name__ == "__main__":
    app.run()