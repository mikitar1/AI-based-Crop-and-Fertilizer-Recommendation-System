from flask import Flask,request,render_template
from flask import Flask
import numpy as np
import pandas as pd
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))
model2 = pickle.load(open('model1.pkl','rb'))
sc2 = pickle.load(open('standscaler1.pkl','rb'))
ms2 = pickle.load(open('minmaxscaler1.pkl','rb'))

# creating flask app
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/home')
def home():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    #cheking
    feature_list2 = [N, P, K, temp, humidity, ph, rainfall]
    #feature_list2 = [N, P, K, ph, rainfall,temp]
    single_pred = np.array(feature_list).reshape(1, -1)
    single_pred2 = np.array(feature_list2).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)
    
    scaled_features2 = ms2.transform(single_pred2)
    final_features2 = sc2.transform(scaled_features2)
    prediction2 = model2.predict(final_features2)
   
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
    fert_dict = {
    1: "Urea",
    2:"DAP",
    3:"MOP",
    4:"19-19-19 NPK",
    5:"SSP",
    6:"Magnesium Sulphate",
    7:"10-26-26 NPK",
    8:"50-26-26 NPK",
    9:"Chilated Micronutrient",
    10:"12-32-16 NPK",
    11:"Ferrous Sulphate",
    12:"13-32-26 NPK",
    13:"Ammonium Sulphate",
    14:"10-10-10 NPK",
    15:"Hydrated Lime", 
    16:"White Potash" ,
    17:"20-20-20 NPK",
    18:"18-46-00 NPK",
    19:"Sulphur"

}
    

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
        if prediction2[0] in fert_dict:
            fertilizer=fert_dict[prediction2[0]]
            result1= "{} is the best Fertilizer to be used right there".format(fertilizer)
            
            
       
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        result1="Sorry, we could not determine the best Fertilizer to be Used  with the provided data."
        
    return render_template('index.html',result = result,result1=result1)



    
    


# python main
if __name__ == "__main__":
    app.run(debug=True)