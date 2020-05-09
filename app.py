import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
app = Flask(__name__)

model = pickle.load(open('model1.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    #df = pd.DataFrame(int_features)
    #df[0].replace('male',1)
    #df[0].replace('female',0)
    #print(df)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(final_features[0])
    #output = round(prediction[0], 2)
    if prediction == 0:
        prediction = "Extremely Weak"
    elif prediction == 1:
        prediction = "Weak"
    elif prediction == 2:
        prediction = "Normal"
    elif prediction == 3:
        prediction = "Overweight"
    elif prediction == 4:
        prediction = "Obese"
    elif prediction == 5:
        prediction = "Extremely obese"

    if int_features[0] == 0:
        asq = "Female"
    elif int_features[0] == 1:
        asq = "Male"


    return render_template('index.html', pred='The person is {}'.format(prediction),a='{}'.format(asq),b='{}'.format(int_features[1]),c='{}'.format(int_features[2]))

#@app.route('/predict_api',methods=['POST'])
if __name__ == '__main__':
    app.run()
