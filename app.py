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
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(final_features)
    prediction = str(prediction)
    pr = prediction[2:-2]
    if int_features[0] == 0:
        asq = "Female"
    elif int_features[0] == 1:
        asq = "Male"

    #return render_template('index.html', pred='The person is {}'.format(prediction))
    return render_template('index.html', pred='The person is {}'.format(pr),a='{}'.format(asq),b='{}'.format(int_features[1]),c='{}'.format(int_features[2]))

if __name__ == "__main__":
    app.run(debug=True)