import numpy as np
import pandas as pd
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")



data = pd.read_csv("bmi.csv")
data['Gender'] = data['Gender'].replace('Male',1)
data['Gender'] = data['Gender'].replace('Female',0)

X = data.iloc[:,:3]
Y = data.iloc[:,-1]
x = X.astype('int')
y = Y.astype('int')

xt,xts,yt,yts = train_test_split(x,y, test_size=0.3,random_state=0)
log_reg = LogisticRegression()
log_reg.fit(xt,yt)

pickle.dump(log_reg,open('model1.pkl','wb'))
model=pickle.load(open('model1.pkl','rb'))