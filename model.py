import numpy as np
import pandas as pd
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

#Data Gathering
data = pd.read_csv("bmi.csv")

#Data cleaning
data['Gender'] = data['Gender'].replace('Male',1)
data['Gender'] = data['Gender'].replace('Female',0)

data['Index'] = data['Index'].replace(0,'Extremely weak')
data['Index'] = data['Index'].replace(1,'Weak')
data['Index'] = data['Index'].replace(2,'Fit')
data['Index'] = data['Index'].replace(3,'Overweight')
data['Index'] = data['Index'].replace(4,'Obese')
data['Index'] = data['Index'].replace(5,'Extremely Obese')

#Prepare dataset
X = data.iloc[:,:3]
y = data.iloc[:,-1]
x = X.astype('int')

#Train test split 
xt,xts,yt,yts = train_test_split(x,y, test_size=0.3,random_state=0)

#Model fitting
k = 3  
kNN_model = KNeighborsClassifier(n_neighbors=k).fit(xt,yt)
y_pred=kNN_model.predict(xts)

#Accuracy
aa = accuracy_score(yts,y_pred)

#Storing the model to a pickle file
pickle.dump(kNN_model,open('model1.pkl','wb'))
model=pickle.load(open('model1.pkl','rb'))
print("Model is trained with " +str(aa*100)+ " % of accuracy using KNN classifier")