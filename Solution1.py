# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:51:44 2021

@author: HP
"""

import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from geopy.distance import vincenty
from matplotlib.collections import PatchCollection
from IPython.display import Image
from sklearn import preprocessing
warnings.filterwarnings('ignore')


#impoRting data from the source
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")
print(data.head(5))


#scatter-plot of the scores
plt.scatter(data['Hours'],data['Scores'],c='blue')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Scored')  
plt.show()


#selected attributes and spliting the data
x= data['Hours'].values.reshape(-1,1)
y= data['Scores'].values.reshape(-1,1) 
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 


#Training the model and plotting the intercept
from sklearn.linear_model import LinearRegression  
reg = LinearRegression()  
reg.fit(x_train, y_train) 
print("Model Trained")

intercept = reg.coef_*x+reg.intercept_
plt.scatter(x, y)
plt.plot(x, intercept, c='red');
plt.show()


#Predciting for the test case
pred_hour =9.25
score=pred_hour*reg.coef_[0][0]+reg.intercept_[0]
print("No of Hours = ",pred_hour)
print("Predicted Score = ", score)


#Calculating the efficiency of the model
from sklearn import metrics  
y_pred = reg.predict(x_test)
print('Mean Absolute Error:',  metrics.mean_absolute_error(y_test, y_pred)) 