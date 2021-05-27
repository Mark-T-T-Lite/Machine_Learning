# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:16:21 2020

@author: Mark_T

Data Preprocessing Template
"""

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#import the dataset and create matrix
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values     #independent variables col
y = dataset.iloc[:, 3].values       #dependent variable col
"""
#Visual to Check for nonlinearity. Can also use Excel 
plt.scatter(x.astype(int), y )
plt.show()
 
#Handle missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
#ML models may recognize 0,2,1 as one greater than the other so we make dummy variables
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray() 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
"""
#Splitting into training set and test set to check ML model
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)  #random_state can affect the results

"""
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

