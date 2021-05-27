# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 19:43:33 2020

@author: Mark_T
@title: Multiple_Linear_Regression
@info: Notes.docx explains this code 
"""
# import the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd 

#import the dataset and create matrix
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values     #independent variables col
y = dataset.iloc[:, 4].values       #dependent variable col

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
#ML models may recognize 0,2,1 as one greater than the other so we make dummy variables
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#Avoiding the Dummy Variable Trap 
x = x[:, 1:]        #removes a redundant dependency of index 0

#Splitting into training set and test set to check ML model
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 1/5, random_state = 0)  #random_state can affect the results

#Feature Scaling Is Handled for us in Multiple Linear Regression

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test set results
y_pred = regressor.predict(x_test) 

#Building optimal model using Backward Elimination
import statsmodels.formula.api as sm
x = np.append(arr=np.ones((50,1)).astype(int), values=x, axis=1) #caters for b0x0 in formula
x_opt = x[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
x_opt = x[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
x_opt = x[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
x_opt = x[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
x_opt = x[:, [0,3]]
regressor_OLS = sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
