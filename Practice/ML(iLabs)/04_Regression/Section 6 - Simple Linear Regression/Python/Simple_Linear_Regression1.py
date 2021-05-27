# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:49:31 2020

@author: Mark_T
"""

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#import the dataset and create matrix
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values     #independent variables col
y = dataset.iloc[:, 1].values       #dependent variable col

#Splitting into training set and test set to check ML model
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 1/3, random_state = 0)

#Fitting simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test set results
y_pred = regressor.predict(x_test)

#Visualizing the training set reslts
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,regressor.predict(x_train), color='black')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set reslts
plt.scatter(x_test,y_test, color='red')
plt.plot(x_train,regressor.predict(x_train), color='black')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()