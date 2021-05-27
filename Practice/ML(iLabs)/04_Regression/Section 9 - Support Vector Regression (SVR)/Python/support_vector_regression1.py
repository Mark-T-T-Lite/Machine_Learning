# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 03:18:29 2020

@author: Mark_T
@title: Support Vector Regression Template 
"""

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#import the dataset and create the matrices
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values     #independent variables col
y = dataset.iloc[:, 2].values       #dependent variable col

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1,1))
y = y.ravel()

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

# Predicting a new result 
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

# Visualising the Regression results(Better resolution, smoother)
x_grid = np.arange(min(x),max(x), 0.1)       #returns a vector 
x_grid = x_grid.reshape(len(x_grid),1)          #returns a matrix
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('SVR results')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

