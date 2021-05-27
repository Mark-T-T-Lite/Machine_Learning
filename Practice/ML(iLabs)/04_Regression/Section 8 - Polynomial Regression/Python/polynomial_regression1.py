# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 01:07:03 2020

@author: Mark_T
"""

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#import the dataset and create the matrices
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values     #independent variables col
y = dataset.iloc[:, 2].values       #dependent variable col

#Visual to Check for nonlinearity. Can also use Excel 
plt.scatter(x.astype(int), y )
plt.show()

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_regressor = PolynomialFeatures(degree=4)
x_poly = poly_regressor.fit_transform(x)
poly_lin_reg = LinearRegression()
poly_lin_reg.fit(x_poly,y)  

# Visualising the Polynomial Regression results
x_grid = np.arange(min(x),max(x), 0.1)       #returns a vector 
x_grid = x_grid.reshape(len(x_grid),1)          #returns a matrix
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, poly_lin_reg.predict(poly_regressor.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Polynomial Regression
poly_lin_reg.predict(poly_regressor.fit_transform(6.5))