# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:56:52 2020

@author: Mark_T
"""

import pandas as pd
from pandas_datareader import data as web
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

start = datetime.datetime(2010,1,1)
end = datetime.datetime(2015,1,1)

df = web.DataReader("XOM", "yahoo",start,end)

print(df.head())


df['Adj Close'].plot()

plt.show()  