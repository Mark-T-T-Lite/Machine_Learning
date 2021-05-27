# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:14:12 2020

@author: Mark_T
"""

import pandas as pd
#import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot') 

web_stats = {'Day' : [1,2,3,4,5,6],
             'Visitors' : [23,56,77,55,44,54],
             'Bounce_Rate' : [54,66,44,56,21,43]}

df = pd.DataFrame(web_stats)

#print(df) 
#print(df.head(2))
#print(df.tail(2))

#print(df.set_index('Day'))
#df = df.set_index('Day')
#print(df)

#plt.plot(df) 
#plt.show()

#df.set_index('Day', inplace=True)
#print(df) 

#print(df['Visitors'])
#print(df.Visitors)

#print(df[['Visitors','Bounce_Rate']])
#print(df.to_dict())
print(df.Visitors.tolist())

#print(df[['Visitors','Bounce_Rate']].tolist())
print(np.array(df[['Visitors','Bounce_Rate']]))