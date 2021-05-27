'''
    CSV: READ and plot
'''

import matplotlib.pyplot as plt
import numpy as np

x,y = np.loadtxt('example_csv.csv', delimiter=',', unpack = True)        
plt.plot(x,y, color='c', label='Loaded from csv')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('CSV Values')
plt.legend()
plt.show()
