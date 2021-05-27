'''
    Visualizing basic Math funcions
'''

import matplotlib.pyplot as plt
import math


x = [x for x in range(-2,11)]
y = []
for eachnum in x:
    #formula
    y.append(math.exp(eachnum))

plt.xlabel('x')
plt.ylabel('cosh(x)' )
plt.title('Formula Visual')
plt.legend()
plt.plot(x,y)
plt.show()

    

