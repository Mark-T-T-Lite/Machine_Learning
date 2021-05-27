'''
    Complex No. Sequence Visualization:
    For example The sequence (1-1/(n**2)) + i(2+4/n); n = 1,2,3,...  
'''

import matplotlib.pyplot as plt
#import math

x,y = [],[]

for n in range(1,200):
    x.append(1-1/(n**2))
    y.append(2+(4/n))

print(x)
print(y)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Complex No. Sequence Visual')
plt.legend()
plt.plot(x,y)
plt.show()
    
