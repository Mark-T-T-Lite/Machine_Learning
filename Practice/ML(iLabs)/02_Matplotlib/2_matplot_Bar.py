'''
    Bar Chart
'''

import matplotlib.pyplot as plt

x1 = [1,3,5,7,9]
y1 = [10,20,30,40,50]

x2 = [2,4,6,8,10]
y2 = [13,25,30,45,58]

plt.bar(x1,y1,label ='Bars1', color = 'r') 
plt.bar(x2,y2,label ='Bars2', color = 'c')




plt.xlabel('x')
plt.ylabel('y')
plt.title('Bar Chart')
plt.legend()
plt.show()
