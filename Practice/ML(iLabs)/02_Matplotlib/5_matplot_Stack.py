'''
    StackPlot: Relative Percentage in the whole 
'''

import matplotlib.pyplot as plt

days = [1,2,3,4,5,6,7]

sleeping = [8,7,8,6,9,7,8]
eating =   [2,1,2,1,2,2,2]
work =     [10,10,10,10,10,10,10]
exercise = [4,6,4,7,3,5,4]

plt.plot([],[],color='m', label='sleeping', linewidth =5)
plt.plot([],[],color='c', label='eating', linewidth =5)
plt.plot([],[],color='k', label='work', linewidth =5)
plt.plot([],[],color='r', label='playing', linewidth =5)


plt.stackplot(days, sleeping,eating,work,exercise, colors = ['m','c','k','r']) 
#plt.bar(x2,y2,label ='Bars2', color = 'c')


plt.xlabel('Days')
plt.ylabel('Hours')
plt.title('STACK PLOT')
plt.legend()
plt.show()
