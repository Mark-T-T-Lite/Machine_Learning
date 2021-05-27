'''
    ScatterPlot: Correlating two or more variables 
'''

import matplotlib.pyplot as plt

population_ages = [45,67,34,56,12,45,10,90,78,34,78,90,12,55,153,56,90,11,6,5,34,9,34,5,62,34]
disease = [0,6,4,0,3,0,2,0,4,8,0,2,9,9,9,4,8,0,2,9,5,4,7,0,2,1]

plt.scatter(population_ages, disease, label ='scat1', color ='k', s=10) 
#plt.bar(x2,y2,label ='Bars2', color = 'c')


plt.xlabel('AGE')
plt.ylabel('Disease')
plt.title('AGE - DISEASE')
plt.legend()
plt.show()
