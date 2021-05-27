'''
    Histogram: Grouping data 
'''

import matplotlib.pyplot as plt

population_ages = [45,67,34,56,12,45,10,90,78,34,78,90,12,55,153,56,90,11,6,5,34,9,34,5,62,34]
grps = [10,20,30,40,50,60,70,80,90,100]

plt.hist(population_ages, grps, histtype ='bar', rwidth = 0.8) 
#plt.bar(x2,y2,label ='Bars2', color = 'c')


plt.xlabel('AGE')
plt.ylabel('NUMBER OF PEOPLE')
plt.title('Population  Age Groups')
plt.legend()
plt.show()
