'''
    PieChart: Slice of the Pie 
'''

import matplotlib.pyplot as plt

'''days = [1,2,3,4,5,6,7]

sleeping = [8,7,8,6,9,7,8]
eating =   [2,1,2,1,2,2,2]
work =     [10,10,10,10,10,10,10]
exercise = [4,6,4,7,3,5,4]'''

slices = [0.35,0.15,0.3,0.2]
activities = ['Mental Work','Reading','Spelling','Writing']
colors = ['c','m','r','g']

plt.pie(slices,
        labels=activities,
        colors=colors,
        startangle=0,
        shadow = True,
        explode = (0,0.1,0,0),
        autopct = '%1.1f%%')
#plt.bar(x2,y2,label ='Bars2', color = 'c')


#plt.xlabel('Days')
#plt.ylabel('Hours')
plt.title('PIE - CHART')
#plt.legend()
plt.show()
