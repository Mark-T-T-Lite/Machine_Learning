'''
    CSV: READ and plot using csv lib
'''

import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('example_csv.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y.append(int(row[1]))         

plt.plot(x,y, color='k', label='Loaded from csv')



plt.xlabel('X')
plt.ylabel('Y')
plt.title('CSV Values')
plt.legend()
plt.show()
