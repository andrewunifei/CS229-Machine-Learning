# A client file for tests

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import regression

X = genfromtxt('data/data1.csv', delimiter=',', usecols=0, skip_header=1)
y = genfromtxt('data/data1.csv', delimiter=',', usecols=1, skip_header=1)

fig, ax = plt.subplots()
plt.scatter(X, y)

tau = 0.05
linear_obj = regression.Linear(X, y, tau)
prediction = [linear_obj.locally_weighted(x) for x in X]
plt.scatter(X, prediction)
ax.set_title(r'$\tau$ = ' + str(tau), loc="left")
plt.show()