# A client file for tests

import regression
from numpy import genfromtxt
import matplotlib.pyplot as plt

X_train = genfromtxt('data/ds4_train.csv', delimiter=',', usecols=(0, 1, 2, 3), skip_header=1)
y_train = genfromtxt('data/ds4_train.csv', delimiter=',', usecols=4, skip_header=1)
X_valid = genfromtxt('data/ds4_valid.csv', delimiter=',', usecols=(0, 1, 2, 3), skip_header=1)
y_valid = genfromtxt('data/ds4_valid.csv', delimiter=',', usecols=4, skip_header=1)

model = regression.PoissonRegression(X_train, y_train, learning_rate=0.0001)
model.fit()
predictions = [model.predict(x) for x in X_valid]

#fig, ax = plt.subplots()
#plt.scatter(y_valid, predictions)
#plt.show()