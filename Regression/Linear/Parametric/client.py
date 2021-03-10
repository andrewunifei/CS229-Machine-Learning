# A client file for tests

import regression
from numpy import genfromtxt
import plot2D

X = genfromtxt('data/data3.csv', delimiter=',', usecols=0, skip_header=1)
y = genfromtxt('data/data3.csv', delimiter=',', usecols=1, skip_header=1)

model = regression.Linear(X, y, learning_rate=0.00001)
#model.batch_gradient_descent()
#model.stochastic_gradient_descent()
thetas = model.normal_equation()

plot_model = plot2D.Model(X, y, thetas)
plot_model.animate('np2')