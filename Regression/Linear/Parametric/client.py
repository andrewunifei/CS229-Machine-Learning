# A client file for tests

import regression
from numpy import genfromtxt
import plot2D as plot

X = genfromtxt('data/data1.csv', delimiter=',', usecols=0, skip_header=1)
y = genfromtxt('data/data1.csv', delimiter=',', usecols=1, skip_header=1)

model = regression.Linear(X, y, learning_rate=0.00001)
model.batch_gradient_descent()
#model.stochastic_gradient_descent()
thetas = model.get_thetas()

plot_model = plot.Model(X, y, thetas)
plot_model.animate('np2')