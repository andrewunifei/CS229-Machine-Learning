# A client file for tests

import regression
import plot2D as plot
import dummy_data

X, y = dummy_data.get(1)

model = regression.Linear(X, y, learning_rate=0.00001)
model.batch_gradient_descent()
#model.stochastic_gradient_descent()
thetas = model.get_thetas()

plot_model = plot.Model(X, y, thetas)
plot_model.animate('np2')