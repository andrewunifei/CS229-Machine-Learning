from numpy import genfromtxt
import classification

X_train = genfromtxt('data/data1_train.csv', delimiter=',', usecols=(0, 1), skip_header=1)
y_train = genfromtxt('data/data1_train.csv', delimiter=',', usecols=2, skip_header=1)
X_valid = genfromtxt('data/data1_valid.csv', delimiter=',', usecols=(0, 1), skip_header=1)

model = classification.Binary(X_train, y_train, X_toPredict=X_train)
predictions = model.logistic_regression()
model.plot_binary_classification(predictions)