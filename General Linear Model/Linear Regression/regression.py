# Andrew Enrique Oliveira
# Computer Science at Universidade Federal de Itajubá (2017 - )
# 01/03/2021
#
# I wrote this as a learning exercise. The focus is not performance but to explore the concepts around this theme

import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression

class Linear():
    def __init__(self, raw_X: object, y: object, learning_rate: float = 0.00001) -> None:
        '''
        Constructor takes
        raw_X as a list of features,
        y as a list of correspondent values to X and
        learning_rate as the learning rate.
        '''
        self.raw_X = raw_X
        self.X = self.__preprocessing()
        self.y = y
        self.learning_rate = learning_rate
        self.m = len(y)
        self.theta = np.zeros(np.shape(self.X)[1]) # num of parameters is determined by quantity of column in X (num of features)
        self.retrieved_thetas = list()

    def get_thetas(self):
        '''
        Return a list of all thetas generated in the calculation
        '''
        return self.retrieved_thetas

    def __preprocessing(self) -> object:
        '''
        Adds intercept 
        '''
        if len(np.shape(self.raw_X)) == 1:
            self.raw_X = self.raw_X[:, np.newaxis]
        ones = np.ones(np.shape(self.raw_X)[0])
        ones = ones[:, np.newaxis]

        return np.concatenate((ones, self.raw_X), axis=1)

    def __hypothesis(self, x: object):
        # h(x) = theta_0 + theta_1 * x_1 + ... + theta_n * x_n
        # return sum(h(x)) for all values of x
        return sum(x * self.theta)

    def batch_gradient_descent(self) -> None:
        '''
        Function that utilizes the method of batch gradient descent
        to evaluate the parameters (otherwise known as weights) of a linear function
        that will map a given X to y based on a set of data points (training data)
        '''
        for j in range(len(self.theta)):
            for i in range(self.m):
                self.theta[j] = self.theta[j] + (self.learning_rate * (np.sum(self.y) - np.sum(self.X * self.theta)) * self.X[i][j])
                self.retrieved_thetas.append(self.theta.copy())

    def stochastic_gradient_descent(self) -> None:
        '''
        Function that utilizes the method of stochastic gradient descent
        to evaluate the parameters (otherwise known as weights) of a linear function
        that will map a given X to y based on a set of data points (training data)
        '''
        FLAG = 1
        NUM_STEPS = 1000
        steps = [0] * len(self.theta)

        while(FLAG and NUM_STEPS):
            for i in range(self.m):
                for j in range(len(self.theta)):
                    step = self.learning_rate * (self.y[i] - self.__hypothesis(self.X[i])) * self.X[i][j]
                    self.theta[j] = self.theta[j] + step
                    steps[j] = step

                self.retrieved_thetas.append(self.theta.copy())

                if all(abs(steps[i]) < 0.0001 for i in range(len(steps))):
                    FLAG = 0
                NUM_STEPS -=1

    def normal_equation(self) -> None:
        '''
        Function that utilizes the method of normal equation to evaluate the parameters (otherwise known as weights) of a linear function.
        Returns thetas values
        '''
        # This method instantly finds the appropriated values of theta by setting ∇J(θ)=0

        y = self.y[:, np.newaxis]
        theta = ((inv(self.X.T@self.X)@self.X.T@y).flatten())

        self.retrieved_thetas = np.array([theta])
    
    def sklearn_linear_reg(self) -> None:
        model = LinearRegression().fit(self.X, self.y)
        self.retrieved_thetas = np.array([np.insert(np.array(model.coef_[1]), 0, model.intercept_)])


if __name__ == '__main__':
    exit()