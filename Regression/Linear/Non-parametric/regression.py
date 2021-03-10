import numpy as np
from numpy.linalg import inv

class Linear():
    def __init__(self, X, y, tau):
        self.raw_X = X
        self.X = self.__preprocessing()
        self.y = y
        self.tau = tau
        self.m = len(y)
    
    def __preprocessing(self):
        if len(np.shape(self.raw_X)) == 1:
            self.raw_X = self.raw_X[:, np.newaxis]
        ones = np.ones(np.shape(self.raw_X)[0])
        ones = ones[:, np.newaxis]

        return np.concatenate((ones, self.raw_X), axis=1)

    def __w(self, i, x, tau):
        return np.exp(-(np.sum((self.X[i] - x)**2))/(2*tau**2))

    def locally_weighted(self, x):
        '''
        Function that utilizes the method of locally weighted regression
        to evaluate the parameters (otherwise known as weights) of a linear function
        around a given value of x
        '''
        x = np.r_[1, x]
        y = self.y[:, np.newaxis]
        W = np.zeros((self.m, self.m))

        np.fill_diagonal(W, list(0.5 * self.__w(i, x, self.tau) for i in range(self.m)))
        
        thetas = (inv(self.X.T@W@self.X)@self.X.T@W@y)

        return thetas.T@x.T