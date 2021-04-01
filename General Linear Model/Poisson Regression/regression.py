import numpy as np

class PoissonRegression():
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

    def get_thetas(self) -> object:
        '''
        Return a list of all thetas generated in the calculation
        '''
        return self.theta

    def __preprocessing(self) -> object:
        '''
        Adds intercept 
        '''
        if len(np.shape(self.raw_X)) == 1:
            self.raw_X = self.raw_X[:, np.newaxis]
        
        # returns input array with intercept x0
        return np.c_[np.ones((np.shape(self.raw_X)[0])), self.raw_X]
    
    def __linear_combination(self, x) -> None:
        return sum(x * self.theta)

    def fit(self):
        FLAG = 1
        NUM_STEPS = 1000
        steps = [0] * len(self.theta)

        while(FLAG and NUM_STEPS):
            for i in range(self.m):
                for j in range(len(self.theta)):
                    self.theta[j] = self.theta[j] + (self.learning_rate * (self.y[i] - np.exp(self.__linear_combination(self.X[i]))) * self.X[i][j])

                if all(abs(steps[i]) < 0.0001 for i in range(len(steps))):
                    FLAG = 0
                NUM_STEPS -=1
        
    def predict(self, x: object) -> object:
        x = np.r_[1, x]
        return np.exp(self.__linear_combination(x))