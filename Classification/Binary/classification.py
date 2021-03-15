import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Binary():
    def __init__(self, X_train: object, y_train: object , X_toPredict: object) -> None:
        '''
        Args:
        Set of datas X and y to train,
        Set of datas to predict
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.X_toPredict = X_toPredict
        self.coef = np.empty((0,0)) 
        self.intercept = np.empty((0,0)) 

    def logistic_regression(self) -> object:
        model = LogisticRegression(random_state=0)
        model.fit(self.X_train, self.y_train.ravel())
        predictions = model.predict(self.X_toPredict)
        self.coef = model.coef_[0]
        self.intercept = model.intercept_

        return predictions

    def GDA(self) -> object:
        """
        Gaussian discriminant analysis
        """
        model = LinearDiscriminantAnalysis()
        model.fit(self.X_train, self.y_train.ravel())
        predictions = model.predict(self.X_toPredict)

        return predictions
    
    def plot_binary_classification(self, predictions: object, boundary: object = False) -> None:
        x1 = self.X_toPredict[:,0]
        x2 = self.X_toPredict[:,1]

        # Data points classified by colors
        fig, ax = plt.subplots()
        ax.scatter(x1, x2, c=predictions, marker='x')

        if boundary:
            # Decision boundary
            x = np.linspace(np.amin(x1), np.amax(x1))
            y = -self.coef[0]/self.coef[1] * x - self.intercept/self.coef[1]
            line, = ax.plot(x, y)
            line.set_label('Decision Boundary')
            ax.legend(loc='upper left')

        plt.show()
    
    def plot_raw(self):
        fig, ax = plt.subplots()
        ax.scatter(self.X_toPredict[:,0], self.X_toPredict[:, 1], marker='x')
        plt.show()
    
    def retrieve_parameters(self):
        return self.coef, self.intercept
    
    def save_predictions(self, predictions: object) -> None:
        data = np.concatenate((self.X_toPredict[:,0].reshape((-1, 1)), self.X_toPredict[:,1].reshape((-1, 1)), predictions.reshape((-1, 1))), axis=1)
        np.savetxt('predictions.csv', data, delimiter=',', header='x_1,x_2,y', comments="")
