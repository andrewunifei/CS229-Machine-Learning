import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Binary():
    def __init__(self, X_train, y_train , X_toPredict):
        '''
        Args:
        Set of datas X and y to train,
        Set of datas to predict
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.X_toPredict = X_toPredict

    def logistic_regression(self):
        model = LogisticRegression(random_state=0)
        model.fit(self.X_train, self.y_train.ravel())
        predictions = model.predict(self.X_toPredict)

        return predictions

    def GDA(self):
        """
        Gaussian discriminant analysis
        """
        model = LinearDiscriminantAnalysis()
        model.fit(self.X_train, self.y_train.ravel())
        predictions = model.predict(self.X_toPredict)

        return predictions
    
    def plot_binary_classification(self, predictions):
        fig, ax = plt.subplots()
        ax.scatter(self.X_toPredict[:,0], self.X_toPredict[:,1], c=predictions, marker='x')
        plt.show()
    
    def save_predictions(self, predictions):
        data = np.concatenate((self.X_toPredict[:,0].reshape((-1, 1)), self.X_toPredict[:,1].reshape((-1, 1)), predictions.reshape((-1, 1))), axis=1)
        np.savetxt('predictions.csv', data, delimiter=',', header='x_1,x_2,y', comments="")