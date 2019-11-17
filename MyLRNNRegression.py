from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
from sklearn.metrics import mean_squared_error as mse


class LR_NN_Regressor(BaseEstimator):
    """
    Linear and Neural Network regression
    Output average prediction
    """
    input_dim = 3

    def __init__(self, input_dim=input_dim):
        """Initialize LR and NN"""
        self.lr = LinearRegression()
        self.nn = Sequential()
        # self.arma_obj = None
        self.nn.add(Dense(32, input_dim=input_dim, activation='relu'))
        self.nn.add(Dropout(0.2))
        self.nn.add(Dense(16, activation='relu'))
        self.nn.add(Dropout(0.2))
        self.nn.add(Dense(1, activation='linear'))
        self.nn.compile(loss='mean_squared_error', optimizer='adam',
                        metrics=['mean_squared_error'])

    def fit(self, X, y):
        """Fit estimator"""
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        self.lr_obj = self.lr.fit(X, y)

        self.nn.fit(X, y, epochs=30, batch_size=32,
                    validation_split=0.2)
        self.nn_obj = self.nn
        # Return the classifier
        return self

    def predict(self, X):
        """Predict using average of LR and NN"""

        # Input validation
        X = check_array(X)
        y_h = 0.5 * self.lr_obj.predict(X) + \
            0.5 * self.nn_obj.predict(X).flatten()
        return y_h

    def score(self, X, y):
        """Returns the normalised root mean square error"""

        # Input validation
        X = check_array(X)
        y_h = 0.5 * self.lr_obj.predict(X) + \
            0.5 * self.nn_obj.predict(X).flatten()
        y_true = y
        nrmse = np.sqrt(mse(y_true, y_h))/y_true.mean()
        return nrmse
