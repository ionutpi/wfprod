from MyLRNNRegression import LR_NN_Regressor
import numpy as np


def test_predict():
    lr_nn = LR_NN_Regressor().fit(np.array([[1, 1, 1],
                                            [1, 1, 1],
                                            [1, 1, 1]]),
                                  np.array([1, 1, 1]))

    y = lr_nn.predict(np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]))
    assert(np.isfinite(y).any())
