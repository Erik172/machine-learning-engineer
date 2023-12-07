import numpy as np
import math

class LogisticRegression:
    """
    Logistic Regression classifier.

    Parameters:
    - learning_rate (float): The learning rate for gradient descent. Default is 0.01.
    - n_iters (int): The number of iterations for gradient descent. Default is 1000.

    Attributes:
    - lr (float): The learning rate for gradient descent.
    - n_iters (int): The number of iterations for gradient descent.
    - weights (ndarray): The learned weights for the logistic regression model.
    - bias (float): The learned bias for the logistic regression model.
    - costs (list): The list of costs during training.

    Methods:
    - fit(X, y): Fit the logistic regression model to the training data.
    - predict(X): Predict the class labels for the input data.
    - score(X, y): Calculate the accuracy of the model on the input data.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.costs = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _get_cost(self, X, y, w, b):
        z = np.dot(X, w) + b
        y_hat = self._sigmoid(z)
        cost = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        return cost
    
    def _get_gradients(self, X, y, w, b):
        m, n = X.shape
        d_dw = np.zeros((n, 1))
        d_db = 0

        for i in range(m):
            y_hat = self._sigmoid(np.dot(X[i], w) + b)
            error = y_hat - y[i]
            for j in range(n):
                d_dw[j] += error * X[i][j]
            d_db += error

        d_dw /= m
        d_db /= m

        return d_dw, d_db
    
    def _gradient_descent(self, X, y, w, b):
        w = np.copy(w)
        b = np.copy(b)

        for _ in range(self.n_iters):
            d_dw, d_db = self._get_gradients(X, y, w, b)
            w -= self.lr * d_dw
            b -= self.lr * d_db
            
            if _ < 1000000:  # prevent resource exhaustion
                self.costs.append(self._get_cost(X, y, w, b))

            if math.ceil(self.n_iters / 10) == 0:
                print(f'Iteration {_}: Cost {self.costs[-1]}')

        return w, b, self.costs
    
    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.

        Parameters:
        - X (ndarray): The input features of shape (m, n), where m is the number of samples and n is the number of features.
        - y (ndarray): The target labels of shape (m,).

        Returns:
        - self: The fitted LogisticRegression object.
        """
        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = 0

        self.weights, self.bias, self.costs = self._gradient_descent(X, y, self.weights, self.bias)

        return self
    
    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        - X (ndarray): The input features of shape (m, n), where m is the number of samples and n is the number of features.

        Returns:
        - ndarray: The predicted class labels of shape (m,).
        """
        y_hat = self._sigmoid(np.dot(X, self.weights) + self.bias)
        y_hat = y_hat > 0.5

        return y_hat.astype(int)
    
    def score(self, X, y):
        """
        Calculate the accuracy of the model on the input data.

        Parameters:
        - X (ndarray): The input features of shape (m, n), where m is the number of samples and n is the number of features.
        - y (ndarray): The target labels of shape (m,).

        Returns:
        - float: The accuracy of the model.
        """
        y_hat = self.predict(X)
        accuracy = np.sum(y_hat == y) / len(y)

        return accuracy
    
    def __repr__(self):
        if self.weights is None:
            return f'LogisticRegression(lr={self.lr}, n_iters={self.n_iters})'
        else:
            return f'LogisticRegression(lr={self.lr}, n_iters={self.n_iters}, weights={self.weights}, bias={self.bias})'
    
    def __str__(self):
        return self.__repr__()