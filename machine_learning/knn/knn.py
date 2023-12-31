import numpy as np

class KnnClassifier:
    """
    K-nearest neighbors classifier.

    Parameters:
    -----------
    k : int, optional (default=3)
        The number of nearest neighbors to consider.
    type_dist : str, optional (default='euclidean')
        The type of distance metric to use. Supported values are 'euclidean', 'manhattan', and 'cosine'.

    Attributes:
    -----------
    X : array-like, shape (n_samples, n_features)
        The training data.
    y : array-like, shape (n_samples,)
        The target values.

    Methods:
    --------
    fit(X, y)
        Fit the model with the training data.
    predict(X_test)
        Predicts the class labels for the given test data using the KNN algorithm.
    distance(x1, x2, type_dist='euclidean')
        Computes the distance between two points x1 and x2 using the specified distance metric.
    accuracy(X, y)
        Computes the accuracy of the model on the given test data.

    """

    def __init__(self, k=3, type_dist='euclidean'):
        self.k = k
        self.type_dist = type_dist

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        """
        Predicts the class labels for the given test data using the KNN algorithm.

        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            The test data to predict the class labels for.

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            The predicted class labels for the test data.
        """
        y_pred = []
        for x in X_test:
            distances = []
            for i, x_train in enumerate(self.X):
                distances.append((self.distance(x, x_train, self.type_dist), self.y[i]))
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.k]
            counts = {}
            for neighbor in neighbors:
                if neighbor[1] in counts:
                    counts[neighbor[1]] += 1
                else:
                    counts[neighbor[1]] = 1
            y_pred.append(max(counts, key=counts.get))

        return np.array(y_pred)

    def distance(self, x1, x2, type_dist='euclidean'):
        """
        Computes the distance between two points x1 and x2 using the specified distance metric.

        Parameters:
        -----------
        x1 (numpy.ndarray): The first point.
        x2 (numpy.ndarray): The second point.
        type_dist (str): The type of distance metric to use. Supported values are 'euclidean', 'manhattan', and 'cosine'.

        Returns:
        --------
        float: The distance between x1 and x2.
        """
        if type_dist == 'euclidean':
            return np.sqrt(np.sum((x1 - x2)**2))
        elif type_dist == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif type_dist == 'cosine':
            return 1 - np.dot(x1, x2) / (np.sqrt(np.dot(x1, x1)) * np.sqrt(np.dot(x2, x2)))
        else:
            raise ValueError('Type of distance is not supported')

    def accuracy(self, X, y):
        """
        Computes the accuracy of the model on the given test data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The test data.
        y : array-like, shape (n_samples,)
            The true labels for the test data.

        Returns:
        --------
        float: The accuracy of the model on the test data.
        """
        return np.sum(self.predict(X) == y) / len(y)