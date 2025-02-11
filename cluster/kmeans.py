import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        if type(k) != int:
            raise TypeError('Input for k must be a integer')
        if k <= 0:
            raise TypeError('Input for k can not be equal to 0')
        if type(tol) != float:
            raise TypeError('Input for tol must be a float')
        if type(max_iter) != int:
            raise TypeError('Input for max_iter must be a integer')
        
        self.k = k 
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.error = None
        self.n_features = None
        self.n_samples = None

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        if len(mat.shape) != 2:
            raise ValueError('Input matrix must be two dimensional')
        if mat.shape[0] < self.k:
            raise ValueError('Number of points must be larger than number of k')
        
        prev_error = 0
        
        self.n_samples = mat.shape[0]
        self.n_features = mat.shape[1]

        random_number_generator = np.random.default_rng()
        initial_indices = random_number_generator.choice(self.n_samples, size=self.k, replace=False)
        self.centroids = mat[initial_indices].copy()

        initial_error = float('inf')

        for data_point in range(self.max_iter):
            distances = cdist(mat, self.centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.k):
                if np.sum(labels == i) > 0:  # Avoid empty clusters
                    new_centroids[i] = np.mean(mat[labels == i], axis=0)
                else:
                    # If empty cluster, reinitialize randomly
                    new_centroids[i] = mat[rng.choice(mat.shape[0])].copy()

            self.error = np.mean(np.min(distances, axis=1) ** 2)

            if abs(prev_error - self.error) < self.tol:
                break
        else:    
            self.centroids = new_centroids
            prev_error = self.error

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        if not isinstance(mat, np.ndarray):
            raise TypeError('Input must be a numpy array')
        if len(mat.shape) != 2:
            raise ValueError('Input matrix must be 2-D')
        if mat.shape[1] != self.n_features:
            raise ValueError(f'Expected number of features: {self.n_features}, input number of features {mat.shape[1]}')
        
        distances = cdist(mat, self.centroids)
        return np.argmin(distances, axis=1)

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        return self.centroids.copy()