import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        if len(y.shape) != 1:
            raise ValueError("y must be 1-dimensional")
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y must match")

        n_samples = X.shape[0]
        silhouette_scores = np.zeros(n_samples)
        unique_labels = np.unique(y)
        
        distances = cdist(X, X)

        for i in range(n_samples):
            # Get current point's cluster
            current_cluster = y[i]

            same_cluster_mask = y == current_cluster
            same_cluster_mask[i] = False  # Exclude the point itself
            
            if np.sum(same_cluster_mask) > 0:
                a_i = np.mean(distances[i, same_cluster_mask])
            else:
                a_i = 0

            b_i = float('inf')
            for label in unique_labels:
                if label != current_cluster:
                    other_cluster_mask = y == label
                    if np.sum(other_cluster_mask) > 0:
                        cluster_distances = np.mean(distances[i, other_cluster_mask])
                        b_i = min(b_i, cluster_distances)
                    
            if a_i == 0 and b_i == float('inf'):
                silhouette_scores[i] = 0
            else:
                silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

        return np.average(silhouette_scores)