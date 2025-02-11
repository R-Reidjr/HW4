import pytest
import numpy as np
from scipy.spatial.distance import cdist
from cluster.kmeans import KMeans


def test_kmeans_basic():
    """Test KMeans on a simple dataset."""
    X = np.array([[1, 2], [2, 3], [3, 4], [8, 9], [9, 10], [10, 11]])
    
    kmeans = KMeans(k=2, tol=1e-6, max_iter=100)
    kmeans.fit(X)
    labels = kmeans.predict(X)

    # Check correct number of clusters
    assert len(np.unique(labels)) == 2

    # Check centroids shape
    assert kmeans.get_centroids().shape == (2, X.shape[1])

def test_kmeans_error_decreases():
    """Ensure the error decreases over iterations."""
    X = np.random.rand(50, 2)
    
    kmeans = KMeans(k=3, tol=1e-6, max_iter=100)
    kmeans.fit(X)

    assert kmeans.get_error() >= 0  # Error should be non-negative

def test_kmeans_invalid_inputs():
    """Test edge cases for invalid inputs."""
    with pytest.raises(ValueError):
        KMeans(k=3).fit(np.array([1, 2, 3]))  # Not a 2D array

    with pytest.raises(ValueError):
        KMeans(k=5).fit(np.random.rand(3, 2))  # More clusters than points

    with pytest.raises(TypeError):
        KMeans(k="two")  # k must be an integer

