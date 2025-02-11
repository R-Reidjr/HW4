# write your silhouette score unit tests here

import pytest
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette
from cluster.utils import make_clusters



data, y_true = make_clusters(n=1000, m=2, k=6, seed=151) 

# Fit custom KMeans model
kmeans = KMeans(k=6, tol=1e-6, max_iter=100)
kmeans.fit(data)

# Get cluster labels from KMeans implementation
y_pred = kmeans.predict(data)

# Step 3: Compute silhouette score for KMeans implementation
silhouette = Silhouette()
model_silhouette = silhouette.score(X=data, y=y_pred)

# Compute silhouette score from sklearn
sklearn_silhouette = silhouette_score(data, y_true)

# Step 5: Calculate the percent error
percent_error = abs(model_silhouette - sklearn_silhouette) / sklearn_silhouette * 100

print(f"Your silhouette score: {model_silhouette}")
print(f"Sklearn silhouette score: {sklearn_silhouette}")
print(f"Percent error: {percent_error}%")
