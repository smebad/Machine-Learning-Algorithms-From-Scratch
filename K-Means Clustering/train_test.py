from knn import KMeans
import numpy as np
from sklearn.datasets import make_blobs

np.random.seed(42) # for reproducibility
# Generate synthetic data
X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
print(X.shape)

# X is a 2D numpy array with shape (n_samples, n_features)
clusters = len(np.unique(y))
print(clusters)

# Create an instance of KMeans with the number of clusters
k = KMeans(K=clusters, max_iters=150, plot_steps=True)
y_pred = k.predict(X)

k.plot()