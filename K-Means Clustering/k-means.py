# K-Means Clustering from scratch

import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2): # Euclidean distance
    return np.sqrt(np.sum((x1-x2)**2)) # distance formula

class KMeans: 

    def __init__(self, K=5, max_iters=100, plot_steps=False): # K is the number of clusters
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)] # each cluster is a list of sample indices

        # the centers (mean vector) for each cluster
        self.centroids = [] # list of centroids, each centroid is a vector


    def predict(self, X): # X is the input data, a 2D numpy array
        self.X = X
        self.n_samples, self.n_features = X.shape # number of samples and features
 
        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False) # randomly select K samples from the dataset
        self.centroids = [self.X[idx] for idx in random_sample_idxs] # initialize centroids with random samples

        # optimize clusters
        for _ in range(self.max_iters): # iterate for a maximum number of iterations
            self.clusters = self._create_clusters(self.centroids) # assign samples to closest centroids (create clusters)

            if self.plot_steps: # if plotting is enabled
                self.plot() # plot the current state of clusters

            # calculate new centroids from the clusters
            centroids_old = self.centroids # store old centroids for convergence check
            self.centroids = self._get_centroids(self.clusters) # calculate new centroids as the mean of the clusters

            if self._is_converged(centroids_old, self.centroids): # check if centroids have converged
                break

            if self.plot_steps:
                self.plot()

        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters) # return labels for each sample indicating which cluster it belongs to


    def _get_cluster_labels(self, clusters): # clusters is a list of lists, where each inner list contains indices of samples in that cluster
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters): # iterate over each cluster
            for sample_idx in cluster: # iterate over each sample in the cluster
                labels[sample_idx] = cluster_idx # assign the cluster index as the label for the sample

        return labels # return the labels for all samples, where each label corresponds to the cluster index


    def _create_clusters(self, centroids): # centroids is a list of current centroids
        # assign the samples to the closest centroids
        clusters = [[] for _ in range(self.K)] # initialize empty clusters for each centroid
        for idx, sample in enumerate(self.X): # iterate over each sample in the dataset
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx) # add the sample index to the corresponding cluster
        return clusters # return the clusters, where each cluster is a list of sample indices

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids] # calculate distances from the sample to each centroid
        closest_idx = np.argmin(distances) # find the index of the closest centroid (the one with the minimum distance)
        return closest_idx # return the index of the closest centroid to the sample


    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features)) # initialize centroids as a 2D array with shape (K, n_features)
        for cluster_idx, cluster in enumerate(clusters): # iterate over each cluster
            cluster_mean = np.mean(self.X[cluster], axis=0) # calculate the mean of the samples in the cluster
            centroids[cluster_idx] = cluster_mean # assign the mean to the corresponding centroid
        return centroids # return the updated centroids

    def _is_converged(self, centroids_old, centroids): # check if the centroids have converged
        # distances between old and new centroids, for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0 # return True if the sum of distances is zero, indicating no change in centroids

    def plot(self): # plot the clusters and centroids
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

