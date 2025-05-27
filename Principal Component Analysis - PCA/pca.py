# Implementing PCA (Principal Component Analysis) from scratch
import numpy as np

class PCA:

  def __init__(self, n_components): # Initialize PCA with the number of components
    self.n_components = n_components
    self.components_ = None
    self.mean = None

  def fit(self, X): # Fit the PCA model to the data

    # Centering the mean of the data
    self.mean = np.mean(X, axis=0) # Calculate the mean of each feature
    X = X - self.mean # Center the data by subtracting the mean

    # Calculating the covariance matrix
    covariance_matrix = np.cov(X.T) # Calculate the covariance matrix of the centered data

    # Calculating the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix) # Calculate eigenvalues and eigenvectors of the covariance matrix

    # Tranposing eigenvectors to match the shape of the data
    eigenvectors = eigenvectors.T

    # Sorting the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1] # Get indices that would sort the eigenvalues in descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[sorted_indices]

    # Selecting the top n_components eigenvectors
    self.components_ = eigenvectors[:self.n_components]

  def transform(self, X): # Transform the data to the new PCA space
    
    X = X - self.mean # Center the data by subtracting the mean
    return np.dot(X, self.components_.T) # Project the data onto the PCA components