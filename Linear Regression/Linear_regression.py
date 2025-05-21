import numpy as np

class LinearRegression:
  
  def __init__(self, lr = 0.001, n_iters = 1000):
    self.lr = lr # learning rate
    self.n_iters = n_iters # number of iterations
    self.weights = None 
    self.bias = None

  def fit(self, X, y):
    n_samples, n_features = X.shape # number of samples and number of features    
    self.weights = np.zeros(n_features) # initialize weights
    self.bias = 0 # initialize bias

    # Gradient descent
    for _ in range(self.n_iters):
      # Dot product of X and weights
      y_pred = np.dot(X, self.weights) + self.bias  # y_pred = X * w + b from the equation

      # Gradient calculation
      dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
      db = (1/n_samples) * np.sum(y_pred - y)

      # Updating weights and bias
      self.weights = self.weights - self.lr * dw
      self.bias = self.bias - self.lr * db


  def predict(self, X):
    # Predicting the output
    y_pred = np.dot(X, self.weights) + self.bias
    return y_pred
  