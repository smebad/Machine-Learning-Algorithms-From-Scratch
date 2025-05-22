# Logistic Regression From Scratch
import numpy as np


def sigmoid(x): # activation function
  return 1/(1 + np.exp(-x)) # sigmoid function

class LogisticRegression():

  # Logistic Regression Classifier
  def __init__(self, lr = 0.001, n_iters = 1000):
    self.lr = lr
    self.n_iters = n_iters
    self.weights = None
    self.bias = None

  # fit method to train the model
  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    # Gradient Descent loop
    for _ in range(self.n_iters):
      l_pred = np.dot(X, self.weights) + self.bias
      predictions = sigmoid(l_pred)

      # Gradient Descent
      dw = (1/n_samples) * np.dot(X.T, (predictions - y))
      db = (1/n_samples) * np.sum(predictions - y)

      # Update weights and bias
      self.weights = self.weights - self.lr * dw
      self.bias = self.bias - self.lr * db 

  # predict method to make predictions
  def predict(self, X):
    l_pred = np.dot(X, self.weights) + self.bias
    y_pred = sigmoid(l_pred)
    class_pred = [0 if y <= 0.5 else 1 for y in y_pred] # thresholding
    # class_pred = np.where(y_pred > 0.5, 1, 0) # thresholding
    return class_pred
    