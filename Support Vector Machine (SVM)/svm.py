# Support Vector Machine (SVM) from scratch
import numpy as np

class SVM:
  
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate # learning rate
        self.lambda_param = lambda_param # regularization parameter
        self.n_iters = n_iters # number of iterations
        # weights and bias
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape # number of samples and features

        y_ = np.where(y <= 0, -1, 1) # convert labels to -1 and 1

        # initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0
        
        # gradient descent
        for _ in range(self.n_iters): # iterate over the number of iterations
            for idx, x_i in enumerate(X): # iterate over each sample
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1 # check the condition for SVM
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w) # if the condition is satisfied, update weights
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])) # if the condition is not satisfied, update weights and bias
                    self.b -= self.lr * y_[idx] # update bias


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b # calculate the approximation
        return np.sign(approx) # return the sign of the approximation, which gives the predicted labels